#!/usr/bin/env python3
"""
AIVA AI 組件協調器
專注於與核心服務協調，保持簡單穩定的運行狀態
"""

import os
import sys
import time
import signal
import subprocess
import threading
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import psutil

class AIComponentCoordinator:
    """AIVA AI 組件協調器 - 簡化版本專注於穩定運行"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.running_components = {}
        self.shutdown_requested = False
        self.core_service_pid = None
        
        # 設置日誌
        self.setup_logging()
        
        # 設置信號處理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🚀 AIVA AI 組件協調器初始化完成")
    
    def setup_logging(self):
        """設置日誌系統"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"aiva_coordinator_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("AICoordinator")
    
    def find_core_service(self) -> Optional[int]:
        """查找正在運行的AIVA核心服務"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'aiva_launcher.py' in ' '.join(cmdline):
                        if '--mode' in ' '.join(cmdline) and 'core' in ' '.join(cmdline):
                            self.logger.info(f"✅ 發現核心服務 PID: {proc.info['pid']}")
                            return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ 搜尋核心服務時出錯: {e}")
            return None
    
    def start_ai_component(self, component_name: str, script_path: str, 
                          args: List[str] = None) -> bool:
        """啟動AI組件"""
        try:
            if args is None:
                args = []
            
            cmd = [sys.executable, script_path] + args
            self.logger.info(f"🚀 啟動組件 {component_name}: {' '.join(cmd)}")
            
            # 使用簡單的Popen避免編碼問題
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.DEVNULL,  # 避免輸出處理
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # 短暫等待確認啟動
            time.sleep(2)
            
            if process.poll() is None:
                self.running_components[component_name] = {
                    'process': process,
                    'pid': process.pid,
                    'start_time': datetime.now(),
                    'script': script_path
                }
                self.logger.info(f"✅ 組件 {component_name} 啟動成功 (PID: {process.pid})")
                return True
            else:
                self.logger.error(f"❌ 組件 {component_name} 啟動失敗")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 啟動組件 {component_name} 時出錯: {e}")
            return False
    
    def check_component_health(self) -> Dict[str, bool]:
        """檢查組件健康狀態"""
        health_status = {}
        
        for name, info in self.running_components.items():
            try:
                process = info['process']
                if process.poll() is None:
                    # 檢查進程是否真的存在
                    if psutil.pid_exists(process.pid):
                        health_status[name] = True
                    else:
                        health_status[name] = False
                        self.logger.warning(f"⚠️ 組件 {name} 進程不存在")
                else:
                    health_status[name] = False
                    self.logger.warning(f"⚠️ 組件 {name} 已退出")
            except Exception as e:
                health_status[name] = False
                self.logger.error(f"❌ 檢查組件 {name} 健康狀態時出錯: {e}")
        
        return health_status
    
    def restart_component(self, component_name: str) -> bool:
        """重啟組件"""
        if component_name not in self.running_components:
            self.logger.warning(f"⚠️ 組件 {component_name} 不在運行列表中")
            return False
        
        info = self.running_components[component_name]
        script_path = info['script']
        
        # 停止現有進程
        self.stop_component(component_name)
        
        # 短暫等待
        time.sleep(1)
        
        # 重新啟動
        return self.start_ai_component(component_name, script_path)
    
    def stop_component(self, component_name: str):
        """停止組件"""
        if component_name not in self.running_components:
            return
        
        try:
            process = self.running_components[component_name]['process']
            if process.poll() is None:
                process.terminate()
                # 等待終止
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
            del self.running_components[component_name]
            self.logger.info(f"✅ 組件 {component_name} 已停止")
            
        except Exception as e:
            self.logger.error(f"❌ 停止組件 {component_name} 時出錯: {e}")
    
    def stop_all_components(self):
        """停止所有組件"""
        self.logger.info("🛑 停止所有AI組件...")
        
        for component_name in list(self.running_components.keys()):
            self.stop_component(component_name)
    
    def signal_handler(self, signum, frame):
        """信號處理器"""
        self.logger.info(f"🛑 收到停止信號 {signum}，開始優雅關閉...")
        self.shutdown_requested = True
    
    def run_coordination_loop(self):
        """運行協調循環"""
        self.logger.info("🔄 開始AI組件協調循環")
        
        # 查找核心服務
        self.core_service_pid = self.find_core_service()
        if not self.core_service_pid:
            self.logger.warning("⚠️ 未找到AIVA核心服務，建議先啟動核心服務")
        
        # 啟動基本AI組件
        self.start_basic_components()
        
        # 主循環
        check_interval = 30  # 30秒檢查一次
        last_health_check = time.time()
        
        while not self.shutdown_requested:
            try:
                current_time = time.time()
                
                # 定期健康檢查
                if current_time - last_health_check >= check_interval:
                    self.logger.info("📊 執行健康檢查...")
                    health_status = self.check_component_health()
                    
                    # 重啟失敗的組件
                    for component_name, is_healthy in health_status.items():
                        if not is_healthy:
                            self.logger.warning(f"🔄 重啟不健康的組件: {component_name}")
                            self.restart_component(component_name)
                    
                    # 檢查核心服務狀態
                    if self.core_service_pid and not psutil.pid_exists(self.core_service_pid):
                        self.logger.warning("⚠️ 核心服務已停止，建議重啟")
                        self.core_service_pid = self.find_core_service()
                    
                    self.log_status_summary()
                    last_health_check = current_time
                
                time.sleep(5)  # 5秒休眠
                
            except KeyboardInterrupt:
                self.logger.info("🛑 收到鍵盤中斷，開始關閉...")
                break
            except Exception as e:
                self.logger.error(f"❌ 協調循環出錯: {e}")
                time.sleep(10)
        
        # 清理
        self.stop_all_components()
        self.logger.info("✅ AI組件協調器已關閉")
    
    def start_basic_components(self):
        """啟動基本AI組件"""
        basic_components = [
            ("autonomous_testing", "ai_autonomous_testing_loop.py"),
            ("system_explorer", "ai_system_explorer_v3.py", ["--continuous"]),
            ("functionality_validator", "ai_functionality_validator.py")
        ]
        
        for component_info in basic_components:
            component_name = component_info[0]
            script_path = component_info[1]
            args = component_info[2] if len(component_info) > 2 else []
            
            # 檢查腳本是否存在
            script_file = self.project_root / script_path
            if script_file.exists():
                self.start_ai_component(component_name, script_path, args)
            else:
                self.logger.warning(f"⚠️ 組件腳本不存在: {script_path}")
    
    def log_status_summary(self):
        """記錄狀態摘要"""
        running_count = len(self.running_components)
        core_status = "運行中" if self.core_service_pid and psutil.pid_exists(self.core_service_pid) else "未知"
        
        self.logger.info(f"📊 狀態摘要: 核心服務={core_status}, AI組件={running_count}個運行中")
        
        for name, info in self.running_components.items():
            uptime = datetime.now() - info['start_time']
            self.logger.info(f"   • {name}: PID={info['pid']}, 運行時間={str(uptime).split('.')[0]}")

def main():
    """主函數"""
    print("🚀 AIVA AI 組件協調器")
    print("=" * 50)
    print("功能:")
    print("  • 與AIVA核心服務協調")
    print("  • 管理AI組件持續運作")
    print("  • 自動重啟失敗的組件")
    print("  • 簡化設計確保穩定性")
    print("=" * 50)
    print("🛑 按 Ctrl+C 關閉")
    print()
    
    coordinator = AIComponentCoordinator()
    try:
        coordinator.run_coordination_loop()
    except Exception as e:
        coordinator.logger.error(f"❌ 協調器運行失敗: {e}")
        coordinator.stop_all_components()

if __name__ == "__main__":
    main()