#!/usr/bin/env python3
"""
AIVA 企業級AI組件管理器
解決Windows編碼問題、環境配置和組件協調

修復要點：
1. Windows CP950編碼兼容
2. 完整環境變數配置 
3. 穩定的子進程管理
4. 生產級錯誤處理
5. 組件健康監控
"""

import os
import sys
import json
import time
import signal
import locale
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 設置編碼兼容性
def setup_encoding_compatibility():
    """設置Windows編碼兼容性"""
    # 獲取系統首選編碼
    system_encoding = locale.getpreferredencoding()
    
    # 設置環境變數確保編碼一致性
    os.environ.setdefault('PYTHONIOENCODING', system_encoding)
    os.environ.setdefault('PYTHONUTF8', '0')  # 不強制UTF-8
    
    return system_encoding

class ComponentStatus(Enum):
    """組件狀態枚舉"""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    ERROR = "error"
    RESTARTING = "restarting"

@dataclass
class ComponentInfo:
    """組件資訊類"""
    name: str
    script_path: str
    args: List[str]
    status: ComponentStatus
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None

class EnterpriseAIManager:
    """企業級AI組件管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.components: Dict[str, ComponentInfo] = {}
        self.shutdown_requested = False
        self.system_encoding = setup_encoding_compatibility()
        
        # 設置日誌
        self.setup_logging()
        
        # 設置完整環境
        self.setup_complete_environment()
        
        # 設置信號處理
        self.setup_signal_handlers()
        
        self.logger.info("🚀 AIVA 企業級AI組件管理器初始化完成")
        self.logger.info(f"📋 系統編碼: {self.system_encoding}")
    
    def setup_logging(self):
        """設置企業級日誌系統"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"enterprise_ai_manager_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 使用系統編碼寫入日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding=self.system_encoding),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("EnterpriseAIManager")
    
    def setup_complete_environment(self):
        """設置完整的環境變數"""
        # 基礎環境設置
        env_vars = {
            'ENVIRONMENT': 'production', 
            'LOG_LEVEL': 'INFO',
            
            # 研發階段直接使用預設值
            'DATABASE_URL': 'postgresql://postgres:postgres@localhost:5432/aiva_db',
            'RABBITMQ_URL': 'amqp://guest:guest@localhost:5672/',
            
            # Python路徑
            'PYTHONPATH': str(self.project_root),
            'PYTHONIOENCODING': self.system_encoding,
        }
        
        # 構建RabbitMQ URL
        rabbitmq_user = env_vars['RABBITMQ_USER']
        rabbitmq_pass = env_vars['RABBITMQ_PASSWORD'] 
        rabbitmq_host = env_vars['RABBITMQ_HOST']
        rabbitmq_port = env_vars['RABBITMQ_PORT']
        
        env_vars['RABBITMQ_URL'] = f"amqp://{rabbitmq_user}:{rabbitmq_pass}@{rabbitmq_host}:{rabbitmq_port}/"
        
        # 設置環境變數
        for key, value in env_vars.items():
            os.environ[key] = value
        
        self.logger.info("✅ 完整環境變數配置完成")
    
    def setup_signal_handlers(self):
        """設置信號處理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 收到信號 {signum}，開始優雅關閉...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_component_process(self, component: ComponentInfo) -> Optional[subprocess.Popen]:
        """創建組件進程（使用編碼兼容方法）"""
        try:
            cmd = [sys.executable, component.script_path] + component.args
            
            # Windows編碼兼容的進程創建
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=self.system_encoding,  # 使用系統編碼
                errors='replace',  # 替換無法解碼的字符
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                env=os.environ.copy()  # 傳遞完整環境
            )
            
            return process
            
        except Exception as e:
            self.logger.error(f"❌ 創建進程失敗: {e}")
            return None
    
    def start_component(self, name: str, script_path: str, args: List[str] = None) -> bool:
        """啟動組件"""
        if args is None:
            args = []
        
        # 檢查腳本是否存在
        script_file = self.project_root / script_path
        if not script_file.exists():
            self.logger.error(f"❌ 腳本不存在: {script_path}")
            return False
        
        # 檢查是否已在運行
        if name in self.components and self.components[name].status == ComponentStatus.RUNNING:
            self.logger.warning(f"⚠️ 組件 {name} 已在運行")
            return True
        
        self.logger.info(f"🚀 啟動組件: {name}")
        
        # 創建組件資訊
        component = ComponentInfo(
            name=name,
            script_path=script_path,
            args=args,
            status=ComponentStatus.STARTING
        )
        
        # 創建進程
        process = self.create_component_process(component)
        if not process:
            component.status = ComponentStatus.ERROR
            component.last_error = "進程創建失敗"
            self.components[name] = component
            return False
        
        # 更新組件資訊
        component.process = process
        component.pid = process.pid
        component.start_time = datetime.now()
        
        # 等待啟動確認
        time.sleep(2)
        
        if process.poll() is None:
            component.status = ComponentStatus.RUNNING
            self.components[name] = component
            self.logger.info(f"✅ 組件 {name} 啟動成功 (PID: {process.pid})")
            return True
        else:
            # 獲取錯誤資訊（使用編碼兼容方法）
            try:
                stdout, stderr = process.communicate(timeout=1)
                error_msg = f"啟動失敗: {stderr[:200]}..." if stderr else "進程意外退出"
            except:
                error_msg = "啟動失敗且無法獲取錯誤資訊"
            
            component.status = ComponentStatus.ERROR
            component.last_error = error_msg
            self.components[name] = component
            self.logger.error(f"❌ 組件 {name} 啟動失敗: {error_msg}")
            return False
    
    def stop_component(self, name: str) -> bool:
        """停止組件"""
        if name not in self.components:
            self.logger.warning(f"⚠️ 組件 {name} 不存在")
            return False
        
        component = self.components[name]
        if not component.process or component.status == ComponentStatus.STOPPED:
            return True
        
        try:
            self.logger.info(f"🛑 停止組件: {name}")
            
            # 嘗試優雅終止
            component.process.terminate()
            
            # 等待終止
            try:
                component.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 強制終止
                component.process.kill()
                component.process.wait()
            
            component.status = ComponentStatus.STOPPED
            component.process = None
            component.pid = None
            
            self.logger.info(f"✅ 組件 {name} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 停止組件 {name} 失敗: {e}")
            return False
    
    def check_component_health(self, name: str) -> bool:
        """檢查組件健康狀態"""
        if name not in self.components:
            return False
        
        component = self.components[name]
        if not component.process:
            return False
        
        # 檢查進程是否存在
        if component.process.poll() is None:
            return True
        else:
            # 進程已退出，更新狀態
            component.status = ComponentStatus.ERROR
            component.last_error = f"進程意外退出 (返回碼: {component.process.returncode})"
            return False
    
    def restart_component(self, name: str) -> bool:
        """重啟組件"""
        if name not in self.components:
            return False
        
        component = self.components[name]
        self.logger.info(f"🔄 重啟組件: {name}")
        
        # 停止現有進程
        self.stop_component(name)
        
        # 等待一會
        time.sleep(2)
        
        # 重新啟動
        component.restart_count += 1
        return self.start_component(name, component.script_path, component.args)
    
    def start_basic_components(self):
        """啟動基本AI組件"""
        # 定義基本組件（已知能工作的）
        basic_components = [
            # 暫時只啟動已驗證存在的組件
            ("core_service", "aiva_launcher.py", ["--mode", "core_only"]),
        ]
        
        # AI組件需要特殊處理（因為有依賴問題）
        ai_components = [
            # 這些組件需要修復後才能啟動
            # ("autonomous_testing", "ai_autonomous_testing_loop.py", []),
            # ("system_explorer", "ai_system_explorer_v3.py", ["--continuous"]), 
            # ("functionality_validator", "ai_functionality_validator.py", [])
        ]
        
        # 先啟動基本組件
        for name, script, args in basic_components:
            self.start_component(name, script, args)
        
        # AI組件需要等基本服務啟動後再嘗試
        # self.logger.info("⏳ AI組件需要修復依賴問題後才能啟動")
    
    def monitor_components(self):
        """監控組件狀態"""
        while not self.shutdown_requested:
            try:
                for name, component in self.components.items():
                    if component.status == ComponentStatus.RUNNING:
                        if not self.check_component_health(name):
                            self.logger.warning(f"⚠️ 組件 {name} 健康檢查失敗，嘗試重啟")
                            if component.restart_count < 3:  # 最多重啟3次
                                self.restart_component(name) 
                            else:
                                self.logger.error(f"❌ 組件 {name} 重啟次數過多，停止重啟")
                                component.status = ComponentStatus.ERROR
                
                # 記錄狀態
                self.log_status_summary()
                
                # 等待30秒再次檢查
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"❌ 監控循環出錯: {e}")
                time.sleep(10)
    
    def log_status_summary(self):
        """記錄狀態摘要"""
        running_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.RUNNING)
        error_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.ERROR)
        
        self.logger.info(f"📊 狀態摘要: 運行中={running_count}, 錯誤={error_count}, 總數={len(self.components)}")
        
        for name, component in self.components.items():
            status_emoji = {
                ComponentStatus.RUNNING: "✅",
                ComponentStatus.ERROR: "❌", 
                ComponentStatus.STARTING: "🚀",
                ComponentStatus.STOPPED: "⏹️",
                ComponentStatus.RESTARTING: "🔄"
            }.get(component.status, "❓")
            
            if component.status == ComponentStatus.RUNNING and component.start_time:
                uptime = datetime.now() - component.start_time
                self.logger.info(f"   {status_emoji} {name}: PID={component.pid}, 運行時間={str(uptime).split('.')[0]}")
            elif component.status == ComponentStatus.ERROR:
                self.logger.info(f"   {status_emoji} {name}: 錯誤={component.last_error}")
            else:
                self.logger.info(f"   {status_emoji} {name}: {component.status.value}")
    
    def stop_all_components(self):
        """停止所有組件"""
        self.logger.info("🛑 停止所有組件...")
        
        for name in list(self.components.keys()):
            self.stop_component(name)
        
        self.logger.info("✅ 所有組件已停止")
    
    def run(self):
        """運行管理器主循環"""
        try:
            self.logger.info("🔄 開始企業級AI組件管理")
            
            # 啟動基本組件
            self.start_basic_components()
            
            # 啟動監控線程
            monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
            monitor_thread.start()
            
            # 主循環（等待關閉信號）
            while not self.shutdown_requested:
                time.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 收到鍵盤中斷信號")
        except Exception as e:
            self.logger.error(f"❌ 管理器運行失敗: {e}")
        finally:
            self.stop_all_components()
            self.logger.info("✅ 企業級AI組件管理器已關閉")

def main():
    """主函數"""
    print("🚀 AIVA 企業級AI組件管理器")
    print("=" * 60)
    print("✨ 特性:")
    print("  • Windows CP950編碼兼容")
    print("  • 完整環境變數配置")
    print("  • 企業級錯誤處理")
    print("  • 自動組件重啟")
    print("  • 實時健康監控")
    print("=" * 60)
    print("🛑 按 Ctrl+C 關閉")
    print()
    
    manager = EnterpriseAIManager()
    manager.run()

if __name__ == "__main__":
    main()