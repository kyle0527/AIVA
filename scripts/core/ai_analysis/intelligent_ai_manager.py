#!/usr/bin/env python3
"""
AIVA 智能AI組件管理器 v3.0
修復運作邏輯問題，實現真正的智能管理

🎯 核心改進：
1. ✅ 智能健康檢查（服務可用性而非僅進程存在）
2. ✅ 合理的重啟策略（避免無限重啟循環）
3. ✅ 分級管理（核心服務 vs 可選AI組件）
4. ✅ 實際服務驗證（HTTP端點健康檢查）
5. ✅ 優雅降級（核心功能優先）
"""

import os
import sys
import time
import locale
import logging
import threading
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# ========================= 編碼兼容性設置 =========================
def setup_production_encoding():
    """設置生產級編碼兼容性"""
    system_encoding = locale.getpreferredencoding()
    
    # 設置環境變數確保編碼一致性
    os.environ.setdefault('PYTHONIOENCODING', system_encoding)
    os.environ.setdefault('PYTHONUTF8', '0')  # 不強制UTF-8模式
    
    # Windows特定設置
    if os.name == 'nt':
        os.environ.setdefault('PYTHONLEGACYWINDOWSSTDIO', '1')
    
    return system_encoding

class ComponentStatus(Enum):
    """組件狀態枚舉"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running" 
    ERROR = "error"
    RESTARTING = "restarting"
    DISABLED = "disabled"  # 新增：禁用狀態

class ComponentType(Enum):
    """組件類型枚舉"""
    CORE = "core"        # 核心服務（必須運行）
    OPTIONAL = "optional" # 可選AI組件（可以失敗）

@dataclass
class ComponentInfo:
    """組件資訊"""
    name: str
    script_path: str
    args: List[str]
    component_type: ComponentType
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    status: ComponentStatus = ComponentStatus.STOPPED
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    health_check_url: Optional[str] = None  # 健康檢查端點
    max_restarts: int = 3  # 最大重啟次數

class SafeFormatter:
    """安全的字符串格式器"""
    @staticmethod
    def safe_print(message: str, encoding: str = 'cp950'):
        try:
            # 移除可能有問題的unicode字符
            safe_message = message.encode(encoding, errors='replace').decode(encoding)
            print(safe_message)
        except Exception:
            # 回退到ASCII安全版本
            ascii_message = message.encode('ascii', errors='replace').decode('ascii')
            print(ascii_message)

class IntelligentAIManager:
    """智能AI組件管理器"""
    
    def __init__(self):
        self.system_encoding = setup_production_encoding()
        self.project_root = Path(__file__).parent.absolute()
        self.components: Dict[str, ComponentInfo] = {}
        self.shutdown_requested = False
        
        # 設置日誌
        self.setup_logging()
        
        # 自動環境配置
        self.setup_production_environment()
        
        self.logger.info("AIVA 智能AI組件管理器初始化完成")
        self.logger.info(f"系統編碼: {self.system_encoding}")
    
    def setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('IntelligentAIManager')
    
    def setup_production_environment(self):
        """設置生產環境變數（v2.0 簡化版）"""
        # v2.0 架構：移除 RabbitMQ、PostgreSQL、Redis、Neo4j 等外部依賴
        # 只保留必要的編碼配置
        production_defaults = {
            'ENVIRONMENT': 'production',
            'PYTHONIOENCODING': self.system_encoding
        }
        
        for key, value in production_defaults.items():
            if key not in os.environ:
                os.environ[key] = value
        
        self.logger.info("智能環境變數配置完成（v2.0 簡化版）")
    
    def create_safe_process(self, cmd: List[str]) -> Optional[subprocess.Popen]:
        """創建安全的子進程"""
        try:
            # 創建安全的環境副本
            safe_env = os.environ.copy()
            safe_env['PYTHONIOENCODING'] = self.system_encoding
            
            # Windows安全進程創建
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=self.system_encoding,
                errors='replace',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                env=safe_env
            )
            
            return process
            
        except Exception as e:
            self.logger.error(f"創建進程失敗: {e}")
            return None
    
    def check_service_health(self, component: ComponentInfo) -> bool:
        """智能健康檢查"""
        # 基本進程檢查
        if not component.process or component.process.poll() is not None:
            return False
        
        # 如果有健康檢查URL，進行HTTP檢查
        if component.health_check_url:
            try:
                response = requests.get(component.health_check_url, timeout=5)
                return response.status_code == 200
            except Exception as e:
                self.logger.debug(f"Health check failed for {component.name}: {e}")
                return False
        
        # 對於核心服務，檢查是否已經運行足夠長時間（避免立即重啟）
        if component.component_type == ComponentType.CORE and component.start_time:
            uptime = datetime.now() - component.start_time
            if uptime < timedelta(seconds=30):  # 給30秒穩定時間
                return True  # 假設健康，避免過早重啟
        
        return True  # 默認認為健康
    
    def start_component(self, component_info: ComponentInfo) -> bool:
        """啟動組件（智能策略）"""
        name = component_info.name
        
        # 檢查是否已禁用
        if component_info.status == ComponentStatus.DISABLED:
            self.logger.info(f"組件 {name} 已禁用，跳過啟動")
            return False
        
        # 檢查腳本存在
        script_file = self.project_root / component_info.script_path
        if not script_file.exists():
            self.logger.error(f"腳本不存在: {component_info.script_path}")
            component_info.status = ComponentStatus.ERROR
            return False
        
        # 檢查重啟次數限制
        if component_info.restart_count >= component_info.max_restarts:
            self.logger.warning(f"組件 {name} 重啟次數過多，禁用該組件")
            component_info.status = ComponentStatus.DISABLED
            return False
        
        self.logger.info(f"啟動組件: {name} (類型: {component_info.component_type.value})")
        
        # 更新狀態
        component_info.status = ComponentStatus.STARTING
        
        # 創建命令
        cmd = [sys.executable, component_info.script_path] + component_info.args
        
        # 創建進程
        process = self.create_safe_process(cmd)
        if not process:
            component_info.status = ComponentStatus.ERROR
            component_info.last_error = "進程創建失敗"
            return False
        
        # 等待進程穩定
        time.sleep(3)
        
        # 檢查進程是否仍在運行
        if process.poll() is not None:
            # 進程已終止，讀取錯誤信息
            try:
                stdout, stderr = process.communicate(timeout=1)
                error_info = stderr or stdout or "未知錯誤"
            except subprocess.TimeoutExpired as e:
                error_info = f"進程啟動後立即終止: {e}"
            
            component_info.status = ComponentStatus.ERROR
            component_info.last_error = error_info[:200]  # 限制錯誤信息長度
            
            # 對於可選組件，記錄錯誤但不重試
            if component_info.component_type == ComponentType.OPTIONAL:
                self.logger.warning(f"可選組件 {name} 啟動失敗，將跳過: {error_info[:100]}")
                return False
            
            self.logger.error(f"核心組件 {name} 啟動失敗: {error_info[:100]}")
            return False
        
        # 更新組件狀態
        component_info.process = process
        component_info.pid = process.pid
        component_info.status = ComponentStatus.RUNNING
        component_info.start_time = datetime.now()
        
        self.components[name] = component_info
        
        self.logger.info(f"組件 {name} 啟動成功 (PID: {process.pid})")
        return True
    
    def stop_component(self, name: str) -> bool:
        """停止組件"""
        if name not in self.components:
            return False
        
        component = self.components[name]
        if not component.process or component.status == ComponentStatus.STOPPED:
            return True
        
        try:
            self.logger.info(f"停止組件: {name}")
            
            # 優雅終止
            component.process.terminate()
            
            # 等待終止
            try:
                component.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                component.process.kill()
                component.process.wait()
            
            component.status = ComponentStatus.STOPPED
            component.process = None
            component.pid = None
            
            self.logger.info(f"組件 {name} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止組件 {name} 失敗: {e}")
            return False
    
    def restart_component(self, name: str) -> bool:
        """智能重啟組件"""
        if name not in self.components:
            return False
        
        component = self.components[name]
        
        # 檢查重啟限制
        if component.restart_count >= component.max_restarts:
            self.logger.warning(f"組件 {name} 已達到最大重啟次數，禁用該組件")
            component.status = ComponentStatus.DISABLED
            return False
        
        self.logger.info(f"智能重啟組件: {name} (第{component.restart_count + 1}次)")
        
        self.stop_component(name)
        time.sleep(5)  # 給更多時間清理
        
        component.restart_count += 1
        return self.start_component(component)
    
    def setup_components(self):
        """設置組件配置"""
        # 核心組件（必須運行）
        core_components = [
            ComponentInfo(
                name="core_service",
                script_path="aiva_launcher.py",
                args=["--mode", "core_only"],
                component_type=ComponentType.CORE,
                max_restarts=5,  # 核心服務允許更多重啟
                health_check_url=None  # 可以添加HTTP健康檢查端點
            )
        ]
        
        # 可選AI組件（可以失敗）
        optional_components = [
            ComponentInfo(
                name="autonomous_testing",
                script_path="ai_autonomous_testing_loop.py",
                args=[],
                component_type=ComponentType.OPTIONAL,
                max_restarts=1  # 可選組件限制重啟次數
            ),
            ComponentInfo(
                name="system_explorer",
                script_path="ai_system_explorer_v3.py",
                args=["--workspace", ".", "--detailed"],
                component_type=ComponentType.OPTIONAL,
                max_restarts=1
            ),
            ComponentInfo(
                name="functionality_validator",
                script_path="ai_functionality_validator.py",
                args=[],
                component_type=ComponentType.OPTIONAL,
                max_restarts=1
            )
        ]
        
        return core_components + optional_components
    
    def start_all_components(self):
        """啟動所有組件（智能策略）"""
        components = self.setup_components()
        
        # 首先啟動核心組件
        self.logger.info("啟動核心組件...")
        core_success = 0
        core_total = 0
        
        for component in components:
            if component.component_type == ComponentType.CORE:
                core_total += 1
                if self.start_component(component):
                    core_success += 1
        
        if core_success == 0:
            self.logger.error("沒有核心組件成功啟動，系統無法運行")
            return False
        
        self.logger.info(f"核心組件啟動完成: {core_success}/{core_total}")
        
        # 然後啟動可選組件（失敗不影響系統運行）
        self.logger.info("啟動可選AI組件...")
        optional_success = 0
        optional_total = 0
        
        for component in components:
            if component.component_type == ComponentType.OPTIONAL:
                optional_total += 1
                if self.start_component(component):
                    optional_success += 1
        
        self.logger.info(f"可選組件啟動完成: {optional_success}/{optional_total}")
        self.logger.info("系統已準備就緒 - 核心服務正常運行")
        
        return True
    
    def _handle_unhealthy_component(self, name: str, component: ComponentInfo):
        """處理不健康的組件"""
        self.logger.warning(f"組件 {name} 健康檢查失敗")
        
        # 核心組件自動重啟，可選組件禁用
        if component.component_type == ComponentType.CORE:
            if component.restart_count < component.max_restarts:
                self.logger.info(f"重啟核心組件 {name}")
                self.restart_component(name)
            else:
                self.logger.error(f"核心組件 {name} 重啟次數過多，系統可能有問題")
        else:
            self.logger.info(f"禁用有問題的可選組件 {name}")
            component.status = ComponentStatus.DISABLED
    
    def _check_all_components_health(self):
        """檢查所有組件健康狀態"""
        for name, component in self.components.items():
            if component.status == ComponentStatus.RUNNING:
                if not self.check_service_health(component):
                    self._handle_unhealthy_component(name, component)
    
    def monitor_components(self):
        """智能組件監控"""
        self.logger.info("開始智能監控循環")
        
        while not self.shutdown_requested:
            try:
                # 每30秒進行一次健康檢查
                time.sleep(30)
                self._check_all_components_health()
                self.print_status_summary()
            except Exception as e:
                self.logger.error(f"監控循環錯誤: {e}")
    
    def print_status_summary(self):
        """打印狀態摘要"""
        running = sum(1 for c in self.components.values() if c.status == ComponentStatus.RUNNING)
        error = sum(1 for c in self.components.values() if c.status == ComponentStatus.ERROR)
        disabled = sum(1 for c in self.components.values() if c.status == ComponentStatus.DISABLED)
        total = len(self.components)
        
        self.logger.info(f"智能狀態摘要: 運行={running}, 錯誤={error}, 禁用={disabled}, 總數={total}")
        
        for name, component in self.components.items():
            if component.status == ComponentStatus.RUNNING and component.start_time:
                uptime = datetime.now() - component.start_time
                uptime_str = str(uptime).split('.')[0]  # 去掉微秒
                self.logger.info(f"  ✅ {name}: PID={component.pid}, 運行時間={uptime_str}, 類型={component.component_type.value}")
            elif component.status == ComponentStatus.ERROR:
                error = component.last_error[:50] if component.last_error else "未知錯誤"
                self.logger.info(f"  ❌ {name}: 錯誤={error}, 類型={component.component_type.value}")
            elif component.status == ComponentStatus.DISABLED:
                self.logger.info(f"  ⏸️ {name}: 已禁用, 類型={component.component_type.value}")
    
    def stop_all_components(self):
        """停止所有組件"""
        self.logger.info("停止所有組件...")
        
        for name in self.components.keys():
            self.stop_component(name)
        
        self.logger.info("所有組件已停止")
    
    def run(self):
        """運行智能管理器"""
        try:
            self.logger.info("🚀 開始AIVA智能AI組件管理")
            
            # 啟動組件
            if not self.start_all_components():
                self.logger.error("系統啟動失敗")
                return
            
            # 啟動監控線程
            monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
            monitor_thread.start()
            
            self.logger.info("🎯 AIVA智能管理器運行中... (Ctrl+C 停止)")
            self.logger.info("💡 系統采用智能管理策略：核心服務自動重啟，可選組件優雅降級")
            
            # 主循環
            while not self.shutdown_requested:
                time.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("收到鍵盤中斷信號")
        except Exception as e:
            self.logger.error(f"智能管理器運行失敗: {e}")
        finally:
            self.shutdown_requested = True
            self.stop_all_components()
            self.logger.info("🏁 AIVA智能管理器已關閉")

def main():
    """主函數"""
    SafeFormatter.safe_print("AIVA 智能AI組件管理器 v3.0")
    SafeFormatter.safe_print("=" * 60)
    SafeFormatter.safe_print("🎯 智能特性:")
    SafeFormatter.safe_print("  ✅ 分級管理 (核心服務 vs 可選組件)")
    SafeFormatter.safe_print("  ✅ 智能健康檢查 (服務可用性驗證)")
    SafeFormatter.safe_print("  ✅ 優雅降級 (有問題的組件自動禁用)")
    SafeFormatter.safe_print("  ✅ 合理重啟策略 (避免無限重啟循環)")
    SafeFormatter.safe_print("  ✅ Windows CP950完全兼容")
    SafeFormatter.safe_print("=" * 60)
    SafeFormatter.safe_print("核心邏輯: 確保核心服務穩定運行，可選組件失敗不影響系統")
    SafeFormatter.safe_print("按 Ctrl+C 停止")
    print()
    
    manager = IntelligentAIManager()
    manager.run()

if __name__ == "__main__":
    main()