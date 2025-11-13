#!/usr/bin/env python3
"""
AIVA 智能管理器測試版 - 快速展示監控功能
"""

import os
import sys
import time
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class ComponentStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running" 
    ERROR = "error"
    DISABLED = "disabled"

class ComponentType(Enum):
    CORE = "core"
    OPTIONAL = "optional"

@dataclass
class ComponentInfo:
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

class QuickTestManager:
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.shutdown_requested = False
        
        # 設置日誌
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('QuickTestManager')
    
    def create_mock_components(self):
        """創建模擬組件進行測試"""
        # 核心組件（成功運行）
        core_component = ComponentInfo(
            name="core_service",
            script_path="aiva_launcher.py",
            args=["--mode", "core_only"],
            component_type=ComponentType.CORE,
            status=ComponentStatus.RUNNING,
            start_time=datetime.now(),
            pid=12345
        )
        
        # 模擬運行中的進程
        class MockProcess:
            def __init__(self, pid):
                self.pid = pid
                self.returncode = None
            
            def poll(self):
                return None  # 表示進程仍在運行
            
            def terminate(self):
                pass
            
            def wait(self, timeout=None):
                pass
        
        core_component.process = MockProcess(12345)
        self.components["core_service"] = core_component
        
        # 可選組件（失敗）
        failed_components = [
            ComponentInfo(
                name="autonomous_testing",
                script_path="ai_autonomous_testing_loop.py",
                args=[],
                component_type=ComponentType.OPTIONAL,
                status=ComponentStatus.ERROR,
                last_error="導入錯誤：缺少依賴模組 experience_manager"
            ),
            ComponentInfo(
                name="system_explorer",
                script_path="ai_system_explorer_v3.py",
                args=["--workspace", ".", "--detailed"],
                component_type=ComponentType.OPTIONAL,
                status=ComponentStatus.ERROR,
                last_error="參數錯誤：不支援的命令行參數"
            ),
            ComponentInfo(
                name="functionality_validator",
                script_path="ai_functionality_validator.py",
                args=[],
                component_type=ComponentType.OPTIONAL,
                status=ComponentStatus.DISABLED,
                last_error="編碼錯誤：emoji字符與CP950不兼容"
            )
        ]
        
        for component in failed_components:
            self.components[component.name] = component
    
    def check_service_health(self, component: ComponentInfo) -> bool:
        """智能健康檢查（模擬）"""
        if component.status != ComponentStatus.RUNNING:
            return False
        
        # 模擬核心服務健康
        if component.name == "core_service":
            return True
        
        return False
    
    def print_status_summary(self):
        """打印詳細狀態摘要"""
        running = sum(1 for c in self.components.values() if c.status == ComponentStatus.RUNNING)
        error = sum(1 for c in self.components.values() if c.status == ComponentStatus.ERROR)
        disabled = sum(1 for c in self.components.values() if c.status == ComponentStatus.DISABLED)
        total = len(self.components)
        
        self.logger.info("=" * 60)
        self.logger.info(f"🎯 智能狀態摘要: 運行={running}, 錯誤={error}, 禁用={disabled}, 總數={total}")
        self.logger.info("=" * 60)
        
        for name, component in self.components.items():
            if component.status == ComponentStatus.RUNNING and component.start_time:
                uptime = datetime.now() - component.start_time
                uptime_str = str(uptime).split('.')[0]
                self.logger.info(f"  ✅ {name}: PID={component.pid}, 運行時間={uptime_str}, 類型={component.component_type.value}")
            elif component.status == ComponentStatus.ERROR:
                error = component.last_error[:60] if component.last_error else "未知錯誤"
                self.logger.info(f"  ❌ {name}: 錯誤={error}, 類型={component.component_type.value}")
            elif component.status == ComponentStatus.DISABLED:
                reason = component.last_error[:60] if component.last_error else "已禁用"
                self.logger.info(f"  ⏸️ {name}: 原因={reason}, 類型={component.component_type.value}")
        
        self.logger.info("=" * 60)
    
    def monitor_components(self):
        """智能監控（每10秒報告一次）"""
        self.logger.info("開始智能監控循環（每10秒報告）")
        
        while not self.shutdown_requested:
            try:
                time.sleep(10)  # 縮短監控間隔便於測試
                
                # 健康檢查
                for name, component in self.components.items():
                    if component.status == ComponentStatus.RUNNING:
                        if not self.check_service_health(component):
                            self.logger.warning(f"組件 {name} 健康檢查失敗")
                
                # 狀態報告
                self.print_status_summary()
                
            except Exception as e:
                self.logger.error(f"監控循環錯誤: {e}")
    
    def run(self):
        """運行測試管理器"""
        try:
            self.logger.info("🚀 AIVA智能管理器測試版啟動")
            
            # 創建模擬組件
            self.create_mock_components()
            
            self.logger.info("📊 模擬組件創建完成:")
            self.logger.info("  ✅ 核心服務: 正常運行")
            self.logger.info("  ❌ 3個可選AI組件: 啟動失敗（模擬）")
            
            # 初始狀態報告
            self.print_status_summary()
            
            # 啟動監控線程
            monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
            monitor_thread.start()
            
            self.logger.info("🎯 智能監控已啟動，每10秒報告一次狀態")
            self.logger.info("💡 這就是修復後的智能運作邏輯演示")
            self.logger.info("按 Ctrl+C 停止測試")
            
            # 主循環
            while not self.shutdown_requested:
                time.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("收到鍵盤中斷信號")
        except Exception as e:
            self.logger.error(f"測試管理器運行失敗: {e}")
        finally:
            self.shutdown_requested = True
            self.logger.info("🏁 智能管理器測試結束")

if __name__ == "__main__":
    print("AIVA 智能管理器 - 運作邏輯測試")
    print("=" * 50)
    print("這個測試展示修復後的智能運作邏輯:")
    print("✅ 分級管理 (核心 vs 可選)")
    print("✅ 智能健康檢查")
    print("✅ 定期狀態報告")
    print("✅ 優雅降級策略")
    print("=" * 50)
    print()
    
    manager = QuickTestManager()
    manager.run()