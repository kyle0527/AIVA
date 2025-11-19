#!/usr/bin/env python3
"""
AIVA 持續運作 AI 組件管理器
基於單一事實來源(SOT)原則的持續運作改進方案

設計原則:
1. 單一事實來源(SOT) - 所有配置來自統一配置文件
2. 持續運作 - 啟動後持續運行直到手動關閉
3. 自動恢復 - 組件異常時自動重啟
4. 資源監控 - 實時監控系統資源使用
5. 智能調度 - 根據系統負載智能調整
6. 統一日誌 - 集中化日誌管理
"""

import os
import sys
import asyncio
import json
import time
import threading
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import subprocess

# 設置離線模式環境變數(遵循現有資源配置)
if not os.getenv("RABBITMQ_USER"):
    os.environ["RABBITMQ_USER"] = "admin"
    os.environ["RABBITMQ_PASSWORD"] = "password123"
    os.environ["RABBITMQ_HOST"] = "localhost"
    os.environ["RABBITMQ_PORT"] = "5672"
    os.environ["ENVIRONMENT"] = "continuous"

# 設置路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class ComponentStatus(Enum):
    """組件狀態枚舉"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    RESTARTING = "restarting"

@dataclass
class ComponentHealth:
    """組件健康狀態"""
    name: str
    status: ComponentStatus
    pid: Optional[int]
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    restart_count: int
    last_error: Optional[str]
    timestamp: datetime

@dataclass
class SystemMetrics:
    """系統指標"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    timestamp: datetime

class AIComponentManager:
    """AI 組件持續運作管理器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.is_running = False
        self.shutdown_requested = False
        
        # 設置日誌
        self.setup_logging()
        
        # 組件配置 - 基於SOT原則的統一定義
        self.components_config = self.load_sot_configuration()
        
        # 運行時狀態
        self.components: Dict[str, subprocess.Popen] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics: Optional[SystemMetrics] = None
        
        # 監控線程
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        # 統計信息
        self.start_time = datetime.now()
        self.total_restarts = 0
        self.error_count = 0
        
        # 設置信號處理
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """設置統一日誌系統"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 創建日誌文件路徑
        log_file = log_dir / f"aiva_continuous_manager_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 配置日誌格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("AIComponentManager")
        self.logger.info("🚀 AIVA 持續運作 AI 組件管理器初始化")
    
    def load_sot_configuration(self) -> Dict[str, Any]:
        """載入SOT配置 - 單一事實來源原則"""
        sot_config_file = self.project_root / "services" / "aiva_common" / "continuous_components_sot.json"
        
        # 預設配置(如果SOT文件不存在)
        default_config = {
            "ai_components": {
                "autonomous_testing": {
                    "command": [sys.executable, "AI_AUTONOMOUS_TESTING_LOOP.py"],
                    "cwd": str(self.project_root),
                    "enabled": True,
                    "restart_policy": "always",
                    "max_restarts": 5,
                    "restart_delay": 30,
                    "health_check_interval": 60,
                    "resource_limits": {
                        "max_cpu_percent": 80,
                        "max_memory_mb": 2048
                    }
                },
                "system_explorer": {
                    "command": [sys.executable, "ai_system_explorer_v3.py", "--continuous"],
                    "cwd": str(self.project_root),
                    "enabled": True,
                    "restart_policy": "on-failure",
                    "max_restarts": 3,
                    "restart_delay": 60,
                    "health_check_interval": 120,
                    "resource_limits": {
                        "max_cpu_percent": 60,
                        "max_memory_mb": 1024
                    }
                },
                "ai_security_monitor": {
                    "command": [sys.executable, "ai_security_test.py", "--monitor-mode"],
                    "cwd": str(self.project_root),
                    "enabled": True,
                    "restart_policy": "always",
                    "max_restarts": 10,
                    "restart_delay": 15,
                    "health_check_interval": 30,
                    "resource_limits": {
                        "max_cpu_percent": 70,
                        "max_memory_mb": 1536
                    }
                },
                "functionality_validator": {
                    "command": [sys.executable, "ai_functionality_validator.py", "--continuous"],
                    "cwd": str(self.project_root),
                    "enabled": True,
                    "restart_policy": "on-failure",
                    "max_restarts": 3,
                    "restart_delay": 45,
                    "health_check_interval": 90,
                    "resource_limits": {
                        "max_cpu_percent": 50,
                        "max_memory_mb": 768
                    }
                }
            },
            "system_settings": {
                "global_restart_delay": 10,
                "health_check_timeout": 30,
                "max_concurrent_restarts": 2,
                "system_resource_threshold": {
                    "cpu_percent": 90,
                    "memory_percent": 85,
                    "disk_percent": 95
                },
                "log_retention_days": 7,
                "metrics_collection_interval": 60
            }
        }
        
        # 嘗試讀取SOT配置文件
        if sot_config_file.exists():
            try:
                with open(sot_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"✅ 已載入SOT配置: {sot_config_file}")
                return config
            except Exception as e:
                self.logger.warning(f"⚠️ 讀取SOT配置失敗: {e}，使用預設配置")
        else:
            # 創建SOT配置文件
            sot_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sot_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📝 已創建SOT配置文件: {sot_config_file}")
        
        return default_config
    
    def setup_signal_handlers(self):
        """設置信號處理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 收到信號 {signum}，開始優雅關閉...")
            self.shutdown_requested = True
            self.stop_all_components()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_continuous_operation(self):
        """開始持續運作"""
        self.logger.info("🔄 開始 AIVA AI 組件持續運作管理")
        self.is_running = True
        
        # 啟動所有組件
        await self.start_all_components()
        
        # 啟動監控線程
        self.start_monitoring_threads()
        
        # 主循環
        await self.main_management_loop()
    
    async def start_all_components(self):
        """啟動所有已啟用的組件"""
        ai_components = self.components_config.get("ai_components", {})
        
        for component_name, config in ai_components.items():
            if config.get("enabled", False):
                await self.start_component(component_name, config)
            else:
                self.logger.info(f"⏭️ 組件 {component_name} 已停用，跳過啟動")
    
    async def start_component(self, component_name: str, config: Dict[str, Any]) -> bool:
        """啟動單個組件"""
        try:
            self.logger.info(f"🚀 啟動組件: {component_name}")
            
            # 檢查是否已在運行
            if component_name in self.components:
                process = self.components[component_name]
                if process.poll() is None:
                    self.logger.warning(f"⚠️ 組件 {component_name} 已在運行")
                    return True
            
            # 創建進程 - 修復編碼問題
            process = subprocess.Popen(
                config["command"],
                cwd=config["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'  # 處理編碼錯誤
            )
            
            # 記錄進程
            self.components[component_name] = process
            
            # 初始化健康狀態
            self.component_health[component_name] = ComponentHealth(
                name=component_name,
                status=ComponentStatus.STARTING,
                pid=process.pid,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                restart_count=0,
                last_error=None,
                timestamp=datetime.now()
            )
            
            # 等待啟動確認
            await asyncio.sleep(2)
            
            if process.poll() is None:
                self.component_health[component_name].status = ComponentStatus.RUNNING
                self.logger.info(f"✅ 組件 {component_name} 啟動成功 (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                error_msg = f"啟動失敗: {stderr}"
                self.component_health[component_name].status = ComponentStatus.ERROR
                self.component_health[component_name].last_error = error_msg
                self.logger.error(f"❌ 組件 {component_name} 啟動失敗: {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"啟動異常: {str(e)}"
            self.logger.error(f"❌ 組件 {component_name} 啟動異常: {error_msg}")
            if component_name in self.component_health:
                self.component_health[component_name].status = ComponentStatus.ERROR
                self.component_health[component_name].last_error = error_msg
            return False
    
    def start_monitoring_threads(self):
        """啟動監控線程"""
        # 組件健康監控線程
        self.monitor_thread = threading.Thread(
            target=self.component_monitor_loop,
            daemon=True,
            name="ComponentMonitor"
        )
        self.monitor_thread.start()
        
        # 系統指標收集線程
        self.metrics_thread = threading.Thread(
            target=self.metrics_collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self.metrics_thread.start()
        
        self.logger.info("📊 監控線程已啟動")
    
    def component_monitor_loop(self):
        """組件監控循環"""
        while self.is_running and not self.shutdown_requested:
            try:
                self.check_all_components_health()
                time.sleep(30)  # 每30秒檢查一次
            except Exception as e:
                self.logger.error(f"❌ 組件監控錯誤: {e}")
                time.sleep(60)  # 錯誤時延長檢查間隔
    
    def metrics_collection_loop(self):
        """系統指標收集循環"""
        while self.is_running and not self.shutdown_requested:
            try:
                self.collect_system_metrics()
                time.sleep(60)  # 每分鐘收集一次
            except Exception as e:
                self.logger.error(f"❌ 指標收集錯誤: {e}")
                time.sleep(120)  # 錯誤時延長收集間隔
    
    def check_all_components_health(self):
        """檢查所有組件健康狀態"""
        for component_name in list(self.components.keys()):
            self.check_component_health(component_name)
    
    def check_component_health(self, component_name: str):
        """檢查單個組件健康狀態"""
        try:
            process = self.components.get(component_name)
            if not process:
                return
            
            health = self.component_health.get(component_name)
            if not health:
                return
            
            # 檢查進程是否存活
            if process.poll() is not None:
                # 進程已退出
                health.status = ComponentStatus.ERROR
                health.last_error = f"進程退出，返回碼: {process.returncode}"
                self.logger.warning(f"⚠️ 組件 {component_name} 進程已退出")
                
                # 嘗試重啟
                asyncio.create_task(self.restart_component(component_name))
                return
            
            # 更新資源使用情況
            try:
                proc = psutil.Process(process.pid)
                health.cpu_percent = proc.cpu_percent()
                health.memory_mb = proc.memory_info().rss / 1024 / 1024
                health.uptime_seconds = (datetime.now() - health.timestamp).total_seconds()
                
                # 檢查資源限制
                config = self.components_config["ai_components"][component_name]
                limits = config.get("resource_limits", {})
                
                if (limits.get("max_cpu_percent", 100) < health.cpu_percent or
                    limits.get("max_memory_mb", 4096) < health.memory_mb):
                    self.logger.warning(
                        f"⚠️ 組件 {component_name} 資源使用過高: "
                        f"CPU={health.cpu_percent:.1f}%, MEM={health.memory_mb:.1f}MB"
                    )
                
            except psutil.NoSuchProcess:
                health.status = ComponentStatus.ERROR
                health.last_error = "進程不存在"
            
            health.timestamp = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ 檢查組件 {component_name} 健康狀態失敗: {e}")
    
    async def restart_component(self, component_name: str):
        """重啟組件"""
        try:
            health = self.component_health.get(component_name)
            if not health:
                return
            
            config = self.components_config["ai_components"][component_name]
            restart_policy = config.get("restart_policy", "on-failure")
            max_restarts = config.get("max_restarts", 3)
            
            # 檢查重啟策略
            if restart_policy == "never":
                self.logger.info(f"📋 組件 {component_name} 重啟策略為 never，不重啟")
                return
            
            if health.restart_count >= max_restarts:
                self.logger.error(f"❌ 組件 {component_name} 已達到最大重啟次數 ({max_restarts})")
                return
            
            # 停止舊進程
            self.stop_component(component_name)
            
            # 等待重啟延遲
            restart_delay = config.get("restart_delay", 30)
            self.logger.info(f"⏳ 等待 {restart_delay} 秒後重啟組件 {component_name}")
            await asyncio.sleep(restart_delay)
            
            # 更新狀態
            health.status = ComponentStatus.RESTARTING
            health.restart_count += 1
            self.total_restarts += 1
            
            # 重新啟動
            if await self.start_component(component_name, config):
                self.logger.info(f"🔄 組件 {component_name} 重啟成功 (第 {health.restart_count} 次)")
            else:
                self.logger.error(f"❌ 組件 {component_name} 重啟失敗")
                self.error_count += 1
                
        except Exception as e:
            self.logger.error(f"❌ 重啟組件 {component_name} 失敗: {e}")
    
    def collect_system_metrics(self):
        """收集系統指標"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 記憶體使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁碟使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 網路IO
            network_io = psutil.net_io_counters()._asdict()
            
            self.system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                timestamp=datetime.now()
            )
            
            # 檢查系統資源閾值
            thresholds = self.components_config["system_settings"]["system_resource_threshold"]
            
            if (cpu_percent > thresholds["cpu_percent"] or
                memory_percent > thresholds["memory_percent"] or
                disk_percent > thresholds["disk_percent"]):
                
                self.logger.warning(
                    f"⚠️ 系統資源使用過高: "
                    f"CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%, DISK={disk_percent:.1f}%"
                )
            
        except Exception as e:
            self.logger.error(f"❌ 收集系統指標失敗: {e}")
    
    async def main_management_loop(self):
        """主管理循環"""
        self.logger.info("🔄 進入主管理循環")
        
        try:
            while self.is_running and not self.shutdown_requested:
                # 每5分鐘生成狀態報告
                await self.generate_status_report()
                
                # 等待5分鐘
                for _ in range(300):  # 5分鐘 = 300秒
                    if self.shutdown_requested:
                        break
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 收到中斷信號")
        except Exception as e:
            self.logger.error(f"❌ 主管理循環錯誤: {e}")
        finally:
            await self.shutdown()
    
    async def generate_status_report(self):
        """生成狀態報告"""
        try:
            current_time = datetime.now()
            uptime = current_time - self.start_time
            
            # 統計組件狀態
            status_counts = {}
            for health in self.component_health.values():
                status = health.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # 生成報告
            report = {
                "timestamp": current_time.isoformat(),
                "uptime_hours": uptime.total_seconds() / 3600,
                "total_components": len(self.components),
                "component_status": status_counts,
                "total_restarts": self.total_restarts,
                "error_count": self.error_count,
                "system_metrics": asdict(self.system_metrics) if self.system_metrics else None,
                "component_details": {
                    name: asdict(health) for name, health in self.component_health.items()
                }
            }
            
            # 保存報告
            report_dir = self.project_root / "logs" / "status_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"status_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # 控制台輸出簡要狀態
            self.logger.info(
                f"📊 狀態報告: 運行時間={uptime.total_seconds()/3600:.1f}h, "
                f"組件={len(self.components)}, 重啟={self.total_restarts}, "
                f"錯誤={self.error_count}, CPU={self.system_metrics.cpu_percent:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"❌ 生成狀態報告失敗: {e}")
    
    def stop_component(self, component_name: str):
        """停止單個組件"""
        try:
            process = self.components.get(component_name)
            if not process:
                return
            
            self.logger.info(f"🛑 停止組件: {component_name}")
            
            # 嘗試優雅關閉
            process.terminate()
            
            # 等待最多10秒
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # 強制關閉
                process.kill()
                process.wait()
                self.logger.warning(f"⚠️ 組件 {component_name} 被強制關閉")
            
            # 更新狀態
            if component_name in self.component_health:
                self.component_health[component_name].status = ComponentStatus.STOPPED
            
            # 移除記錄
            del self.components[component_name]
            
            self.logger.info(f"✅ 組件 {component_name} 已停止")
            
        except Exception as e:
            self.logger.error(f"❌ 停止組件 {component_name} 失敗: {e}")
    
    def stop_all_components(self):
        """停止所有組件"""
        self.logger.info("🛑 停止所有組件...")
        
        for component_name in list(self.components.keys()):
            self.stop_component(component_name)
        
        self.logger.info("✅ 所有組件已停止")
    
    async def shutdown(self):
        """優雅關閉"""
        self.logger.info("🔄 開始優雅關閉...")
        
        # 設置關閉標誌
        self.is_running = False
        self.shutdown_requested = True
        
        # 停止所有組件
        self.stop_all_components()
        
        # 等待監控線程結束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)
        
        # 生成最終報告
        await self.generate_status_report()
        
        self.logger.info("✅ AIVA 持續運作 AI 組件管理器已關閉")

async def main():
    """主函數"""
    print("🚀 AIVA 持續運作 AI 組件管理器")
    print("=" * 60)
    print("📋 功能特點:")
    print("   • 基於SOT原則的統一配置管理")
    print("   • 持續運作直到手動關閉")
    print("   • 自動組件健康監控和重啟")
    print("   • 實時系統資源監控")
    print("   • 智能故障恢復機制")
    print("   • 集中化日誌和狀態報告")
    print("=" * 60)
    print("🛑 按 Ctrl+C 優雅關閉")
    print()
    
    manager = AIComponentManager()
    await manager.start_continuous_operation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 收到中斷信號，正在關閉...")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
    finally:
        print("👋 AIVA 持續運作管理器已退出")