"""Resource Watchdog - 系統資源監控與動態調整 (P3)

實現：
1. 實時監控 CPU、RAM、磁盤使用率
2. 動態調整掃描併發數與速率
3. 自動垃圾回收與資源釋放
4. 過載保護機制
"""

import asyncio
import psutil
import gc
from typing import Dict, Any, Callable, Optional
from datetime import datetime, UTC
from dataclasses import dataclass

from aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceThresholds:
    """資源閾值配置"""
    cpu_warning: float = 80.0  # CPU 使用率警告閾值 (%)
    cpu_critical: float = 90.0  # CPU 使用率危險閾值 (%)
    
    memory_warning: float = 75.0  # 記憶體使用率警告閾值 (%)
    memory_critical: float = 85.0  # 記憶體使用率危險閾值 (%)
    
    disk_warning: float = 85.0  # 磁盤使用率警告閾值 (%)
    disk_critical: float = 95.0  # 磁盤使用率危險閾值 (%)


@dataclass
class ResourceStatus:
    """資源狀態"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    timestamp: datetime
    
    def is_healthy(self, thresholds: ResourceThresholds) -> bool:
        """檢查資源是否健康"""
        return (
            self.cpu_percent < thresholds.cpu_warning and
            self.memory_percent < thresholds.memory_warning and
            self.disk_percent < thresholds.disk_warning
        )
    
    def is_critical(self, thresholds: ResourceThresholds) -> bool:
        """檢查資源是否處於危險狀態"""
        return (
            self.cpu_percent >= thresholds.cpu_critical or
            self.memory_percent >= thresholds.memory_critical or
            self.disk_percent >= thresholds.disk_critical
        )


class ResourceWatchdog:
    """系統資源監控看門狗
    
    職責：
    1. 每 5 秒檢查一次系統資源
    2. 動態調整掃描策略：
       - RAM > 80%: 暫停新 Browser 啟動，強制 GC
       - CPU > 90%: 降低掃描速率
       - 資源回穩: 恢復原始設定
    3. 記錄資源使用歷史
    """
    
    def __init__(
        self,
        check_interval: float = 5.0,
        thresholds: Optional[ResourceThresholds] = None
    ):
        self.check_interval = check_interval
        self.thresholds = thresholds or ResourceThresholds()
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._status_history: list[ResourceStatus] = []
        self._max_history = 100  # 保留最近 100 次檢查
        
        # 動態調整回調
        self._adjustment_callbacks: list[Callable[[ResourceStatus], None]] = []
        
        logger.info(f"ResourceWatchdog initialized (interval={check_interval}s)")
    
    def register_adjustment_callback(self, callback: Callable[[ResourceStatus], None]) -> None:
        """註冊資源調整回調函數
        
        當資源超過閾值時，會調用這些回調進行動態調整
        
        Args:
            callback: 回調函數，接收 ResourceStatus 參數
        """
        self._adjustment_callbacks.append(callback)
        logger.info(f"Registered adjustment callback: {callback.__name__}")
    
    def get_current_status(self) -> ResourceStatus:
        """獲取當前資源狀態"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return ResourceStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            disk_percent=disk.percent,
            timestamp=datetime.now(UTC)
        )
    
    def start(self) -> None:
        """啟動監控"""
        if self._running:
            logger.warning("Watchdog already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("✅ ResourceWatchdog started")
    
    async def stop(self) -> None:
        """停止監控"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("ResourceWatchdog was cancelled")
                raise
        
        logger.info("✅ ResourceWatchdog stopped")
    
    async def _monitor_loop(self) -> None:
        """監控循環"""
        while self._running:
            try:
                # 獲取資源狀態
                status = self.get_current_status()
                
                # 記錄歷史
                self._status_history.append(status)
                if len(self._status_history) > self._max_history:
                    self._status_history.pop(0)
                
                # 檢查閾值並執行調整
                self._check_and_adjust(status)
                
                # 等待下次檢查
                await asyncio.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    def _check_and_adjust(self, status: ResourceStatus) -> None:
        """檢查資源並執行動態調整"""
        
        # 記憶體危險：強制 GC
        if status.memory_percent >= self.thresholds.memory_critical:
            logger.warning(
                f"⚠️ CRITICAL: Memory usage {status.memory_percent:.1f}% "
                f"(available: {status.memory_available_mb:.1f} MB)"
            )
            logger.info("🗑️ Forcing garbage collection...")
            gc.collect()
            
            # 通知所有回調
            for callback in self._adjustment_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # 記憶體警告
        elif status.memory_percent >= self.thresholds.memory_warning:
            logger.warning(
                f"⚠️ WARNING: Memory usage {status.memory_percent:.1f}% "
                f"(available: {status.memory_available_mb:.1f} MB)"
            )
        
        # CPU 危險：降低掃描速率
        if status.cpu_percent >= self.thresholds.cpu_critical:
            logger.warning(
                f"⚠️ CRITICAL: CPU usage {status.cpu_percent:.1f}%"
            )
            # 通知回調（例如降低線程數）
            for callback in self._adjustment_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # CPU 警告
        elif status.cpu_percent >= self.thresholds.cpu_warning:
            logger.warning(f"⚠️ WARNING: CPU usage {status.cpu_percent:.1f}%")
        
        # 磁盤警告
        if status.disk_percent >= self.thresholds.disk_warning:
            logger.warning(f"⚠️ WARNING: Disk usage {status.disk_percent:.1f}%")
        
        # 資源健康：記錄 debug 信息
        if status.is_healthy(self.thresholds):
            logger.debug(
                f"✅ Resources healthy: "
                f"CPU={status.cpu_percent:.1f}%, "
                f"MEM={status.memory_percent:.1f}%, "
                f"DISK={status.disk_percent:.1f}%"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取資源統計信息"""
        if not self._status_history:
            return {"error": "No data collected yet"}
        
        cpu_values = [s.cpu_percent for s in self._status_history]
        mem_values = [s.memory_percent for s in self._status_history]
        
        return {
            "total_checks": len(self._status_history),
            "cpu": {
                "current": self._status_history[-1].cpu_percent,
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current": self._status_history[-1].memory_percent,
                "avg": sum(mem_values) / len(mem_values),
                "max": max(mem_values),
                "min": min(mem_values),
                "available_mb": self._status_history[-1].memory_available_mb
            },
            "disk": {
                "current": self._status_history[-1].disk_percent
            },
            "last_check": self._status_history[-1].timestamp.isoformat()
        }
