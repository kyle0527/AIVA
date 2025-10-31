"""
性能優化模組
包含並行處理、記憶體管理和監控功能
"""

from .memory_manager import ComponentPool, MemoryManager
from .monitoring import Metric, MetricsCollector, metrics_collector, monitor_performance
from .parallel_processor import ParallelMessageProcessor

__all__ = [
    "ParallelMessageProcessor",
    "ComponentPool",
    "MemoryManager",
    "MetricsCollector",
    "Metric",
    "monitor_performance",
    "metrics_collector",
]
