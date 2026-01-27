"""
性能優化模組
包含並行處理、記憶體管理、監控、健康檢查和診斷功能
"""

from .unified_memory_manager import ComponentPool, MemoryManager, UnifiedMemoryManager
from .monitoring import Metric, MetricsCollector, metrics_collector, monitor_performance
from .parallel_processor import ParallelMessageProcessor
# 新增能力 (2026-01-21)
from .health_check import check_schemas, check_tools, check_directories
from .diagnose import check_engines, check_docker

__all__ = [
    "ParallelMessageProcessor",
    "ComponentPool",
    "MemoryManager",
    "UnifiedMemoryManager",
    "MetricsCollector",
    "Metric",
    "monitor_performance",
    "metrics_collector",
    # 新增匯出
    "check_schemas",
    "check_tools",
    "check_directories",
    "check_engines",
    "check_docker",
]
