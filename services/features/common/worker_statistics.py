"""
Worker統計模組 - 用於收集和管理工作器執行統計資訊
"""

import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import threading


class ErrorCategory(Enum):
    """錯誤類別枚舉"""
    NETWORK_ERROR = "network_error"
    NETWORK = "network_error"  # alias
    TIMEOUT_ERROR = "timeout_error"
    TIMEOUT = "timeout_error"  # alias
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN_ERROR = "unknown_error"
    UNKNOWN = "unknown_error"  # alias


@dataclass
class ExecutionStats:
    """執行統計資料"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_execution_time: float = 0.0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_execution_time(self) -> float:
        """平均執行時間"""
        if self.total_requests == 0:
            return 0.0
        return self.total_execution_time / self.total_requests


class StatisticsCollector:
    """統計收集器 - 線程安全的統計資料收集"""
    
    def __init__(self):
        self._stats = ExecutionStats()
        self._lock = threading.Lock()
    
    def record_request_start(self) -> float:
        """記錄請求開始時間"""
        return time.time()
    
    def record_request_success(self, start_time: float):
        """記錄成功請求"""
        execution_time = time.time() - start_time
        with self._lock:
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.total_execution_time += execution_time
    
    def record_request_error(self, start_time: float, category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR):
        """記錄失敗請求"""
        execution_time = time.time() - start_time
        with self._lock:
            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.total_execution_time += execution_time
            self._stats.errors_by_category[category] = self._stats.errors_by_category.get(category, 0) + 1
    
    def record_request(self, success: bool, timeout: bool = False):
        """記錄請求結果"""
        with self._lock:
            self._stats.total_requests += 1
            if success:
                self._stats.successful_requests += 1
            else:
                self._stats.failed_requests += 1
                if timeout:
                    category = ErrorCategory.TIMEOUT_ERROR
                    self._stats.errors_by_category[category] = self._stats.errors_by_category.get(category, 0) + 1
    
    def record_error(self, category: ErrorCategory, error_msg: str = "", request_info: Optional[Dict[str, Any]] = None):
        """記錄錯誤"""
        with self._lock:
            self._stats.errors_by_category[category] = self._stats.errors_by_category.get(category, 0) + 1
            # error_msg和request_info可用於未來的詳細日誌記錄
            _ = error_msg, request_info  # 避免未使用參數警告
    
    def record_payload_test(self, success: bool):
        """記錄payload測試結果"""
        # 可以在後續版本中添加更詳細的payload統計
        pass
    
    def record_vulnerability(self, false_positive: bool = False):
        """記錄漏洞發現"""
        # 可以在後續版本中添加漏洞統計
        pass
    
    def set_module_specific(self, key: str, value: Any):
        """設置模組特定數據"""
        # 可以在後續版本中添加模組特定統計
        pass
    
    def get_stats(self) -> ExecutionStats:
        """獲取統計資料副本"""
        with self._lock:
            # 返回統計資料的深拷貝
            return ExecutionStats(
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                total_execution_time=self._stats.total_execution_time,
                errors_by_category=self._stats.errors_by_category.copy(),
                start_time=self._stats.start_time
            )
    
    def reset_stats(self):
        """重置統計資料"""
        with self._lock:
            self._stats = ExecutionStats()
    
    def get_summary(self) -> Dict[str, Any]:
        """獲取統計摘要"""
        stats = self.get_stats()
        uptime = time.time() - stats.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": stats.success_rate,
            "average_execution_time": stats.average_execution_time,
            "errors_by_category": {cat.value: count for cat, count in stats.errors_by_category.items()},
            "requests_per_second": stats.total_requests / uptime if uptime > 0 else 0
        }


# 全域統計收集器實例
_global_collector: Optional[StatisticsCollector] = None


def get_global_collector() -> StatisticsCollector:
    """獲取全域統計收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = StatisticsCollector()
    return _global_collector


def reset_global_collector():
    """重置全域統計收集器"""
    global _global_collector
    _global_collector = StatisticsCollector()