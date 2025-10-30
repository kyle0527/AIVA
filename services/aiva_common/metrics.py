"""
AIVA 統一統計收集模組 - Python 實現
日期: 2025-01-07
目的: 提供跨語言一致的性能監控和統計收集功能
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


# MetricType 已移至 aiva_common.observability 模組統一管理
# 遵循修復原則：避免重複定義，使用官方標準版本
from .observability import MetricType


class SeverityLevel(Enum):
    """嚴重程度級別"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class MetricData:
    """指標數據結構"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class WorkerMetrics:
    """Worker 統計指標集合"""
    worker_id: str
    
    # 任務處理統計
    tasks_received: int = 0
    tasks_processed: int = 0
    tasks_failed: int = 0
    tasks_retried: int = 0
    
    # 時間統計
    total_processing_time: float = 0.0
    total_queue_wait_time: float = 0.0
    
    # 檢測結果統計
    findings_created: int = 0
    vulnerabilities_found: int = 0
    false_positives: int = 0
    
    # 嚴重程度分佈
    severity_distribution: Dict[str, int] = field(default_factory=lambda: {
        "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
    })
    
    # 系統資源 (瞬時值)
    current_memory_usage: float = 0.0
    current_cpu_usage: float = 0.0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "worker_id": self.worker_id,
            "timestamp": time.time(),
            "task_metrics": {
                "tasks_received": self.tasks_received,
                "tasks_processed": self.tasks_processed,
                "tasks_failed": self.tasks_failed,
                "tasks_retried": self.tasks_retried,
                "average_processing_time": (
                    self.total_processing_time / max(self.tasks_processed, 1)
                ),
                "average_queue_wait_time": (
                    self.total_queue_wait_time / max(self.tasks_received, 1)
                )
            },
            "detection_metrics": {
                "findings_created": self.findings_created,
                "vulnerabilities_found": self.vulnerabilities_found,
                "false_positives": self.false_positives,
                "severity_distribution": self.severity_distribution.copy()
            },
            "system_metrics": {
                "memory_usage": self.current_memory_usage,
                "cpu_usage": self.current_cpu_usage,
                "active_connections": self.active_connections
            }
        }


class MetricsExporter(ABC):
    """指標導出器介面"""
    
    @abstractmethod
    def export(self, metrics: Dict[str, Any]) -> bool:
        """導出指標數據"""
        pass


class JSONMetricsExporter(MetricsExporter):
    """JSON 格式指標導出器"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.metrics_history = deque(maxlen=1000)  # 保留最近1000條記錄
    
    def export(self, metrics: Dict[str, Any]) -> bool:
        """導出為 JSON 格式"""
        try:
            metrics_with_timestamp = {
                **metrics,
                "exported_at": datetime.now().isoformat()
            }
            
            # 加入歷史記錄
            self.metrics_history.append(metrics_with_timestamp)
            
            # 如果指定了輸出文件，寫入文件
            if self.output_file:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics_with_timestamp, ensure_ascii=False) + '\n')
            
            logger.debug(f"Exported metrics: {json.dumps(metrics_with_timestamp, indent=2)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """獲取最近的指標記錄"""
        return list(self.metrics_history)[-count:]


class MetricsCollector:
    """統一指標收集器"""
    
    def __init__(
        self,
        worker_id: str,
        collection_interval: int = 60,
        exporters: Optional[List[MetricsExporter]] = None
    ):
        self.worker_id = worker_id
        self.collection_interval = collection_interval
        self.exporters = exporters or [JSONMetricsExporter()]
        
        # 指標數據
        self.metrics = WorkerMetrics(worker_id=worker_id)
        self._lock = threading.RLock()
        
        # 性能追蹤
        self._task_start_times: Dict[str, float] = {}
        self._last_export_time = time.time()
        
        # 後台導出線程
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info(f"MetricsCollector initialized for worker: {worker_id}")
    
    def start_background_export(self):
        """啟動後台指標導出"""
        if self._export_thread and self._export_thread.is_alive():
            logger.warning("Background export already running")
            return
        
        def export_loop():
            while not self._stop_event.wait(self.collection_interval):
                try:
                    self.export_metrics()
                except Exception as e:
                    logger.error(f"Error in background export: {e}")
        
        self._export_thread = threading.Thread(target=export_loop, daemon=True)
        self._export_thread.start()
        logger.info("Background metrics export started")
    
    def stop_background_export(self):
        """停止後台指標導出"""
        if self._export_thread:
            self._stop_event.set()
            self._export_thread.join(timeout=5)
            logger.info("Background metrics export stopped")
    
    # 任務處理指標
    def record_task_received(self, task_id: str):
        """記錄接收到任務"""
        with self._lock:
            self.metrics.tasks_received += 1
            self._task_start_times[task_id] = time.time()
    
    def record_task_completed(self, task_id: str, findings_count: int = 0):
        """記錄任務完成"""
        with self._lock:
            self.metrics.tasks_processed += 1
            self.metrics.findings_created += findings_count
            
            # 計算處理時間
            if task_id in self._task_start_times:
                processing_time = time.time() - self._task_start_times[task_id]
                self.metrics.total_processing_time += processing_time
                del self._task_start_times[task_id]
    
    def record_task_failed(self, task_id: str, will_retry: bool = False):
        """記錄任務失敗"""
        with self._lock:
            self.metrics.tasks_failed += 1
            if will_retry:
                self.metrics.tasks_retried += 1
            
            # 清理開始時間
            self._task_start_times.pop(task_id, None)
    
    # 檢測結果指標
    def record_vulnerability_found(self, severity: SeverityLevel):
        """記錄發現漏洞"""
        with self._lock:
            self.metrics.vulnerabilities_found += 1
            self.metrics.severity_distribution[severity.value] += 1
    
    def record_false_positive(self):
        """記錄誤報"""
        with self._lock:
            self.metrics.false_positives += 1
    
    # 系統資源指標
    def update_system_metrics(
        self,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        active_connections: Optional[int] = None
    ):
        """更新系統資源指標"""
        with self._lock:
            if memory_usage is not None:
                self.metrics.current_memory_usage = memory_usage
            if cpu_usage is not None:
                self.metrics.current_cpu_usage = cpu_usage
            if active_connections is not None:
                self.metrics.active_connections = active_connections
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """獲取當前指標快照"""
        with self._lock:
            return self.metrics.to_dict()
    
    def export_metrics(self) -> bool:
        """導出指標到所有配置的導出器"""
        try:
            metrics_data = self.get_current_metrics()
            
            success = True
            for exporter in self.exporters:
                if not exporter.export(metrics_data):
                    success = False
            
            self._last_export_time = time.time()
            return success
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def reset_counters(self):
        """重置計數器 (保留瞬時值)"""
        with self._lock:
            self.metrics.tasks_received = 0
            self.metrics.tasks_processed = 0
            self.metrics.tasks_failed = 0
            self.metrics.tasks_retried = 0
            self.metrics.total_processing_time = 0.0
            self.metrics.total_queue_wait_time = 0.0
            self.metrics.findings_created = 0
            self.metrics.vulnerabilities_found = 0
            self.metrics.false_positives = 0
            self.metrics.severity_distribution = {
                "critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0
            }
            logger.info("Metrics counters reset")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """生成摘要報告"""
        with self._lock:
            metrics = self.metrics
            
            # 計算成功率
            total_tasks = metrics.tasks_received
            success_rate = (
                metrics.tasks_processed / total_tasks if total_tasks > 0 else 0.0
            )
            
            # 計算平均處理時間
            avg_processing_time = (
                metrics.total_processing_time / metrics.tasks_processed
                if metrics.tasks_processed > 0 else 0.0
            )
            
            return {
                "worker_id": self.worker_id,
                "report_generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_tasks": total_tasks,
                    "success_rate": success_rate,
                    "failure_rate": (metrics.tasks_failed / total_tasks) if total_tasks > 0 else 0.0,
                    "retry_rate": (metrics.tasks_retried / total_tasks) if total_tasks > 0 else 0.0,
                    "average_processing_time_ms": avg_processing_time * 1000,
                    "total_findings": metrics.findings_created,
                    "total_vulnerabilities": metrics.vulnerabilities_found,
                    "false_positive_rate": (
                        metrics.false_positives / max(metrics.findings_created, 1)
                    )
                },
                "current_system_status": {
                    "memory_usage_mb": metrics.current_memory_usage / (1024 * 1024),
                    "cpu_usage_percent": metrics.current_cpu_usage,
                    "active_connections": metrics.active_connections
                },
                "vulnerability_distribution": metrics.severity_distribution.copy()
            }


# 全局指標收集器實例 (單例模式)
_global_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(worker_id: Optional[str] = None) -> Optional[MetricsCollector]:
    """獲取全局指標收集器實例"""
    global _global_collector
    
    if _global_collector is None and worker_id:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = MetricsCollector(worker_id=worker_id)
    
    return _global_collector


def initialize_metrics(
    worker_id: str,
    collection_interval: int = 60,
    output_file: Optional[str] = None,
    start_background_export: bool = True
) -> MetricsCollector:
    """初始化全局指標收集器"""
    global _global_collector
    
    with _collector_lock:
        if _global_collector is not None:
            logger.warning(f"Metrics already initialized for {_global_collector.worker_id}")
            return _global_collector
        
        # 創建導出器
        exporters = [JSONMetricsExporter(output_file=output_file)]
        
        # 創建收集器
        _global_collector = MetricsCollector(
            worker_id=worker_id,
            collection_interval=collection_interval,
            exporters=exporters
        )
        
        if start_background_export:
            _global_collector.start_background_export()
        
        logger.info(f"Global metrics collector initialized for {worker_id}")
        return _global_collector


def cleanup_metrics():
    """清理指標收集器"""
    global _global_collector
    
    with _collector_lock:
        if _global_collector:
            _global_collector.stop_background_export()
            _global_collector.export_metrics()  # 最後一次導出
            _global_collector = None
            logger.info("Metrics collector cleaned up")


# 便捷函數
def record_task_received(task_id: str):
    """記錄任務接收 (便捷函數)"""
    collector = get_metrics_collector()
    if collector:
        collector.record_task_received(task_id)


def record_task_completed(task_id: str, findings_count: int = 0):
    """記錄任務完成 (便捷函數)"""
    collector = get_metrics_collector()
    if collector:
        collector.record_task_completed(task_id, findings_count)


def record_task_failed(task_id: str, will_retry: bool = False):
    """記錄任務失敗 (便捷函數)"""
    collector = get_metrics_collector()
    if collector:
        collector.record_task_failed(task_id, will_retry)


def record_vulnerability_found(severity: SeverityLevel):
    """記錄發現漏洞 (便捷函數)"""
    collector = get_metrics_collector()
    if collector:
        collector.record_vulnerability_found(severity)


def update_system_metrics(**kwargs):
    """更新系統指標 (便捷函數)"""
    collector = get_metrics_collector()
    if collector:
        collector.update_system_metrics(**kwargs)


# 裝飾器
def track_task_metrics(task_id_func=None):
    """任務執行追蹤裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成任務ID
            if task_id_func:
                task_id = task_id_func(*args, **kwargs)
            else:
                task_id = f"{func.__name__}_{int(time.time() * 1000)}"
            
            collector = get_metrics_collector()
            if not collector:
                return func(*args, **kwargs)
            
            collector.record_task_received(task_id)
            
            try:
                result = func(*args, **kwargs)
                
                # 嘗試從結果中提取發現數量
                findings_count = 0
                if isinstance(result, dict):
                    findings_count = result.get('findings_count', 0)
                elif hasattr(result, 'findings_count'):
                    findings_count = result.findings_count
                
                collector.record_task_completed(task_id, findings_count)
                return result
                
            except Exception as e:
                collector.record_task_failed(task_id, will_retry=False)
                raise
        
        return wrapper
    return decorator