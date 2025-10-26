"""
現代化觀測性與監控模組
基於 OpenTelemetry 和結構化日誌的最佳實踐
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# 嘗試導入 OpenTelemetry（可選依賴）
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class LogLevel(str, Enum):
    """日誌級別枚舉"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(str, Enum):
    """指標類型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class TraceContext:
    """追蹤上下文"""
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid4())[:8])
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: list[Dict[str, Any]] = field(default_factory=list)


class StructuredLog(BaseModel):
    """結構化日誌模型"""
    timestamp: float = Field(default_factory=time.time)
    level: LogLevel
    message: str
    module: str
    function: str
    line: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class Metric(BaseModel):
    """指標模型"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = Field(default_factory=time.time)
    tags: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None


class PerformanceMetrics(BaseModel):
    """性能指標"""
    operation_name: str
    duration_ms: float
    success: bool = True
    error_message: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    timestamp: float = Field(default_factory=time.time)
    trace_id: Optional[str] = None


class AIVALogger:
    """AIVA 結構化日誌器"""
    
    def __init__(self, name: str, enable_tracing: bool = True):
        self.name = name
        self.enable_tracing = enable_tracing
        self._logger = logging.getLogger(name)
        self._current_trace: Optional[TraceContext] = None
    
    def _create_structured_log(
        self, 
        level: LogLevel, 
        message: str,
        **extra_fields: Any
    ) -> StructuredLog:
        """創建結構化日誌"""
        import inspect
        frame = inspect.currentframe()
        try:
            # 向上查找調用方的框架
            caller_frame = frame.f_back.f_back if frame and frame.f_back else None
            if caller_frame:
                module = caller_frame.f_globals.get('__name__', 'unknown')
                function = caller_frame.f_code.co_name
                line = caller_frame.f_lineno
            else:
                module = function = 'unknown'
                line = 0
        finally:
            del frame
        
        return StructuredLog(
            level=level,
            message=message,
            module=module,
            function=function,
            line=line,
            trace_id=self._current_trace.trace_id if self._current_trace else None,
            span_id=self._current_trace.span_id if self._current_trace else None,
            extra_fields=extra_fields
        )
    
    def debug(self, message: str, **extra_fields: Any) -> None:
        """DEBUG 級別日誌"""
        log_entry = self._create_structured_log(LogLevel.DEBUG, message, **extra_fields)
        self._logger.debug(log_entry.model_dump_json())
    
    def info(self, message: str, **extra_fields: Any) -> None:
        """INFO 級別日誌"""
        log_entry = self._create_structured_log(LogLevel.INFO, message, **extra_fields)
        self._logger.info(log_entry.model_dump_json())
    
    def warning(self, message: str, **extra_fields: Any) -> None:
        """WARNING 級別日誌"""
        log_entry = self._create_structured_log(LogLevel.WARNING, message, **extra_fields)
        self._logger.warning(log_entry.model_dump_json())
    
    def error(self, message: str, **extra_fields: Any) -> None:
        """ERROR 級別日誌"""
        log_entry = self._create_structured_log(LogLevel.ERROR, message, **extra_fields)
        self._logger.error(log_entry.model_dump_json())
    
    def critical(self, message: str, **extra_fields: Any) -> None:
        """CRITICAL 級別日誌"""
        log_entry = self._create_structured_log(LogLevel.CRITICAL, message, **extra_fields)
        self._logger.critical(log_entry.model_dump_json())
    
    @contextmanager
    def trace_operation(
        self, 
        operation_name: str, 
        **tags: Any
    ) -> Generator[TraceContext, None, None]:
        """追蹤操作上下文管理器"""
        trace_context = TraceContext(
            operation_name=operation_name,
            parent_span_id=self._current_trace.span_id if self._current_trace else None,
            tags=tags
        )
        
        previous_trace = self._current_trace
        self._current_trace = trace_context
        
        # OpenTelemetry 整合
        if OTEL_AVAILABLE:
            tracer = trace.get_tracer(self.name)
            with tracer.start_as_current_span(operation_name) as span:
                # 設置標籤
                for key, value in tags.items():
                    span.set_attribute(key, str(value))
                
                try:
                    self.info(f"Starting operation: {operation_name}", **tags)
                    yield trace_context
                    span.set_status(Status(StatusCode.OK))
                    self.info(f"Completed operation: {operation_name}", 
                             duration_ms=(time.time() - trace_context.start_time) * 1000)
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    self.error(f"Operation failed: {operation_name}", 
                              error=str(e), 
                              duration_ms=(time.time() - trace_context.start_time) * 1000)
                    raise
        else:
            # 簡化版追蹤（無 OpenTelemetry）
            try:
                self.info(f"Starting operation: {operation_name}", **tags)
                yield trace_context
                self.info(f"Completed operation: {operation_name}", 
                         duration_ms=(time.time() - trace_context.start_time) * 1000)
            except Exception as e:
                self.error(f"Operation failed: {operation_name}", 
                          error=str(e), 
                          duration_ms=(time.time() - trace_context.start_time) * 1000)
                raise
            finally:
                self._current_trace = previous_trace


class MetricCollector:
    """指標收集器"""
    
    def __init__(self):
        self._metrics: list[Metric] = []
        self._performance_metrics: list[PerformanceMetrics] = []
    
    def counter(self, name: str, value: Union[int, float] = 1, 
                tags: Optional[Dict[str, str]] = None, 
                description: Optional[str] = None) -> None:
        """計數器指標"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {},
            description=description
        )
        self._metrics.append(metric)
    
    def gauge(self, name: str, value: Union[int, float], 
              tags: Optional[Dict[str, str]] = None,
              description: Optional[str] = None) -> None:
        """儀表指標"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {},
            description=description
        )
        self._metrics.append(metric)
    
    def histogram(self, name: str, value: Union[int, float], 
                  tags: Optional[Dict[str, str]] = None,
                  description: Optional[str] = None) -> None:
        """直方圖指標"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags or {},
            description=description
        )
        self._metrics.append(metric)
    
    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """記錄性能指標"""
        self._performance_metrics.append(metrics)
    
    def get_metrics(self) -> list[Metric]:
        """獲取所有指標"""
        return self._metrics.copy()
    
    def get_performance_metrics(self) -> list[PerformanceMetrics]:
        """獲取性能指標"""
        return self._performance_metrics.copy()
    
    def clear(self) -> None:
        """清空指標"""
        self._metrics.clear()
        self._performance_metrics.clear()


# 全域實例
default_logger = AIVALogger("aiva")
default_metrics = MetricCollector()


def get_logger(name: str) -> AIVALogger:
    """獲取日誌器實例"""
    return AIVALogger(name)


def get_metrics_collector() -> MetricCollector:
    """獲取指標收集器實例"""
    return default_metrics


# 裝飾器支援
def trace_function(operation_name: Optional[str] = None):
    """函數追蹤裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)
            
            with logger.trace_operation(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def measure_performance(operation_name: Optional[str] = None):
    """性能測量裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                metrics = PerformanceMetrics(
                    operation_name=name,
                    duration_ms=duration_ms,
                    success=True
                )
                default_metrics.record_performance(metrics)
                
                return result
            
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                metrics = PerformanceMetrics(
                    operation_name=name,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e)
                )
                default_metrics.record_performance(metrics)
                
                raise
        
        return wrapper
    return decorator


__all__ = [
    "LogLevel",
    "MetricType", 
    "TraceContext",
    "StructuredLog",
    "Metric",
    "PerformanceMetrics",
    "AIVALogger",
    "MetricCollector",
    "get_logger",
    "get_metrics_collector",
    "trace_function",
    "measure_performance",
    "default_logger",
    "default_metrics",
]