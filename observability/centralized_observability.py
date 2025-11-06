"""
AIVA 集中化可觀測性框架
基於 OpenTelemetry 標準實施分散式追蹤、集中化日誌記錄和即時監控

功能特性：
1. 分散式追蹤 (Distributed Tracing)
2. 結構化日誌記錄 (Structured Logging)
3. 度量指標收集 (Metrics Collection)
4. 效能監控 (Performance Monitoring)
5. 錯誤追蹤與警報 (Error Tracking & Alerting)
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
from datetime import datetime, timezone
from enum import Enum

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.semantic_conventions.trace import SpanAttributes
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    print("⚠️ OpenTelemetry 未安裝，使用簡化版本")

# AIVA 內部導入
from services.aiva_common.utils import get_logger

class ObservabilityLevel(Enum):
    """可觀測性級別"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """度量類型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class TraceContext:
    """追蹤上下文"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    service_name: str = ""
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogEntry:
    """日誌條目"""
    timestamp: datetime
    level: ObservabilityLevel
    service_name: str
    operation: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass 
class MetricEntry:
    """度量條目"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    service_name: str
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

class StructuredLogger:
    """結構化日誌記錄器"""
    
    def __init__(self, service_name: str, output_path: Optional[str] = None):
        self.service_name = service_name
        self.output_path = Path(output_path) if output_path else Path("./logs")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 設置標準日誌
        self.logger = get_logger(f"observability.{service_name}")
        
        # 設置檔案處理器
        log_file = self.output_path / f"{service_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    def log(self, level: ObservabilityLevel, operation: str, message: str, 
            trace_context: Optional[TraceContext] = None, **attributes):
        """記錄結構化日誌"""
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            service_name=self.service_name,
            operation=operation,
            message=message,
            trace_id=trace_context.trace_id if trace_context else None,
            span_id=trace_context.span_id if trace_context else None,
            attributes=attributes
        )
        
        # 轉換為 JSON 格式
        log_data = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "service": log_entry.service_name,
            "operation": log_entry.operation,
            "message": log_entry.message,
            "trace_id": log_entry.trace_id,
            "span_id": log_entry.span_id,
            "attributes": log_entry.attributes
        }
        
        # 寫入檔案
        log_file = self.output_path / f"{self.service_name}-structured.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        
        # 標準日誌輸出
        log_level = getattr(logging, level.value.upper())
        self.logger.log(log_level, f"[{operation}] {message} {attributes}")

class MetricsCollector:
    """度量收集器"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics: List[MetricEntry] = []
        self.lock = threading.Lock()
        
        # OpenTelemetry 設置
        if OPENTELEMETRY_AVAILABLE:
            self.meter = metrics.get_meter(service_name)
            self._setup_metrics()
        
    def _setup_metrics(self):
        """設置度量指標"""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        # 請求計數器
        self.request_counter = self.meter.create_counter(
            name=f"{self.service_name}_requests_total",
            description="Total number of requests",
        )
        
        # 響應時間直方圖
        self.response_time_histogram = self.meter.create_histogram(
            name=f"{self.service_name}_response_time_seconds",
            description="Response time in seconds",
            unit="s"
        )
        
        # 錯誤計數器
        self.error_counter = self.meter.create_counter(
            name=f"{self.service_name}_errors_total",
            description="Total number of errors",
        )
        
        # 活躍連線數
        self.active_connections_gauge = self.meter.create_up_down_counter(
            name=f"{self.service_name}_active_connections",
            description="Number of active connections",
        )
    
    def increment_counter(self, name: str, value: float = 1.0, **tags):
        """增加計數器"""
        metric = MetricEntry(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=datetime.now(timezone.utc),
            service_name=self.service_name,
            tags=tags
        )
        
        with self.lock:
            self.metrics.append(metric)
            
        if OPENTELEMETRY_AVAILABLE and hasattr(self, 'request_counter'):
            self.request_counter.add(value, tags)
    
    def record_histogram(self, name: str, value: float, **tags):
        """記錄直方圖值"""
        metric = MetricEntry(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(timezone.utc),
            service_name=self.service_name,
            tags=tags
        )
        
        with self.lock:
            self.metrics.append(metric)
            
        if OPENTELEMETRY_AVAILABLE and hasattr(self, 'response_time_histogram'):
            self.response_time_histogram.record(value, tags)
    
    def set_gauge(self, name: str, value: float, **tags):
        """設置儀表值"""
        metric = MetricEntry(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(timezone.utc),
            service_name=self.service_name,
            tags=tags
        )
        
        with self.lock:
            self.metrics.append(metric)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """獲取度量摘要"""
        with self.lock:
            total_metrics = len(self.metrics)
            metric_types = {}
            
            for metric in self.metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metric_types:
                    metric_types[metric_type] = 0
                metric_types[metric_type] += 1
            
            return {
                "service": self.service_name,
                "total_metrics": total_metrics,
                "metric_types": metric_types,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

class DistributedTracer:
    """分散式追蹤器"""
    
    def __init__(self, service_name: str, jaeger_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.active_spans: Dict[str, TraceContext] = {}
        
        # OpenTelemetry 設置
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing(jaeger_endpoint)
        
    def _setup_tracing(self, jaeger_endpoint: Optional[str]):
        """設置追蹤"""
        # 設置追蹤器提供者
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # 設置 Jaeger 導出器（如果提供端點）
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(self.service_name)
    
    def start_span(self, operation_name: str, parent_context: Optional[TraceContext] = None) -> TraceContext:
        """開始新的跨度"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_context.span_id if parent_context else None,
            service_name=self.service_name,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self.active_spans[span_id] = context
        
        return context
    
    def add_span_attribute(self, span_id: str, key: str, value: Any):
        """添加跨度屬性"""
        if span_id in self.active_spans:
            self.active_spans[span_id].attributes[key] = value
    
    def finish_span(self, span_id: str, status: str = "ok", error: Optional[str] = None):
        """結束跨度"""
        if span_id in self.active_spans:
            context = self.active_spans[span_id]
            context.attributes["duration"] = time.time() - context.start_time
            context.attributes["status"] = status
            
            if error:
                context.attributes["error"] = error
                
            # 移除活躍跨度
            del self.active_spans[span_id]
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_context: Optional[TraceContext] = None):
        """追蹤操作的上下文管理器"""
        context = self.start_span(operation_name, parent_context)
        try:
            yield context
            self.finish_span(context.span_id, "ok")
        except Exception as e:
            self.finish_span(context.span_id, "error", str(e))
            raise

class ObservabilityFramework:
    """可觀測性框架 - 主要協調器"""
    
    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        
        # 初始化元件
        self.logger = StructuredLogger(service_name, self.config.get("log_path"))
        self.metrics = MetricsCollector(service_name)
        self.tracer = DistributedTracer(
            service_name, 
            self.config.get("jaeger_endpoint")
        )
        
        # 健康檢查
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
    async def record_request(self, operation: str, duration: float, status: str = "success", **attributes):
        """記錄請求"""
        self.request_count += 1
        
        # 記錄度量
        self.metrics.increment_counter("requests_total", tags={"operation": operation, "status": status})
        self.metrics.record_histogram("request_duration_seconds", duration, operation=operation)
        
        # 記錄日誌
        level = ObservabilityLevel.INFO if status == "success" else ObservabilityLevel.ERROR
        self.logger.log(
            level, 
            operation, 
            f"Request completed in {duration:.3f}s",
            operation=operation,
            duration=duration,
            status=status,
            **attributes
        )
        
        if status == "error":
            self.error_count += 1
    
    async def record_error(self, operation: str, error: Exception, trace_context: Optional[TraceContext] = None):
        """記錄錯誤"""
        self.error_count += 1
        
        # 記錄度量
        self.metrics.increment_counter("errors_total", tags={"operation": operation, "error_type": type(error).__name__})
        
        # 記錄日誌
        self.logger.log(
            ObservabilityLevel.ERROR,
            operation,
            f"Error occurred: {str(error)}",
            trace_context,
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    @asynccontextmanager
    async def observe_operation(self, operation_name: str, **attributes):
        """觀察操作的上下文管理器"""
        start_time = time.time()
        
        async with self.tracer.trace_operation(operation_name) as trace_context:
            # 添加屬性到跨度
            for key, value in attributes.items():
                self.tracer.add_span_attribute(trace_context.span_id, key, value)
            
            try:
                yield trace_context
                duration = time.time() - start_time
                await self.record_request(operation_name, duration, "success", **attributes)
                
            except Exception as e:
                duration = time.time() - start_time
                await self.record_request(operation_name, duration, "error", **attributes)
                await self.record_error(operation_name, e, trace_context)
                raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """獲取健康狀態"""
        uptime = time.time() - self.start_time
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        
        return {
            "service": self.service_name,
            "status": "healthy" if error_rate < 5 else "degraded" if error_rate < 20 else "unhealthy",
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": error_rate,
            "metrics_summary": self.metrics.get_metrics_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """獲取儀表板數據"""
        return {
            "service_info": {
                "name": self.service_name,
                "uptime": time.time() - self.start_time,
                "version": "1.0.0"
            },
            "health": self.get_health_status(),
            "metrics": self.metrics.get_metrics_summary(),
            "active_traces": len(self.tracer.active_spans)
        }

# 全域觀測性管理器
_observability_frameworks: Dict[str, ObservabilityFramework] = {}

def get_observability_framework(service_name: str, config: Optional[Dict[str, Any]] = None) -> ObservabilityFramework:
    """獲取或創建可觀測性框架"""
    if service_name not in _observability_frameworks:
        _observability_frameworks[service_name] = ObservabilityFramework(service_name, config)
    return _observability_frameworks[service_name]

def setup_aiva_observability():
    """設置 AIVA 五大模組的可觀測性"""
    modules = ["core", "features", "integration", "scan", "aiva_common"]
    
    config = {
        "log_path": "./logs",
        "jaeger_endpoint": "http://localhost:14268/api/traces"
    }
    
    frameworks = {}
    for module in modules:
        frameworks[module] = get_observability_framework(module, config)
    
    return frameworks

# 裝飾器支援
def observe_function(operation_name: str = None, service_name: str = "default"):
    """函數可觀測性裝飾器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            framework = get_observability_framework(service_name)
            
            async with framework.observe_operation(op_name, function=func.__name__):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            framework = get_observability_framework(service_name)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                asyncio.create_task(framework.record_request(op_name, duration, "success"))
                return result
            except Exception as e:
                duration = time.time() - start_time
                asyncio.create_task(framework.record_request(op_name, duration, "error"))
                asyncio.create_task(framework.record_error(op_name, e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 測試和驗證
async def test_observability_framework():
    """測試可觀測性框架"""
    print("📊 測試 AIVA 可觀測性框架")
    
    # 設置框架
    frameworks = setup_aiva_observability()
    
    # 測試各個元件
    test_framework = frameworks["core"]
    
    # 測試操作觀察
    async with test_framework.observe_operation("test_operation", test_param="value"):
        await asyncio.sleep(0.1)  # 模擬操作
        
    # 測試錯誤記錄
    try:
        async with test_framework.observe_operation("test_error"):
            raise ValueError("測試錯誤")
    except ValueError:
        pass
    
    # 獲取健康狀態
    health = test_framework.get_health_status()
    dashboard = test_framework.get_dashboard_data()
    
    print(f"✅ 健康狀態: {health['status']}")
    print(f"✅ 總請求數: {health['total_requests']}")
    print(f"✅ 錯誤率: {health['error_rate_percent']:.1f}%")
    print(f"✅ 度量摘要: {health['metrics_summary']['total_metrics']} 個度量")
    
    return {
        "health": health,
        "dashboard": dashboard,
        "frameworks_count": len(frameworks)
    }

if __name__ == "__main__":
    asyncio.run(test_observability_framework())