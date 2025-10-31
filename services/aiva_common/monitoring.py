"""
AIVA Monitoring and Logging System
AIVA 監控和日誌系統

實施 TODO 項目 12: 建立監控和日誌系統
- 分佈式追蹤和鏈路監控
- 效能指標收集和監控
- 統一日誌聚合和查詢
- 跨語言組件監控支持
- 實時告警和通知系統

特性：
1. 分佈式追蹤：支持請求鏈路追蹤
2. 性能監控：CPU、內存、網絡、延遲監控
3. 日誌聚合：統一收集和存儲各組件日誌
4. 指標收集：業務指標和技術指標
5. 告警系統：閾值監控和實時通知
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
)

import psutil


class MetricType(Enum):
    """指標類型"""

    COUNTER = "counter"  # 計數器
    GAUGE = "gauge"  # 儀表
    HISTOGRAM = "histogram"  # 直方圖
    SUMMARY = "summary"  # 摘要
    TIMER = "timer"  # 計時器


class LogLevel(Enum):
    """日誌級別"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertSeverity(Enum):
    """告警嚴重程度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TraceStatus(Enum):
    """追蹤狀態"""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class MetricData:
    """指標數據"""

    name: str
    type: MetricType
    value: int | float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass
class LogEntry:
    """日誌條目"""

    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    trace_id: str | None = None
    span_id: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    extra_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "labels": self.labels,
            "extra_data": self.extra_data,
        }


@dataclass
class TraceSpan:
    """追蹤跨度"""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    status: TraceStatus = TraceStatus.OK
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def finish(self, status: TraceStatus = TraceStatus.OK):
        """完成跨度"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status

    def add_tag(self, key: str, value: Any):
        """添加標籤"""
        self.tags[key] = value

    def add_log(self, message: str, **kwargs):
        """添加日誌"""
        self.logs.append({"timestamp": time.time(), "message": message, **kwargs})

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs,
        }


@dataclass
class SystemMetrics:
    """系統指標"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "disk_usage_percent": self.disk_usage_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "active_connections": self.active_connections,
            "load_average": self.load_average,
        }


class MetricsCollector:
    """指標收集器"""

    def __init__(self):
        self.metrics: dict[str, list[MetricData]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retention_period = 3600 * 24  # 24小時

    def record_counter(
        self, name: str, value: int | float = 1, labels: dict[str, str] = None
    ):
        """記錄計數器指標"""
        metric = MetricData(
            name=name, type=MetricType.COUNTER, value=value, labels=labels or {}
        )
        self._add_metric(metric)

    def record_gauge(
        self, name: str, value: int | float, labels: dict[str, str] = None
    ):
        """記錄儀表指標"""
        metric = MetricData(
            name=name, type=MetricType.GAUGE, value=value, labels=labels or {}
        )
        self._add_metric(metric)

    def record_timer(self, name: str, duration: float, labels: dict[str, str] = None):
        """記錄計時器指標"""
        metric = MetricData(
            name=name, type=MetricType.TIMER, value=duration, labels=labels or {}
        )
        self._add_metric(metric)

    def _add_metric(self, metric: MetricData):
        """添加指標"""
        with self.lock:
            self.metrics[metric.name].append(metric)
            self._cleanup_old_metrics(metric.name)

    def _cleanup_old_metrics(self, metric_name: str):
        """清理舊指標"""
        cutoff_time = time.time() - self.retention_period
        self.metrics[metric_name] = [
            m for m in self.metrics[metric_name] if m.timestamp > cutoff_time
        ]

    def get_metrics(
        self, name: str = None, start_time: float = None, end_time: float = None
    ) -> list[MetricData]:
        """獲取指標"""
        with self.lock:
            if name:
                metrics = self.metrics.get(name, [])
            else:
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)

            # 時間過濾
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                return filtered_metrics

            return metrics.copy()

    def get_metric_names(self) -> list[str]:
        """獲取所有指標名稱"""
        with self.lock:
            return list(self.metrics.keys())


class SystemMonitor:
    """系統監控器"""

    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self._monitor_task = None

    async def start(self):
        """啟動監控"""
        if self.is_running:
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("系統監控已啟動")

    async def stop(self):
        """停止監控"""
        if not self.is_running:
            return

        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("系統監控已停止")

    async def _monitor_loop(self):
        """監控循環"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                self._record_system_metrics(metrics)
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"系統監控錯誤: {e}")
                await asyncio.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系統指標"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 內存使用率
            memory = psutil.virtual_memory()

            # 磁盤使用率
            disk = psutil.disk_usage("/")

            # 網絡統計
            net_io = psutil.net_io_counters()

            # 網絡連接數
            connections = len(psutil.net_connections())

            # 負載平均值（僅Linux/Unix）
            load_avg = []
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows 不支持
                pass

            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used,
                memory_total=memory.total,
                disk_usage_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_connections=connections,
                load_average=load_avg,
            )

        except Exception as e:
            self.logger.error(f"收集系統指標失敗: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0,
                memory_percent=0,
                memory_used=0,
                memory_total=0,
                disk_usage_percent=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
            )

    def _record_system_metrics(self, metrics: SystemMetrics):
        """記錄系統指標"""
        self.metrics_collector.record_gauge("system.cpu.percent", metrics.cpu_percent)
        self.metrics_collector.record_gauge(
            "system.memory.percent", metrics.memory_percent
        )
        self.metrics_collector.record_gauge("system.memory.used", metrics.memory_used)
        self.metrics_collector.record_gauge(
            "system.disk.percent", metrics.disk_usage_percent
        )
        self.metrics_collector.record_gauge(
            "system.network.bytes_sent", metrics.network_bytes_sent
        )
        self.metrics_collector.record_gauge(
            "system.network.bytes_recv", metrics.network_bytes_recv
        )
        self.metrics_collector.record_gauge(
            "system.connections.active", metrics.active_connections
        )

        if metrics.load_average:
            for i, load in enumerate(metrics.load_average):
                self.metrics_collector.record_gauge(f"system.load.avg_{i+1}min", load)


class TraceManager:
    """追蹤管理器"""

    def __init__(self):
        self.active_traces: dict[str, list[TraceSpan]] = defaultdict(list)
        self.completed_traces: dict[str, list[TraceSpan]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retention_period = 3600 * 24  # 24小時

    def start_trace(
        self, operation_name: str, trace_id: str = None, parent_span_id: str = None
    ) -> TraceSpan:
        """開始新的追蹤跨度"""
        if not trace_id:
            trace_id = str(uuid.uuid4())

        span_id = str(uuid.uuid4())
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
        )

        with self.lock:
            self.active_traces[trace_id].append(span)

        return span

    def finish_span(self, span: TraceSpan, status: TraceStatus = TraceStatus.OK):
        """完成追蹤跨度"""
        span.finish(status)

        with self.lock:
            # 從活動追蹤移到完成追蹤
            if span.trace_id in self.active_traces:
                if span in self.active_traces[span.trace_id]:
                    self.active_traces[span.trace_id].remove(span)
                    if not self.active_traces[span.trace_id]:
                        del self.active_traces[span.trace_id]

            self.completed_traces[span.trace_id].append(span)
            self._cleanup_old_traces()

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """獲取追蹤信息"""
        with self.lock:
            active = self.active_traces.get(trace_id, [])
            completed = self.completed_traces.get(trace_id, [])
            return active + completed

    def get_active_traces(self) -> dict[str, list[TraceSpan]]:
        """獲取所有活動追蹤"""
        with self.lock:
            return {k: v.copy() for k, v in self.active_traces.items()}

    def _cleanup_old_traces(self):
        """清理舊追蹤"""
        cutoff_time = time.time() - self.retention_period

        # 清理完成的追蹤
        traces_to_remove = []
        for trace_id, spans in self.completed_traces.items():
            # 如果追蹤中的所有跨度都太舊，則刪除整個追蹤
            if all(span.start_time < cutoff_time for span in spans):
                traces_to_remove.append(trace_id)

        for trace_id in traces_to_remove:
            del self.completed_traces[trace_id]


class LogAggregator:
    """日誌聚合器"""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.log_entries: deque = deque(maxlen=max_entries)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_log(self, log_entry: LogEntry):
        """添加日誌條目"""
        with self.lock:
            self.log_entries.append(log_entry)

    def query_logs(
        self,
        level: LogLevel = None,
        logger_name: str = None,
        trace_id: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """查詢日誌"""
        with self.lock:
            filtered_logs = []

            for entry in reversed(self.log_entries):  # 最新的在前
                # 應用過濾條件
                if level and entry.level != level:
                    continue
                if logger_name and entry.logger_name != logger_name:
                    continue
                if trace_id and entry.trace_id != trace_id:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

                filtered_logs.append(entry)

                if len(filtered_logs) >= limit:
                    break

            return filtered_logs

    def get_log_stats(self) -> dict[str, Any]:
        """獲取日誌統計"""
        with self.lock:
            if not self.log_entries:
                return {"total": 0}

            level_counts = defaultdict(int)
            logger_counts = defaultdict(int)

            for entry in self.log_entries:
                level_counts[entry.level.value] += 1
                logger_counts[entry.logger_name] += 1

            return {
                "total": len(self.log_entries),
                "level_distribution": dict(level_counts),
                "logger_distribution": dict(logger_counts),
                "oldest_timestamp": self.log_entries[0].timestamp,
                "newest_timestamp": self.log_entries[-1].timestamp,
            }


class AlertRule:
    """告警規則"""

    def __init__(
        self,
        name: str,
        condition: Callable[[list[MetricData]], bool],
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        cooldown: float = 300.0,
    ):  # 5分鐘冷卻期
        self.name = name
        self.condition = condition
        self.severity = severity
        self.cooldown = cooldown
        self.last_alert_time = 0.0

    def should_alert(self, metrics: list[MetricData]) -> bool:
        """檢查是否應該告警"""
        # 檢查冷卻期
        if time.time() - self.last_alert_time < self.cooldown:
            return False

        # 檢查條件
        if self.condition(metrics):
            self.last_alert_time = time.time()
            return True

        return False


@dataclass
class Alert:
    """告警"""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        """解決告警"""
        self.resolved = True
        self.resolved_timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp,
            "metadata": self.metadata,
        }


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.alert_handlers: list[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule):
        """添加告警規則"""
        with self.lock:
            self.rules.append(rule)
            self.logger.info(f"添加告警規則: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除告警規則"""
        with self.lock:
            self.rules = [r for r in self.rules if r.name != rule_name]
            self.logger.info(f"移除告警規則: {rule_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警處理器"""
        self.alert_handlers.append(handler)

    def check_alerts(self, metrics_collector: MetricsCollector):
        """檢查告警條件"""
        with self.lock:
            for rule in self.rules:
                try:
                    # 獲取相關指標
                    recent_time = time.time() - 300  # 最近5分鐘
                    metrics = metrics_collector.get_metrics(start_time=recent_time)

                    if rule.should_alert(metrics):
                        alert = Alert(
                            id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"告警觸發: {rule.name}",
                            timestamp=time.time(),
                        )

                        self._trigger_alert(alert)

                except Exception as e:
                    self.logger.error(f"檢查告警規則失敗 {rule.name}: {e}")

    def _trigger_alert(self, alert: Alert):
        """觸發告警"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        self.logger.warning(f"觸發告警: {alert.message}")

        # 調用告警處理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"告警處理器錯誤: {e}")

    def resolve_alert(self, alert_id: str):
        """解決告警"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                del self.active_alerts[alert_id]
                self.logger.info(f"告警已解決: {alert.message}")

    def get_active_alerts(self) -> list[Alert]:
        """獲取活動告警"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """獲取告警歷史"""
        with self.lock:
            return self.alert_history[-limit:]


class MonitoringService:
    """監控服務"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.trace_manager = TraceManager()
        self.log_aggregator = LogAggregator()
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self._alert_check_task = None

        # 設置默認告警規則
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """設置默認告警規則"""
        # CPU使用率過高
        cpu_rule = AlertRule(
            name="high_cpu_usage",
            condition=lambda metrics: any(
                m.name == "system.cpu.percent" and m.value > 80 for m in metrics
            ),
            severity=AlertSeverity.HIGH,
        )
        self.alert_manager.add_rule(cpu_rule)

        # 內存使用率過高
        memory_rule = AlertRule(
            name="high_memory_usage",
            condition=lambda metrics: any(
                m.name == "system.memory.percent" and m.value > 90 for m in metrics
            ),
            severity=AlertSeverity.CRITICAL,
        )
        self.alert_manager.add_rule(memory_rule)

        # 磁盤使用率過高
        disk_rule = AlertRule(
            name="high_disk_usage",
            condition=lambda metrics: any(
                m.name == "system.disk.percent" and m.value > 85 for m in metrics
            ),
            severity=AlertSeverity.HIGH,
        )
        self.alert_manager.add_rule(disk_rule)

    async def start(self):
        """啟動監控服務"""
        if self.is_running:
            return

        self.is_running = True

        # 啟動系統監控
        await self.system_monitor.start()

        # 啟動告警檢查任務
        self._alert_check_task = asyncio.create_task(self._alert_check_loop())

        self.logger.info("監控服務已啟動")

    async def stop(self):
        """停止監控服務"""
        if not self.is_running:
            return

        self.is_running = False

        # 停止系統監控
        await self.system_monitor.stop()

        # 停止告警檢查
        if self._alert_check_task:
            self._alert_check_task.cancel()
            try:
                await self._alert_check_task
            except asyncio.CancelledError:
                pass

        self.logger.info("監控服務已停止")

    async def _alert_check_loop(self):
        """告警檢查循環"""
        while self.is_running:
            try:
                self.alert_manager.check_alerts(self.metrics_collector)
                await asyncio.sleep(60)  # 每分鐘檢查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"告警檢查錯誤: {e}")
                await asyncio.sleep(60)

    def record_metric(
        self,
        name: str,
        value: int | float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: dict[str, str] = None,
    ):
        """記錄指標"""
        if metric_type == MetricType.COUNTER:
            self.metrics_collector.record_counter(name, value, labels)
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.record_gauge(name, value, labels)
        elif metric_type == MetricType.TIMER:
            self.metrics_collector.record_timer(name, value, labels)

    def start_trace(
        self, operation_name: str, trace_id: str = None, parent_span_id: str = None
    ) -> TraceSpan:
        """開始追蹤"""
        return self.trace_manager.start_trace(operation_name, trace_id, parent_span_id)

    def finish_trace(self, span: TraceSpan, status: TraceStatus = TraceStatus.OK):
        """完成追蹤"""
        self.trace_manager.finish_span(span, status)

    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "default",
        trace_id: str = None,
        span_id: str = None,
        **kwargs,
    ):
        """記錄日誌"""
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=logger_name,
            trace_id=trace_id,
            span_id=span_id,
            extra_data=kwargs,
        )
        self.log_aggregator.add_log(log_entry)

    def get_metrics(
        self, name: str = None, start_time: float = None, end_time: float = None
    ) -> list[MetricData]:
        """獲取指標"""
        return self.metrics_collector.get_metrics(name, start_time, end_time)

    def get_traces(
        self, trace_id: str = None
    ) -> list[TraceSpan] | dict[str, list[TraceSpan]]:
        """獲取追蹤信息"""
        if trace_id:
            return self.trace_manager.get_trace(trace_id)
        else:
            return self.trace_manager.get_active_traces()

    def query_logs(self, **kwargs) -> list[LogEntry]:
        """查詢日誌"""
        return self.log_aggregator.query_logs(**kwargs)

    def get_alerts(self) -> list[Alert]:
        """獲取告警"""
        return self.alert_manager.get_active_alerts()

    def get_system_status(self) -> dict[str, Any]:
        """獲取系統狀態"""
        recent_time = time.time() - 300  # 最近5分鐘
        recent_metrics = self.metrics_collector.get_metrics(start_time=recent_time)

        # 計算平均值
        cpu_values = [m.value for m in recent_metrics if m.name == "system.cpu.percent"]
        memory_values = [
            m.value for m in recent_metrics if m.name == "system.memory.percent"
        ]

        return {
            "timestamp": time.time(),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "avg_memory_percent": (
                sum(memory_values) / len(memory_values) if memory_values else 0
            ),
            "active_traces": len(self.trace_manager.get_active_traces()),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "total_metrics": len(self.metrics_collector.get_metric_names()),
            "log_entries": len(self.log_aggregator.log_entries),
        }


# 全局監控服務實例
_monitoring_service: MonitoringService | None = None


def get_monitoring_service() -> MonitoringService:
    """獲取監控服務實例"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


def create_monitoring_service() -> MonitoringService:
    """創建新的監控服務實例"""
    return MonitoringService()


# 便捷裝飾器和上下文管理器
class TraceContext:
    """追蹤上下文管理器"""

    def __init__(
        self, operation_name: str, monitoring_service: MonitoringService = None
    ):
        self.operation_name = operation_name
        self.monitoring_service = monitoring_service or get_monitoring_service()
        self.span: TraceSpan | None = None

    def __enter__(self) -> TraceSpan:
        self.span = self.monitoring_service.start_trace(self.operation_name)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            status = TraceStatus.ERROR if exc_type else TraceStatus.OK
            if exc_type:
                self.span.add_log(f"Exception: {exc_val}")
            self.monitoring_service.finish_trace(self.span, status)


def trace_operation(
    operation_name: str = None, monitoring_service: MonitoringService = None
):
    """追蹤操作裝飾器"""

    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with TraceContext(operation_name, monitoring_service) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.add_tag("success", True)
                        return result
                    except Exception as e:
                        span.add_tag("success", False)
                        span.add_tag("error", str(e))
                        raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with TraceContext(operation_name, monitoring_service) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.add_tag("success", True)
                        return result
                    except Exception as e:
                        span.add_tag("success", False)
                        span.add_tag("error", str(e))
                        raise

            return sync_wrapper

    return decorator


class TimerContext:
    """計時器上下文管理器"""

    def __init__(
        self,
        metric_name: str,
        monitoring_service: MonitoringService = None,
        labels: dict[str, str] = None,
    ):
        self.metric_name = metric_name
        self.monitoring_service = monitoring_service or get_monitoring_service()
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitoring_service.record_metric(
                self.metric_name, duration, MetricType.TIMER, self.labels
            )


def timer_metric(
    metric_name: str = None,
    monitoring_service: MonitoringService = None,
    labels: dict[str, str] = None,
):
    """計時器指標裝飾器"""

    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"{func.__module__}.{func.__name__}.duration"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with TimerContext(metric_name, monitoring_service, labels):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with TimerContext(metric_name, monitoring_service, labels):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
