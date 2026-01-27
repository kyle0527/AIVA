"""
AIVA Monitoring Log Handlers
AIVA 監控日誌處理器

提供與監控系統集成的日誌處理器，實現統一的日誌收集和分析。
"""

import logging
import threading
from typing import Any

from .monitoring import (
    LogEntry,
    LogLevel,
    MonitoringService,
    get_monitoring_service,
)


class MonitoringLogHandler(logging.Handler):
    """監控日誌處理器"""

    def __init__(self, monitoring_service: MonitoringService | None = None):
        super().__init__()
        self.monitoring_service = monitoring_service or get_monitoring_service()
        self.local = threading.local()

    def emit(self, record: logging.LogRecord):
        """發送日誌記錄"""
        try:
            # 轉換日誌級別
            level_mapping = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL,
            }

            log_level = level_mapping.get(record.levelno, LogLevel.INFO)

            # 獲取追蹤信息（如果有的話）
            trace_id = getattr(record, "trace_id", None)
            span_id = getattr(record, "span_id", None)

            # 創建日誌條目
            log_entry = LogEntry(
                timestamp=record.created,
                level=log_level,
                message=self.format(record),
                logger_name=record.name,
                module=record.module if hasattr(record, "module") else "",
                function=record.funcName if record.funcName else "",
                line_number=record.lineno,
                trace_id=trace_id,
                span_id=span_id,
                extra_data=self._extract_extra_data(record),
            )

            # 添加到監控系統
            self.monitoring_service.log_aggregator.add_log(log_entry)

        except Exception:
            # 避免日誌處理器本身出錯影響主程序
            self.handleError(record)

    def _extract_extra_data(self, record: logging.LogRecord) -> dict[str, Any]:
        """提取額外數據"""
        extra_data = {}

        # 提取標準字段之外的數據
        standard_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "exc_info",
            "exc_text",
            "stack_info",
            "trace_id",
            "span_id",
        }

        for key, value in record.__dict__.items():
            if key not in standard_fields:
                try:
                    # 確保值可以序列化
                    if isinstance(
                        value, (str, int, float, bool, list, dict, type(None))
                    ):
                        extra_data[key] = value
                    else:
                        extra_data[key] = str(value)
                except Exception:
                    # 如果無法處理，跳過
                    pass

        # 添加異常信息
        if record.exc_info:
            extra_data["exception"] = self.format(record)

        return extra_data


def setup_monitoring_logging(
    logger_name: str | None = None,
    level: int = logging.INFO,
    monitoring_service: MonitoringService | None = None,
) -> logging.Logger:
    """設置監控日誌"""

    # 獲取或創建 logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # 設置級別
    logger.setLevel(level)

    # 添加監控處理器
    monitoring_handler = MonitoringLogHandler(monitoring_service)
    monitoring_handler.setLevel(level)

    # 設置格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    monitoring_handler.setFormatter(formatter)

    # 檢查是否已經添加了監控處理器
    has_monitoring_handler = any(
        isinstance(handler, MonitoringLogHandler) for handler in logger.handlers
    )

    if not has_monitoring_handler:
        logger.addHandler(monitoring_handler)

    return logger


def get_logger_with_monitoring(
    name: str,
    level: int = logging.INFO,
    monitoring_service: MonitoringService | None = None,
) -> logging.Logger:
    """獲取帶監控的 logger"""
    return setup_monitoring_logging(name, level, monitoring_service)


class TraceLoggerAdapter(logging.LoggerAdapter):
    """追蹤日誌適配器"""

    def __init__(
        self,
        logger: logging.Logger,
        trace_id: str | None = None,
        span_id: str | None = None,
    ):
        super().__init__(logger, {})
        self.trace_id = trace_id
        self.span_id = span_id

    def process(self, msg, kwargs):
        """處理日誌消息"""
        # 添加追蹤信息到額外字段
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if self.trace_id:
            kwargs["extra"]["trace_id"] = self.trace_id
        if self.span_id:
            kwargs["extra"]["span_id"] = self.span_id

        return msg, kwargs

    def update_trace_info(
        self, trace_id: str | None = None, span_id: str | None = None
    ):
        """更新追蹤信息"""
        if trace_id:
            self.trace_id = trace_id
        if span_id:
            self.span_id = span_id


def create_trace_logger(
    logger_name: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    monitoring_service: MonitoringService | None = None,
) -> TraceLoggerAdapter:
    """創建帶追蹤信息的 logger"""
    base_logger = get_logger_with_monitoring(
        logger_name, monitoring_service=monitoring_service
    )
    return TraceLoggerAdapter(base_logger, trace_id, span_id)
