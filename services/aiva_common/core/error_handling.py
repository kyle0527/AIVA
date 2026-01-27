"""AIVA Common - 統一錯誤處理機制

提供跨組件的統一錯誤處理，符合 AIVA 開發規範。

功能特性：
- 統一的異常類型和嚴重程度分類
- 自動堆棧追蹤和上下文捕獲
- 結構化的錯誤記錄和報告
- 支持跨組件的錯誤傳播和追蹤
- 整合系統監控和告警機制
"""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """錯誤類型枚舉"""

    SYSTEM = "system"
    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """錯誤嚴重程度枚舉"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ErrorContext:
    """錯誤上下文信息"""

    timestamp: datetime = field(default_factory=datetime.now)
    module: str | None = None
    function: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


class AIVAError(Exception):
    """AIVA 系統統一異常類"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception

        # 自動捕獲堆棧信息
        self.stack_trace = traceback.format_exc() if original_exception else None

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "message": self.message,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "module": self.context.module,
            "function": self.context.function,
            "user_id": self.context.user_id,
            "session_id": self.context.session_id,
            "request_id": self.context.request_id,
            "additional_data": self.context.additional_data,
            "stack_trace": self.stack_trace,
        }

    def __str__(self) -> str:
        return f"[{self.error_type.value.upper()}:{self.severity.value.upper()}] {self.message}"


class ErrorHandler:
    """錯誤處理器"""

    def __init__(self, logger_name: str | None = None):
        self.logger = logging.getLogger(logger_name or __name__)
        self.error_callbacks = []

    def add_error_callback(self, callback):
        """添加錯誤回調函數"""
        self.error_callbacks.append(callback)

    def handle_error(
        self, error: Exception | AIVAError, context: ErrorContext | None = None
    ) -> AIVAError:
        """處理錯誤"""

        # 如果已經是 AIVAError，直接使用
        if isinstance(error, AIVAError):
            aiva_error = error
        else:
            # 轉換為 AIVAError
            aiva_error = AIVAError(
                message=str(error),
                error_type=self._classify_error(error),
                severity=self._determine_severity(error),
                context=context,
                original_exception=error,
            )

        # 記錄錯誤
        self._log_error(aiva_error)

        # 執行回調
        for callback in self.error_callbacks:
            try:
                callback(aiva_error)
            except Exception as callback_error:
                self.logger.error(f"Error in error callback: {callback_error}")

        return aiva_error

    def _classify_error(self, error: Exception) -> ErrorType:
        """分類錯誤類型"""
        error_type_name = type(error).__name__

        if "Network" in error_type_name or "Connection" in error_type_name:
            return ErrorType.NETWORK
        elif "Database" in error_type_name or "SQL" in error_type_name:
            return ErrorType.DATABASE
        elif "Auth" in error_type_name:
            return ErrorType.AUTHENTICATION
        elif "Permission" in error_type_name:
            return ErrorType.AUTHORIZATION
        elif "Validation" in error_type_name or "ValueError" in error_type_name:
            return ErrorType.VALIDATION
        elif "Config" in error_type_name:
            return ErrorType.CONFIGURATION
        else:
            return ErrorType.SYSTEM

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """確定錯誤嚴重程度"""
        error_type_name = type(error).__name__

        # 關鍵錯誤
        if any(
            keyword in error_type_name for keyword in ["System", "Critical", "Fatal"]
        ):
            return ErrorSeverity.CRITICAL

        # 高優先級錯誤
        elif any(
            keyword in error_type_name for keyword in ["Connection", "Database", "Auth"]
        ):
            return ErrorSeverity.HIGH

        # 中等優先級錯誤
        elif any(
            keyword in error_type_name for keyword in ["Validation", "Permission"]
        ):
            return ErrorSeverity.MEDIUM

        # 低優先級錯誤
        else:
            return ErrorSeverity.LOW

    def _log_error(self, error: AIVAError):
        """記錄錯誤"""
        log_method = {
            ErrorSeverity.CRITICAL: self.logger.critical,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.LOW: self.logger.info,
            ErrorSeverity.INFO: self.logger.info,
        }.get(error.severity, self.logger.error)

        log_method(f"{error} | Context: {error.context.additional_data}")

        if error.stack_trace and error.severity in [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.HIGH,
        ]:
            self.logger.debug(f"Stack trace: {error.stack_trace}")


# 全局錯誤處理器實例
_global_error_handler = None


def get_error_handler(logger_name: str | None = None) -> ErrorHandler:
    """獲取錯誤處理器實例"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(logger_name)
    return _global_error_handler


def create_error_context(
    module: str | None = None,
    function: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    **kwargs,
) -> ErrorContext:
    """創建錯誤上下文"""
    return ErrorContext(
        module=module,
        function=function,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        additional_data=kwargs,
    )


def handle_exception(
    exception: Exception,
    context: ErrorContext | None = None,
    logger_name: str | None = None,
) -> AIVAError:
    """便捷的異常處理函數"""
    handler = get_error_handler(logger_name)
    return handler.handle_error(exception, context)


# 裝飾器
def error_handler(
    error_type: ErrorType = ErrorType.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    logger_name: str | None = None,
):
    """錯誤處理裝飾器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = create_error_context(
                    module=func.__module__, function=func.__name__
                )

                aiva_error = AIVAError(
                    message=str(e),
                    error_type=error_type,
                    severity=severity,
                    context=context,
                    original_exception=e,
                )

                handler = get_error_handler(logger_name)
                handler.handle_error(aiva_error)
                raise aiva_error

        return wrapper

    return decorator


# 模組導出
__all__ = [
    "ErrorType",
    "ErrorSeverity",
    "ErrorContext",
    "AIVAError",
    "ErrorHandler",
    "get_error_handler",
    "create_error_context",
    "handle_exception",
    "error_handler",
]
