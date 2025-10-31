"""
AIVA Cross-Language Error Handling System
跨語言統一錯誤處理系統

此模組提供：
1. 統一錯誤碼映射
2. 跨語言錯誤轉換
3. 錯誤分類與處理策略
4. 錯誤統計與監控
5. 自動錯誤恢復機制
"""

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AIVAErrorCode(Enum):
    """AIVA 統一錯誤碼"""

    # 通用錯誤 (0-99)
    SUCCESS = 1
    UNKNOWN = 2
    INTERNAL_ERROR = 3
    INVALID_ARGUMENT = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    TIMEOUT = 16
    CANCELLED = 17
    CONFIGURATION_ERROR = 20

    # 網路相關錯誤 (100-199)
    NETWORK_UNREACHABLE = 100
    CONNECTION_REFUSED = 101
    CONNECTION_TIMEOUT = 102
    DNS_RESOLUTION_FAILED = 104
    SSL_HANDSHAKE_FAILED = 105
    HTTP_UNAUTHORIZED = 108
    HTTP_FORBIDDEN = 109
    HTTP_NOT_FOUND = 110
    HTTP_INTERNAL_SERVER_ERROR = 112
    RATE_LIMITED = 117

    # 資料庫相關錯誤 (200-299)
    DATABASE_CONNECTION_FAILED = 200
    DATABASE_TIMEOUT = 201
    DATABASE_CONSTRAINT_VIOLATION = 203
    DATABASE_INTEGRITY_ERROR = 204

    # 認證授權錯誤 (300-399)
    AUTH_INVALID_CREDENTIALS = 300
    AUTH_TOKEN_EXPIRED = 301
    AUTH_TOKEN_INVALID = 302
    AUTH_INSUFFICIENT_PERMISSIONS = 304

    # 檔案系統錯誤 (400-499)
    FILE_NOT_FOUND = 400
    FILE_ACCESS_DENIED = 401
    FILE_TOO_LARGE = 403
    FILE_CORRUPTED = 404
    DISK_FULL = 408

    # 掃描相關錯誤 (500-599)
    SCAN_INITIALIZATION_FAILED = 500
    SCAN_TARGET_UNREACHABLE = 501
    SCAN_TIMEOUT = 504
    SCAN_INTERRUPTED = 505
    SCAN_ENGINE_ERROR = 508

    # AI 相關錯誤 (700-799)
    AI_MODEL_NOT_LOADED = 700
    AI_MODEL_INFERENCE_FAILED = 701
    AI_INSUFFICIENT_CONTEXT = 702
    AI_MEMORY_EXHAUSTED = 704
    AI_SERVICE_UNAVAILABLE = 708

    # 跨語言特定錯誤 (1400-1499)
    RUST_PANIC = 1400
    RUST_COMPILATION_FAILED = 1401
    PYTHON_IMPORT_ERROR = 1403
    PYTHON_SYNTAX_ERROR = 1404
    PYTHON_RUNTIME_ERROR = 1405
    GO_PANIC = 1406
    JAVASCRIPT_RUNTIME_ERROR = 1410

    # gRPC 錯誤 (1500-1599)
    GRPC_UNAVAILABLE = 1500
    GRPC_DEADLINE_EXCEEDED = 1501
    GRPC_INTERNAL = 1502
    GRPC_INVALID_ARGUMENT = 1503
    GRPC_NOT_FOUND = 1504
    GRPC_PERMISSION_DENIED = 1506


class ErrorSeverity(Enum):
    """錯誤嚴重程度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class ErrorContext:
    """錯誤上下文信息"""

    service_name: str
    function_name: str
    file_name: str
    line_number: int
    language: str
    timestamp: float
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIVAError:
    """AIVA 統一錯誤結構"""

    error_code: AIVAErrorCode
    message: str
    severity: ErrorSeverity
    context: ErrorContext
    original_error: Exception | None = None
    is_recoverable: bool = True
    requires_user_action: bool = False
    suggested_actions: list[str] = field(default_factory=list)
    error_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return {
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "message": self.message,
            "severity": self.severity.value,
            "context": {
                "service_name": self.context.service_name,
                "function_name": self.context.function_name,
                "file_name": self.context.file_name,
                "line_number": self.context.line_number,
                "language": self.context.language,
                "timestamp": self.context.timestamp,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "additional_data": self.context.additional_data,
            },
            "is_recoverable": self.is_recoverable,
            "requires_user_action": self.requires_user_action,
            "suggested_actions": self.suggested_actions,
            "error_id": self.error_id,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class LanguageErrorMapper:
    """語言特定錯誤映射器"""

    def __init__(self, language: str):
        self.language = language
        self.logger = setup_logger(f"error_mapper.{language}")
        self._mappings: dict[type[Exception], AIVAErrorCode] = {}
        self._setup_default_mappings()

    def _setup_default_mappings(self):
        """設置默認錯誤映射"""
        if self.language == "python":
            self._mappings.update(
                {
                    ImportError: AIVAErrorCode.PYTHON_IMPORT_ERROR,
                    SyntaxError: AIVAErrorCode.PYTHON_SYNTAX_ERROR,
                    RuntimeError: AIVAErrorCode.PYTHON_RUNTIME_ERROR,
                    FileNotFoundError: AIVAErrorCode.FILE_NOT_FOUND,
                    PermissionError: AIVAErrorCode.FILE_ACCESS_DENIED,
                    TimeoutError: AIVAErrorCode.TIMEOUT,
                    ConnectionError: AIVAErrorCode.CONNECTION_REFUSED,
                    ValueError: AIVAErrorCode.INVALID_ARGUMENT,
                    KeyError: AIVAErrorCode.NOT_FOUND,
                    OSError: AIVAErrorCode.INTERNAL_ERROR,
                    MemoryError: AIVAErrorCode.RESOURCE_EXHAUSTED,
                }
            )

    def register_mapping(
        self, exception_type: type[Exception], error_code: AIVAErrorCode
    ):
        """註冊自定義錯誤映射"""
        self._mappings[exception_type] = error_code
        self.logger.debug(
            f"Registered mapping: {exception_type.__name__} -> {error_code.name}"
        )

    def map_error(self, error: Exception) -> AIVAErrorCode:
        """映射錯誤到 AIVA 錯誤碼"""
        error_type = type(error)

        # 直接映射
        if error_type in self._mappings:
            return self._mappings[error_type]

        # 繼承映射
        for mapped_type, error_code in self._mappings.items():
            if isinstance(error, mapped_type):
                return error_code

        # 默認錯誤
        return AIVAErrorCode.UNKNOWN


class ErrorHandler:
    """統一錯誤處理器"""

    def __init__(self):
        self.logger = setup_logger("error_handler")
        self._language_mappers: dict[str, LanguageErrorMapper] = {}
        self._error_stats: dict[AIVAErrorCode, int] = {}
        self._recovery_strategies: dict[AIVAErrorCode, list[Callable]] = {}
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """設置默認恢復策略"""
        self._recovery_strategies.update(
            {
                AIVAErrorCode.CONNECTION_REFUSED: [self._retry_connection],
                AIVAErrorCode.TIMEOUT: [self._increase_timeout, self._retry_operation],
                AIVAErrorCode.RATE_LIMITED: [self._wait_and_retry],
                AIVAErrorCode.AI_SERVICE_UNAVAILABLE: [self._switch_to_fallback_ai],
                AIVAErrorCode.DATABASE_CONNECTION_FAILED: [self._reconnect_database],
            }
        )

    def register_language_mapper(self, language: str, mapper: LanguageErrorMapper):
        """註冊語言錯誤映射器"""
        self._language_mappers[language] = mapper
        self.logger.info(f"Registered error mapper for {language}")

    def get_or_create_mapper(self, language: str) -> LanguageErrorMapper:
        """獲取或創建語言映射器"""
        if language not in self._language_mappers:
            self._language_mappers[language] = LanguageErrorMapper(language)
        return self._language_mappers[language]

    def handle_error(
        self, error: Exception, context: ErrorContext, auto_recover: bool = True
    ) -> AIVAError:
        """處理錯誤"""
        # 獲取映射器並映射錯誤
        mapper = self.get_or_create_mapper(context.language)
        error_code = mapper.map_error(error)

        # 確定嚴重程度
        severity = self._determine_severity(error_code)

        # 創建 AIVA 錯誤
        aiva_error = AIVAError(
            error_code=error_code,
            message=str(error),
            severity=severity,
            context=context,
            original_error=error,
            is_recoverable=self._is_recoverable(error_code),
            requires_user_action=self._requires_user_action(error_code),
            suggested_actions=self._get_suggested_actions(error_code),
            error_id=self._generate_error_id(),
        )

        # 記錄錯誤
        self._log_error(aiva_error)

        # 更新統計
        self._update_stats(error_code)

        # 嘗試自動恢復
        if auto_recover and aiva_error.is_recoverable:
            asyncio.create_task(self._attempt_recovery(aiva_error))

        return aiva_error

    def _determine_severity(self, error_code: AIVAErrorCode) -> ErrorSeverity:
        """確定錯誤嚴重程度"""
        critical_errors = {
            AIVAErrorCode.INTERNAL_ERROR,
            AIVAErrorCode.RESOURCE_EXHAUSTED,
            AIVAErrorCode.AI_MEMORY_EXHAUSTED,
            AIVAErrorCode.RUST_PANIC,
            AIVAErrorCode.GO_PANIC,
        }

        high_errors = {
            AIVAErrorCode.PERMISSION_DENIED,
            AIVAErrorCode.AUTH_INVALID_CREDENTIALS,
            AIVAErrorCode.DATABASE_CONNECTION_FAILED,
            AIVAErrorCode.AI_SERVICE_UNAVAILABLE,
        }

        medium_errors = {
            AIVAErrorCode.TIMEOUT,
            AIVAErrorCode.CONNECTION_REFUSED,
            AIVAErrorCode.FILE_NOT_FOUND,
            AIVAErrorCode.SCAN_ENGINE_ERROR,
        }

        if error_code in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_code in high_errors:
            return ErrorSeverity.HIGH
        elif error_code in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _is_recoverable(self, error_code: AIVAErrorCode) -> bool:
        """判斷錯誤是否可恢復"""
        non_recoverable = {
            AIVAErrorCode.PERMISSION_DENIED,
            AIVAErrorCode.AUTH_INVALID_CREDENTIALS,
            AIVAErrorCode.PYTHON_SYNTAX_ERROR,
            AIVAErrorCode.RUST_COMPILATION_FAILED,
            AIVAErrorCode.INVALID_ARGUMENT,
        }
        return error_code not in non_recoverable

    def _requires_user_action(self, error_code: AIVAErrorCode) -> bool:
        """判斷是否需要用戶操作"""
        user_action_required = {
            AIVAErrorCode.PERMISSION_DENIED,
            AIVAErrorCode.AUTH_INVALID_CREDENTIALS,
            AIVAErrorCode.AUTH_TOKEN_EXPIRED,
            AIVAErrorCode.CONFIGURATION_ERROR,
            AIVAErrorCode.DISK_FULL,
        }
        return error_code in user_action_required

    def _get_suggested_actions(self, error_code: AIVAErrorCode) -> list[str]:
        """獲取建議操作"""
        suggestions = {
            AIVAErrorCode.CONNECTION_REFUSED: [
                "檢查網路連接",
                "確認服務是否運行",
                "檢查防火牆設置",
            ],
            AIVAErrorCode.AUTH_TOKEN_EXPIRED: [
                "重新登錄獲取新的身份驗證令牌",
                "檢查系統時間是否正確",
            ],
            AIVAErrorCode.FILE_NOT_FOUND: [
                "檢查文件路徑是否正確",
                "確認文件是否存在",
                "檢查文件權限",
            ],
            AIVAErrorCode.AI_SERVICE_UNAVAILABLE: [
                "稍後重試",
                "檢查 AI 服務狀態",
                "使用備用 AI 服務",
            ],
            AIVAErrorCode.DISK_FULL: [
                "清理磁盤空間",
                "刪除不必要的文件",
                "移動數據到其他磁盤",
            ],
        }
        return suggestions.get(error_code, ["聯繫系統管理員"])

    def _generate_error_id(self) -> str:
        """生成錯誤 ID"""
        import uuid

        return str(uuid.uuid4())[:8]

    def _log_error(self, error: AIVAError):
        """記錄錯誤"""
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.INFORMATIONAL: logging.DEBUG,
        }.get(error.severity, logging.ERROR)

        self.logger.log(
            log_level,
            f"[{error.error_id}] {error.error_code.name}: {error.message} "
            f"({error.context.service_name}:{error.context.function_name})",
        )

    def _update_stats(self, error_code: AIVAErrorCode):
        """更新錯誤統計"""
        self._error_stats[error_code] = self._error_stats.get(error_code, 0) + 1

    async def _attempt_recovery(self, error: AIVAError):
        """嘗試錯誤恢復"""
        if error.error_code not in self._recovery_strategies:
            return

        strategies = self._recovery_strategies[error.error_code]
        for strategy in strategies:
            try:
                success = await strategy(error)
                if success:
                    self.logger.info(
                        f"Successfully recovered from {error.error_code.name}"
                    )
                    return
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {e}")

        self.logger.warning(
            f"All recovery strategies failed for {error.error_code.name}"
        )

    # 恢復策略實現
    async def _retry_connection(self, error: AIVAError) -> bool:
        """重試連接策略"""
        await asyncio.sleep(1)
        # 實際的重連邏輯
        return False

    async def _increase_timeout(self, error: AIVAError) -> bool:
        """增加超時時間策略"""
        # 實際的超時調整邏輯
        return False

    async def _retry_operation(self, error: AIVAError) -> bool:
        """重試操作策略"""
        await asyncio.sleep(0.5)
        # 實際的重試邏輯
        return False

    async def _wait_and_retry(self, error: AIVAError) -> bool:
        """等待後重試策略"""
        await asyncio.sleep(5)
        # 實際的重試邏輯
        return False

    async def _switch_to_fallback_ai(self, error: AIVAError) -> bool:
        """切換到備用 AI 服務策略"""
        # 實際的切換邏輯
        return False

    async def _reconnect_database(self, error: AIVAError) -> bool:
        """重新連接資料庫策略"""
        # 實際的重連邏輯
        return False

    def get_error_stats(self) -> dict[str, Any]:
        """獲取錯誤統計"""
        return {
            "total_errors": sum(self._error_stats.values()),
            "error_counts": {
                code.name: count for code, count in self._error_stats.items()
            },
            "registered_languages": list(self._language_mappers.keys()),
        }


# 全局錯誤處理器
_global_error_handler: ErrorHandler | None = None


def get_error_handler() -> ErrorHandler:
    """獲取全局錯誤處理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def create_error_context(
    service_name: str, function_name: str = "", language: str = "python"
) -> ErrorContext:
    """創建錯誤上下文"""
    frame = sys._getframe(1) if function_name == "" else None

    return ErrorContext(
        service_name=service_name,
        function_name=function_name or (frame.f_code.co_name if frame else "unknown"),
        file_name=frame.f_code.co_filename if frame else "unknown",
        line_number=frame.f_lineno if frame else 0,
        language=language,
        timestamp=time.time(),
    )


def handle_error(
    error: Exception,
    service_name: str,
    function_name: str = "",
    language: str = "python",
) -> AIVAError:
    """便利函數：處理錯誤"""
    context = create_error_context(service_name, function_name, language)
    handler = get_error_handler()
    return handler.handle_error(error, context)


# 裝飾器
def error_handler(service_name: str, language: str = "python"):
    """錯誤處理裝飾器"""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    aiva_error = handle_error(e, service_name, func.__name__, language)
                    raise AIVAException(aiva_error) from e

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    aiva_error = handle_error(e, service_name, func.__name__, language)
                    raise AIVAException(aiva_error) from e

            return sync_wrapper

    return decorator


class AIVAException(Exception):
    """AIVA 統一異常類"""

    def __init__(self, aiva_error: AIVAError):
        self.aiva_error = aiva_error
        super().__init__(aiva_error.message)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式"""
        return self.aiva_error.to_dict()

    def __str__(self) -> str:
        return f"[{self.aiva_error.error_code.name}] {self.aiva_error.message}"


if __name__ == "__main__":
    # 測試錯誤處理
    def test_error_handling():
        try:
            raise FileNotFoundError("Test file not found")
        except Exception as e:
            aiva_error = handle_error(e, "test_service")
            print("Error handled:", aiva_error.to_dict())

    test_error_handling()
