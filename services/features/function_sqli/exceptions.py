"""
SQLi 檢測專用異常類別
提供具體的錯誤類型和詳細的錯誤信息
"""

from __future__ import annotations


class SqliError(Exception):
    """SQLi檢測基礎異常"""

    def __init__(self, message: str, engine_name: str = "", payload: str = ""):
        self.engine_name = engine_name
        self.payload = payload
        super().__init__(message)

    def __str__(self) -> str:
        parts = []
        if self.engine_name:
            parts.append(f"[{self.engine_name}]")
        parts.append(super().__str__())
        if self.payload:
            parts.append(f"(payload: {self.payload[:50]}...)")
        return " ".join(parts)


class EngineExecutionError(SqliError):
    """引擎執行異常"""

    def __init__(self, engine_name: str, original_error: Exception, payload: str = ""):
        self.original_error = original_error
        message = f"Engine execution failed: {original_error}"
        super().__init__(message, engine_name, payload)


class PayloadGenerationError(SqliError):
    """載荷生成異常"""

    def __init__(self, message: str, payload_type: str = ""):
        self.payload_type = payload_type
        super().__init__(f"Payload generation error: {message}")


class ConfigurationError(SqliError):
    """配置錯誤異常"""

    def __init__(self, message: str, config_key: str = ""):
        self.config_key = config_key
        super().__init__(f"Configuration error: {message}")


class NetworkError(SqliError):
    """網路請求異常"""

    def __init__(self, message: str, url: str = "", status_code: int = 0):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Network error: {message}")

    def __str__(self) -> str:
        parts = [super(SqliError, self).__str__()]
        if self.url:
            parts.append(f"(URL: {self.url})")
        if self.status_code:
            parts.append(f"(Status: {self.status_code})")
        return " ".join(parts)


class TimeoutError(SqliError):
    """超時異常"""

    def __init__(self, message: str, timeout_seconds: float = 0):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Timeout error: {message}")


class ValidationError(SqliError):
    """驗證錯誤異常"""

    def __init__(self, message: str, field_name: str = ""):
        self.field_name = field_name
        super().__init__(f"Validation error: {message}")


class DetectionResultError(SqliError):
    """檢測結果處理異常"""

    def __init__(self, message: str, result_data: str = ""):
        self.result_data = result_data
        super().__init__(f"Detection result error: {message}")


# 向後兼容的別名
SqliDetectionError = SqliError
SQLiException = SqliError  # 匹配 __init__.py 中的導入
