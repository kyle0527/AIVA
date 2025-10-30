"""
Scan 模組專用數據合約
定義掃描引擎相關的所有數據結構，基於 Pydantic v2.12.0

修復記錄 (2025-10-29):
- 使用 aiva_common.enums.HttpMethod 替代硬編碼 HTTP 方法驗證
- 遵循統一枚舉原則，提升代碼一致性
"""



from typing import Any

from pydantic import BaseModel, Field, field_validator

# 使用 aiva_common 統一的 HTTP 方法枚舉
from ...aiva_common.enums import HttpMethod

from services.aiva_common.enums import Location, SensitiveInfoType, Severity


class SensitiveMatch(BaseModel):
    """敏感信息匹配結果 - 官方 Pydantic BaseModel"""

    info_type: SensitiveInfoType
    value: str
    location: Location
    context: str
    line_number: int | None = None
    severity: Severity = Severity.MEDIUM
    description: str = ""
    recommendation: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        # 自動設置描述和建議（如果為空）
        if not self.description:
            self.description = self._get_default_description()
        if not self.recommendation:
            self.recommendation = self._get_default_recommendation()

    def _get_default_description(self) -> str:
        """獲取默認描述"""
        descriptions = {
            SensitiveInfoType.API_KEY: "API key exposed in response",
            SensitiveInfoType.ACCESS_TOKEN: "Access token exposed in response",
            SensitiveInfoType.PASSWORD: "Password exposed in response",
            SensitiveInfoType.EMAIL: "Email address exposed",
            SensitiveInfoType.CREDIT_CARD: "Credit card number exposed",
            SensitiveInfoType.DATABASE_CONNECTION: "Database connection string exposed",
            SensitiveInfoType.INTERNAL_IP: "Internal IP address exposed",
            SensitiveInfoType.AWS_KEY: "AWS credentials exposed",
            SensitiveInfoType.FILE_PATH: "Internal file path exposed",
            SensitiveInfoType.STACK_TRACE: "Stack trace exposed",
            SensitiveInfoType.DEBUG_INFO: "Debug information exposed",
        }
        return descriptions.get(self.info_type, f"{self.info_type.value} exposed")

    def _get_default_recommendation(self) -> str:
        """獲取默認建議"""
        if self.info_type in [
            SensitiveInfoType.API_KEY,
            SensitiveInfoType.ACCESS_TOKEN,
            SensitiveInfoType.SECRET_KEY,
            SensitiveInfoType.PASSWORD,
        ]:
            return (
                "Remove credentials from client-side code. "
                "Use secure server-side authentication."
            )
        elif self.info_type in [
            SensitiveInfoType.EMAIL,
            SensitiveInfoType.PHONE,
            SensitiveInfoType.CREDIT_CARD,
        ]:
            return "Mask or remove personal information from responses."
        elif self.info_type in [
            SensitiveInfoType.STACK_TRACE,
            SensitiveInfoType.DEBUG_INFO,
            SensitiveInfoType.ERROR_MESSAGE,
        ]:
            return "Disable debug mode in production. Use generic error messages."
        elif self.info_type == SensitiveInfoType.FILE_PATH:
            return "Remove internal file paths from responses."
        else:
            return f"Remove {self.info_type.value} from responses."


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 代碼分析結果 - 官方 Pydantic BaseModel"""

    url: str
    has_sensitive_data: bool = False
    api_endpoints: list[str] = Field(default_factory=list)
    dom_sinks: list[str] = Field(default_factory=list)
    sensitive_functions: list[str] = Field(default_factory=list)
    external_requests: list[str] = Field(default_factory=list)
    cookies_accessed: list[str] = Field(default_factory=list)

    @field_validator("api_endpoints", "dom_sinks", "external_requests")
    def limit_list_size(cls, v: list[str]) -> list[str]:
        """限制列表大小以防止過大的響應"""
        max_items = 1000
        if len(v) > max_items:
            raise ValueError(f"List too large (max {max_items} items)")
        return v


class NetworkRequest(BaseModel):
    """網絡請求記錄 - 官方 Pydantic BaseModel"""

    url: str
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    timestamp: str = ""
    request_type: str = "xhr"  # "xhr", "fetch", "websocket"

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        """驗證 HTTP 方法 - 使用統一枚舉"""
        # 使用 aiva_common.HttpMethod 枚舉值進行驗證
        allowed = {method.value for method in HttpMethod}
        if v.upper() not in allowed:
            raise ValueError(f"Invalid HTTP method: {v}. Must be one of {sorted(allowed)}")
        return v.upper()

    @field_validator("request_type")
    def validate_request_type(cls, v: str) -> str:
        """驗證請求類型"""
        allowed = {"xhr", "fetch", "websocket", "beacon"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid request_type: {v}. Must be one of {allowed}")
        return v.lower()


class InteractionResult(BaseModel):
    """JavaScript 交互模擬結果 - 官方 Pydantic BaseModel"""

    success: bool
    new_urls: list[str] = Field(default_factory=list)
    network_requests: list[NetworkRequest] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dom_changes: int = 0
    events_triggered: list[str] = Field(default_factory=list)

    @field_validator("new_urls")
    def limit_urls(cls, v: list[str]) -> list[str]:
        """限制 URL 數量"""
        max_urls = 500
        if len(v) > max_urls:
            raise ValueError(f"Too many URLs (max {max_urls})")
        return v


# TypeScript 兼容的動態掃描數據結構
class DynamicScanTask(BaseModel):
    """動態掃描任務 - 與 TypeScript 介面對應"""

    task_id: str
    scan_id: str
    url: str
    extraction_config: dict[str, Any] = Field(default_factory=dict)
    interaction_config: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(ge=1000, le=300000, default=30000)


class DynamicScanResult(BaseModel):
    """動態掃描結果 - 與 TypeScript 介面對應"""

    task_id: str
    scan_id: str
    url: str
    status: str  # "completed", "failed", "timeout"
    contents: list[dict[str, Any]] = Field(default_factory=list)
    interactions: list[dict[str, Any]] = Field(default_factory=list)
    network_requests: list[dict[str, Any]] = Field(default_factory=list)
    dom_changes: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
# 導出所有公共類
__all__ = [
    "SensitiveInfoType",
    "Location",
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
    "NetworkRequest",
    "InteractionResult",
    "DynamicScanTask",
    "DynamicScanResult",
]
