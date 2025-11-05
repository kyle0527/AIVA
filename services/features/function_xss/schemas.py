"""
XSS 模組專用數據合約
定義 XSS 檢測相關的所有數據結構，基於 Pydantic v2.12.0

⚠️ DEPRECATION WARNING - V1 架構 ⚠️
此文件為 V1 架構的 schema 定義，正在逐步遷移到 V2 統一架構。
V2 單一事實來源: services/aiva_common/core_schema_sot.yaml
新功能開發請使用 V2 架構。
"""

import warnings
from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.aiva_common.schemas import (
    FindingPayload,
    FunctionTelemetry,
)

# V1 架構棄用警告
warnings.warn(
    "services/features/function_xss/schemas.py is deprecated (V1 architecture). "
    "Migrate to V2: services/aiva_common/core_schema_sot.yaml",
    DeprecationWarning,
    stacklevel=2,
)


class XssDetectionResult(BaseModel):
    """XSS 檢測結果 - 官方 Pydantic BaseModel"""

    payload: str
    request_url: str
    request_method: str = "GET"
    response_status: int
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_text: str = ""
    reflection_found: bool = False
    context: str | None = None
    sink_type: str | None = None

    @field_validator("request_method")
    def validate_method(cls, v: str) -> str:
        """驗證 HTTP 方法"""
        allowed = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if v.upper() not in allowed:
            raise ValueError(f"Invalid HTTP method: {v}")
        return v.upper()

    @field_validator("response_status")
    def validate_status(cls, v: int) -> int:
        """驗證 HTTP 狀態碼"""
        if not 100 <= v < 600:
            raise ValueError(f"Invalid HTTP status code: {v}")
        return v


class XssTelemetry(FunctionTelemetry):
    """XSS 專用遙測數據 - 繼承自 FunctionTelemetry"""

    reflections: int = 0
    dom_escalations: int = 0
    blind_callbacks: int = 0
    stored_xss_found: int = 0
    contexts_tested: list[str] = Field(default_factory=list)

    def record_reflection(self) -> None:
        """記錄反射發現"""
        self.reflections += 1

    def record_dom_escalation(self) -> None:
        """記錄 DOM XSS 升級"""
        self.dom_escalations += 1

    def record_blind_callback(self) -> None:
        """記錄盲 XSS 回調"""
        self.blind_callbacks += 1

    def record_stored_xss(self) -> None:
        """記錄存儲型 XSS"""
        self.stored_xss_found += 1

    def record_context(self, context: str) -> None:
        """記錄測試的上下文"""
        if context not in self.contexts_tested:
            self.contexts_tested.append(context)

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details = super().to_details(findings_count)
        details.update(
            {
                "reflections": self.reflections,
                "dom_escalations": self.dom_escalations,
                "blind_callbacks": self.blind_callbacks,
                "stored_xss": self.stored_xss_found,
                "contexts_tested": len(self.contexts_tested),
            }
        )
        return details


class DomDetectionResult(BaseModel):
    """DOM XSS 檢測結果 - 官方 Pydantic BaseModel"""

    vulnerable: bool
    sink_type: str  # "innerHTML", "eval", "document.write", "location.href"
    source_type: str  # "location.hash", "location.search", "postMessage"
    payload: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("sink_type")
    def validate_sink_type(cls, v: str) -> str:
        """驗證 sink 類型"""
        allowed = {
            "innerHTML",
            "outerHTML",
            "eval",
            "document.write",
            "document.writeln",
            "location.href",
            "location.assign",
            "setTimeout",
            "setInterval",
        }
        if v not in allowed:
            raise ValueError(f"Invalid sink_type: {v}. Must be one of {allowed}")
        return v

    @field_validator("source_type")
    def validate_source_type(cls, v: str) -> str:
        """驗證 source 類型"""
        allowed = {
            "location.hash",
            "location.search",
            "location.href",
            "postMessage",
            "document.referrer",
            "window.name",
        }
        if v not in allowed:
            raise ValueError(f"Invalid source_type: {v}. Must be one of {allowed}")
        return v


class StoredXssResult(BaseModel):
    """存儲型 XSS 檢測結果 - 官方 Pydantic BaseModel"""

    vulnerable: bool
    storage_url: str  # 存儲 payload 的 URL
    trigger_url: str  # 觸發 payload 的 URL
    payload: str
    storage_response_status: int
    trigger_response_status: int
    evidence: str
    delay_seconds: float = 0.0

    @field_validator("storage_response_status", "trigger_response_status")
    def validate_status(cls, v: int) -> int:
        """驗證 HTTP 狀態碼"""
        if not 100 <= v < 600:
            raise ValueError(f"Invalid HTTP status code: {v}")
        return v


class XssExecutionError(BaseModel):
    """XSS 執行錯誤 - 官方 Pydantic BaseModel"""

    payload: str
    vector: str
    message: str
    attempts: int = 1

    def to_detail(self) -> str:
        """轉換為詳細字符串"""
        prefix = f"[{self.vector}]"
        return f"{prefix} {self.payload!r} failed after {self.attempts} attempts: {self.message}"


class TaskExecutionResult(BaseModel):
    """XSS 任務執行結果 - 官方 Pydantic BaseModel"""

    findings: list[FindingPayload]
    telemetry: XssTelemetry


# 向後兼容別名
XssExecutionTelemetry = XssTelemetry
