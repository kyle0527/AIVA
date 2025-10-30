"""
SSRF 模組專用數據合約
定義 SSRF 檢測相關的所有數據結構，基於 Pydantic v2.12.0
"""



from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.aiva_common.schemas import (
    FindingPayload,
    FunctionTelemetry,
    OastEvent,
)


class SsrfTestVector(BaseModel):
    """SSRF 測試向量 - 官方 Pydantic BaseModel"""

    payload: str
    vector_type: str  # "internal", "cloud_metadata", "oast", "cross_protocol"
    priority: int = Field(default=5, ge=1, le=10)
    requires_oast: bool = False
    protocol: str = "http"  # "http", "https", "ftp", "gopher", "file"

    @field_validator("vector_type")
    def validate_vector_type(cls, v: str) -> str:
        """驗證向量類型"""
        allowed = {"internal", "cloud_metadata", "oast", "cross_protocol", "dns"}
        if v not in allowed:
            raise ValueError(f"Invalid vector_type: {v}. Must be one of {allowed}")
        return v

    @field_validator("protocol")
    def validate_protocol(cls, v: str) -> str:
        """驗證協議"""
        allowed = {"http", "https", "ftp", "gopher", "file", "dict", "ldap", "smb"}
        if v not in allowed:
            raise ValueError(f"Invalid protocol: {v}. Must be one of {allowed}")
        return v.lower()


class AnalysisPlan(BaseModel):
    """SSRF 分析計劃 - 官方 Pydantic BaseModel"""

    vectors: list[SsrfTestVector]
    param_name: str | None = None
    semantic_hints: list[str] = Field(default_factory=list)
    requires_oast: bool = False
    estimated_tests: int = 0

    def __init__(self, **data):
        super().__init__(**data)
        # 自動計算需要的測試數量
        if self.estimated_tests == 0:
            self.estimated_tests = len(self.vectors)

    @field_validator("vectors")
    def validate_vectors(cls, v: list[SsrfTestVector]) -> list[SsrfTestVector]:
        """驗證測試向量"""
        if not v:
            raise ValueError("At least one test vector required")
        if len(v) > 1000:
            raise ValueError("Too many test vectors (max 1000)")
        return v


class SsrfTelemetry(FunctionTelemetry):
    """SSRF 專用遙測數據 - 繼承自 FunctionTelemetry"""

    oast_callbacks: int = 0
    internal_access: int = 0
    cloud_metadata_access: int = 0
    dns_lookups: int = 0
    protocols_tested: list[str] = Field(default_factory=list)

    def record_oast_callback(self) -> None:
        """記錄 OAST 回調"""
        self.oast_callbacks += 1

    def record_internal_access(self) -> None:
        """記錄內部訪問"""
        self.internal_access += 1

    def record_cloud_metadata_access(self) -> None:
        """記錄雲元數據訪問"""
        self.cloud_metadata_access += 1

    def record_dns_lookup(self) -> None:
        """記錄 DNS 查找"""
        self.dns_lookups += 1

    def record_protocol(self, protocol: str) -> None:
        """記錄測試的協議"""
        if protocol not in self.protocols_tested:
            self.protocols_tested.append(protocol)

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details = super().to_details(findings_count)
        details.update(
            {
                "oast_callbacks": self.oast_callbacks,
                "internal_access": self.internal_access,
                "cloud_metadata_access": self.cloud_metadata_access,
                "dns_lookups": self.dns_lookups,
                "protocols_tested": self.protocols_tested,
            }
        )
        return details


class TaskExecutionResult(BaseModel):
    """SSRF 任務執行結果 - 官方 Pydantic BaseModel"""

    findings: list[FindingPayload]
    telemetry: SsrfTelemetry


class InternalAddressDetectionResult(BaseModel):
    """內部地址檢測結果 - 官方 Pydantic BaseModel"""

    is_internal_access: bool
    target_ip: str | None = None
    response_contains_internal_data: bool = False
    evidence: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


# 導出 OastEvent 以便其他模組使用
__all__ = [
    "SsrfTestVector",
    "AnalysisPlan",
    "SsrfTelemetry",
    "TaskExecutionResult",
    "InternalAddressDetectionResult",
    "OastEvent",
]
