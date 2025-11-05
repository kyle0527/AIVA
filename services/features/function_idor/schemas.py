"""
IDOR 模組專用數據合約
定義 IDOR (Insecure Direct Object References) 檢測相關的所有數據結構，基於 Pydantic v2.12.0

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
    "services/features/function_idor/schemas.py is deprecated (V1 architecture). "
    "Migrate to V2: services/aiva_common/core_schema_sot.yaml",
    DeprecationWarning,
    stacklevel=2,
)


class IdorTestVector(BaseModel):
    """IDOR 測試向量 - 官方 Pydantic BaseModel"""

    vector_id: str
    original_id: str
    test_id: str
    resource_type: str  # "user", "document", "order", "profile", etc.
    id_pattern: str  # "numeric", "uuid", "hash", "sequential"
    requires_auth: bool = True
    http_method: str = "GET"
    endpoint_template: str  # "/api/users/{id}", "/documents/{id}/view"
    headers: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)

    @field_validator("http_method")
    def validate_method(cls, v: str) -> str:
        """驗證 HTTP 方法"""
        allowed = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if v.upper() not in allowed:
            raise ValueError(f"Invalid HTTP method: {v}")
        return v.upper()

    @field_validator("id_pattern")
    def validate_pattern(cls, v: str) -> str:
        """驗證 ID 模式"""
        allowed = {"numeric", "uuid", "hash", "sequential", "base64", "custom"}
        if v not in allowed:
            raise ValueError(f"Invalid ID pattern: {v}. Must be one of {allowed}")
        return v

    @field_validator("resource_type")
    def validate_resource_type(cls, v: str) -> str:
        """驗證資源類型"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Resource type cannot be empty")
        return v.strip().lower()


class IdorDetectionResult(BaseModel):
    """IDOR 檢測結果 - 官方 Pydantic BaseModel"""

    test_vector: IdorTestVector
    vulnerable: bool = False
    original_response_code: int
    test_response_code: int
    original_response_size: int = 0
    test_response_size: int = 0
    access_granted: bool = False
    data_leaked: bool = False
    response_similarity: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    execution_time_ms: float = Field(ge=0.0, default=0.0)

    @field_validator("original_response_code", "test_response_code")
    def validate_status_code(cls, v: int) -> int:
        """驗證 HTTP 狀態碼"""
        if not 100 <= v < 600:
            raise ValueError(f"Invalid HTTP status code: {v}")
        return v

    def is_successful_bypass(self) -> bool:
        """判斷是否成功繞過權限檢查"""
        return (
            self.vulnerable
            and self.access_granted
            and self.test_response_code in {200, 201, 202, 204}
        )

    def get_severity_score(self) -> float:
        """計算嚴重程度評分 (0.0-1.0)"""
        score = 0.0

        if self.vulnerable:
            score += 0.3

        if self.access_granted:
            score += 0.4

        if self.data_leaked:
            score += 0.3

        # 響應相似度越高，代表可能洩露了相同的數據
        if self.response_similarity > 0.8:
            score += 0.2

        return min(score * self.confidence, 1.0)


class ResourceAccessPattern(BaseModel):
    """資源存取模式分析 - 官方 Pydantic BaseModel"""

    resource_type: str
    id_examples: list[str] = Field(default_factory=list)
    pattern_regex: str | None = None
    sequential_pattern: bool = False
    predictable_ids: bool = False
    min_id_value: int | None = None
    max_id_value: int | None = None
    id_length: int | None = None
    authorization_required: bool = True

    def analyze_predictability(self) -> float:
        """分析 ID 可預測性 (0.0-1.0, 1.0 = 完全可預測)"""
        if self.sequential_pattern:
            return 1.0

        if self.predictable_ids:
            return 0.8

        # 根據 ID 範圍計算可預測性
        if self.min_id_value and self.max_id_value:
            range_size = self.max_id_value - self.min_id_value
            if range_size < 1000:  # 小範圍高度可預測
                return 0.9
            elif range_size < 10000:
                return 0.6
            else:
                return 0.3

        return 0.0


class IdorTelemetry(FunctionTelemetry):
    """IDOR 專用遙測數據 - 繼承自 FunctionTelemetry"""

    vectors_tested: int = 0
    successful_bypasses: int = 0
    data_leakages: int = 0
    authorization_bypasses: int = 0
    resource_types_found: list[str] = Field(default_factory=list)
    id_patterns_detected: dict[str, int] = Field(
        default_factory=dict
    )  # pattern -> count
    average_response_time_ms: float = 0.0

    def record_vector_test(self, result: IdorDetectionResult) -> None:
        """記錄向量測試結果"""
        self.vectors_tested += 1

        if result.vulnerable:
            self.successful_bypasses += 1

        if result.data_leaked:
            self.data_leakages += 1

        if result.access_granted:
            self.authorization_bypasses += 1

        # 記錄資源類型
        resource_type = result.test_vector.resource_type
        if resource_type not in self.resource_types_found:
            self.resource_types_found.append(resource_type)

        # 記錄 ID 模式
        id_pattern = result.test_vector.id_pattern
        self.id_patterns_detected[id_pattern] = (
            self.id_patterns_detected.get(id_pattern, 0) + 1
        )

        # 更新平均響應時間
        if self.vectors_tested == 1:
            self.average_response_time_ms = result.execution_time_ms
        else:
            self.average_response_time_ms = (
                self.average_response_time_ms * (self.vectors_tested - 1)
                + result.execution_time_ms
            ) / self.vectors_tested

    def get_success_rate(self) -> float:
        """獲取成功率"""
        if self.vectors_tested == 0:
            return 0.0
        return self.successful_bypasses / self.vectors_tested

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details = super().to_details(findings_count)
        details.update(
            {
                "vectors_tested": self.vectors_tested,
                "successful_bypasses": self.successful_bypasses,
                "data_leakages": self.data_leakages,
                "authorization_bypasses": self.authorization_bypasses,
                "success_rate": self.get_success_rate(),
                "resource_types_count": len(self.resource_types_found),
                "id_patterns_detected": self.id_patterns_detected,
                "average_response_time_ms": round(self.average_response_time_ms, 2),
            }
        )
        return details


class TaskExecutionResult(BaseModel):
    """IDOR 任務執行結果 - 官方 Pydantic BaseModel"""

    findings: list[FindingPayload]
    telemetry: IdorTelemetry
    resource_patterns: list[ResourceAccessPattern] = Field(default_factory=list)


# 導出所有公共類
__all__ = [
    "IdorTestVector",
    "IdorDetectionResult",
    "ResourceAccessPattern",
    "IdorTelemetry",
    "TaskExecutionResult",
]
