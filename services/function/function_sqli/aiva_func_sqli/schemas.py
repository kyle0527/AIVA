"""
SQLi 模組專用數據合約
定義 SQLi 檢測相關的所有數據結構，基於 Pydantic v2.12.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.aiva_common.schemas import (
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    FunctionTaskPayload,
    FunctionTelemetry,
    Vulnerability,
)


class SqliDetectionResult(BaseModel):
    """SQLi 檢測結果 - 官方 Pydantic BaseModel"""

    is_vulnerable: bool
    vulnerability: Vulnerability
    evidence: FindingEvidence
    impact: FindingImpact
    recommendation: FindingRecommendation
    target: FindingTarget
    detection_method: str  # "error", "boolean", "time", "union", "oob"
    payload_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    db_fingerprint: str | None = None
    response_time: float = 0.0

    @field_validator("detection_method")
    def validate_detection_method(cls, v: str) -> str:
        """驗證檢測方法"""
        allowed = {"error", "boolean", "time", "union", "oob", "stacked"}
        if v not in allowed:
            raise ValueError(f"Invalid detection_method: {v}. Must be one of {allowed}")
        return v


class SqliTelemetry(FunctionTelemetry):
    """SQLi 專用遙測數據 - 繼承自 FunctionTelemetry"""

    engines_run: list[str] = Field(default_factory=list)
    blind_detections: int = 0
    error_based_detections: int = 0
    union_based_detections: int = 0
    time_based_detections: int = 0
    oob_detections: int = 0

    def record_engine_execution(self, engine_name: str) -> None:
        """記錄引擎執行"""
        if engine_name not in self.engines_run:
            self.engines_run.append(engine_name)

    def record_payload_sent(self) -> None:
        """記錄載荷發送"""
        self.payloads_sent += 1
        self.attempts += 1

    def record_detection(self, method: str = "generic") -> None:
        """記錄檢測結果"""
        self.detections += 1
        if method == "error":
            self.error_based_detections += 1
        elif method == "boolean":
            self.blind_detections += 1
        elif method == "time":
            self.time_based_detections += 1
        elif method == "union":
            self.union_based_detections += 1
        elif method == "oob":
            self.oob_detections += 1

    def record_error(self, error_message: str) -> None:
        """記錄錯誤訊息"""
        self.errors.append(error_message)

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式（覆蓋父類方法以添加 SQLi 特定信息）"""
        details = super().to_details(findings_count)
        details.update(
            {
                "engines_run": self.engines_run,
                "error_based": self.error_based_detections,
                "boolean_based": self.blind_detections,
                "time_based": self.time_based_detections,
                "union_based": self.union_based_detections,
                "oob_based": self.oob_detections,
            }
        )
        return details


class SqliEngineConfig(BaseModel):
    """SQLi 引擎配置 - 官方 Pydantic BaseModel"""

    timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    enable_error_detection: bool = True
    enable_boolean_detection: bool = True
    enable_time_detection: bool = True
    enable_union_detection: bool = True
    enable_oob_detection: bool = True
    time_threshold_seconds: float = Field(default=5.0, gt=0, le=30)

    @field_validator("timeout_seconds")
    def validate_timeout(cls, v: float, info) -> float:
        """驗證超時設置"""
        if (
            "time_threshold_seconds" in info.data
            and v < info.data["time_threshold_seconds"]
        ):
            raise ValueError("timeout_seconds must be >= time_threshold_seconds")
        return v


class EncodedPayload(BaseModel):
    """編碼後的 SQL Payload - 官方 Pydantic BaseModel"""

    url: str
    method: str
    payload: str
    request_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        """驗證 HTTP 方法"""
        allowed = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if v.upper() not in allowed:
            raise ValueError(f"Invalid HTTP method: {v}")
        return v.upper()

    def build_request_dump(self) -> str:
        """構建請求轉儲字符串 - 用於日誌和證據"""
        lines = [f"{self.method} {self.url}"]

        # 添加標頭
        if "headers" in self.request_kwargs:
            for key, value in self.request_kwargs["headers"].items():
                lines.append(f"{key}: {value}")

        # 添加請求體
        body_parts = []
        if "params" in self.request_kwargs:
            body_parts.append(f"params={self.request_kwargs['params']}")
        if "data" in self.request_kwargs:
            body_parts.append(f"data={self.request_kwargs['data']}")
        if "json" in self.request_kwargs:
            body_parts.append(f"json={self.request_kwargs['json']}")
        if "content" in self.request_kwargs:
            body_parts.append(f"content={self.request_kwargs['content']!r}")

        if body_parts:
            lines.append("\n".join(body_parts))

        return "\n".join(lines)


class SqliDetectionContext(BaseModel):
    """SQLi 檢測上下文 - 官方 Pydantic BaseModel"""

    task: FunctionTaskPayload
    findings: list[FindingPayload] = Field(default_factory=list)
    telemetry: SqliTelemetry = Field(default_factory=SqliTelemetry)
    # http_client 不包含在 Pydantic 模型中，由運行時管理

    class Config:
        arbitrary_types_allowed = True  # 允許任意類型（如果需要）


class DetectionError(BaseModel):
    """檢測錯誤 - 官方 Pydantic BaseModel"""

    payload: str
    vector: str
    message: str
    attempts: int = 1
    engine_name: str = ""

    def __str__(self) -> str:
        """字符串表示"""
        payload_preview = (
            self.payload[:50] + "..." if len(self.payload) > 50 else self.payload
        )
        return f"[{self.engine_name}] {self.message} (payload: {payload_preview})"


# 向後兼容別名 - 保持與舊代碼的兼容性
DetectionResult = SqliDetectionResult
SqliExecutionTelemetry = SqliTelemetry
