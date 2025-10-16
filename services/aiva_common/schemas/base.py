"""
基礎 Schema 模型

此模組包含所有基礎的資料模型，這些模型會被其他 Schema 模組所繼承或引用。
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from ..enums import ModuleName


class MessageHeader(BaseModel):
    """訊息標頭 - 用於所有訊息的統一標頭格式"""

    message_id: str
    trace_id: str
    correlation_id: str | None = None
    source_module: ModuleName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"


class Authentication(BaseModel):
    """認證資訊"""

    method: str = "none"
    credentials: dict[str, str] | None = None


class RateLimit(BaseModel):
    """速率限制"""

    requests_per_second: int = 25
    burst: int = 50

    @field_validator("requests_per_second", "burst")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limit must be non-negative")
        return v


class ScanScope(BaseModel):
    """掃描範圍"""

    exclusions: list[str] = Field(default_factory=list)
    include_subdomains: bool = True
    allowed_hosts: list[str] = Field(default_factory=list)


class Asset(BaseModel):
    """資產基本資訊"""

    asset_id: str
    type: str
    value: str
    parameters: list[str] | None = None
    has_form: bool = False


class Summary(BaseModel):
    """掃描摘要"""

    urls_found: int = 0
    forms_found: int = 0
    apis_found: int = 0
    scan_duration_seconds: int = 0


class Fingerprints(BaseModel):
    """技術指紋"""

    web_server: dict[str, str] | None = None
    framework: dict[str, str] | None = None
    language: dict[str, str] | None = None
    waf_detected: bool = False
    waf_vendor: str | None = None


class ExecutionError(BaseModel):
    """執行錯誤統一格式"""

    error_id: str
    error_type: str
    message: str
    payload: str | None = None
    vector: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 1


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str = Field(description="風險因子名稱")
    weight: float = Field(ge=0.0, le=1.0, description="權重")
    value: float = Field(ge=0.0, le=10.0, description="因子值")
    description: str | None = Field(default=None, description="因子描述")


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str = Field(
        description="依賴類型"
    )  # "prerequisite", "blocker", "input"
    dependent_task_id: str = Field(description="依賴任務ID")
    condition: str | None = Field(default=None, description="依賴條件")
    required: bool = Field(default=True, description="是否必需")
