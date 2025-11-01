"""
基礎 Schema 模型

此模組包含所有基礎的資料模型，這些模型會被其他 Schema 模組所繼承或引用。
"""

from datetime import UTC, datetime

# 前向聲明用於避免循環導入
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from ..enums import ModuleName

if TYPE_CHECKING:
    from .findings import Target


class MessageHeader(BaseModel):
    """訊息標頭 - 用於所有訊息的統一標頭格式"""

    message_id: str
    trace_id: str
    correlation_id: str | None = None
    source_module: ModuleName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"


class APIResponse(BaseModel):
    """標準API響應格式 - 統一所有API端點的響應結構"""
    
    success: bool = Field(description="請求是否成功")
    message: str = Field(description="響應消息")
    data: dict | list | None = Field(None, description="響應數據")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="響應時間戳")
    trace_id: str | None = Field(None, description="追蹤ID")
    errors: list[str] | None = Field(None, description="錯誤列表")
    metadata: dict | None = Field(None, description="額外的元數據")


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


class Task(BaseModel):
    """任務基礎模型 - 定義所有任務的通用結構"""

    task_id: str = Field(description="任務唯一識別符")
    task_type: str = Field(description="任務類型，例如：function_ssrf、function_sqli")
    status: str = Field(
        default="pending", description="任務狀態：pending、running、completed、failed"
    )
    priority: int = Field(default=5, ge=1, le=10, description="任務優先級（1-10）")

    # 目標資訊 - 支援兩種方式，保持向後相容
    target_url: str | None = Field(default=None, description="目標URL（簡化版）")
    target_params: dict[str, str] | None = Field(default=None, description="目標參數")

    # 完整目標物件 - 用於複雜場景
    target: "Target | None" = Field(default=None, description="完整目標物件")

    # 掃描配置
    scan_strategy: str = Field(
        default="normal", description="掃描策略：fast、normal、deep、comprehensive"
    )
    strategy: str | None = Field(default=None, description="策略別名（向後相容）")
    scan_id: str | None = Field(default=None, description="掃描ID")
    max_execution_time: int = Field(default=300, description="最大執行時間（秒）")

    # 時間戳記
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = Field(default=None, description="開始執行時間")
    completed_at: datetime | None = Field(default=None, description="完成時間")

    # 結果資訊
    result: dict | None = Field(default=None, description="任務執行結果")
    error_message: str | None = Field(default=None, description="錯誤訊息")

    # 依賴和元數據
    dependencies: list[TaskDependency] = Field(
        default_factory=list, description="任務依賴列表"
    )
    metadata: dict[str, str] | None = Field(default=None, description="額外元數據")


# 解決前向引用
from .findings import Target

Task.model_rebuild()
