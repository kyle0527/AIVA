"""
異步工具 Schema 定義

此模組定義了異步任務管理、重試策略、資源限制等相關的資料模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..enums import AsyncTaskStatus


class RetryConfig(BaseModel):
    """重試配置"""

    max_attempts: int = Field(default=3, ge=1, le=10, description="最大重試次數")
    backoff_base: float = Field(default=1.0, gt=0, description="退避基礎時間(秒)")
    backoff_factor: float = Field(default=2.0, gt=1, description="退避倍數")
    max_backoff: float = Field(default=60.0, gt=0, description="最大退避時間(秒)")
    exponential_backoff: bool = Field(default=True, description="是否使用指數退避")


class ResourceLimits(BaseModel):
    """資源限制配置"""

    max_memory_mb: int | None = Field(
        default=None, ge=1, description="最大內存限制(MB)"
    )
    max_cpu_percent: float | None = Field(
        default=None, ge=0.1, le=100.0, description="最大CPU使用率(%)"
    )
    max_execution_time: int | None = Field(
        default=None, ge=1, description="最大執行時間(秒)"
    )
    max_concurrent_tasks: int = Field(
        default=10, ge=1, le=100, description="最大並發任務數"
    )


class AsyncTaskConfig(BaseModel):
    """異步任務配置"""

    task_name: str = Field(description="任務名稱")
    timeout_seconds: int = Field(default=30, ge=1, le=3600, description="超時時間(秒)")
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig, description="重試配置"
    )
    priority: int = Field(default=5, ge=1, le=10, description="任務優先級")
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits, description="資源限制"
    )
    tags: list[str] = Field(default_factory=list, description="任務標籤")
    metadata: dict[str, Any] = Field(default_factory=dict, description="任務元數據")

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        """驗證任務名稱"""
        if not v.strip():
            raise ValueError("任務名稱不能為空")
        if len(v) > 100:
            raise ValueError("任務名稱長度不能超過100個字符")
        return v.strip()


class AsyncTaskResult(BaseModel):
    """異步任務結果"""

    task_id: str = Field(description="任務ID")
    task_name: str = Field(description="任務名稱")
    status: AsyncTaskStatus = Field(description="任務狀態")
    result: dict[str, Any] | None = Field(default=None, description="執行結果")
    error_message: str | None = Field(default=None, description="錯誤信息")
    execution_time_ms: float = Field(ge=0, description="執行時間(毫秒)")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="開始時間"
    )
    end_time: datetime | None = Field(default=None, description="結束時間")
    retry_count: int = Field(default=0, ge=0, description="重試次數")
    resource_usage: dict[str, Any] = Field(
        default_factory=dict, description="資源使用情況"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="結果元數據")

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, v: str | None) -> str | None:
        """驗證錯誤信息"""
        if v is not None and len(v) > 1000:
            return v[:1000] + "... (截斷)"
        return v


class AsyncBatchConfig(BaseModel):
    """異步批次任務配置"""

    batch_id: str = Field(description="批次ID")
    batch_name: str = Field(description="批次名稱")
    tasks: list[AsyncTaskConfig] = Field(description="任務列表")
    max_concurrent: int = Field(default=5, ge=1, le=50, description="最大並發數")
    stop_on_first_error: bool = Field(default=False, description="遇到第一個錯誤時停止")
    batch_timeout_seconds: int = Field(
        default=3600, ge=1, description="批次超時時間(秒)"
    )

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: list[AsyncTaskConfig]) -> list[AsyncTaskConfig]:
        """驗證任務列表"""
        if not v:
            raise ValueError("任務列表不能為空")
        if len(v) > 1000:
            raise ValueError("批次任務數量不能超過1000個")
        return v


class AsyncBatchResult(BaseModel):
    """異步批次任務結果"""

    batch_id: str = Field(description="批次ID")
    batch_name: str = Field(description="批次名稱")
    total_tasks: int = Field(ge=0, description="總任務數")
    completed_tasks: int = Field(default=0, ge=0, description="已完成任務數")
    failed_tasks: int = Field(default=0, ge=0, description="失敗任務數")
    task_results: list[AsyncTaskResult] = Field(
        default_factory=list, description="任務結果列表"
    )
    batch_status: str = Field(description="批次狀態")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="開始時間"
    )
    end_time: datetime | None = Field(default=None, description="結束時間")
    total_execution_time_ms: float = Field(
        default=0, ge=0, description="總執行時間(毫秒)"
    )

    @field_validator("batch_status")
    @classmethod
    def validate_batch_status(cls, v: str) -> str:
        """驗證批次狀態"""
        allowed = {"pending", "running", "completed", "failed", "cancelled", "partial"}
        if v not in allowed:
            raise ValueError(f"批次狀態必須是 {allowed} 之一")
        return v
