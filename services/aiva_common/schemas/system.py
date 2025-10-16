"""
系統和編排相關 Schemas

此模組定義了系統編排、會話管理、任務隊列、Webhook 等系統級別的資料模型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..enums import ModuleName

# ============================================================================
# 會話管理
# ============================================================================


class SessionState(BaseModel):
    """會話狀態 - 多步驟攻擊鏈的會話管理"""

    session_id: str
    plan_id: str
    scan_id: str
    status: str  # "active", "paused", "completed", "failed", "aborted"
    current_step_index: int = 0
    completed_steps: list[str] = Field(default_factory=list)  # step_ids
    pending_steps: list[str] = Field(default_factory=list)  # step_ids
    context: dict[str, Any] = Field(default_factory=dict)  # 動態上下文
    variables: dict[str, Any] = Field(
        default_factory=dict
    )  # 會話變數（用於步驟間傳遞數據）
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.startswith("session_"):
            raise ValueError("session_id must start with 'session_'")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"active", "paused", "completed", "failed", "aborted"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ============================================================================
# AI 訓練結果
# ============================================================================


class ModelTrainingResult(BaseModel):
    """模型訓練結果"""

    training_id: str
    config: dict[str, Any]  # 簡化避免循環引用
    model_version: str
    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    average_reward: float | None = None  # 強化學習平均獎勵
    training_duration_seconds: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 任務隊列
# ============================================================================


class TaskQueue(BaseModel):
    """任務隊列"""

    queue_id: str = Field(description="隊列ID")
    queue_name: str = Field(description="隊列名稱")

    # 隊列配置
    max_concurrent_tasks: int = Field(default=5, ge=1, description="最大併發任務數")
    task_timeout: int = Field(default=3600, ge=60, description="任務超時(秒)")

    # 隊列狀態
    pending_tasks: list[str] = Field(default_factory=list, description="等待任務")
    running_tasks: list[str] = Field(default_factory=list, description="運行任務")
    completed_tasks: list[str] = Field(
        default_factory=list, description="完成任務")

    # 統計信息
    total_processed: int = Field(ge=0, description="總處理數")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    average_execution_time: float = Field(ge=0.0, description="平均執行時間")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ============================================================================
# 系統編排
# ============================================================================


class EnhancedModuleStatus(BaseModel):
    """增強模組狀態"""

    module_name: ModuleName = Field(description="模組名稱")
    version: str = Field(description="模組版本")

    # 狀態信息
    status: str = Field(
        description="運行狀態"
    )  # "running", "stopped", "error", "maintenance"
    health_score: float = Field(ge=0.0, le=1.0, description="健康評分")

    # 性能指標
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU使用率")
    memory_usage: float = Field(ge=0.0, description="內存使用(MB)")
    active_connections: int = Field(ge=0, description="活躍連接數")

    # 任務統計
    tasks_processed: int = Field(ge=0, description="處理任務數")
    tasks_pending: int = Field(ge=0, description="待處理任務數")
    error_count: int = Field(ge=0, description="錯誤次數")

    # 時間信息
    started_at: datetime = Field(description="啟動時間")
    last_heartbeat: datetime = Field(description="最後心跳")
    uptime_seconds: int = Field(ge=0, description="運行時間(秒)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class SystemOrchestration(BaseModel):
    """系統編排"""

    orchestration_id: str = Field(description="編排ID")
    orchestration_name: str = Field(description="編排名稱")

    # 模組狀態
    module_statuses: list[EnhancedModuleStatus] = Field(description="模組狀態列表")

    # 系統配置
    scan_configuration: dict[str, Any] = Field(
        default_factory=dict, description="掃描配置"
    )
    resource_allocation: dict[str, Any] = Field(
        default_factory=dict, description="資源分配"
    )

    # 編排狀態
    overall_status: str = Field(description="整體狀態")
    active_scans: int = Field(ge=0, description="活躍掃描數")
    queued_tasks: int = Field(ge=0, description="排隊任務數")

    # 性能指標
    system_cpu: float = Field(ge=0.0, le=100.0, description="系統CPU")
    system_memory: float = Field(ge=0.0, description="系統內存(MB)")
    network_throughput: float = Field(ge=0.0, description="網絡吞吐量(Mbps)")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ============================================================================
# Webhook
# ============================================================================


class WebhookPayload(BaseModel):
    """Webhook載荷"""

    webhook_id: str = Field(description="Webhook ID")
    event_type: str = Field(description="事件類型")
    source: str = Field(description="來源系統")

    # 時間戳
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 負載數據
    data: dict[str, Any] = Field(default_factory=dict, description="事件數據")

    # 配置
    delivery_url: str | None = Field(default=None, description="交付URL")
    retry_count: int = Field(default=0, ge=0, description="重試次數")
    max_retries: int = Field(default=3, ge=0, description="最大重試次數")

    # 狀態
    status: str = Field(
        default="pending", description="狀態"
    )  # pending, delivered, failed
    delivered_at: datetime | None = Field(default=None, description="交付時間")
    error_message: str | None = Field(default=None, description="錯誤消息")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
