"""
遙測與監控相關 Schema

此模組定義了模組狀態、心跳、性能指標等監控相關的資料模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...enums import ErrorCategory, ModuleName, Severity, StoppingReason


class HeartbeatPayload(BaseModel):
    """心跳 Payload"""

    module: ModuleName
    worker_id: str
    capacity: int


class ModuleStatus(BaseModel):
    """模組狀態報告 - 用於模組健康檢查和監控"""

    module: ModuleName
    status: str  # "running", "stopped", "error", "initializing"
    worker_id: str
    worker_count: int = 1
    queue_size: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float = 0.0

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"running", "stopped", "error", "initializing", "degraded"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class FunctionTelemetry(BaseModel):
    """功能模組遙測數據基礎類"""

    payloads_sent: int = 0
    detections: int = 0
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式"""
        details: dict[str, Any] = {
            "payloads_sent": self.payloads_sent,
            "detections": self.detections,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
        }
        if findings_count is not None:
            details["findings"] = findings_count
        if self.errors:
            details["errors"] = self.errors
        return details


# ============================================================================
# Enhanced Function Telemetry (統一擴展版)
# ============================================================================


class ErrorRecord(BaseModel):
    """錯誤記錄"""

    category: ErrorCategory
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class OastCallbackDetail(BaseModel):
    """OAST 回調詳情"""

    callback_type: str  # "http", "dns", "smtp" 等
    token: str
    source_ip: str
    timestamp: datetime
    protocol: str | None = None
    raw_data: dict[str, Any] = Field(default_factory=dict)


class EarlyStoppingInfo(BaseModel):
    """提前停止信息"""

    reason: StoppingReason
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_tests: int
    completed_tests: int
    remaining_tests: int
    details: dict[str, Any] = Field(default_factory=dict)


class AdaptiveBehaviorInfo(BaseModel):
    """自適應行為信息"""

    initial_batch_size: int = 10
    final_batch_size: int = 10
    rate_adjustments: int = 0
    protection_detections: int = 0
    bypass_attempts: int = 0
    success_rate: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class EnhancedFunctionTelemetry(FunctionTelemetry):
    """
    增強版功能模組遙測 - 統一擴展所有 Worker 模組

    繼承自 FunctionTelemetry，新增:
    - 結構化錯誤記錄 (分類、時間戳、詳情)
    - OAST 回調追蹤 (支持 HTTP/DNS/SMTP 等)
    - 提前停止檢測 (8種原因，含剩餘測試數)
    - 自適應行為監控 (批次大小、成功率、繞過嘗試)
    """

    # 結構化錯誤記錄
    error_records: list[ErrorRecord] = Field(default_factory=list)

    # OAST 回調追蹤
    oast_callbacks: list[OastCallbackDetail] = Field(default_factory=list)

    # 提前停止檢測
    early_stopping: EarlyStoppingInfo | None = None

    # 自適應行為
    adaptive_behavior: AdaptiveBehaviorInfo | None = None

    def record_error(
        self,
        category: ErrorCategory,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """記錄結構化錯誤"""
        error = ErrorRecord(
            category=category,
            message=message,
            details=details or {},
        )
        self.error_records.append(error)
        # 保持向後兼容，也添加到 errors 列表
        self.errors.append(f"[{category.value}] {message}")

    def record_oast_callback(
        self,
        callback_type: str,
        token: str,
        source_ip: str,
        timestamp: datetime,
        protocol: str | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> None:
        """記錄 OAST 回調"""
        callback = OastCallbackDetail(
            callback_type=callback_type,
            token=token,
            source_ip=source_ip,
            timestamp=timestamp,
            protocol=protocol,
            raw_data=raw_data or {},
        )
        self.oast_callbacks.append(callback)

    def record_early_stopping(
        self,
        reason: StoppingReason,
        total_tests: int,
        completed_tests: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """記錄提前停止"""
        self.early_stopping = EarlyStoppingInfo(
            reason=reason,
            total_tests=total_tests,
            completed_tests=completed_tests,
            remaining_tests=total_tests - completed_tests,
            details=details or {},
        )

    def update_adaptive_behavior(
        self,
        initial_batch_size: int | None = None,
        final_batch_size: int | None = None,
        rate_adjustments: int | None = None,
        protection_detections: int | None = None,
        bypass_attempts: int | None = None,
        success_rate: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """更新自適應行為信息"""
        if self.adaptive_behavior is None:
            self.adaptive_behavior = AdaptiveBehaviorInfo()

        if initial_batch_size is not None:
            self.adaptive_behavior.initial_batch_size = initial_batch_size
        if final_batch_size is not None:
            self.adaptive_behavior.final_batch_size = final_batch_size
        if rate_adjustments is not None:
            self.adaptive_behavior.rate_adjustments = rate_adjustments
        if protection_detections is not None:
            self.adaptive_behavior.protection_detections = protection_detections
        if bypass_attempts is not None:
            self.adaptive_behavior.bypass_attempts = bypass_attempts
        if success_rate is not None:
            self.adaptive_behavior.success_rate = success_rate
        if details:
            self.adaptive_behavior.details.update(details)

    def to_details(self, findings_count: int | None = None) -> dict[str, Any]:
        """轉換為詳細報告格式 (擴展版)"""
        details = super().to_details(findings_count)

        # 添加結構化錯誤統計
        if self.error_records:
            error_summary: dict[str, int] = {}
            for err in self.error_records:
                category = err.category.value
                error_summary[category] = error_summary.get(category, 0) + 1
            details["error_categories"] = error_summary
            details["error_details"] = [
                {
                    "category": err.category.value,
                    "message": err.message,
                    "timestamp": err.timestamp.isoformat(),
                }
                for err in self.error_records
            ]

        # 添加 OAST 回調統計
        if self.oast_callbacks:
            details["oast_callbacks_count"] = len(self.oast_callbacks)
            callback_summary: dict[str, int] = {}
            for cb in self.oast_callbacks:
                callback_summary[cb.callback_type] = (
                    callback_summary.get(cb.callback_type, 0) + 1
                )
            details["oast_callback_types"] = callback_summary

        # 添加提前停止信息
        if self.early_stopping:
            details["early_stopping"] = {
                "reason": self.early_stopping.reason.value,
                "completed_tests": self.early_stopping.completed_tests,
                "total_tests": self.early_stopping.total_tests,
                "completion_rate": (
                    self.early_stopping.completed_tests
                    / self.early_stopping.total_tests
                    if self.early_stopping.total_tests > 0
                    else 0.0
                ),
            }

        # 添加自適應行為信息
        if self.adaptive_behavior:
            details["adaptive_behavior"] = {
                "batch_size_change": (
                    self.adaptive_behavior.final_batch_size
                    - self.adaptive_behavior.initial_batch_size
                ),
                "rate_adjustments": self.adaptive_behavior.rate_adjustments,
                "protection_detections": self.adaptive_behavior.protection_detections,
                "bypass_attempts": self.adaptive_behavior.bypass_attempts,
                "success_rate": self.adaptive_behavior.success_rate,
            }

        return details


class FunctionExecutionResult(BaseModel):
    """功能模組執行結果統一格式"""

    findings: list[dict[str, Any]]
    telemetry: dict[str, Any]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OastEvent(BaseModel):
    """OAST (Out-of-Band Application Security Testing) 事件數據合約"""

    event_id: str
    probe_token: str
    event_type: str  # "http", "dns", "smtp", "ftp"
    source_ip: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    protocol: str | None = None
    raw_request: str | None = None
    raw_data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = {"http", "dns", "smtp", "ftp", "ldap", "other"}
        if v not in allowed:
            raise ValueError(f"Invalid event_type: {v}. Must be one of {allowed}")
        return v


class OastProbe(BaseModel):
    """OAST 探針數據合約"""

    probe_id: str
    token: str
    callback_url: str
    task_id: str
    scan_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    status: str = "active"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"active", "triggered", "expired", "cancelled"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class SIEMEventPayload(BaseModel):
    """SIEM 事件 Payload"""

    event_id: str
    event_type: str
    severity: str
    source: str
    destination: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NotificationPayload(BaseModel):
    """通知 Payload - 用於 Slack/Teams/Email"""

    notification_id: str
    notification_type: str  # "slack", "teams", "email", "webhook"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    recipients: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# SIEM 事件 (詳細版)
# ============================================================================


class SIEMEvent(BaseModel):
    """SIEM事件記錄"""

    event_id: str = Field(description="事件ID")
    event_type: str = Field(description="事件類型")
    source_system: str = Field(description="來源系統")

    # 時間信息
    timestamp: datetime = Field(description="事件時間戳")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 事件屬性
    severity: Severity = Field(description="嚴重程度")
    category: str = Field(description="事件分類")
    subcategory: str | None = Field(default=None, description="事件子分類")

    # 來源信息
    source_ip: str | None = Field(default=None, description="來源IP")
    source_port: int | None = Field(default=None, description="來源端口")
    destination_ip: str | None = Field(default=None, description="目標IP")
    destination_port: int | None = Field(default=None, description="目標端口")

    # 用戶和資產
    username: str | None = Field(default=None, description="用戶名")
    asset_id: str | None = Field(default=None, description="資產ID")
    hostname: str | None = Field(default=None, description="主機名")

    # 事件詳情
    description: str = Field(description="事件描述")
    raw_log: str | None = Field(default=None, description="原始日誌")

    # 關聯分析
    correlation_rules: list[str] = Field(
        default_factory=list, description="觸發的關聯規則"
    )
    related_events: list[str] = Field(default_factory=list, description="相關事件ID")

    # 處理狀態
    status: str = Field(
        default="new", description="處理狀態"
    )  # "new", "investigating", "resolved", "false_positive"
    assigned_to: str | None = Field(default=None, description="分配給")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
