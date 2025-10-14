"""
AIVA Integration Models - 整合服務模組

此文件包含與外部服務整合相關的所有數據模型。

職責範圍：
1. 威脅情報整合 (ThreatIntel, IOC)
2. SIEM 事件整合
3. 通知系統
4. Webhook 處理
5. 第三方服務API整合
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..aiva_common.enums import IntelSource, IOCType, Severity, ThreatLevel

# ==================== 威脅情報整合 ====================


class ThreatIntelLookupPayload(BaseModel):
    """威脅情報查詢載荷"""

    lookup_id: str = Field(description="查詢ID")
    indicator_type: IOCType = Field(description="指標類型")
    indicator_value: str = Field(description="指標值")
    sources: list[IntelSource] = Field(description="情報來源列表")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ThreatIntelResultPayload(BaseModel):
    """威脅情報結果載荷"""

    lookup_id: str = Field(description="查詢ID")
    indicator_type: IOCType = Field(description="指標類型")
    indicator_value: str = Field(description="指標值")
    threat_level: ThreatLevel = Field(description="威脅級別")
    is_malicious: bool = Field(description="是否惡意")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    sources: list[dict[str, Any]] = Field(description="情報來源詳情")
    threat_details: dict[str, Any] = Field(default_factory=dict, description="威脅詳情")
    related_indicators: list[str] = Field(default_factory=list, description="相關指標")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedIOCRecord(BaseModel):
    """增強威脅指標記錄 (Indicator of Compromise)"""

    ioc_id: str = Field(description="IOC唯一標識符")
    ioc_type: str = Field(description="IOC類型")  # "ip", "domain", "url", "hash", "email"
    value: str = Field(description="IOC值")

    # 威脅信息
    threat_type: str | None = Field(default=None, description="威脅類型")
    malware_family: str | None = Field(default=None, description="惡意軟體家族")
    campaign: str | None = Field(default=None, description="攻擊活動")

    # 評級信息
    severity: Severity = Field(description="嚴重程度")
    confidence: int = Field(ge=0, le=100, description="可信度 0-100")
    reputation_score: int = Field(ge=0, le=100, description="聲譽分數")

    # 時間信息
    first_seen: datetime | None = Field(default=None, description="首次發現時間")
    last_seen: datetime | None = Field(default=None, description="最後發現時間")
    expires_at: datetime | None = Field(default=None, description="過期時間")

    # 標籤和分類
    tags: list[str] = Field(default_factory=list, description="標籤")
    mitre_techniques: list[str] = Field(default_factory=list, description="MITRE ATT&CK技術")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== SIEM 整合 ====================


class SIEMEventPayload(BaseModel):
    """SIEM事件載荷"""

    event_id: str = Field(description="事件ID")
    event_type: str = Field(description="事件類型")
    severity: Severity = Field(description="嚴重程度")
    source_system: str = Field(description="來源系統")
    event_data: dict[str, Any] = Field(description="事件數據")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


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
    correlation_rules: list[str] = Field(default_factory=list, description="觸發的關聯規則")
    related_events: list[str] = Field(default_factory=list, description="相關事件ID")

    # 處理狀態
    status: str = Field(default="new", description="處理狀態")  # "new", "investigating", "resolved", "false_positive"
    assigned_to: str | None = Field(default=None, description="分配給")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 通知系統 ====================


class NotificationPayload(BaseModel):
    """通知載荷"""

    notification_id: str = Field(description="通知ID")
    notification_type: str = Field(description="通知類型")  # "email", "slack", "webhook", "sms"
    severity: Severity = Field(description="嚴重程度")
    title: str = Field(description="通知標題")
    message: str = Field(description="通知消息")
    recipients: list[str] = Field(description="接收者列表")
    channels: list[str] = Field(description="通知通道")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    attachment_urls: list[str] = Field(default_factory=list, description="附件URL")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== Webhook ====================


class WebhookPayload(BaseModel):
    """Webhook載荷"""

    webhook_id: str = Field(description="Webhook ID")
    event_type: str = Field(description="事件類型")
    source: str = Field(description="來源系統")

    # 時間戳
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 載荷數據
    payload: dict[str, Any] = Field(description="事件載荷")

    # 處理狀態
    processed: bool = Field(default=False, description="是否已處理")
    processing_attempts: int = Field(default=0, ge=0, description="處理嘗試次數")
    last_error: str | None = Field(default=None, description="最後錯誤")

    # 驗證信息
    signature: str | None = Field(default=None, description="簽名")
    verified: bool = Field(default=False, description="是否已驗證")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


__all__ = [
    # 威脅情報
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "EnhancedIOCRecord",
    # SIEM
    "SIEMEventPayload",
    "SIEMEvent",
    # 通知
    "NotificationPayload",
    # Webhook
    "WebhookPayload",
]
