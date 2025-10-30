"""
AIVA Integration Models - 整合服務模組

此文件包含與外部服務整合相關的所有數據模型。

職責範圍：
1. Integration 專屬的增強記錄類 (EnhancedIOCRecord, SIEMEvent)

共享類從 aiva_common 導入:
- ThreatIntelLookupPayload, ThreatIntelResultPayload
- SIEMEventPayload, NotificationPayload
- WebhookPayload
"""



from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..aiva_common.enums import Severity
from ..aiva_common.schemas import (
    NotificationPayload,
    SIEMEventPayload,
    ThreatIntelLookupPayload,
    ThreatIntelResultPayload,
    WebhookPayload,
)

# ==================== Integration 專屬的增強類 ====================


class EnhancedIOCRecord(BaseModel):
    """
    增強威脅指標記錄 (Indicator of Compromise)
    
    Integration 模組專屬,用於威脅情報的詳細記錄和擴展。
    """

    ioc_id: str = Field(description="IOC唯一標識符")
    ioc_type: str = Field(description="IOC類型")  # "ip", "domain", "url", "hash", "email"
    value: str = Field(description="IOC值")

    # 威脅信息
    threat_type: Optional[str] = Field(default=None, description="威脅類型")
    malware_family: Optional[str] = Field(default=None, description="惡意軟體家族")
    campaign: Optional[str] = Field(default=None, description="攻擊活動")

    # 評級信息
    severity: Severity = Field(description="嚴重程度")
    confidence: int = Field(ge=0, le=100, description="可信度 0-100")
    reputation_score: int = Field(ge=0, le=100, description="聲譽分數")

    # 時間信息
    first_seen: Optional[datetime] = Field(default=None, description="首次發現時間")
    last_seen: Optional[datetime] = Field(default=None, description="最後發現時間")
    expires_at: Optional[datetime] = Field(default=None, description="過期時間")

    # 標籤和分類
    tags: List[str] = Field(default_factory=list, description="標籤")
    mitre_techniques: List[str] = Field(default_factory=list, description="MITRE ATT&CK技術")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="元數據")


class SIEMEvent(BaseModel):
    """
    SIEM事件記錄
    
    Integration 模組專屬,用於 SIEM 系統的詳細事件記錄。
    與 aiva_common.schemas.SIEMEventPayload 互補使用。
    """

    event_id: str = Field(description="事件ID")
    event_type: str = Field(description="事件類型")
    source_system: str = Field(description="來源系統")

    # 時間信息
    timestamp: datetime = Field(description="事件時間戳")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 事件屬性
    severity: Severity = Field(description="嚴重程度")
    category: str = Field(description="事件分類")
    subcategory: Optional[str] = Field(default=None, description="事件子分類")

    # 來源信息
    source_ip: Optional[str] = Field(default=None, description="來源IP")
    source_port: Optional[int] = Field(default=None, description="來源端口")
    destination_ip: Optional[str] = Field(default=None, description="目標IP")
    destination_port: Optional[int] = Field(default=None, description="目標端口")

    # 用戶和資產
    username: Optional[str] = Field(default=None, description="用戶名")
    asset_id: Optional[str] = Field(default=None, description="資產ID")
    hostname: Optional[str] = Field(default=None, description="主機名")

    # 事件詳情
    description: str = Field(description="事件描述")
    raw_log: Optional[str] = Field(default=None, description="原始日誌")

    # 關聯分析
    correlation_rules: List[str] = Field(default_factory=list, description="觸發的關聯規則")
    related_events: List[str] = Field(default_factory=list, description="相關事件ID")

    # 處理狀態
    status: str = Field(default="new", description="處理狀態")  # "new", "investigating", "resolved", "false_positive"
    assigned_to: Optional[str] = Field(default=None, description="分配給")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="元數據")


__all__ = [
    # ==================== 從 aiva_common 導入的共享類 ====================
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "SIEMEventPayload",
    "NotificationPayload",
    "WebhookPayload",
    # ==================== Integration 專屬類 ====================
    "EnhancedIOCRecord",
    "SIEMEvent",
]
