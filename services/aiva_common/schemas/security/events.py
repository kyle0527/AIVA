"""
AIVA 安全事件統一標準模型

提供SIEM事件、攻擊路徑等安全相關模型的統一架構，
確保跨服務的一致性和可維護性。
"""

from datetime import datetime, UTC
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ...enums.common import Severity


class EventStatus(str, Enum):
    """SIEM事件狀態枚舉"""
    NEW = "new"
    ANALYZING = "analyzing"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class SkillLevel(str, Enum):
    """攻擊技能等級枚舉"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Priority(str, Enum):
    """優先級枚舉"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AttackPathNodeType(str, Enum):
    """攻擊路徑節點類型枚舉"""
    ASSET = "asset"
    VULNERABILITY = "vulnerability"
    EXPLOIT = "exploit"
    PRIVILEGE = "privilege"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class AttackPathEdgeType(str, Enum):
    """攻擊路徑邊類型枚舉"""
    EXPLOITS = "exploits"
    LEADS_TO = "leads_to"
    REQUIRES = "requires"
    ENABLES = "enables"
    COMPROMISES = "compromises"


# ==================== 基礎安全事件模型 ====================

class BaseSIEMEvent(BaseModel):
    """所有SIEM事件的基礎模型
    
    提供統一的SIEM事件結構，支援所有安全監控場景。
    遵循 Pydantic v2 最佳實踐和安全事件標準。
    """
    
    # 核心識別欄位
    event_id: str = Field(description="事件唯一識別ID")
    event_type: str = Field(description="事件類型 (e.g., 'intrusion_attempt', 'malware_detection')")
    source_system: str = Field(description="來源系統名稱")
    
    # 時間信息
    timestamp: datetime = Field(description="事件發生時間戳 (UTC)")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="事件接收處理時間 (UTC)"
    )
    
    # 分類和嚴重程度
    severity: Severity = Field(description="事件嚴重程度")
    category: str = Field(description="事件主分類")
    subcategory: str | None = Field(default=None, description="事件子分類")
    
    # 網路信息
    source_ip: str | None = Field(default=None, description="來源IP位址")
    source_port: int | None = Field(
        default=None, ge=1, le=65535,
        description="來源端口號"
    )
    destination_ip: str | None = Field(default=None, description="目標IP位址")
    destination_port: int | None = Field(
        default=None, ge=1, le=65535,
        description="目標端口號"
    )
    
    # 身份和資產信息
    username: str | None = Field(default=None, description="相關用戶名")
    asset_id: str | None = Field(default=None, description="相關資產識別ID")
    hostname: str | None = Field(default=None, description="主機名稱")
    
    # 事件詳情
    description: str = Field(default="", description="事件描述")
    raw_log: str | None = Field(default=None, description="原始日誌內容")
    
    # 擴展元數據
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="擴展屬性和自定義欄位"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True
        validate_assignment = True


# ==================== 攻擊路徑基礎模型 ====================

class BaseAttackPathNode(BaseModel):
    """攻擊路徑節點基礎模型
    
    表示攻擊路徑中的單一節點，可以是資產、漏洞、或攻擊步驟。
    """
    
    node_id: str = Field(description="節點唯一識別ID")
    node_type: AttackPathNodeType = Field(description="節點類型")
    name: str = Field(description="節點名稱")
    description: str = Field(default="", description="節點詳細描述")
    
    # 風險評估
    risk_score: float = Field(
        ge=0.0, le=10.0, default=0.0,
        description="節點風險評分 (0-10)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="評估置信度 (0-1)"
    )
    
    # 攻擊屬性
    exploit_difficulty: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="利用難度 (0=簡單, 1=困難)"
    )
    detection_probability: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="被檢測機率 (0=難檢測, 1=易檢測)"
    )
    
    # 擴展屬性
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="節點特定屬性和元數據"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True


class BaseAttackPathEdge(BaseModel):
    """攻擊路徑邊基礎模型
    
    表示攻擊路徑中節點間的關係和轉換條件。
    """
    
    edge_id: str = Field(description="邊唯一識別ID")
    source_node_id: str = Field(description="源節點ID")
    target_node_id: str = Field(description="目標節點ID")
    edge_type: AttackPathEdgeType = Field(description="邊關係類型")
    
    # 攻擊轉換評估
    attack_complexity: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="攻擊複雜度 (0=簡單, 1=複雜)"
    )
    success_probability: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description="成功機率 (0-1)"
    )
    time_required_hours: float = Field(
        ge=0.0, default=1.0,
        description="預估所需時間 (小時)"
    )
    
    # 條件和需求
    prerequisites: list[str] = Field(
        default_factory=list,
        description="前提條件列表"
    )
    tools_required: list[str] = Field(
        default_factory=list,
        description="所需工具列表"
    )
    
    # 擴展屬性
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="邊特定屬性和元數據"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True


class BaseAttackPath(BaseModel):
    """攻擊路徑基礎模型
    
    表示完整的攻擊路徑，包含節點、邊和整體評估信息。
    """
    
    path_id: str = Field(description="路徑唯一識別ID")
    name: str = Field(description="攻擊路徑名稱")
    target_asset: str = Field(description="目標資產識別")
    
    # 路徑組成
    nodes: list[BaseAttackPathNode] = Field(description="路徑節點列表")
    edges: list[BaseAttackPathEdge] = Field(description="路徑邊列表")
    
    # 整體評估
    overall_risk_score: float = Field(
        ge=0.0, le=10.0, default=0.0,
        description="整體風險評分 (0-10)"
    )
    path_feasibility: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="路徑可行性 (0-1)"
    )
    estimated_time_hours: float = Field(
        ge=0.0, default=0.0,
        description="預估總攻擊時間 (小時)"
    )
    
    # 技能和資源需求
    skill_level_required: SkillLevel = Field(description="所需技能等級")
    resources_required: list[str] = Field(
        default_factory=list,
        description="所需資源列表"
    )
    
    # 時間信息
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="路徑發現時間"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="最後更新時間"
    )
    
    # 擴展元數據
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="路徑擴展屬性和分析結果"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True


# ==================== 增強版安全事件模型 ====================

class EnhancedSIEMEvent(BaseSIEMEvent):
    """增強版SIEM事件模型
    
    支援威脅情報整合、關聯分析和響應管理的高級SIEM事件。
    """
    
    # 威脅情報整合
    threat_indicators: list[str] = Field(
        default_factory=list,
        description="關聯的威脅指標 (IoCs)"
    )
    threat_actor: str | None = Field(default=None, description="威脅行為者")
    attack_pattern: str | None = Field(default=None, description="攻擊模式 (MITRE ATT&CK)")
    
    # 關聯分析
    related_events: list[str] = Field(
        default_factory=list,
        description="相關事件ID列表"
    )
    correlation_score: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="事件關聯評分"
    )
    attack_chain_position: int | None = Field(
        default=None, ge=1,
        description="在攻擊鏈中的位置"
    )
    
    # 響應和處理
    response_actions: list[str] = Field(
        default_factory=list,
        description="已執行的響應動作"
    )
    status: EventStatus = Field(default=EventStatus.NEW, description="事件處理狀態")
    assigned_analyst: str | None = Field(default=None, description="指派分析師")
    
    # 業務影響
    business_impact: Priority = Field(default=Priority.MEDIUM, description="業務影響程度")
    affected_systems: list[str] = Field(
        default_factory=list,
        description="影響系統列表"
    )
    
    class Config:
        """Pydantic v2 配置"""
        use_enum_values = True


# ==================== 向後兼容適配器 ====================

class LegacySIEMEventAdapter:
    """SIEM事件向後兼容適配器
    
    支援從舊版本模型轉換到新的統一標準。
    """
    
    @staticmethod
    def from_integration_models(legacy_data: dict[str, Any]) -> BaseSIEMEvent:
        """從 services/integration/models.py 的 SIEMEvent 轉換"""
        return BaseSIEMEvent(
            event_id=legacy_data.get("event_id", ""),
            event_type=legacy_data.get("event_type", ""),
            source_system=legacy_data.get("source_system", ""),
            timestamp=legacy_data.get("timestamp", datetime.now(UTC)),
            severity=legacy_data.get("severity", Severity.MEDIUM),
            category=legacy_data.get("category", ""),
            subcategory=legacy_data.get("subcategory"),
            source_ip=legacy_data.get("source_ip"),
            source_port=legacy_data.get("source_port"),
            destination_ip=legacy_data.get("destination_ip"),
            destination_port=legacy_data.get("destination_port"),
            username=legacy_data.get("username"),
            description=legacy_data.get("description", ""),
            metadata=legacy_data.get("metadata", {})
        )
    
    @staticmethod
    def from_telemetry_schemas(legacy_data: dict[str, Any]) -> BaseSIEMEvent:
        """從 services/aiva_common/schemas/telemetry.py 的 SIEMEvent 轉換"""
        return BaseSIEMEvent(
            event_id=legacy_data.get("event_id", ""),
            event_type=legacy_data.get("event_type", ""),
            source_system=legacy_data.get("source_system", ""),
            timestamp=legacy_data.get("timestamp", datetime.now(UTC)),
            severity=legacy_data.get("severity", Severity.MEDIUM),
            category=legacy_data.get("category", ""),
            subcategory=legacy_data.get("subcategory"),
            source_ip=legacy_data.get("source_ip"),
            source_port=legacy_data.get("source_port"),
            destination_ip=legacy_data.get("destination_ip"),
            destination_port=legacy_data.get("destination_port"),
            username=legacy_data.get("username"),
            description=legacy_data.get("description", ""),
            metadata={}
        )


class LegacyAttackPathAdapter:
    """攻擊路徑向後兼容適配器"""
    
    @staticmethod
    def from_risk_schemas(legacy_node: dict[str, Any]) -> BaseAttackPathNode:
        """從 risk.py 的 AttackPathNode 轉換"""
        return BaseAttackPathNode(
            node_id=legacy_node.get("node_id", ""),
            node_type=legacy_node.get("node_type", AttackPathNodeType.ASSET),
            name=legacy_node.get("name", ""),
            properties=legacy_node.get("properties", {})
        )


# 導出統一接口
__all__ = [
    # 基礎模型
    "BaseSIEMEvent",
    "BaseAttackPathNode", 
    "BaseAttackPathEdge",
    "BaseAttackPath",
    # 增強模型
    "EnhancedSIEMEvent",
    # 枚舉
    "EventStatus",
    "SkillLevel",
    "Priority",
    "AttackPathNodeType",
    "AttackPathEdgeType",
    # 適配器
    "LegacySIEMEventAdapter",
    "LegacyAttackPathAdapter",
]