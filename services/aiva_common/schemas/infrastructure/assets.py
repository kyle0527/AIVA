"""
資產與 EASM 相關 Schema

此模組定義了資產探索、資產生命週期管理、EASM 等相關的資料模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ...enums import (
    AssetExposure,
    AssetType,
    BusinessCriticality,
    ComplianceFramework,
    Confidence,
    DataSensitivity,
    Environment,
    Exploitability,
    Severity,
    VulnerabilityStatus,
    VulnerabilityType,
)
from ..risk.references import TechnicalFingerprint  # 從 risk 域引用，避免重複定義


class AssetLifecyclePayload(BaseModel):
    """資產生命週期管理 Payload"""

    asset_id: str
    asset_type: AssetType
    value: str
    environment: Environment
    business_criticality: BusinessCriticality
    data_sensitivity: DataSensitivity | None = None
    asset_exposure: AssetExposure | None = None
    owner: str | None = None
    team: str | None = None
    compliance_tags: list[ComplianceFramework] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VulnerabilityLifecyclePayload(BaseModel):
    """漏洞生命週期管理 Payload"""

    vulnerability_id: str
    finding_id: str
    asset_id: str
    vulnerability_type: VulnerabilityType
    severity: Severity
    confidence: Confidence
    status: VulnerabilityStatus
    exploitability: Exploitability | None = None
    assigned_to: str | None = None
    due_date: datetime | None = None
    first_detected: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolution_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VulnerabilityUpdatePayload(BaseModel):
    """漏洞狀態更新 Payload"""

    vulnerability_id: str
    status: VulnerabilityStatus
    assigned_to: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_by: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DiscoveredAsset(BaseModel):
    """探索到的資產"""

    asset_id: str
    asset_type: AssetType
    value: str
    discovery_method: str
    confidence: Confidence
    metadata: dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# 資產清單和 EASM
# ============================================================================


class AssetInventoryItem(BaseModel):
    """資產清單項目"""

    asset_id: str = Field(description="資產唯一標識")
    asset_type: str = Field(description="資產類型")
    name: str = Field(description="資產名稱")

    # 網路信息
    ip_address: str | None = Field(default=None, description="IP地址")
    hostname: str | None = Field(default=None, description="主機名")
    domain: str | None = Field(default=None, description="域名")
    ports: list[int] = Field(default_factory=list, description="開放端口")

    # 技術棧
    fingerprints: list[TechnicalFingerprint] = Field(
        default_factory=list, description="技術指紋"
    )

    # 業務信息
    business_criticality: str = Field(
        description="業務重要性"
    )  # "critical", "high", "medium", "low"
    owner: str | None = Field(default=None, description="負責人")
    environment: str = Field(
        description="環境類型"
    )  # "production", "staging", "development"

    # 安全狀態
    last_scanned: datetime | None = Field(default=None, description="最後掃描時間")
    vulnerability_count: int = Field(ge=0, description="漏洞數量")
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")

    # 時間戳
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EASMAsset(BaseModel):
    """外部攻擊面管理資產"""

    asset_id: str = Field(description="資產ID")
    asset_type: str = Field(
        description="資產類型"
    )  # "domain", "subdomain", "ip", "service", "certificate"
    value: str = Field(description="資產值")

    # 發現信息
    discovery_method: str = Field(description="發現方法")
    discovery_source: str = Field(description="發現來源")
    first_discovered: datetime = Field(description="首次發現時間")
    last_seen: datetime = Field(description="最後發現時間")

    # 資產屬性
    status: str = Field(
        description="資產狀態"
    )  # "active", "inactive", "monitoring", "expired"
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")

    # 技術信息
    technologies: list[str] = Field(default_factory=list, description="檢測到的技術")
    services: list[dict] = Field(default_factory=list, description="運行的服務")
    certificates: list[dict] = Field(default_factory=list, description="SSL證書信息")

    # 安全評估
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")
    vulnerability_count: int = Field(ge=0, description="漏洞數量")
    exposure_level: str = Field(
        description="暴露級別"
    )  # "public", "internal", "restricted"

    # 業務關聯
    business_unit: str | None = Field(default=None, description="業務單位")
    owner: str | None = Field(default=None, description="負責人")
    criticality: str = Field(
        description="重要性"
    )  # "critical", "high", "medium", "low"

    # 合規性
    compliance_status: dict[str, bool] = Field(
        default_factory=dict, description="合規狀態"
    )
    policy_violations: list[str] = Field(default_factory=list, description="政策違規")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
