"""
Enhanced Database Models for Asset and Vulnerability Lifecycle Management

這個模組包含增強的資料庫模型，支援資產與漏洞的完整生命週期管理。

Compliance Note (遵循 aiva_common 設計原則):
- 所有 enum 定義已移除,改為從 aiva_common.enums import (4-layer priority 原則)
- BusinessCriticality, Environment, AssetType, AssetStatus → aiva_common.enums.assets
- VulnerabilityStatus, Exploitability → aiva_common.enums.security
- Severity, Confidence → aiva_common.enums.common
- 修正日期: 2025-10-25
"""



from datetime import datetime
from typing import Any

from sqlalchemy import (  # type: ignore[import-not-found]
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB  # type: ignore[import-not-found]
from sqlalchemy.ext.declarative import (  # type: ignore[import-not-found]
    declarative_base,
)
from sqlalchemy.orm import relationship  # type: ignore[import-not-found]

# Import enums from aiva_common (Single Source of Truth)
from services.aiva_common.enums.assets import (
    AssetStatus,
    AssetType,
    BusinessCriticality,
    Environment,
)
from services.aiva_common.enums.common import Confidence, Severity
from services.aiva_common.enums.security import Exploitability, VulnerabilityStatus

Base = declarative_base()  # type: ignore[misc]


class Asset(Base):  # type: ignore[misc, valid-type]
    """資產模型 - 統一管理所有被掃描的目標資產"""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(255), unique=True, nullable=False, index=True)

    # 資產基本資訊
    name = Column(String(500))
    type = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text)

    # 業務與環境上下文
    business_criticality = Column(
        String(50), default=BusinessCriticality.MEDIUM.value, index=True
    )
    environment = Column(String(50), default=Environment.DEVELOPMENT.value, index=True)
    owner = Column(String(255), index=True)
    tags = Column(JSONB, default=[])

    # 技術資訊
    technology_stack = Column(JSONB, default={})
    metadata = Column(JSONB, default={})

    # 狀態與時間
    status = Column(String(50), default=AssetStatus.ACTIVE.value, index=True)
    first_discovered_at = Column(DateTime)
    last_scanned_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 關聯
    vulnerabilities = relationship(
        "Vulnerability", back_populates="asset", cascade="all, delete-orphan"
    )
    findings = relationship(
        "FindingRecord", back_populates="asset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Asset(asset_id={self.asset_id}, name={self.name}, type={self.type})>"

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "description": self.description,
            "business_criticality": self.business_criticality,
            "environment": self.environment,
            "owner": self.owner,
            "tags": self.tags,
            "technology_stack": self.technology_stack,
            "metadata": self.metadata,
            "status": self.status,
            "first_discovered_at": (
                self.first_discovered_at.isoformat()
                if self.first_discovered_at
                else None
            ),
            "last_scanned_at": (
                self.last_scanned_at.isoformat() if self.last_scanned_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Vulnerability(Base):  # type: ignore[misc, valid-type]
    """漏洞模型 - 去重後的漏洞總表，管理完整生命週期"""

    __tablename__ = "vulnerabilities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vulnerability_id = Column(String(255), unique=True, nullable=False, index=True)

    # 漏洞基本資訊
    name = Column(String(255), nullable=False)
    vulnerability_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(50), nullable=False, index=True)
    confidence = Column(String(50), nullable=False)

    # 標準參考
    cwe = Column(String(50))
    cve = Column(String(50))
    owasp_category = Column(String(100))

    # 關聯資產
    asset_id = Column(
        String(255), ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False
    )

    # 位置資訊
    location = Column(JSONB, nullable=False)

    # 漏洞詳細資訊
    description = Column(Text)
    impact = Column(Text)
    remediation = Column(Text)

    # 風險評估
    cvss_score = Column(Numeric(3, 1))
    risk_score = Column(Numeric(5, 2), index=True)
    exploitability = Column(String(50), default=Exploitability.MEDIUM.value)
    business_impact = Column(String(50), index=True)

    # 生命週期狀態
    status = Column(String(50), default=VulnerabilityStatus.NEW.value, index=True)
    resolution = Column(String(50))

    # 時間追蹤
    first_detected_at = Column(DateTime, nullable=False)
    last_detected_at = Column(DateTime, nullable=False)
    fixed_at = Column(DateTime)
    verified_fixed_at = Column(DateTime)

    # 處理資訊
    assigned_to = Column(String(255), index=True)
    sla_deadline = Column(DateTime)
    notes = Column(Text)

    # 關聯與根因
    root_cause_vulnerability_id = Column(String(255))
    related_vulnerability_ids = Column(JSONB, default=[])

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # 關聯
    asset = relationship("Asset", back_populates="vulnerabilities")
    findings = relationship(
        "FindingRecord", back_populates="vulnerability", cascade="all, delete-orphan"
    )
    history = relationship(
        "VulnerabilityHistory", back_populates="vulnerability", cascade="all, delete-orphan"
    )
    tags = relationship(
        "VulnerabilityTag", back_populates="vulnerability", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Vulnerability(vulnerability_id={self.vulnerability_id}, name={self.name}, severity={self.severity})>"

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "vulnerability_id": self.vulnerability_id,
            "name": self.name,
            "vulnerability_type": self.vulnerability_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "cwe": self.cwe,
            "cve": self.cve,
            "owasp_category": self.owasp_category,
            "asset_id": self.asset_id,
            "location": self.location,
            "description": self.description,
            "impact": self.impact,
            "remediation": self.remediation,
            "cvss_score": float(self.cvss_score) if self.cvss_score else None,
            "risk_score": float(self.risk_score) if self.risk_score else None,
            "exploitability": self.exploitability,
            "business_impact": self.business_impact,
            "status": self.status,
            "resolution": self.resolution,
            "first_detected_at": (
                self.first_detected_at.isoformat() if self.first_detected_at else None
            ),
            "last_detected_at": (
                self.last_detected_at.isoformat() if self.last_detected_at else None
            ),
            "fixed_at": self.fixed_at.isoformat() if self.fixed_at else None,
            "verified_fixed_at": (
                self.verified_fixed_at.isoformat() if self.verified_fixed_at else None
            ),
            "assigned_to": self.assigned_to,
            "sla_deadline": (
                self.sla_deadline.isoformat() if self.sla_deadline else None
            ),
            "notes": self.notes,
            "root_cause_vulnerability_id": self.root_cause_vulnerability_id,
            "related_vulnerability_ids": self.related_vulnerability_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class VulnerabilityHistory(Base):  # type: ignore[misc, valid-type]
    """漏洞歷史模型 - 追蹤所有狀態和屬性變更"""

    __tablename__ = "vulnerability_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vulnerability_id = Column(
        String(255),
        ForeignKey("vulnerabilities.vulnerability_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 變更資訊
    changed_by = Column(String(255))
    change_type = Column(String(50), nullable=False)
    old_value = Column(Text)
    new_value = Column(Text)
    comment = Column(Text)

    created_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # 關聯
    vulnerability = relationship("Vulnerability", back_populates="history")

    def __repr__(self) -> str:
        return f"<VulnerabilityHistory(vulnerability_id={self.vulnerability_id}, change_type={self.change_type})>"


class VulnerabilityTag(Base):  # type: ignore[misc, valid-type]
    """漏洞標籤模型 - 靈活的分類和過濾系統"""

    __tablename__ = "vulnerability_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vulnerability_id = Column(
        String(255),
        ForeignKey("vulnerabilities.vulnerability_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tag = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 關聯
    vulnerability = relationship("Vulnerability", back_populates="tags")

    __table_args__ = (
        UniqueConstraint("vulnerability_id", "tag", name="uq_vulnerability_tag"),
    )

    def __repr__(self) -> str:
        return f"<VulnerabilityTag(vulnerability_id={self.vulnerability_id}, tag={self.tag})>"


# 擴展現有的 FindingRecord 模型
class FindingRecord(Base):  # type: ignore[misc, valid-type]
    """漏洞發現記錄表 - 擴展版本"""

    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    finding_id = Column(String(255), unique=True, nullable=False, index=True)
    scan_id = Column(String(255), nullable=False, index=True)
    task_id = Column(String(255), nullable=False, index=True)

    # 漏洞信息
    vulnerability_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False, index=True)
    confidence = Column(String(50), nullable=False)
    cwe = Column(String(50))

    # 目標信息
    target_url = Column(Text, nullable=False)
    target_parameter = Column(String(255))
    target_method = Column(String(10))

    # 狀態
    status = Column(String(50), nullable=False, index=True)

    # 完整的 JSON 數據
    raw_data = Column(Text, nullable=False)

    # 關聯到增強的模型
    vulnerability_id = Column(
        String(255), ForeignKey("vulnerabilities.vulnerability_id", ondelete="SET NULL")
    )
    asset_id = Column(
        String(255), ForeignKey("assets.asset_id", ondelete="CASCADE")
    )

    # 時間戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 關聯
    vulnerability = relationship("Vulnerability", back_populates="findings")
    asset = relationship("Asset", back_populates="findings")

    def __repr__(self) -> str:
        return f"<FindingRecord(finding_id={self.finding_id}, vulnerability_name={self.vulnerability_name})>"
