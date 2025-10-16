"""
參考資料 Schemas

此模組定義了各種標準參考資料的數據模型，包括 CVE、CWE、
技術指紋和漏洞發現記錄等。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..enums import Confidence, Severity


# ============================================================================
# 安全標準參考
# ============================================================================

class CVEReference(BaseModel):
    """CVE 參考資訊

    符合標準: CVE Numbering Authority (https://www.cve.org/)
    """

    cve_id: str = Field(
        description="CVE ID (格式: CVE-YYYY-NNNNN)",
        pattern=r"^CVE-\d{4}-\d{4,}$",
    )
    description: str | None = None
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0)
    cvss_vector: str | None = None
    references: list[str] = Field(default_factory=list)
    published_date: datetime | None = None
    last_modified_date: datetime | None = None


class CWEReference(BaseModel):
    """CWE 參考資訊

    符合標準: Common Weakness Enumeration (https://cwe.mitre.org/)
    """

    cwe_id: str = Field(description="CWE ID (格式: CWE-XXX)", pattern=r"^CWE-\d+$")
    name: str | None = None
    description: str | None = None
    weakness_category: str | None = None  # "Class", "Base", "Variant", "Compound"
    likelihood_of_exploit: str | None = None  # "High", "Medium", "Low"


# ============================================================================
# 技術指紋
# ============================================================================

class TechnicalFingerprint(BaseModel):
    """技術指紋識別"""

    technology: str = Field(description="技術名稱")
    version: str | None = Field(default=None, description="版本信息")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    detection_method: str = Field(description="檢測方法")
    evidence: list[str] = Field(default_factory=list, description="檢測證據")

    # 技術分類
    category: str = Field(description="技術分類")  # "web_server", "framework", "cms", "database"
    subcategory: str | None = Field(default=None, description="子分類")

    # 安全相關
    known_vulnerabilities: list[str] = Field(default_factory=list, description="已知漏洞")
    eol_status: bool | None = Field(default=None, description="是否已停止支持")

    metadata: dict[str, Any] = Field(default_factory=dict, description="額外信息")


# ============================================================================
# 漏洞發現
# ============================================================================

class VulnerabilityDiscovery(BaseModel):
    """漏洞發現記錄"""

    discovery_id: str = Field(description="發現ID")
    vulnerability_id: str = Field(description="漏洞ID")
    asset_id: str = Field(description="相關資產ID")

    # 漏洞基本信息
    title: str = Field(description="漏洞標題")
    description: str = Field(description="漏洞描述")
    severity: Severity = Field(description="嚴重程度")
    confidence: Confidence = Field(description="置信度")

    # 技術細節
    vulnerability_type: str = Field(description="漏洞類型")
    affected_component: str | None = Field(default=None, description="受影響組件")
    attack_vector: str | None = Field(default=None, description="攻擊向量")

    # 檢測信息
    detection_method: str = Field(description="檢測方法")
    scanner_name: str = Field(description="掃描器名稱")
    scan_rule_id: str | None = Field(default=None, description="掃描規則ID")

    # 證據和驗證
    evidence: list[str] = Field(default_factory=list, description="漏洞證據")
    proof_of_concept: str | None = Field(default=None, description="概念驗證")
    false_positive_likelihood: float = Field(ge=0.0, le=1.0, description="誤報可能性")

    # 影響評估
    impact_assessment: str | None = Field(default=None, description="影響評估")
    exploitability: str | None = Field(default=None, description="可利用性")

    # 修復建議
    remediation_advice: str | None = Field(default=None, description="修復建議")
    remediation_priority: str | None = Field(default=None, description="修復優先級")

    # 標準映射
    cve_ids: list[str] = Field(default_factory=list, description="CVE標識符")
    cwe_ids: list[str] = Field(default_factory=list, description="CWE標識符")
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0, description="CVSS評分")

    # 時間戳
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
