"""
AIVA Scan Models - 掃描發現模組

此文件包含與掃描、資產發現、技術指紋識別、漏洞檢測相關的所有數據模型。

職責範圍：
1. 掃描配置和控制 (ScanScope, ScanStartPayload)
2. 資產發現和清點 (Asset, AssetInventoryItem)
3. 技術指紋識別 (TechnicalFingerprint, Fingerprints)
4. 漏洞發現記錄 (Vulnerability, VulnerabilityDiscovery)
5. 掃描結果匯總 (Summary, ScanCompletedPayload)
6. 外部攻擊面管理 (EASMDiscoveryPayload, EASMAsset)
7. 資產和漏洞生命週期管理
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..aiva_common.enums import (
    AssetType,
    Confidence,
    Severity,
    VulnerabilityStatus,
    VulnerabilityType,
)
from ..aiva_common.models import (
    Authentication,
    RateLimit,
)

# ==================== 掃描配置和控制 ====================


class ScanScope(BaseModel):
    """掃描範圍定義"""

    exclusions: list[str] = Field(default_factory=list, description="排除列表")
    include_subdomains: bool = Field(default=True, description="是否包含子域名")
    allowed_hosts: list[str] = Field(default_factory=list, description="允許的主機")


class ScanStartPayload(BaseModel):
    """掃描啟動載荷"""

    scan_id: str = Field(description="掃描ID")
    targets: list[HttpUrl] = Field(description="目標URL列表")
    scope: ScanScope = Field(default_factory=ScanScope, description="掃描範圍")
    authentication: Authentication = Field(default_factory=Authentication, description="認證配置")
    strategy: str = Field(default="deep", description="掃描策略")
    rate_limit: RateLimit = Field(default_factory=RateLimit, description="速率限制")
    custom_headers: dict[str, str] = Field(default_factory=dict, description="自定義HTTP標頭")
    x_forwarded_for: str | None = Field(default=None, description="X-Forwarded-For標頭")

    @field_validator("scan_id")
    @classmethod
    def validate_scan_id(cls, v: str) -> str:
        """驗證掃描 ID 格式"""
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        if len(v) < 10:
            raise ValueError("scan_id too short (minimum 10 characters)")
        return v

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        """驗證目標列表"""
        if not v:
            raise ValueError("At least one target required")
        if len(v) > 100:
            raise ValueError("Too many targets (maximum 100)")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """驗證掃描策略"""
        allowed = {"quick", "normal", "deep", "stealth"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


class EnhancedScanScope(BaseModel):
    """增強掃描範圍定義"""

    included_hosts: list[str] = Field(default_factory=list, description="包含的主機")
    excluded_hosts: list[str] = Field(default_factory=list, description="排除的主機")
    included_paths: list[str] = Field(default_factory=list, description="包含的路徑")
    excluded_paths: list[str] = Field(default_factory=list, description="排除的路徑")
    max_depth: int = Field(default=5, ge=1, le=20, description="最大掃描深度")


class EnhancedScanRequest(BaseModel):
    """增強掃描請求"""

    scan_id: str = Field(description="掃描ID", pattern=r"^scan_[a-zA-Z0-9_]+$")
    targets: list[HttpUrl] = Field(description="目標URL列表", min_length=1)
    scope: EnhancedScanScope = Field(description="掃描範圍")
    strategy: str = Field(description="掃描策略", pattern=r"^[a-zA-Z0-9_]+$")
    priority: int = Field(default=5, ge=1, le=10, description="優先級 1-10")
    max_duration: int = Field(default=3600, ge=60, description="最大執行時間(秒)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")

    @field_validator("scan_id")
    @classmethod
    def validate_scan_id(cls, v: str) -> str:
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v


# ==================== 資產管理 ====================


class Asset(BaseModel):
    """資產基本信息"""

    asset_type: AssetType = Field(description="資產類型")
    url: HttpUrl = Field(description="資產URL")
    discovery_method: str = Field(description="發現方法")
    confidence: Confidence = Field(description="置信度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


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


class Fingerprints(BaseModel):
    """指紋集合"""

    server: str | None = Field(default=None, description="服務器信息")
    technologies: list[str] = Field(default_factory=list, description="檢測到的技術")
    frameworks: list[str] = Field(default_factory=list, description="框架列表")
    cms: str | None = Field(default=None, description="內容管理系統")
    metadata: dict[str, Any] = Field(default_factory=dict, description="其他指紋信息")


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
    fingerprints: list[TechnicalFingerprint] = Field(default_factory=list, description="技術指紋")

    # 業務信息
    business_criticality: str = Field(description="業務重要性")  # "critical", "high", "medium", "low"
    owner: str | None = Field(default=None, description="負責人")
    environment: str = Field(description="環境類型")  # "production", "staging", "development"

    # 安全狀態
    last_scanned: datetime | None = Field(default=None, description="最後掃描時間")
    vulnerability_count: int = Field(ge=0, description="漏洞數量")
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")

    # 時間戳
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 掃描結果 ====================


class Summary(BaseModel):
    """掃描結果摘要"""

    total_requests: int = Field(ge=0, description="總請求數")
    total_assets: int = Field(ge=0, description="總資產數")
    total_vulnerabilities: int = Field(ge=0, description="總漏洞數")
    duration_seconds: float = Field(ge=0, description="掃描時長(秒)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="其他統計信息")


class ScanCompletedPayload(BaseModel):
    """掃描完成載荷"""

    scan_id: str = Field(description="掃描ID")
    status: str = Field(description="掃描狀態")
    summary: Summary = Field(description="掃描摘要")
    assets: list[Asset] = Field(description="發現的資產")
    fingerprints: Fingerprints = Field(description="技術指紋")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 漏洞管理 ====================


class Vulnerability(BaseModel):
    """漏洞基本信息"""

    vuln_id: str = Field(description="漏洞ID")
    title: str = Field(description="漏洞標題")
    description: str = Field(description="漏洞描述")
    severity: Severity = Field(description="嚴重程度")
    confidence: Confidence = Field(description="置信度")
    vuln_type: VulnerabilityType = Field(description="漏洞類型")

    # 位置信息
    url: HttpUrl = Field(description="漏洞URL")
    parameter: str | None = Field(default=None, description="受影響參數")
    method: str = Field(default="GET", description="HTTP方法")

    # 技術細節
    evidence: list[str] = Field(default_factory=list, description="證據列表")
    payload: str | None = Field(default=None, description="測試載荷")
    cwe_ids: list[str] = Field(default_factory=list, description="CWE標識符")
    cve_ids: list[str] = Field(default_factory=list, description="CVE標識符")

    # CVSS評分
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0, description="CVSS評分")
    cvss_vector: str | None = Field(default=None, description="CVSS向量")

    # 修復建議
    remediation: str | None = Field(default=None, description="修復建議")
    references: list[str] = Field(default_factory=list, description="參考鏈接")

    # 元數據
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="其他信息")


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


class VulnerabilityLifecyclePayload(BaseModel):
    """漏洞生命週期載荷"""

    vuln_id: str = Field(description="漏洞ID")
    status: VulnerabilityStatus = Field(description="漏洞狀態")
    previous_status: VulnerabilityStatus | None = Field(default=None, description="前一狀態")
    changed_by: str = Field(description="變更者")
    change_reason: str | None = Field(default=None, description="變更原因")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class VulnerabilityUpdatePayload(BaseModel):
    """漏洞更新載荷"""

    vuln_id: str = Field(description="漏洞ID")
    updates: dict[str, Any] = Field(description="更新內容")
    updated_by: str = Field(description="更新者")
    update_reason: str | None = Field(default=None, description="更新原因")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 外部攻擊面管理 (EASM) ====================


class DiscoveredAsset(BaseModel):
    """EASM發現的資產"""

    asset_id: str = Field(description="資產ID")
    asset_type: AssetType = Field(description="資產類型")
    value: str = Field(description="資產值")
    discovery_source: str = Field(description="發現來源")
    confidence: Confidence = Field(description="置信度")
    first_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EASMDiscoveryPayload(BaseModel):
    """EASM發現載荷"""

    discovery_id: str = Field(description="發現ID")
    target_domain: str = Field(description="目標域名")
    discovery_method: str = Field(description="發現方法")
    assets: list[DiscoveredAsset] = Field(description="發現的資產列表")
    started_at: datetime = Field(description="開始時間")
    completed_at: datetime = Field(description="完成時間")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EASMDiscoveryResult(BaseModel):
    """EASM發現結果"""

    result_id: str = Field(description="結果ID")
    discovery_id: str = Field(description="關聯發現ID")
    total_assets: int = Field(ge=0, description="總資產數")
    new_assets: int = Field(ge=0, description="新發現資產數")
    updated_assets: int = Field(ge=0, description="更新資產數")
    summary: dict[str, int] = Field(default_factory=dict, description="分類統計")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EASMAsset(BaseModel):
    """外部攻擊面管理資產"""

    asset_id: str = Field(description="資產ID")
    asset_type: str = Field(description="資產類型")  # "domain", "subdomain", "ip", "service", "certificate"
    value: str = Field(description="資產值")

    # 發現信息
    discovery_method: str = Field(description="發現方法")
    discovery_source: str = Field(description="發現來源")
    first_discovered: datetime = Field(description="首次發現時間")
    last_seen: datetime = Field(description="最後發現時間")

    # 資產屬性
    status: str = Field(description="資產狀態")  # "active", "inactive", "monitoring", "expired"
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")

    # 技術信息
    technologies: list[str] = Field(default_factory=list, description="檢測到的技術")
    services: list[dict] = Field(default_factory=list, description="運行的服務")
    certificates: list[dict] = Field(default_factory=list, description="SSL證書信息")

    # 安全評估
    risk_score: float = Field(ge=0.0, le=10.0, description="風險評分")
    vulnerability_count: int = Field(ge=0, description="漏洞數量")
    exposure_level: str = Field(description="暴露級別")  # "public", "internal", "restricted"

    # 業務關聯
    business_unit: str | None = Field(default=None, description="業務單位")
    owner: str | None = Field(default=None, description="負責人")
    criticality: str = Field(description="重要性")  # "critical", "high", "medium", "low"

    # 合規性
    compliance_status: dict[str, bool] = Field(default_factory=dict, description="合規狀態")
    policy_violations: list[str] = Field(default_factory=list, description="政策違規")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 資產生命週期 ====================


class AssetLifecyclePayload(BaseModel):
    """資產生命週期載荷"""

    asset_id: str = Field(description="資產ID")
    event_type: str = Field(description="事件類型")  # "discovered", "updated", "decommissioned", "verified"
    previous_state: dict[str, Any] = Field(default_factory=dict, description="前一狀態")
    current_state: dict[str, Any] = Field(description="當前狀態")
    changed_by: str = Field(description="變更者")
    change_reason: str | None = Field(default=None, description="變更原因")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")



class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""

    analysis_id: str = Field(description="分析ID")
    url: str = Field(description="分析URL")
    source_size_bytes: int = Field(ge=0, description="源代碼大小(字節)")

    # 詳細分析結果
    dangerous_functions: list[str] = Field(
        default_factory=list, description="危險函數列表"
    )
    external_resources: list[str] = Field(default_factory=list, description="外部資源")
    data_leaks: list[dict[str, str]] = Field(default_factory=list, description="數據洩漏")

    # 通用欄位
    findings: list[str] = Field(default_factory=list, description="發現列表")
    apis_called: list[str] = Field(default_factory=list, description="API調用")
    ajax_endpoints: list[str] = Field(default_factory=list, description="AJAX端點")
    suspicious_patterns: list[str] = Field(default_factory=list, description="可疑模式")

    # 評分欄位
    risk_score: float = Field(ge=0.0, le=10.0, default=0.0, description="風險評分")
    security_score: int = Field(ge=0, le=100, default=100, description="安全評分")




__all__ = [
    # 掃描控制
    "ScanScope",
    "ScanStartPayload",
    "EnhancedScanScope",
    "EnhancedScanRequest",
    "ScanCompletedPayload",
    "Summary",
    # 資產管理
    "Asset",
    "TechnicalFingerprint",
    "AssetInventoryItem",
    "AssetLifecyclePayload",
    "DiscoveredAsset",
    # 漏洞發現
    "Vulnerability",
    "VulnerabilityDiscovery",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload",
    # EASM 集成
    "EASMAsset",
    "EASMDiscoveryPayload",
    "EASMDiscoveryResult",
    # 指紋和分析
    "Fingerprints",
    "JavaScriptAnalysisResult",
]
