"""
AIVA 掃描發現模式定義

包含掃描、發現、資產清點、技術指紋等相關的數據模式。
屬於 scan 模組的業務特定定義。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aiva_common.enums import AssetType, Confidence, ScanStatus, Severity
from aiva_common.standards import CVEReference, CVSSv3Metrics, CWEReference
from pydantic import BaseModel, Field, HttpUrl, field_validator

# ==================== 掃描請求和結果 ====================


class ScanScope(BaseModel):
    """掃描範圍定義"""

    included_hosts: list[str] = Field(default_factory=list, description="包含的主機")
    excluded_hosts: list[str] = Field(default_factory=list, description="排除的主機")
    included_paths: list[str] = Field(default_factory=list, description="包含的路徑")
    excluded_paths: list[str] = Field(default_factory=list, description="排除的路徑")
    max_depth: int = Field(default=5, ge=1, le=20, description="最大掃描深度")


class ScanRequest(BaseModel):
    """掃描請求"""

    scan_id: str = Field(description="掃描ID", pattern=r"^scan_[a-zA-Z0-9_]+$")
    targets: list[HttpUrl] = Field(description="目標URL列表", min_items=1)
    scope: ScanScope = Field(description="掃描範圍")
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

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if len(v) > 100:
            raise ValueError("Cannot scan more than 100 targets at once")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed_strategies = ["passive", "active", "aggressive", "stealth"]
        if v not in allowed_strategies:
            raise ValueError(f"Strategy must be one of {allowed_strategies}")
        return v


class Asset(BaseModel):
    """資產信息"""

    url: str = Field(description="資產URL")
    asset_type: AssetType = Field(description="資產類型")
    title: str | None = Field(default=None, description="頁面標題")
    status_code: int = Field(description="HTTP狀態碼")
    content_type: str | None = Field(default=None, description="內容類型")
    content_length: int | None = Field(default=None, description="內容長度")


class TechStackInfo(BaseModel):
    """技術棧信息"""

    name: str = Field(description="技術名稱")
    version: str | None = Field(default=None, description="版本")
    category: str = Field(description="技術分類")
    confidence: Confidence = Field(description="檢測信心度")
    evidence: list[str] = Field(default_factory=list, description="檢測證據")


class ServiceInfo(BaseModel):
    """服務信息"""

    name: str = Field(description="服務名稱")
    version: str | None = Field(default=None, description="服務版本")
    port: int = Field(description="端口號", ge=1, le=65535)
    protocol: str = Field(description="協議類型")
    banner: str | None = Field(default=None, description="服務橫幅")
    is_ssl: bool = Field(default=False, description="是否使用SSL")


class Fingerprints(BaseModel):
    """指紋信息集合"""

    tech_stack: list[TechStackInfo] = Field(default_factory=list, description="技術棧")
    services: list[ServiceInfo] = Field(default_factory=list, description="服務列表")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie信息")


class AssetInventory(BaseModel):
    """資產清單"""

    scan_id: str = Field(description="掃描ID")
    assets: list[Asset] = Field(description="資產列表")
    fingerprints: Fingerprints = Field(description="指紋信息")
    discovery_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_count: int = Field(description="總資產數量")


class ScanResult(BaseModel):
    """掃描結果"""

    scan_id: str = Field(description="掃描ID")
    status: ScanStatus = Field(description="掃描狀態")
    assets: list[Asset] = Field(description="發現的資產")
    fingerprints: Fingerprints = Field(description="技術指紋")
    summary: dict[str, Any] = Field(default_factory=dict, description="掃描摘要")
    start_time: datetime = Field(description="開始時間")
    end_time: datetime | None = Field(default=None, description="結束時間")
    duration: int | None = Field(default=None, description="執行時間(秒)")


# ==================== 漏洞發現相關 ====================


class TargetInfo(BaseModel):
    """目標信息"""

    url: str = Field(description="目標URL")
    method: str = Field(default="GET", description="HTTP方法")
    headers: dict[str, str] = Field(default_factory=dict, description="請求標頭")
    parameters: dict[str, str] = Field(default_factory=dict, description="請求參數")
    body: str | None = Field(default=None, description="請求體")


class FindingEvidence(BaseModel):
    """漏洞證據"""

    request: str | None = Field(default=None, description="原始請求")
    response: str | None = Field(default=None, description="原始響應")
    payload: str | None = Field(default=None, description="測試載荷")
    screenshot: str | None = Field(default=None, description="截圖(base64)")
    additional_data: dict[str, Any] = Field(default_factory=dict, description="額外數據")


class FindingImpact(BaseModel):
    """漏洞影響"""

    description: str = Field(description="影響描述")
    affected_assets: list[str] = Field(default_factory=list, description="受影響資產")
    business_impact: str | None = Field(default=None, description="業務影響")
    technical_impact: str | None = Field(default=None, description="技術影響")


class FindingRecommendation(BaseModel):
    """修復建議"""

    short_term: list[str] = Field(default_factory=list, description="短期建議")
    long_term: list[str] = Field(default_factory=list, description="長期建議")
    references: list[str] = Field(default_factory=list, description="參考資料")
    effort_estimate: str | None = Field(default=None, description="修復工作量估計")


class VulnerabilityFinding(BaseModel):
    """漏洞發現 - 掃描模組的核心數據結構"""

    finding_id: str = Field(description="發現ID", pattern=r"^finding_[a-zA-Z0-9_]+$")
    scan_id: str = Field(description="關聯的掃描ID")

    # 漏洞基本信息
    name: str = Field(description="漏洞名稱")
    description: str = Field(description="漏洞描述")
    severity: Severity = Field(description="嚴重程度")
    confidence: Confidence = Field(description="置信度")

    # 目標和證據
    target: TargetInfo = Field(description="目標信息")
    evidence: FindingEvidence | None = Field(default=None, description="漏洞證據")
    impact: FindingImpact | None = Field(default=None, description="影響評估")
    recommendation: FindingRecommendation | None = Field(default=None, description="修復建議")

    # 標準化引用
    cve: CVEReference | None = Field(default=None, description="CVE引用")
    cwe: CWEReference | None = Field(default=None, description="CWE引用")
    cvss: CVSSv3Metrics | None = Field(default=None, description="CVSS評分")

    # 分類和標籤
    category: str | None = Field(default=None, description="漏洞分類")
    tags: list[str] = Field(default_factory=list, description="標籤")
    owasp_category: str | None = Field(default=None, description="OWASP分類")

    # 時間戳
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verified_at: datetime | None = Field(default=None, description="驗證時間")

    # 元數據
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")

    @field_validator("finding_id")
    @classmethod
    def validate_finding_id(cls, v: str) -> str:
        if not v.startswith("finding_"):
            raise ValueError("finding_id must start with 'finding_'")
        return v


class ScopeDefinition(BaseModel):
    """掃描範圍定義 - 更詳細的版本"""

    name: str = Field(description="範圍名稱")
    description: str | None = Field(default=None, description="範圍描述")

    # 網路範圍
    ip_ranges: list[str] = Field(default_factory=list, description="IP範圍")
    domains: list[str] = Field(default_factory=list, description="域名列表")
    subdomains_allowed: bool = Field(default=True, description="是否包含子域名")

    # 端口範圍
    port_ranges: list[str] = Field(default_factory=list, description="端口範圍")

    # 排除項
    excluded_ips: list[str] = Field(default_factory=list, description="排除的IP")
    excluded_domains: list[str] = Field(default_factory=list, description="排除的域名")
    excluded_paths: list[str] = Field(default_factory=list, description="排除的路徑")

    # 限制條件
    rate_limit: int = Field(default=10, ge=1, description="請求速率限制(每秒)")
    max_requests: int | None = Field(default=None, description="最大請求數")
    timeout: int = Field(default=30, ge=1, description="請求超時時間(秒)")

    # 元數據
    created_by: str | None = Field(default=None, description="創建者")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = Field(default=True, description="是否啟用")


__all__ = [
    "ScanScope",
    "ScanRequest",
    "Asset",
    "TechStackInfo",
    "ServiceInfo",
    "Fingerprints",
    "AssetInventory",
    "ScanResult",
    "TargetInfo",
    "FindingEvidence",
    "FindingImpact",
    "FindingRecommendation",
    "VulnerabilityFinding",
    "ScopeDefinition",
]
