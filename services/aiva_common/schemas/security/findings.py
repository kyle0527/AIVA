"""
漏洞發現相關 Schema

此模組包含與漏洞發現、證據收集、影響評估等相關的資料模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...enums import Confidence, Severity, VulnerabilityType

# ==================== 漏洞基本資訊 ====================


class Vulnerability(BaseModel):
    """漏洞基本資訊 - 用於 Finding 中的漏洞描述

    符合標準：
    - CWE: Common Weakness Enumeration (MITRE)
    - CVE: Common Vulnerabilities and Exposures
    - CVSS: Common Vulnerability Scoring System v3.1/v4.0
    - OWASP: Open Web Application Security Project
    """

    name: VulnerabilityType
    cwe: str | None = Field(
        default=None,
        description="CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/",
        pattern=r"^CWE-\d+$",
    )
    cve: str | None = Field(
        default=None,
        description="CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/",
        pattern=r"^CVE-\d{4}-\d{4,}$",
    )
    severity: Severity
    confidence: Confidence
    description: str | None = None
    cvss_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/",
    )
    cvss_vector: str | None = Field(
        default=None,
        description="CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        pattern=r"^CVSS:3\.[01]/.*",
    )
    owasp_category: str | None = Field(
        default=None,
        description="OWASP Top 10 分類，例如: A03:2021-Injection",
    )


class Target(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any  # Accept arbitrary URL-like values
    parameter: str | None = None
    method: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    body: str | None = None


# 保持向後相容的別名
FindingTarget = Target


class FindingEvidence(BaseModel):
    """漏洞證據"""

    payload: str | None = None
    response_time_delta: float | None = None
    db_version: str | None = None
    request: str | None = None
    response: str | None = None
    proof: str | None = None


class FindingImpact(BaseModel):
    """漏洞影響描述"""

    description: str | None = None
    business_impact: str | None = None
    technical_impact: str | None = None
    affected_users: int | None = None
    estimated_cost: float | None = None


class FindingRecommendation(BaseModel):
    """漏洞修復建議"""

    fix: str | None = None
    priority: str | None = None
    remediation_steps: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class FindingPayload(BaseModel):
    """漏洞發現 Payload - 統一的漏洞報告格式"""

    finding_id: str
    task_id: str
    scan_id: str
    status: str
    vulnerability: Vulnerability
    target: Target
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("finding_id")
    @classmethod
    def validate_finding_id(cls, v: str) -> str:
        if not v.startswith("finding_"):
            raise ValueError("finding_id must start with 'finding_'")
        return v

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v

    @field_validator("scan_id")
    @classmethod
    def validate_scan_id(cls, v: str) -> str:
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"confirmed", "potential", "false_positive", "needs_review"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ==================== Info Gatherer Schemas ====================


class SensitiveMatch(BaseModel):
    """敏感資訊匹配結果"""

    match_id: str
    pattern_name: str  # e.g., "password", "api_key", "credit_card", "private_key"
    matched_text: str
    context: str  # 前後文 (遮蔽敏感部分)
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 - 1.0
    line_number: int | None = None
    file_path: str | None = None
    url: str | None = None
    severity: Severity = Severity.MEDIUM


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""

    analysis_id: str
    url: str
    source_size_bytes: int

    # 詳細分析結果
    dangerous_functions: list[str] = Field(
        default_factory=list
    )  # eval, Function, setTimeout等
    external_resources: list[str] = Field(default_factory=list)  # 外部 URL
    data_leaks: list[dict[str, str]] = Field(default_factory=list)  # 數據洩漏信息

    # 通用欄位 (保持兼容)
    findings: list[str] = Field(
        default_factory=list
    )  # e.g., ["uses_eval", "dom_manipulation"]
    apis_called: list[str] = Field(default_factory=list)  # 發現的 API 端點
    ajax_endpoints: list[str] = Field(default_factory=list)  # AJAX 呼叫端點
    suspicious_patterns: list[str] = Field(default_factory=list)

    # 評分欄位
    risk_score: float = Field(ge=0.0, le=10.0, default=0.0)  # 0.0 - 10.0
    security_score: int = Field(ge=0, le=100, default=100)  # 0-100 分

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 漏洞關聯分析 ====================


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析結果"""

    correlation_id: str
    correlation_type: str  # "code_level", "data_flow", "attack_chain"
    related_findings: list[str]  # finding_ids
    confidence_score: float  # 0.0 - 1.0
    root_cause: str | None = None
    common_components: list[str] = Field(default_factory=list)
    explanation: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CodeLevelRootCause(BaseModel):
    """程式碼層面根因分析結果"""

    analysis_id: str
    vulnerable_component: str  # 共用函式庫、父類別等
    affected_findings: list[str]  # finding_ids
    code_location: str | None = None
    vulnerability_pattern: str | None = None
    fix_recommendation: str | None = None


class SASTDASTCorrelation(BaseModel):
    """SAST-DAST 資料流關聯結果"""

    correlation_id: str
    sast_finding_id: str
    dast_finding_id: str
    data_flow_path: list[str]  # Source -> Sink path
    verification_status: str  # "verified", "partial", "unverified"
    confidence_score: float  # 0.0 - 1.0
    explanation: str | None = None


# ==================== AI 驅動漏洞驗證 ====================


class AIVerificationRequest(BaseModel):
    """AI 驅動漏洞驗證請求"""

    verification_id: str
    finding_id: str
    scan_id: str
    vulnerability_type: VulnerabilityType
    target: Target
    evidence: FindingEvidence
    verification_mode: str = "non_destructive"  # "non_destructive", "safe", "full"
    context: dict[str, Any] = Field(default_factory=dict)


class AIVerificationResult(BaseModel):
    """AI 驅動漏洞驗證結果"""

    verification_id: str
    finding_id: str
    verification_status: str  # "confirmed", "false_positive", "needs_review"
    confidence_score: float  # 0.0 - 1.0
    verification_method: str
    test_steps: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VulnerabilityScorecard(BaseModel):
    """漏洞評分卡 - 綜合漏洞評估報告"""

    vulnerability_id: str = Field(description="漏洞唯一標識")
    name: str = Field(description="漏洞名稱")
    severity: Severity = Field(description="嚴重程度")
    confidence: Confidence = Field(description="信心度")
    cvss_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="CVSS 基礎分數"
    )

    # 影響評估
    impact_score: float = Field(ge=0.0, le=10.0, description="影響評分 (0-10)")
    exploitability_score: float = Field(
        ge=0.0, le=10.0, description="可利用性評分 (0-10)"
    )

    # 風險評估
    risk_level: str = Field(description="風險等級 (Critical/High/Medium/Low)")
    business_impact: str | None = Field(default=None, description="業務影響描述")

    # 修復建議
    remediation_effort: str | None = Field(
        default=None, description="修復工作量 (High/Medium/Low)"
    )
    recommended_actions: list[str] = Field(
        default_factory=list, description="建議修復動作"
    )

    # 元數據
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="評分卡創建時間"
    )
    updated_at: datetime | None = Field(default=None, description="最後更新時間")
    evaluator_version: str | None = Field(default=None, description="評估器版本")
