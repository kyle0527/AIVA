from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .enums import (
    AssetExposure,
    AssetType,
    AttackPathEdgeType,
    AttackPathNodeType,
    BusinessCriticality,
    ComplianceFramework,
    Confidence,
    DataSensitivity,
    Environment,
    Exploitability,
    IntelSource,
    IOCType,
    ModuleName,
    PostExTestType,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    Severity,
    TestStatus,
    ThreatLevel,
    Topic,
    VulnerabilityStatus,
    VulnerabilityType,
)


class MessageHeader(BaseModel):
    message_id: str
    trace_id: str
    correlation_id: str | None = None
    source_module: ModuleName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"


class AivaMessage(BaseModel):
    header: MessageHeader
    topic: Topic
    schema_version: str = "1.0"
    payload: dict[str, Any]


class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 標準指標"""

    model_config = {"str_strip_whitespace": True}

    # Base Metrics (Required)
    attack_vector: Literal["N", "A", "L", "P"] = Field(description="攻擊向量")
    attack_complexity: Literal["L", "H"] = Field(description="攻擊複雜度")
    privileges_required: Literal["N", "L", "H"] = Field(description="所需權限")
    user_interaction: Literal["N", "R"] = Field(description="用戶交互")
    scope: Literal["U", "C"] = Field(description="範圍")
    confidentiality: Literal["N", "L", "H"] = Field(description="機密性影響")
    integrity: Literal["N", "L", "H"] = Field(description="完整性影響")
    availability: Literal["N", "L", "H"] = Field(description="可用性影響")

    # Temporal Metrics (Optional)
    exploit_code_maturity: Literal["X", "H", "F", "P", "U"] = Field(
        default="X", description="漏洞利用代碼成熟度"
    )
    remediation_level: Literal["X", "U", "W", "T", "O"] = Field(default="X", description="修復級別")
    report_confidence: Literal["X", "C", "R", "U"] = Field(default="X", description="報告置信度")

    # Environmental Metrics (Optional)
    confidentiality_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="機密性要求"
    )
    integrity_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="完整性要求"
    )
    availability_requirement: Literal["X", "L", "M", "H"] = Field(
        default="X", description="可用性要求"
    )

    # Calculated Scores
    base_score: float | None = Field(default=None, ge=0.0, le=10.0, description="基本分數")
    temporal_score: float | None = Field(default=None, ge=0.0, le=10.0, description="時間分數")
    environmental_score: float | None = Field(default=None, ge=0.0, le=10.0, description="環境分數")
    vector_string: str | None = Field(default=None, description="CVSS 向量字符串")

    def calculate_base_score(self) -> float:
        """計算 CVSS v3.1 基本分數"""
        # 攻擊向量權重
        av_weights = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
        # 攻擊複雜度權重
        ac_weights = {"L": 0.77, "H": 0.44}
        # 權限要求權重（根據範圍調整）
        if self.scope == "C":
            pr_weights = {"N": 0.85, "L": 0.68, "H": 0.50}
        else:
            pr_weights = {"N": 0.85, "L": 0.62, "H": 0.27}
        # 用戶交互權重
        ui_weights = {"N": 0.85, "R": 0.62}
        # CIA 影響權重
        cia_weights = {"N": 0.0, "L": 0.22, "H": 0.56}

        # 計算影響分數
        impact = 1 - (1 - cia_weights[self.confidentiality]) * (1 - cia_weights[self.integrity]) * (
            1 - cia_weights[self.availability]
        )

        # 調整影響分數
        if self.scope == "C":
            impact_adjusted = 7.52 * (impact - 0.029) - 3.25 * pow(impact - 0.02, 15)
        else:
            impact_adjusted = 6.42 * impact

        # 計算可利用性分數
        exploitability = (
            8.22
            * av_weights[self.attack_vector]
            * ac_weights[self.attack_complexity]
            * pr_weights[self.privileges_required]
            * ui_weights[self.user_interaction]
        )

        # 計算基本分數
        if impact_adjusted <= 0:
            return 0.0
        elif self.scope == "U":
            base = impact_adjusted + exploitability
        else:
            base = 1.08 * (impact_adjusted + exploitability)

        return min(10.0, round(base, 1))


class Authentication(BaseModel):
    method: str = "none"
    credentials: dict[str, str] | None = None


class RateLimit(BaseModel):
    requests_per_second: int = 25
    burst: int = 50

    @field_validator("requests_per_second", "burst")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limit must be non-negative")
        return v


class ScanScope(BaseModel):
    exclusions: list[str] = Field(default_factory=list)
    include_subdomains: bool = True
    allowed_hosts: list[str] = Field(default_factory=list)


class ScanStartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    scope: ScanScope = Field(default_factory=ScanScope)
    authentication: Authentication = Field(default_factory=Authentication)
    strategy: str = "deep"
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    custom_headers: dict[str, str] = Field(default_factory=dict)
    x_forwarded_for: str | None = None

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
    def validate_targets(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        """驗證目標列表"""
        if not v:
            raise ValueError("At least one target required")
        if len(v) > 100:
            raise ValueError("Too many targets (maximum 100)")
        return v

    @field_validator("strategy")
    def validate_strategy(cls, v: str) -> str:
        """驗證掃描策略"""
        allowed = {"quick", "normal", "deep", "full", "custom"}
        if v not in allowed:
            raise ValueError(f"Invalid strategy: {v}. Must be one of {allowed}")
        return v


class Asset(BaseModel):
    asset_id: str
    type: str
    value: str
    parameters: list[str] | None = None
    has_form: bool = False


class Summary(BaseModel):
    urls_found: int = 0
    forms_found: int = 0
    apis_found: int = 0
    scan_duration_seconds: int = 0


class Fingerprints(BaseModel):
    web_server: dict[str, str] | None = None
    framework: dict[str, str] | None = None
    language: dict[str, str] | None = None
    waf_detected: bool = False
    waf_vendor: str | None = None


class ScanCompletedPayload(BaseModel):
    scan_id: str
    status: str
    summary: Summary
    assets: list[Asset] = []
    fingerprints: Fingerprints | None = None
    error_info: str | None = None


class FunctionTaskTarget(BaseModel):
    # Accept arbitrary URL-like values; runtime code will cast to str as needed
    url: Any
    parameter: str | None = None
    method: str = "GET"
    parameter_location: str = "query"
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    form_data: dict[str, Any] = Field(default_factory=dict)
    json_data: dict[str, Any] | None = None
    body: str | None = None


class FunctionTaskContext(BaseModel):
    db_type_hint: str | None = None
    waf_detected: bool = False
    related_findings: list[str] | None = None


class FunctionTaskTestConfig(BaseModel):
    payloads: list[str] = Field(default_factory=lambda: ["basic"])
    custom_payloads: list[str] = Field(default_factory=list)
    blind_xss: bool = False
    dom_testing: bool = False
    timeout: float | None = None


class FunctionTaskPayload(BaseModel):
    task_id: str
    scan_id: str
    priority: int = 5
    target: FunctionTaskTarget
    context: FunctionTaskContext = Field(default_factory=FunctionTaskContext)
    strategy: str = "full"
    custom_payloads: list[str] | None = None
    test_config: FunctionTaskTestConfig = Field(default_factory=FunctionTaskTestConfig)

    @field_validator("task_id")
    def validate_task_id(cls, v: str) -> str:
        """驗證任務 ID 格式"""
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v

    @field_validator("scan_id")
    def validate_scan_id(cls, v: str) -> str:
        """驗證掃描 ID 格式"""
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v

    @field_validator("priority")
    def validate_priority(cls, v: int) -> int:
        """驗證優先級範圍"""
        if not 1 <= v <= 10:
            raise ValueError("priority must be between 1 and 10")
        return v


class FeedbackEventPayload(BaseModel):
    task_id: str
    scan_id: str
    event_type: str
    details: dict[str, Any] = {}
    form_url: HttpUrl | None = None


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
    """目標資訊 - FindingTarget 的別名，保持向後相容"""

    url: Any  # Accept arbitrary URL-like values
    parameter: str | None = None
    method: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    body: str | None = None


# 保持向後相容的別名
FindingTarget = Target


class FindingEvidence(BaseModel):
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
    target: Target  # 使用 Target 而不是 FindingTarget
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("finding_id")
    def validate_finding_id(cls, v: str) -> str:
        """驗證漏洞 ID 格式"""
        if not v.startswith("finding_"):
            raise ValueError("finding_id must start with 'finding_'")
        return v

    @field_validator("task_id")
    def validate_task_id(cls, v: str) -> str:
        """驗證任務 ID 格式"""
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v

    @field_validator("scan_id")
    def validate_scan_id(cls, v: str) -> str:
        """驗證掃描 ID 格式"""
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """驗證狀態"""
        allowed = {"confirmed", "potential", "false_positive", "needs_review"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class TaskUpdatePayload(BaseModel):
    task_id: str
    scan_id: str
    status: str
    worker_id: str
    details: dict[str, Any] | None = None


class HeartbeatPayload(BaseModel):
    module: ModuleName
    worker_id: str
    capacity: int


class ConfigUpdatePayload(BaseModel):
    update_id: str
    config_items: dict[str, Any] = {}


# ==================== 通用功能模組基礎類 ====================


class FunctionTelemetry(BaseModel):
    """功能模組遙測數據基礎類 - 所有功能模組的遙測數據應繼承此類"""

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


class ExecutionError(BaseModel):
    """執行錯誤統一格式 - 用於記錄檢測過程中的錯誤"""

    error_id: str
    error_type: str
    message: str
    payload: str | None = None
    vector: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 1


class FunctionExecutionResult(BaseModel):
    """功能模組執行結果統一格式 - 所有功能模組應返回此格式"""

    findings: list[FindingPayload]
    telemetry: dict[str, Any]  # 使用 dict 以支持各模組的自定義 Telemetry
    errors: list[ExecutionError] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== OAST 相關數據合約 ====================


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
    def validate_event_type(cls, v: str) -> str:
        """驗證事件類型"""
        allowed = {"http", "dns", "smtp", "ftp", "ldap", "other"}
        if v not in allowed:
            raise ValueError(f"Invalid event_type: {v}. Must be one of {allowed}")
        return v


class OastProbe(BaseModel):
    """OAST 探針數據合約"""

    probe_id: str
    token: str
    callback_url: HttpUrl
    task_id: str
    scan_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    status: str = "active"  # "active", "triggered", "expired"

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """驗證探針狀態"""
        allowed = {"active", "triggered", "expired", "cancelled"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ==================== 模組狀態與任務狀態 ====================


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
    def validate_status(cls, v: str) -> str:
        """驗證模組狀態"""
        allowed = {"running", "stopped", "error", "initializing", "degraded"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ==================== ThreatIntel Payloads ====================


class ThreatIntelLookupPayload(BaseModel):
    """威脅情報查詢 Payload"""

    task_id: str
    scan_id: str
    indicator: str
    indicator_type: IOCType
    sources: list[IntelSource] | None = None
    enrich: bool = True


class ThreatIntelResultPayload(BaseModel):
    """威脅情報查詢結果 Payload"""

    task_id: str
    scan_id: str
    indicator: str
    indicator_type: IOCType
    threat_level: ThreatLevel
    sources: dict[str, Any] = Field(default_factory=dict)
    mitre_techniques: list[str] = Field(default_factory=list)
    enrichment_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== AuthZ Payloads ====================


class AuthZCheckPayload(BaseModel):
    """權限檢查 Payload"""

    task_id: str
    scan_id: str
    user_id: str
    resource: str
    permission: str
    context: dict[str, Any] = Field(default_factory=dict)


class AuthZAnalysisPayload(BaseModel):
    """權限分析 Payload"""

    task_id: str
    scan_id: str
    analysis_type: str  # "coverage", "conflicts", "over_privileged"
    target: str | None = None  # user_id or role_id


class AuthZResultPayload(BaseModel):
    """權限分析結果 Payload"""

    task_id: str
    scan_id: str
    decision: str  # "allow", "deny", "conditional"
    analysis: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== Remediation Payloads ====================


class RemediationGeneratePayload(BaseModel):
    """修復方案生成 Payload"""

    task_id: str
    scan_id: str
    finding_id: str
    vulnerability_type: VulnerabilityType
    remediation_type: RemediationType
    context: dict[str, Any] = Field(default_factory=dict)
    auto_apply: bool = False


class RemediationResultPayload(BaseModel):
    """修復方案結果 Payload"""

    task_id: str
    scan_id: str
    finding_id: str
    remediation_type: RemediationType
    status: RemediationStatus
    patch_content: str | None = None
    instructions: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== PostEx Payloads ====================


class PostExTestPayload(BaseModel):
    """後滲透測試 Payload (僅用於授權測試環境)"""

    task_id: str
    scan_id: str
    test_type: PostExTestType
    target: str  # 目標系統/網絡
    safe_mode: bool = True
    authorization_token: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class PostExResultPayload(BaseModel):
    """後滲透測試結果 Payload"""

    task_id: str
    scan_id: str
    test_type: PostExTestType
    findings: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: ThreatLevel
    safe_mode: bool
    authorization_verified: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    dangerous_functions: list[str] = Field(default_factory=list)  # eval, Function, setTimeout等
    external_resources: list[str] = Field(default_factory=list)  # 外部 URL
    data_leaks: list[dict[str, str]] = Field(default_factory=list)  # 數據洩漏信息

    # 通用欄位 (保持兼容)
    findings: list[str] = Field(default_factory=list)  # e.g., ["uses_eval", "dom_manipulation"]
    apis_called: list[str] = Field(default_factory=list)  # 發現的 API 端點
    ajax_endpoints: list[str] = Field(default_factory=list)  # AJAX 呼叫端點
    suspicious_patterns: list[str] = Field(default_factory=list)

    # 評分欄位
    risk_score: float = Field(ge=0.0, le=10.0, default=0.0)  # 0.0 - 10.0
    security_score: int = Field(ge=0, le=100, default=100)  # 0-100 分

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== BizLogic Payloads ====================


class BizLogicTestPayload(BaseModel):
    """業務邏輯測試 Payload"""

    task_id: str
    scan_id: str
    test_type: str  # price_manipulation, workflow_bypass, race_condition
    target_urls: dict[str, str]  # 目標 URL 字典 {"cart_api": "...", "checkout_api": "..."}
    test_config: dict[str, Any] = Field(default_factory=dict)
    product_id: str | None = None
    workflow_steps: list[dict[str, str]] = Field(default_factory=list)


class BizLogicResultPayload(BaseModel):
    """業務邏輯測試結果 Payload"""

    task_id: str
    scan_id: str
    test_type: str
    status: str  # completed, failed, error
    findings: list[dict[str, Any]] = Field(default_factory=list)  # FindingPayload dicts
    statistics: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 資產與漏洞生命週期管理 Schemas ====================


class AssetLifecyclePayload(BaseModel):
    """資產生命週期管理 Payload"""

    asset_id: str
    asset_type: AssetType
    value: str  # URL, IP, repository URL, etc.
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


# ==================== 風險評估增強 Schemas ====================


class RiskAssessmentContext(BaseModel):
    """風險評估上下文 - 用於增強風險評估"""

    environment: Environment
    business_criticality: BusinessCriticality
    data_sensitivity: DataSensitivity | None = None
    asset_exposure: AssetExposure | None = None
    compliance_tags: list[ComplianceFramework] = Field(default_factory=list)
    asset_value: float | None = None  # 資產價值（金額）
    user_base: int | None = None  # 使用者基數
    sla_hours: int | None = None  # SLA 要求（小時）


class RiskAssessmentResult(BaseModel):
    """風險評估結果"""

    finding_id: str
    technical_risk_score: float  # 技術風險分數 (0-10)
    business_risk_score: float  # 業務風險分數 (0-100)
    risk_level: RiskLevel
    priority_score: float  # 優先級分數 (0-100)
    context_multiplier: float  # 上下文乘數
    business_impact: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    estimated_effort: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskTrendAnalysis(BaseModel):
    """風險趨勢分析"""

    period_start: datetime
    period_end: datetime
    total_vulnerabilities: int
    risk_distribution: dict[str, int]  # {risk_level: count}
    average_risk_score: float
    trend: str  # "increasing", "stable", "decreasing"
    improvement_percentage: float | None = None
    top_risks: list[dict[str, Any]] = Field(default_factory=list)


# ==================== 攻擊路徑分析 Schemas ====================


class AttackPathNode(BaseModel):
    """攻擊路徑節點"""

    node_id: str
    node_type: AttackPathNodeType
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class AttackPathEdge(BaseModel):
    """攻擊路徑邊"""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: AttackPathEdgeType
    risk_score: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class AttackPathPayload(BaseModel):
    """攻擊路徑 Payload"""

    path_id: str
    scan_id: str
    source_node: AttackPathNode
    target_node: AttackPathNode
    nodes: list[AttackPathNode]
    edges: list[AttackPathEdge]
    total_risk_score: float
    path_length: int
    description: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AttackPathRecommendation(BaseModel):
    """攻擊路徑推薦"""

    path_id: str
    risk_level: RiskLevel
    priority_score: float
    executive_summary: str
    technical_explanation: str
    business_impact: str
    remediation_steps: list[str]
    quick_wins: list[str] = Field(default_factory=list)
    affected_assets: list[str] = Field(default_factory=list)
    estimated_effort: str
    estimated_risk_reduction: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 漏洞關聯分析 Schemas ====================


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


# ==================== API 安全測試 Schemas ====================


class APISchemaPayload(BaseModel):
    """API Schema 解析 Payload"""

    schema_id: str
    scan_id: str
    schema_type: str  # "openapi", "graphql", "grpc"
    schema_content: dict[str, Any] | str
    base_url: str
    authentication: Authentication = Field(default_factory=Authentication)


class APITestCase(BaseModel):
    """API 測試案例"""

    test_id: str
    test_type: str  # "bola", "bfla", "data_leak", "mass_assignment"
    endpoint: str
    method: str
    test_vectors: list[dict[str, Any]] = Field(default_factory=list)
    expected_behavior: str | None = None


class APISecurityTestPayload(BaseModel):
    """API 安全測試 Payload"""

    task_id: str
    scan_id: str
    api_type: str  # "rest", "graphql", "grpc"
    api_schema: APISchemaPayload | None = None
    test_cases: list[APITestCase] = Field(default_factory=list)
    authentication: Authentication = Field(default_factory=Authentication)


# ==================== AI 驅動漏洞驗證 Schemas ====================


class AIVerificationRequest(BaseModel):
    """AI 驅動漏洞驗證請求"""

    verification_id: str
    finding_id: str
    scan_id: str
    vulnerability_type: VulnerabilityType
    target: FindingTarget
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


# ==================== SIEM 整合 Schemas ====================


class SIEMEventPayload(BaseModel):
    """SIEM 事件 Payload"""

    event_id: str
    event_type: str  # "vulnerability_detected", "scan_completed", "high_risk_finding"
    severity: Severity
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


# ==================== EASM 資產探索 Schemas ====================


class EASMDiscoveryPayload(BaseModel):
    """EASM 資產探索 Payload"""

    discovery_id: str
    scan_id: str
    discovery_type: str  # "subdomain", "port_scan", "cloud_storage", "certificate"
    targets: list[str]  # 起始目標
    scope: ScanScope = Field(default_factory=ScanScope)
    max_depth: int = 3
    passive_only: bool = False


class DiscoveredAsset(BaseModel):
    """探索到的資產"""

    asset_id: str
    asset_type: AssetType
    value: str
    discovery_method: str  # "subdomain_enum", "port_scan", "certificate_transparency"
    confidence: Confidence
    metadata: dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EASMDiscoveryResult(BaseModel):
    """EASM 探索結果"""

    discovery_id: str
    scan_id: str
    status: str  # "completed", "in_progress", "failed"
    discovered_assets: list[DiscoveredAsset] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 強化學習與自動決策 Schemas ====================


class AttackStep(BaseModel):
    """攻擊步驟 - AST 節點的可執行表示

    符合標準：
    - MITRE ATT&CK: 可選擇性映射到 ATT&CK 技術和戰術
    """

    step_id: str
    action: str  # 動作描述，如 "SSRF Attack", "Validate Response"
    tool_type: str  # 工具類型，如 "function_ssrf_go", "function_sqli"
    target: dict[str, Any] = Field(default_factory=dict)  # 目標參數
    parameters: dict[str, Any] = Field(default_factory=dict)  # 執行參數
    expected_result: str | None = None  # 預期結果描述
    timeout_seconds: float = 30.0
    retry_count: int = 0
    # MITRE ATT&CK 映射（可選）
    mitre_technique_id: str | None = Field(
        default=None,
        description="MITRE ATT&CK 技術 ID，例如: T1190 (Exploit Public-Facing Application)",
        pattern=r"^T\d{4}(\.\d{3})?$",
    )
    mitre_tactic: str | None = Field(
        default=None,
        description="MITRE ATT&CK 戰術，例如: Initial Access, Execution, Persistence",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPlan(BaseModel):
    """攻擊計畫 - 完整的攻擊流程定義

    符合標準：
    - MITRE ATT&CK: 支持映射到攻擊技術和戰術
    - CAPEC: Common Attack Pattern Enumeration and Classification (可選)
    """

    plan_id: str
    scan_id: str
    attack_type: VulnerabilityType  # 攻擊類型，如 SQLI, XSS, SSRF
    steps: list[AttackStep]  # 攻擊步驟序列
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict
    )  # step_id -> [dependency_step_ids]
    context: dict[str, Any] = Field(default_factory=dict)  # 場景上下文
    target_info: dict[str, Any] = Field(default_factory=dict)  # 目標系統資訊
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "ai_planner"  # 創建者（AI 或人工）
    # MITRE ATT&CK 映射（可選）
    mitre_techniques: list[str] = Field(
        default_factory=list,
        description="關聯的 MITRE ATT&CK 技術 ID 列表，例如: ['T1190', 'T1059.001']",
    )
    mitre_tactics: list[str] = Field(
        default_factory=list,
        description="關聯的 MITRE ATT&CK 戰術列表，例如: ['Initial Access', 'Execution']",
    )
    # CAPEC 映射（可選）
    capec_id: str | None = Field(
        default=None,
        description="CAPEC ID (格式: CAPEC-XXX)，參考 https://capec.mitre.org/",
        pattern=r"^CAPEC-\d+$",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        if not v.startswith("plan_"):
            raise ValueError("plan_id must start with 'plan_'")
        return v


class TraceRecord(BaseModel):
    """執行追蹤記錄 - 單個步驟的執行詳情"""

    trace_id: str
    plan_id: str
    step_id: str
    session_id: str  # 會話 ID，用於關聯同一次攻擊鏈
    tool_name: str  # 實際使用的工具模組名稱
    input_data: dict[str, Any] = Field(default_factory=dict)  # 輸入參數
    output_data: dict[str, Any] = Field(default_factory=dict)  # 輸出結果
    status: str  # "success", "failed", "timeout", "skipped"
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    environment_response: dict[str, Any] = Field(default_factory=dict)  # 靶場環境回應
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"success", "failed", "timeout", "skipped", "error"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class PlanExecutionMetrics(BaseModel):
    """計畫執行指標 - AST vs Trace 對比結果"""

    plan_id: str
    session_id: str
    expected_steps: int  # AST 中預期的步驟數
    executed_steps: int  # 實際執行的步驟數
    completed_steps: int  # 成功完成的步驟數
    failed_steps: int  # 失敗的步驟數
    skipped_steps: int  # 跳過的步驟數
    extra_actions: int  # AST 中未規劃的額外動作數
    completion_rate: float  # 完成率 (0.0 - 1.0)
    success_rate: float  # 成功率 (0.0 - 1.0)
    sequence_accuracy: float  # 順序準確度 (0.0 - 1.0)
    goal_achieved: bool  # 是否達成攻擊目標
    reward_score: float  # 獎勵分數（用於強化學習）
    total_execution_time: float  # 總執行時間（秒）
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanExecutionResult(BaseModel):
    """計畫執行結果 - 完整的執行報告"""

    result_id: str
    plan_id: str
    session_id: str
    plan: AttackPlan  # 原始攻擊計畫
    trace: list[TraceRecord]  # 執行追蹤記錄
    metrics: PlanExecutionMetrics  # 執行指標
    findings: list[FindingPayload] = Field(default_factory=list)  # 發現的漏洞
    anomalies: list[str] = Field(default_factory=list)  # 異常事件
    recommendations: list[str] = Field(default_factory=list)  # 改進建議
    status: str  # "completed", "partial", "failed", "aborted"
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"completed", "partial", "failed", "aborted"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class ExperienceSample(BaseModel):
    """經驗樣本 - 用於機器學習訓練"""

    sample_id: str
    plan_id: str
    session_id: str
    context: dict[str, Any]  # 場景上下文（目標系統、漏洞類型等）
    plan: AttackPlan  # 攻擊計畫
    trace: list[TraceRecord]  # 執行軌跡
    metrics: PlanExecutionMetrics  # 執行指標
    result: PlanExecutionResult  # 執行結果
    label: str  # "success", "failure", "partial_success"
    quality_score: float  # 樣本質量分數 (0.0 - 1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    annotations: dict[str, Any] = Field(default_factory=dict)  # 人工標註
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    def validate_label(cls, v: str) -> str:
        allowed = {"success", "failure", "partial_success", "invalid"}
        if v not in allowed:
            raise ValueError(f"Invalid label: {v}. Must be one of {allowed}")
        return v

    @field_validator("quality_score")
    def validate_quality_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
        return v


class SessionState(BaseModel):
    """會話狀態 - 多步驟攻擊鏈的會話管理"""

    session_id: str
    plan_id: str
    scan_id: str
    status: str  # "active", "paused", "completed", "failed", "aborted"
    current_step_index: int = 0
    completed_steps: list[str] = Field(default_factory=list)  # step_ids
    pending_steps: list[str] = Field(default_factory=list)  # step_ids
    context: dict[str, Any] = Field(default_factory=dict)  # 動態上下文
    variables: dict[str, Any] = Field(default_factory=dict)  # 會話變數（用於步驟間傳遞數據）
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    def validate_session_id(cls, v: str) -> str:
        if not v.startswith("session_"):
            raise ValueError("session_id must start with 'session_'")
        return v

    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"active", "paused", "completed", "failed", "aborted"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    config_id: str
    model_type: str  # "supervised", "reinforcement", "hybrid"
    training_mode: str  # "batch", "online", "incremental"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 3
    reward_function: str = "completion_rate"  # 獎勵函數類型
    discount_factor: float = 0.99  # 折扣因子（用於強化學習）
    exploration_rate: float = 0.1  # 探索率
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("learning_rate", "validation_split", "discount_factor", "exploration_rate")
    def validate_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Rate must be between 0.0 and 1.0")
        return v


class ModelTrainingResult(BaseModel):
    """模型訓練結果"""

    training_id: str
    config: ModelTrainingConfig
    model_version: str
    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    average_reward: float | None = None  # 強化學習平均獎勵
    training_duration_seconds: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardScenario(BaseModel):
    """標準靶場場景 - 用於訓練和測試"""

    scenario_id: str
    name: str
    description: str
    vulnerability_type: VulnerabilityType
    difficulty_level: str  # "easy", "medium", "hard", "expert"
    target_config: dict[str, Any]  # 靶場配置
    expected_plan: AttackPlan  # 預期的最佳攻擊計畫
    success_criteria: dict[str, Any]  # 成功標準
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("difficulty_level")
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard", "expert"}
        if v not in allowed:
            raise ValueError(f"Invalid difficulty: {v}. Must be one of {allowed}")
        return v


class ScenarioTestResult(BaseModel):
    """場景測試結果 - 模型在標準場景上的表現"""

    test_id: str
    scenario_id: str
    model_version: str
    generated_plan: AttackPlan
    execution_result: PlanExecutionResult
    score: float  # 綜合評分 (0.0 - 100.0)
    comparison: dict[str, Any]  # 與預期計畫的對比
    passed: bool
    tested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== CVSS 漏洞評分系統 (符合 CVSS v3.1 標準) ====================


class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 評分指標

    符合標準: CVSS v3.1 Specification (https://www.first.org/cvss/v3.1/specification-document)
    """

    # Base Metrics (基礎指標)
    attack_vector: str = Field(
        default="N",
        description="攻擊向量: N(Network), A(Adjacent), L(Local), P(Physical)",
        pattern=r"^[NALP]$",
    )
    attack_complexity: str = Field(
        default="L", description="攻擊複雜度: L(Low), H(High)", pattern=r"^[LH]$"
    )
    privileges_required: str = Field(
        default="N",
        description="所需權限: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    user_interaction: str = Field(
        default="N", description="用戶交互: N(None), R(Required)", pattern=r"^[NR]$"
    )
    scope: str = Field(
        default="U", description="影響範圍: U(Unchanged), C(Changed)", pattern=r"^[UC]$"
    )
    confidentiality_impact: str = Field(
        default="N",
        description="機密性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    integrity_impact: str = Field(
        default="N",
        description="完整性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )
    availability_impact: str = Field(
        default="N",
        description="可用性影響: N(None), L(Low), H(High)",
        pattern=r"^[NLH]$",
    )

    # Temporal Metrics (時間指標 - 可選)
    exploit_code_maturity: str | None = Field(
        default=None,
        description="漏洞利用程式碼成熟度: X(Not Defined), U(Unproven), P(Proof-of-Concept), F(Functional), H(High)",
        pattern=r"^[XUPFH]$",
    )
    remediation_level: str | None = Field(
        default=None,
        description="修復級別: X(Not Defined), O(Official Fix), T(Temporary Fix), W(Workaround), U(Unavailable)",
        pattern=r"^[XOTWU]$",
    )
    report_confidence: str | None = Field(
        default=None,
        description="報告可信度: X(Not Defined), U(Unknown), R(Reasonable), C(Confirmed)",
        pattern=r"^[XURC]$",
    )

    def calculate_base_score(self) -> float:
        """計算 CVSS v3.1 基礎分數

        Returns:
            基礎分數 (0.0 - 10.0)
        """
        # 這裡是簡化版的計算邏輯，完整實現應該嚴格遵循 CVSS v3.1 規範
        # 參考: https://www.first.org/cvss/v3.1/specification-document

        # Impact Sub-Score (ISC)
        impact_values = {"N": 0.0, "L": 0.22, "H": 0.56}
        isc_base = 1 - (1 - impact_values[self.confidentiality_impact]) * (
            1 - impact_values[self.integrity_impact]
        ) * (1 - impact_values[self.availability_impact])

        if self.scope == "U":
            impact = 6.42 * isc_base
        else:  # Changed
            impact = 7.52 * (isc_base - 0.029) - 3.25 * ((isc_base - 0.02) ** 15)

        # Exploitability Sub-Score (ESS)
        av_values = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
        ac_values = {"L": 0.77, "H": 0.44}
        pr_values_unchanged = {"N": 0.85, "L": 0.62, "H": 0.27}
        pr_values_changed = {"N": 0.85, "L": 0.68, "H": 0.5}
        ui_values = {"N": 0.85, "R": 0.62}

        pr_values = pr_values_unchanged if self.scope == "U" else pr_values_changed

        exploitability = (
            8.22
            * av_values[self.attack_vector]
            * ac_values[self.attack_complexity]
            * pr_values[self.privileges_required]
            * ui_values[self.user_interaction]
        )

        # Base Score
        if impact <= 0:
            return 0.0
        elif self.scope == "U":
            base_score = min(impact + exploitability, 10.0)
        else:
            base_score = min(1.08 * (impact + exploitability), 10.0)

        # Round up to one decimal place
        return round(base_score * 10) / 10

    def to_vector_string(self) -> str:
        """生成 CVSS v3.1 向量字串

        Returns:
            CVSS 向量字串，例如: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        """
        vector = (
            f"CVSS:3.1/AV:{self.attack_vector}/AC:{self.attack_complexity}/"
            f"PR:{self.privileges_required}/UI:{self.user_interaction}/"
            f"S:{self.scope}/C:{self.confidentiality_impact}/"
            f"I:{self.integrity_impact}/A:{self.availability_impact}"
        )

        # 添加時間指標（如果定義）
        if self.exploit_code_maturity and self.exploit_code_maturity != "X":
            vector += f"/E:{self.exploit_code_maturity}"
        if self.remediation_level and self.remediation_level != "X":
            vector += f"/RL:{self.remediation_level}"
        if self.report_confidence and self.report_confidence != "X":
            vector += f"/RC:{self.report_confidence}"

        return vector


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


# ==================== SARIF 格式支持 (Static Analysis Results Interchange Format) ====================


class SARIFLocation(BaseModel):
    """SARIF 位置資訊

    符合標準: SARIF v2.1.0 (https://docs.oasis-open.org/sarif/sarif/v2.1.0/)
    """

    uri: str  # 檔案 URI
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    snippet: str | None = None  # 代碼片段


class SARIFResult(BaseModel):
    """SARIF 結果項

    符合標準: SARIF v2.1.0
    """

    rule_id: str  # 規則 ID (可以是 CWE ID 或自定義規則)
    level: str = Field(
        default="warning",
        description="嚴重性級別",
        pattern=r"^(none|note|warning|error)$",
    )
    message: str  # 訊息文本
    locations: list[SARIFLocation] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class SARIFReport(BaseModel):
    """SARIF 報告

    符合標準: SARIF v2.1.0
    完整規範: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    """

    version: str = "2.1.0"
    schema_uri: str = (
        "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
        "master/Schemata/sarif-schema-2.1.0.json"
    )
    runs: list[dict[str, Any]] = Field(default_factory=list)

    def add_run(
        self,
        tool_name: str,
        tool_version: str,
        results: list[SARIFResult],
    ) -> None:
        """添加一個掃描運行結果

        Args:
            tool_name: 工具名稱
            tool_version: 工具版本
            results: 結果列表
        """
        run = {
            "tool": {
                "driver": {
                    "name": tool_name,
                    "version": tool_version,
                    "informationUri": "https://github.com/kyle0527/AIVA",
                }
            },
            "results": [
                {
                    "ruleId": r.rule_id,
                    "level": r.level,
                    "message": {"text": r.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": loc.uri},
                                "region": {
                                    "startLine": loc.start_line,
                                    "startColumn": loc.start_column,
                                    "endLine": loc.end_line,
                                    "endColumn": loc.end_column,
                                    "snippet": ({"text": loc.snippet} if loc.snippet else None),
                                },
                            }
                        }
                        for loc in r.locations
                    ],
                    "properties": r.properties,
                }
                for r in results
            ],
        }
        self.runs.append(run)


# ==================== 增強版漏洞發現 (集成 CVSS、CVE、CWE、SARIF) ====================


class EnhancedVulnerability(BaseModel):
    """增強版漏洞資訊 - 集成多種業界標準

    集成標準:
    - CWE: Common Weakness Enumeration
    - CVE: Common Vulnerabilities and Exposures
    - CVSS: Common Vulnerability Scoring System
    - MITRE ATT&CK: 攻擊技術框架
    """

    name: VulnerabilityType
    severity: Severity
    confidence: Confidence
    description: str | None = None

    # CWE 參考
    cwe: CWEReference | None = None

    # CVE 參考（如果已知）
    cve: CVEReference | None = None

    # CVSS 評分
    cvss: CVSSv3Metrics | None = None

    # MITRE ATT&CK 映射
    mitre_techniques: list[str] = Field(
        default_factory=list,
        description="關聯的 MITRE ATT&CK 技術 ID",
    )

    # OWASP Top 10 分類（可選）
    owasp_category: str | None = Field(
        default=None,
        description="OWASP Top 10 分類，例如: A01:2021-Broken Access Control",
    )


class EnhancedFindingPayload(BaseModel):
    """增強版漏洞發現 Payload - 集成所有業界標準

    此 Schema 擴展了基礎 FindingPayload，添加了完整的標準支持
    """

    finding_id: str
    task_id: str
    scan_id: str
    status: str

    # 使用增強版漏洞資訊
    vulnerability: EnhancedVulnerability

    target: Target
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None

    # SARIF 格式支持
    sarif_result: SARIFResult | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("finding_id")
    def validate_finding_id(cls, v: str) -> str:
        if not v.startswith("finding_"):
            raise ValueError("finding_id must start with 'finding_'")
        return v

    def to_sarif_result(self) -> SARIFResult:
        """轉換為 SARIF 結果格式

        Returns:
            SARIF 結果項
        """
        if self.sarif_result:
            return self.sarif_result

        # 構建 SARIF 結果
        level_mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "warning",
            Severity.INFO: "note",
        }

        locations = []
        if self.target.url:
            locations.append(
                SARIFLocation(
                    uri=str(self.target.url),
                    snippet=self.evidence.payload if self.evidence else None,
                )
            )

        return SARIFResult(
            rule_id=(
                self.vulnerability.cwe.cwe_id
                if self.vulnerability.cwe
                else f"AIVA-{self.vulnerability.name.value}"
            ),
            level=level_mapping.get(self.vulnerability.severity, "warning"),
            message=self.vulnerability.description or f"{self.vulnerability.name.value} detected",
            locations=locations,
            properties={
                "finding_id": self.finding_id,
                "confidence": self.vulnerability.confidence.value,
                "cvss_score": (
                    self.vulnerability.cvss.calculate_base_score()
                    if self.vulnerability.cvss
                    else None
                ),
                "mitre_techniques": self.vulnerability.mitre_techniques,
            },
        )


# ==================== AI 訓練與學習合約 ====================


class AITrainingStartPayload(BaseModel):
    """AI 訓練啟動請求 - 用於啟動新的訓練會話"""

    training_id: str = Field(description="訓練會話 ID")
    training_type: str = Field(description="訓練類型: single|batch|continuous|scenario")
    scenario_id: str | None = Field(default=None, description="靶場場景 ID")
    target_vulnerability: str | None = Field(default=None, description="目標漏洞類型")
    config: ModelTrainingConfig = Field(description="訓練配置")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_id")
    def validate_training_id(cls, v: str) -> str:
        if not v.startswith("training_"):
            raise ValueError("training_id must start with 'training_'")
        return v


class AITrainingProgressPayload(BaseModel):
    """AI 訓練進度報告 - 定期報告訓練進度"""

    training_id: str
    episode_number: int
    total_episodes: int
    successful_episodes: int = 0
    failed_episodes: int = 0
    total_samples: int = 0
    high_quality_samples: int = 0
    avg_reward: float | None = None
    avg_quality: float | None = None
    best_reward: float | None = None
    model_metrics: dict[str, float] = Field(default_factory=dict)
    status: str = "running"
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AITrainingCompletedPayload(BaseModel):
    """AI 訓練完成報告 - 訓練會話完成時的最終報告"""

    training_id: str
    status: str
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    total_duration_seconds: float
    total_samples: int
    high_quality_samples: int
    medium_quality_samples: int
    low_quality_samples: int
    final_avg_reward: float | None = None
    final_avg_quality: float | None = None
    best_episode_reward: float | None = None
    model_checkpoint_path: str | None = None
    model_metrics: dict[str, float] = Field(default_factory=dict)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIExperienceCreatedEvent(BaseModel):
    """AI 經驗樣本創建事件 - 當新的經驗樣本被創建時發送"""

    experience_id: str
    training_id: str | None = None
    trace_id: str
    vulnerability_type: str
    quality_score: float = Field(ge=0.0, le=1.0)
    success: bool
    plan_summary: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AITraceCompletedEvent(BaseModel):
    """AI 執行追蹤完成事件 - 當執行追蹤完成時發送"""

    trace_id: str
    session_id: str | None = None
    training_id: str | None = None
    total_steps: int
    successful_steps: int
    failed_steps: int
    duration_seconds: float
    final_success: bool
    plan_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIModelUpdatedEvent(BaseModel):
    """AI 模型更新事件 - 當模型被訓練更新時發送"""

    model_id: str
    model_version: str
    training_id: str | None = None
    update_type: str  # checkpoint|deployment|fine_tune|architecture
    performance_metrics: dict[str, float] = Field(default_factory=dict)
    model_path: str | None = None
    checkpoint_path: str | None = None
    is_deployed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIModelDeployCommand(BaseModel):
    """AI 模型部署命令 - 用於部署訓練好的模型到生產環境"""

    model_id: str
    model_version: str
    checkpoint_path: str
    deployment_target: str = "production"  # production|staging|testing
    deployment_config: dict[str, Any] = Field(default_factory=dict)
    require_validation: bool = True
    min_performance_threshold: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGKnowledgeUpdatePayload(BaseModel):
    """RAG 知識庫更新請求 - 用於向 RAG 知識庫添加新知識"""

    knowledge_type: str  # vulnerability|payload|technique|scenario|experience|cve|mitre
    content: str
    source_id: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    related_cve: str | None = None
    related_cwe: str | None = None
    mitre_techniques: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryPayload(BaseModel):
    """RAG 查詢請求 - 用於從 RAG 知識庫檢索相關知識"""

    query_id: str
    query_text: str
    top_k: int = Field(default=5, ge=1, le=100)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    knowledge_types: list[str] | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponsePayload(BaseModel):
    """RAG 查詢響應 - RAG 知識庫查詢的結果"""

    query_id: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_results: int
    avg_similarity: float | None = None
    enhanced_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 模組間通訊統一包裝 ====================


class AIVARequest(BaseModel):
    """統一的請求包裝器 - 用於模組間的請求消息"""

    request_id: str
    source_module: ModuleName
    target_module: ModuleName
    request_type: str
    payload: dict[str, Any]
    trace_id: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIVAResponse(BaseModel):
    """統一的響應包裝器 - 用於模組間的響應消息"""

    request_id: str
    response_type: str
    success: bool
    payload: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIVAEvent(BaseModel):
    """統一的事件包裝器 - 用於模組間的事件通知"""

    event_id: str
    event_type: str
    source_module: ModuleName
    payload: dict[str, Any]
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AIVACommand(BaseModel):
    """統一的命令包裝器 - 用於模組間的命令消息"""

    command_id: str
    command_type: str
    source_module: ModuleName
    target_module: ModuleName
    payload: dict[str, Any]
    priority: int = Field(default=0, ge=0, le=10)
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# AI 核心系統相關 schemas (BioNeuronRAGAgent 和強化學習系統)
# ============================================================================


class AttackStep(BaseModel):
    """攻擊步驟 (整合 MITRE ATT&CK)"""

    step_id: str = Field(description="步驟唯一標識")
    name: str = Field(description="步驟名稱")
    description: str = Field(description="步驟描述")

    # MITRE ATT&CK 映射
    mitre_technique_id: str | None = Field(default=None, description="MITRE 技術ID (如 T1055)")
    mitre_tactic: str | None = Field(default=None, description="MITRE 戰術")
    mitre_subtechnique_id: str | None = Field(default=None, description="MITRE 子技術ID")

    # 執行參數
    target: str = Field(description="目標資產")
    parameters: dict[str, Any] = Field(default_factory=dict, description="執行參數")
    payload: str | None = Field(default=None, description="攻擊負載")

    # 依賴關係
    depends_on: list[str] = Field(default_factory=list, description="依賴的步驟ID")
    timeout: int = Field(default=30, ge=1, description="超時時間(秒)")

    # 執行狀態
    status: TestStatus = Field(
        default=TestStatus.PENDING, description="執行狀態"
    )
    start_time: datetime | None = Field(default=None, description="開始時間")
    end_time: datetime | None = Field(default=None, description="結束時間")

    # 結果信息
    success: bool | None = Field(default=None, description="是否成功")
    error_message: str | None = Field(default=None, description="錯誤信息")
    output: dict[str, Any] = Field(default_factory=dict, description="輸出結果")


class AttackPlan(BaseModel):
    """攻擊計劃 (整合 MITRE ATT&CK)"""

    plan_id: str = Field(description="計劃唯一標識")
    name: str = Field(description="計劃名稱")
    description: str = Field(description="計劃描述")

    # 目標信息
    target_url: str = Field(description="目標URL")
    target_type: str = Field(description="目標類型")

    # 計劃結構
    steps: list[AttackStep] = Field(description="攻擊步驟列表")
    parallel_groups: list[list[str]] = Field(default_factory=list, description="可並行執行的步驟組")

    # 計劃屬性
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    estimated_duration: int = Field(default=300, ge=1, description="預估執行時間(秒)")
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="風險級別"
    )

    # 執行狀態
    status: TestStatus = Field(
        default=TestStatus.DRAFT, description="計劃狀態"
    )

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 元數據
    tags: list[str] = Field(default_factory=list, description="標籤")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")


class TraceRecord(BaseModel):
    """執行追蹤記錄"""

    trace_id: str = Field(description="追蹤唯一標識")
    session_id: str = Field(description="會話ID")
    plan_id: str = Field(description="計劃ID")
    step_id: str | None = Field(default=None, description="步驟ID")

    # 執行信息
    action: str = Field(description="執行的動作")
    target: str = Field(description="目標")
    parameters: dict[str, Any] = Field(default_factory=dict, description="參數")

    # 時間信息
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = Field(default=None, ge=0, description="執行時長(毫秒)")

    # 結果信息
    success: bool = Field(description="是否成功")
    result: dict[str, Any] = Field(default_factory=dict, description="執行結果")
    error_message: str | None = Field(default=None, description="錯誤信息")

    # 上下文信息
    context: dict[str, Any] = Field(default_factory=dict, description="執行上下文")
    parent_trace_id: str | None = Field(default=None, description="父追蹤ID")

    # 性能指標
    cpu_usage: float | None = Field(default=None, ge=0.0, le=100.0, description="CPU使用率")
    memory_usage: float | None = Field(default=None, ge=0.0, description="內存使用量(MB)")


class PlanExecutionMetrics(BaseModel):
    """計劃執行指標"""

    plan_id: str = Field(description="計劃ID")
    session_id: str = Field(description="會話ID")

    # 執行統計
    total_steps: int = Field(ge=0, description="總步驟數")
    completed_steps: int = Field(ge=0, description="已完成步驟數")
    failed_steps: int = Field(ge=0, description="失敗步驟數")
    skipped_steps: int = Field(ge=0, description="跳過步驟數")

    # 時間指標
    start_time: datetime = Field(description="開始時間")
    end_time: datetime | None = Field(default=None, description="結束時間")
    total_duration_ms: int | None = Field(default=None, ge=0, description="總執行時長(毫秒)")

    # 成功率計算
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    completion_rate: float = Field(ge=0.0, le=1.0, description="完成率")

    # 性能指標
    avg_step_duration_ms: float | None = Field(default=None, ge=0.0, description="平均步驟時長")
    max_step_duration_ms: int | None = Field(default=None, ge=0, description="最長步驟時長")

    # 資源使用
    peak_cpu_usage: float | None = Field(default=None, ge=0.0, le=100.0, description="峰值CPU")
    peak_memory_usage: float | None = Field(default=None, ge=0.0, description="峰值內存(MB)")

    # 結果統計
    vulnerabilities_found: int = Field(default=0, ge=0, description="發現漏洞數")
    false_positives: int = Field(default=0, ge=0, description="誤報數")

    # 質量指標
    sequence_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, description="序列準確度")
    goal_achievement: float | None = Field(default=None, ge=0.0, le=1.0, description="目標達成度")


class ExperienceSample(BaseModel):
    """經驗樣本 (用於強化學習)"""

    sample_id: str = Field(description="樣本唯一標識")
    session_id: str = Field(description="會話ID")
    plan_id: str = Field(description="計劃ID")

    # 狀態信息
    state_before: dict[str, Any] = Field(description="執行前狀態")
    action_taken: dict[str, Any] = Field(description="採取的行動")
    state_after: dict[str, Any] = Field(description="執行後狀態")

    # 獎勵信息
    reward: float = Field(description="獎勵值")
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict, description="獎勵分解 (completion, success, sequence, goal)"
    )

    # 上下文信息
    context: dict[str, Any] = Field(default_factory=dict, description="環境上下文")
    target_info: dict[str, Any] = Field(default_factory=dict, description="目標信息")

    # 時間信息
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = Field(default=None, ge=0, description="執行時長")

    # 質量標記
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0, description="樣本質量分數")
    is_positive: bool = Field(description="是否為正樣本")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="樣本置信度")

    # 學習標籤
    learning_tags: list[str] = Field(default_factory=list, description="學習標籤")
    difficulty_level: int = Field(default=1, ge=1, le=5, description="難度級別")


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    config_id: str = Field(description="配置唯一標識")
    model_name: str = Field(description="模型名稱")
    model_version: str = Field(description="模型版本")

    # 訓練參數
    learning_rate: float = Field(default=0.001, gt=0.0, description="學習率")
    batch_size: int = Field(default=32, ge=1, description="批次大小")
    epochs: int = Field(default=100, ge=1, description="訓練輪數")

    # 網絡結構
    hidden_layers: list[int] = Field(
        default_factory=lambda: [512, 256, 128], description="隱藏層大小"
    )
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout率")
    activation_function: str = Field(default="relu", description="激活函數")

    # 強化學習參數
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0, description="折扣因子")
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="探索率")
    replay_buffer_size: int = Field(default=10000, ge=1, description="經驗回放緩衝區大小")

    # 訓練策略
    training_strategy: str = Field(default="dqn", description="訓練策略")
    optimizer: str = Field(default="adam", description="優化器")
    loss_function: str = Field(default="mse", description="損失函數")

    # 驗證設置
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0, description="驗證集比例")
    early_stopping_patience: int = Field(default=10, ge=1, description="早停耐心值")

    # 保存設置
    save_frequency: int = Field(default=10, ge=1, description="保存頻率(輪)")
    checkpoint_path: str = Field(description="檢查點路徑")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EnhancedVulnerability(BaseModel):
    """增強漏洞信息 (整合 AI 分析結果)"""

    vulnerability_id: str = Field(description="漏洞唯一標識")
    title: str = Field(description="漏洞標題")
    description: str = Field(description="漏洞描述")

    # 基本信息
    vulnerability_type: str = Field(description="漏洞類型")
    severity: Severity = Field(description="嚴重性")

    # 位置信息
    url: str = Field(description="漏洞URL")
    parameter: str | None = Field(default=None, description="參數名")
    location: str = Field(description="參數位置")

    # CVSS 評分
    cvss_metrics: CVSSv3Metrics | None = Field(default=None, description="CVSS v3.1 指標")

    # AI 分析結果
    ai_confidence: float = Field(ge=0.0, le=1.0, description="AI 置信度")
    ai_risk_assessment: dict[str, Any] = Field(default_factory=dict, description="AI 風險評估")
    exploitability_score: float = Field(ge=0.0, le=1.0, description="可利用性分數")

    # 攻擊路徑
    attack_vector: str = Field(description="攻擊向量")
    attack_complexity: str = Field(description="攻擊複雜度")
    prerequisites: list[str] = Field(default_factory=list, description="利用前提")

    # 影響分析
    business_impact: dict[str, Any] = Field(default_factory=dict, description="業務影響")
    technical_impact: dict[str, Any] = Field(default_factory=dict, description="技術影響")

    # 修復建議
    remediation_effort: str = Field(description="修復難度")
    remediation_priority: int = Field(ge=1, le=5, description="修復優先級")
    fix_recommendations: list[str] = Field(default_factory=list, description="修復建議")

    # 驗證信息
    poc_available: bool = Field(default=False, description="是否有概念驗證")
    verified: bool = Field(default=False, description="是否已驗證")
    false_positive_probability: float = Field(ge=0.0, le=1.0, description="誤報概率")

    # 時間信息
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_verified_at: datetime | None = Field(default=None, description="最後驗證時間")

    # 元數據
    tags: list[str] = Field(default_factory=list, description="標籤")
    references: list[str] = Field(default_factory=list, description="參考資料")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")


# SARIF Report Support
class SARIFLocation(BaseModel):
    """SARIF 位置信息"""

    uri: str = Field(description="資源URI")
    start_line: int | None = Field(default=None, ge=1, description="開始行號")
    start_column: int | None = Field(default=None, ge=1, description="開始列號")
    end_line: int | None = Field(default=None, ge=1, description="結束行號")
    end_column: int | None = Field(default=None, ge=1, description="結束列號")


class SARIFResult(BaseModel):
    """SARIF 結果"""

    rule_id: str = Field(description="規則ID")
    message: str = Field(description="消息")
    level: str = Field(description="級別")
    locations: list[SARIFLocation] = Field(description="位置列表")

    # 可選字段
    partial_fingerprints: dict[str, str] = Field(default_factory=dict, description="部分指紋")
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFRule(BaseModel):
    """SARIF 規則"""

    id: str = Field(description="規則ID")
    name: str = Field(description="規則名稱")
    short_description: str = Field(description="簡短描述")
    full_description: str | None = Field(default=None, description="完整描述")
    help_uri: str | None = Field(default=None, description="幫助URI")

    # 安全等級
    default_level: str = Field(
        default="warning", description="默認級別"
    )

    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFTool(BaseModel):
    """SARIF 工具信息"""

    name: str = Field(description="工具名稱")
    version: str = Field(description="版本")
    information_uri: str | None = Field(default=None, description="信息URI")

    rules: list[SARIFRule] = Field(default_factory=list, description="規則列表")


class SARIFRun(BaseModel):
    """SARIF 運行"""

    tool: SARIFTool = Field(description="工具信息")
    results: list[SARIFResult] = Field(description="結果列表")

    # 可選信息
    invocations: list[dict[str, Any]] = Field(default_factory=list, description="調用信息")
    artifacts: list[dict[str, Any]] = Field(default_factory=list, description="工件信息")
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFReport(BaseModel):
    """SARIF v2.1.0 報告"""

    version: str = Field(default="2.1.0", description="SARIF版本")
    json_schema: str = Field(
        default="https://json.schemastore.org/sarif-2.1.0.json", description="JSON Schema"
    )
    runs: list[SARIFRun] = Field(description="運行列表")

    # 元數據
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


# ==================== 新增：掃描發現模式 (原 scan/discovery_schemas.py) ====================


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
    business_criticality: str = Field(
        description="業務重要性"
    )  # "critical", "high", "medium", "low"
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


# ==================== 新增：功能測試模式 (原 function/test_schemas.py) ====================


class EnhancedFunctionTaskTarget(BaseModel):
    """增強功能測試目標"""

    url: HttpUrl = Field(description="目標URL")
    method: str = Field(default="GET", description="HTTP方法")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie")
    parameters: dict[str, str] = Field(default_factory=dict, description="參數")
    body: str | None = Field(default=None, description="請求體")
    auth_required: bool = Field(default=False, description="是否需要認證")


class ExploitPayload(BaseModel):
    """漏洞利用載荷"""

    payload_id: str = Field(description="載荷ID")
    payload_type: str = Field(description="載荷類型")  # "xss", "sqli", "command_injection"
    payload_content: str = Field(description="載荷內容")

    # 載荷屬性
    encoding: str = Field(default="none", description="編碼方式")
    obfuscation: bool = Field(default=False, description="是否混淆")
    bypass_technique: str | None = Field(default=None, description="繞過技術")

    # 適用條件
    target_technology: list[str] = Field(default_factory=list, description="目標技術")
    required_context: dict[str, Any] = Field(default_factory=dict, description="所需上下文")

    # 效果評估
    effectiveness_score: float = Field(ge=0.0, le=1.0, description="效果評分")
    detection_evasion: float = Field(ge=0.0, le=1.0, description="逃避檢測能力")

    # 使用統計
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    usage_count: int = Field(ge=0, description="使用次數")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class TestExecution(BaseModel):
    """測試執行記錄"""

    execution_id: str = Field(description="執行ID")
    test_case_id: str = Field(description="測試案例ID")
    target_url: str = Field(description="目標URL")

    # 執行配置
    timeout: int = Field(default=30, ge=1, description="超時時間(秒)")
    retry_attempts: int = Field(default=3, ge=0, description="重試次數")

    # 執行狀態
    status: TestStatus = Field(description="執行狀態")
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = Field(default=None, description="結束時間")
    duration: float | None = Field(default=None, ge=0.0, description="執行時間(秒)")

    # 執行結果
    success: bool = Field(description="是否成功")
    vulnerability_found: bool = Field(description="是否發現漏洞")
    confidence_level: Confidence = Field(description="結果置信度")

    # 詳細信息
    request_data: dict[str, Any] = Field(default_factory=dict, description="請求數據")
    response_data: dict[str, Any] = Field(default_factory=dict, description="響應數據")
    evidence: list[str] = Field(default_factory=list, description="證據列表")
    error_message: str | None = Field(default=None, description="錯誤消息")

    # 資源使用
    cpu_usage: float | None = Field(default=None, description="CPU使用率")
    memory_usage: int | None = Field(default=None, description="內存使用(MB)")
    network_traffic: int | None = Field(default=None, description="網絡流量(bytes)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ExploitResult(BaseModel):
    """漏洞利用結果"""

    result_id: str = Field(description="結果ID")
    exploit_id: str = Field(description="利用ID")
    target_id: str = Field(description="目標ID")

    # 利用狀態
    success: bool = Field(description="利用是否成功")
    severity: Severity = Field(description="嚴重程度")
    impact_level: str = Field(description="影響級別")  # "critical", "high", "medium", "low"

    # 利用細節
    exploit_technique: str = Field(description="利用技術")
    payload_used: str = Field(description="使用的載荷")
    execution_time: float = Field(ge=0.0, description="執行時間(秒)")

    # 獲得的訪問
    access_gained: dict[str, Any] = Field(default_factory=dict, description="獲得的訪問權限")
    data_extracted: list[str] = Field(default_factory=list, description="提取的數據")
    system_impact: str | None = Field(default=None, description="系統影響")

    # 檢測規避
    detection_bypassed: bool = Field(description="是否繞過檢測")
    artifacts_left: list[str] = Field(default_factory=list, description="留下的痕跡")

    # 修復驗證
    remediation_verified: bool = Field(default=False, description="修復是否已驗證")
    retest_required: bool = Field(default=True, description="是否需要重測")

    # 時間戳
    executed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 新增：整合服務模式 (原 integration/service_schemas.py) ====================


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
    status: str = Field(
        default="new", description="處理狀態"
    )  # "new", "investigating", "resolved", "false_positive"
    assigned_to: str | None = Field(default=None, description="分配給")

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


# ==================== 新增：核心業務模式 (原 core/business_schemas.py) ====================


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str = Field(description="風險因子名稱")
    weight: float = Field(ge=0.0, le=1.0, description="權重")
    value: float = Field(ge=0.0, le=10.0, description="因子值")
    description: str | None = Field(default=None, description="因子描述")


class EnhancedRiskAssessment(BaseModel):
    """增強風險評估"""

    assessment_id: str = Field(description="評估ID")
    target_id: str = Field(description="目標ID")

    # 風險評分
    overall_risk_score: float = Field(ge=0.0, le=10.0, description="總體風險評分")
    likelihood_score: float = Field(ge=0.0, le=10.0, description="可能性評分")
    impact_score: float = Field(ge=0.0, le=10.0, description="影響評分")

    # 風險分級
    risk_level: Severity = Field(description="風險級別")
    risk_category: str = Field(description="風險分類")

    # 風險因子
    risk_factors: list[RiskFactor] = Field(description="風險因子列表")

    # CVSS 整合
    cvss_metrics: CVSSv3Metrics | None = Field(default=None, description="CVSS評分")

    # 業務影響
    business_impact: str | None = Field(default=None, description="業務影響描述")
    affected_assets: list[str] = Field(default_factory=list, description="受影響資產")

    # 緩解措施
    mitigation_strategies: list[str] = Field(default_factory=list, description="緩解策略")
    residual_risk: float = Field(ge=0.0, le=10.0, description="殘餘風險")

    # 時間戳
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = Field(default=None, description="有效期限")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedAttackPathNode(BaseModel):
    """增強攻擊路徑節點"""

    node_id: str = Field(description="節點ID")
    node_type: str = Field(description="節點類型")  # "asset", "vulnerability", "technique"
    name: str = Field(description="節點名稱")
    description: str | None = Field(default=None, description="節點描述")

    # 節點屬性
    exploitability: float = Field(ge=0.0, le=10.0, description="可利用性")
    impact: float = Field(ge=0.0, le=10.0, description="影響度")
    difficulty: float = Field(ge=0.0, le=10.0, description="難度")

    # MITRE ATT&CK 映射
    mitre_technique: str | None = Field(default=None, description="MITRE技術ID")
    mitre_tactic: str | None = Field(default=None, description="MITRE戰術")

    # 前置條件和後果
    prerequisites: list[str] = Field(default_factory=list, description="前置條件")
    consequences: list[str] = Field(default_factory=list, description="後果")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedAttackPath(BaseModel):
    """增強攻擊路徑"""

    path_id: str = Field(description="路徑ID")
    target_asset: str = Field(description="目標資產")

    # 路徑信息
    nodes: list[EnhancedAttackPathNode] = Field(description="路徑節點")
    edges: list[dict[str, str]] = Field(description="邊關係")

    # 路徑評估
    path_feasibility: float = Field(ge=0.0, le=1.0, description="路徑可行性")
    estimated_time: int = Field(ge=0, description="估計時間(分鐘)")
    skill_level_required: str = Field(description="所需技能等級")

    # 風險評估
    success_probability: float = Field(ge=0.0, le=1.0, description="成功概率")
    detection_probability: float = Field(ge=0.0, le=1.0, description="被檢測概率")
    overall_risk: float = Field(ge=0.0, le=10.0, description="總體風險")

    # 緩解措施
    blocking_controls: list[str] = Field(default_factory=list, description="阻斷控制")
    detection_controls: list[str] = Field(default_factory=list, description="檢測控制")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str = Field(description="依賴類型")  # "prerequisite", "blocker", "input"
    dependent_task_id: str = Field(description="依賴任務ID")
    condition: str | None = Field(default=None, description="依賴條件")
    required: bool = Field(default=True, description="是否必需")


class EnhancedTaskExecution(BaseModel):
    """增強任務執行"""

    task_id: str = Field(description="任務ID")
    task_type: str = Field(description="任務類型")
    module_name: ModuleName = Field(description="執行模組")

    # 任務配置
    priority: int = Field(ge=1, le=10, description="優先級")
    timeout: int = Field(default=3600, ge=60, description="超時時間(秒)")
    retry_count: int = Field(default=3, ge=0, description="重試次數")

    # 依賴關係
    dependencies: list[TaskDependency] = Field(default_factory=list, description="任務依賴")

    # 執行狀態
    status: TestStatus = Field(description="執行狀態")
    progress: float = Field(ge=0.0, le=1.0, description="執行進度")

    # 結果信息
    result_data: dict[str, Any] = Field(default_factory=dict, description="結果數據")
    error_message: str | None = Field(default=None, description="錯誤消息")

    # 資源使用
    cpu_usage: float | None = Field(default=None, description="CPU使用率")
    memory_usage: int | None = Field(default=None, description="內存使用(MB)")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = Field(default=None, description="開始時間")
    completed_at: datetime | None = Field(default=None, description="完成時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v


class TaskQueue(BaseModel):
    """任務隊列"""

    queue_id: str = Field(description="隊列ID")
    queue_name: str = Field(description="隊列名稱")

    # 隊列配置
    max_concurrent_tasks: int = Field(default=5, ge=1, description="最大併發任務數")
    task_timeout: int = Field(default=3600, ge=60, description="任務超時(秒)")

    # 隊列狀態
    pending_tasks: list[str] = Field(default_factory=list, description="等待任務")
    running_tasks: list[str] = Field(default_factory=list, description="運行任務")
    completed_tasks: list[str] = Field(default_factory=list, description="完成任務")

    # 統計信息
    total_processed: int = Field(ge=0, description="總處理數")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    average_execution_time: float = Field(ge=0.0, description="平均執行時間")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class TestStrategy(BaseModel):
    """測試策略"""

    strategy_id: str = Field(description="策略ID")
    strategy_name: str = Field(description="策略名稱")
    target_type: str = Field(description="目標類型")

    # 策略配置
    test_categories: list[str] = Field(description="測試分類")
    test_sequence: list[str] = Field(description="測試順序")
    parallel_execution: bool = Field(default=False, description="是否並行執行")

    # 條件配置
    trigger_conditions: list[str] = Field(default_factory=list, description="觸發條件")
    stop_conditions: list[str] = Field(default_factory=list, description="停止條件")

    # 優先級和資源
    priority_weights: dict[str, float] = Field(default_factory=dict, description="優先級權重")
    resource_limits: dict[str, Any] = Field(default_factory=dict, description="資源限制")

    # 適應性配置
    learning_enabled: bool = Field(default=True, description="是否啟用學習")
    adaptation_threshold: float = Field(ge=0.0, le=1.0, description="適應閾值")

    # 效果評估
    effectiveness_score: float = Field(ge=0.0, le=10.0, description="效果評分")
    usage_count: int = Field(ge=0, description="使用次數")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedModuleStatus(BaseModel):
    """增強模組狀態"""

    module_name: ModuleName = Field(description="模組名稱")
    version: str = Field(description="模組版本")

    # 狀態信息
    status: str = Field(description="運行狀態")  # "running", "stopped", "error", "maintenance"
    health_score: float = Field(ge=0.0, le=1.0, description="健康評分")

    # 性能指標
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU使用率")
    memory_usage: float = Field(ge=0.0, description="內存使用(MB)")
    active_connections: int = Field(ge=0, description="活躍連接數")

    # 任務統計
    tasks_processed: int = Field(ge=0, description="處理任務數")
    tasks_pending: int = Field(ge=0, description="待處理任務數")
    error_count: int = Field(ge=0, description="錯誤次數")

    # 時間信息
    started_at: datetime = Field(description="啟動時間")
    last_heartbeat: datetime = Field(description="最後心跳")
    uptime_seconds: int = Field(ge=0, description="運行時間(秒)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class SystemOrchestration(BaseModel):
    """系統編排"""

    orchestration_id: str = Field(description="編排ID")
    orchestration_name: str = Field(description="編排名稱")

    # 模組狀態
    module_statuses: list[EnhancedModuleStatus] = Field(description="模組狀態列表")

    # 系統配置
    load_balancing: dict[str, Any] = Field(default_factory=dict, description="負載均衡配置")
    failover_rules: dict[str, Any] = Field(default_factory=dict, description="故障轉移規則")

    # 整體狀態
    overall_health: float = Field(ge=0.0, le=1.0, description="整體健康度")
    system_load: float = Field(ge=0.0, le=1.0, description="系統負載")

    # 事件處理
    active_incidents: list[str] = Field(default_factory=list, description="活躍事件")
    maintenance_windows: list[dict] = Field(default_factory=list, description="維護時段")

    # 時間戳
    status_updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedVulnerabilityCorrelation(BaseModel):
    """增強漏洞關聯分析"""

    correlation_id: str = Field(description="關聯分析ID")
    primary_vulnerability: str = Field(description="主要漏洞ID")

    # 關聯漏洞
    related_vulnerabilities: list[str] = Field(description="相關漏洞列表")
    correlation_strength: float = Field(ge=0.0, le=1.0, description="關聯強度")

    # 關聯類型
    correlation_type: str = Field(description="關聯類型")

    # 組合影響
    combined_risk_score: float = Field(ge=0.0, le=10.0, description="組合風險評分")
    exploitation_complexity: float = Field(ge=0.0, le=10.0, description="利用複雜度")

    # 攻擊場景
    attack_scenarios: list[str] = Field(default_factory=list, description="攻擊場景")
    recommended_order: list[str] = Field(default_factory=list, description="建議利用順序")

    # 緩解建議
    coordinated_mitigation: list[str] = Field(default_factory=list, description="協調緩解措施")
    priority_ranking: list[str] = Field(default_factory=list, description="優先級排序")

    # 時間戳
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")

