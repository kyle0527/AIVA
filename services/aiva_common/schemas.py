from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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


class Authentication(BaseModel):
    method: str = "none"
    credentials: dict[str, str] | None = None


class RateLimit(BaseModel):
    requests_per_second: int = 25
    burst: int = 50

    @field_validator("requests_per_second", "burst")
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limit must be non-negative")
        return v


class ScanScope(BaseModel):
    exclusions: list[str] = []
    include_subdomains: bool = True
    allowed_hosts: list[str] = []


class ScanStartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    scope: ScanScope = Field(default_factory=ScanScope)
    authentication: Authentication = Field(default_factory=Authentication)
    strategy: str = "deep"
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    custom_headers: dict[str, str] = {}
    x_forwarded_for: str | None = None

    @field_validator("scan_id")
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


# ==================== BizLogic Payloads ====================


class BizLogicTestPayload(BaseModel):
    """業務邏輯測試 Payload"""

    task_id: str
    scan_id: str
    test_type: str  # price_manipulation, workflow_bypass, race_condition
    target_urls: dict[
        str, str
    ]  # 目標 URL 字典 {"cart_api": "...", "checkout_api": "..."}
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
    schema: APISchemaPayload | None = None
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


class AttackTarget(BaseModel):
    """攻擊目標 - 定義測試的目標系統

    用於 RAG 模組和訓練場景管理
    """

    target_id: str = Field(description="目標唯一標識")
    name: str = Field(description="目標名稱")
    base_url: str = Field(description="基礎 URL")
    target_type: str = Field(
        description="目標類型",
        examples=["web_app", "api", "mobile_backend", "iot_device"],
    )
    description: str | None = Field(default=None, description="目標描述")

    # 環境配置
    environment: str = Field(
        default="development",
        description="環境類型: development, staging, production, sandbox",
    )
    authentication: dict[str, Any] = Field(
        default_factory=dict,
        description="認證信息 (credentials, tokens, etc.)",
    )

    # 技術堆疊
    technologies: list[str] = Field(
        default_factory=list,
        description="技術堆疊: ['PHP', 'MySQL', 'Apache'] 等",
    )
    frameworks: list[str] = Field(
        default_factory=list,
        description="框架列表: ['Laravel', 'WordPress'] 等",
    )

    # 安全特性
    waf_enabled: bool = Field(default=False, description="是否啟用 WAF")
    waf_type: str | None = Field(default=None, description="WAF 類型")
    rate_limiting: bool = Field(default=False, description="是否有速率限制")

    # 測試範圍
    allowed_paths: list[str] = Field(
        default_factory=list,
        description="允許測試的路徑",
    )
    excluded_paths: list[str] = Field(
        default_factory=list,
        description="禁止測試的路徑",
    )

    # 元數據
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        allowed = {"web_app", "api", "mobile_backend", "iot_device", "cloud_service"}
        if v not in allowed:
            raise ValueError(f"Invalid target_type: {v}. Must be one of {allowed}")
        return v


class Scenario(BaseModel):
    """訓練場景 - 標準化的測試場景定義

    用於 AI 訓練和場景管理
    """

    scenario_id: str = Field(description="場景唯一標識")
    name: str = Field(description="場景名稱")
    description: str = Field(description="場景描述")

    # 場景分類
    category: str = Field(
        description="場景類別",
        examples=["owasp_top10", "custom", "ctf", "real_world"],
    )
    vulnerability_type: VulnerabilityType = Field(description="漏洞類型")
    difficulty_level: str = Field(
        default="medium",
        description="難度等級: easy, medium, hard, expert",
    )

    # 目標配置
    target: AttackTarget = Field(description="攻擊目標")

    # 預期計畫
    expected_plan: AttackPlan = Field(description="預期的攻擊計畫")

    # 成功標準
    success_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description="成功標準 (goal_achieved, exploit_successful 等)",
    )

    # 評分權重
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "completion_rate": 0.3,
            "success_rate": 0.3,
            "sequence_accuracy": 0.2,
            "efficiency": 0.2,
        },
        description="評分權重配置",
    )

    # 標籤與元數據
    tags: list[str] = Field(default_factory=list, description="標籤")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="額外元數據 (estimated_duration, author, etc.)",
    )

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("difficulty_level")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard", "expert"}
        if v not in allowed:
            raise ValueError(f"Invalid difficulty_level: {v}. Must be one of {allowed}")
        return v


class ScenarioResult(BaseModel):
    """場景執行結果 - 訓練或測試場景的執行結果

    用於記錄訓練成果和性能評估
    """

    result_id: str = Field(description="結果唯一標識")
    scenario_id: str = Field(description="關聯的場景 ID")
    session_id: str = Field(description="訓練會話 ID")

    # 執行狀態
    status: str = Field(
        description="執行狀態: completed, failed, timeout, cancelled",
    )

    # 執行指標
    metrics: PlanExecutionMetrics = Field(description="執行指標")

    # 計畫與追蹤
    executed_plan: AttackPlan = Field(description="實際執行的計畫")
    trace_records: list[TraceRecord] = Field(
        default_factory=list,
        description="執行追蹤記錄",
    )

    # 結果分析
    goal_achieved: bool = Field(default=False, description="是否達成目標")
    exploit_successful: bool = Field(default=False, description="利用是否成功")
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="總體評分 (0-1)",
    )

    # 錯誤與問題
    errors: list[str] = Field(default_factory=list, description="錯誤列表")
    warnings: list[str] = Field(default_factory=list, description="警告列表")

    # 性能數據
    total_duration_seconds: float = Field(default=0.0, description="總執行時間")
    resource_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="資源使用情況 (cpu, memory, network 等)",
    )

    # 學習數據
    experience_sample_ids: list[str] = Field(
        default_factory=list,
        description="生成的經驗樣本 ID 列表",
    )

    # 元數據
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(default=None)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"completed", "failed", "timeout", "cancelled", "in_progress"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v

    @field_validator("overall_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("overall_score must be between 0.0 and 1.0")
        return v


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
    variables: dict[str, Any] = Field(
        default_factory=dict
    )  # 會話變數（用於步驟間傳遞數據）
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

    @field_validator(
        "learning_rate", "validation_split", "discount_factor", "exploration_rate"
    )
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
                                    "snippet": (
                                        {"text": loc.snippet} if loc.snippet else None
                                    ),
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
            Severity.INFORMATIONAL: "note",
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
            message=self.vulnerability.description
            or f"{self.vulnerability.name.value} detected",
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


# ==================== 補充的 AI 訓練合約 ====================


class AITrainingStopPayload(BaseModel):
    """AI 訓練停止請求 - 用於停止正在進行的訓練會話"""

    training_id: str = Field(description="訓練會話 ID")
    reason: str = Field(default="user_requested", description="停止原因")
    save_checkpoint: bool = Field(default=True, description="是否保存檢查點")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AITrainingFailedPayload(BaseModel):
    """AI 訓練失敗通知 - 訓練過程中發生錯誤"""

    training_id: str
    error_type: str = Field(description="錯誤類型")
    error_message: str = Field(description="錯誤訊息")
    traceback: str | None = Field(default=None, description="錯誤追蹤")
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    partial_results_available: bool = Field(default=False)
    checkpoint_saved: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIScenarioLoadedEvent(BaseModel):
    """標準場景載入事件 - 當 OWASP 靶場場景被載入時觸發"""

    scenario_id: str
    scenario_name: str
    target_system: str
    vulnerability_type: VulnerabilityType
    expected_steps: int
    difficulty_level: str = Field(
        default="medium", description="難度等級: easy|medium|hard"
    )
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== 補充的 Scan 模組合約 ====================


class ScanProgressPayload(BaseModel):
    """掃描進度通知 - 定期報告掃描進度"""

    scan_id: str
    progress_percentage: float = Field(ge=0.0, le=100.0, description="進度百分比")
    current_target: HttpUrl | None = Field(default=None, description="當前掃描目標")
    assets_discovered: int = Field(default=0, description="已發現資產數")
    vulnerabilities_found: int = Field(default=0, description="已發現漏洞數")
    tests_completed: int = Field(default=0, description="已完成測試數")
    tests_total: int = Field(default=0, description="總測試數")
    estimated_time_remaining_seconds: int | None = Field(
        default=None, description="預計剩餘時間（秒）"
    )
    current_phase: str = Field(
        default="discovery", description="當前階段: discovery|fingerprinting|scanning"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScanFailedPayload(BaseModel):
    """掃描失敗通知 - 掃描過程中發生錯誤"""

    scan_id: str
    error_type: str = Field(
        description="錯誤類型: network|timeout|authentication|internal"
    )
    error_message: str = Field(description="錯誤訊息")
    failed_target: HttpUrl | None = Field(default=None, description="失敗的目標 URL")
    failed_at_phase: str = Field(
        default="unknown", description="失敗階段: discovery|fingerprinting|scanning"
    )
    partial_results_available: bool = Field(default=False, description="是否有部分結果")
    assets_discovered: int = Field(default=0, description="已發現資產數")
    vulnerabilities_found: int = Field(default=0, description="已發現漏洞數")
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScanAssetDiscoveredEvent(BaseModel):
    """資產發現事件 - 當新資產被發現時觸發"""

    scan_id: str
    asset: Asset
    discovery_method: str = Field(
        description="發現方法: crawler|dns|port_scan|subdomain|directory"
    )
    confidence: Confidence = Field(default=Confidence.FIRM)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== 補充的 Function 模組合約 ====================


class FunctionTaskProgressPayload(BaseModel):
    """功能測試進度通知 - 定期報告測試進度"""

    task_id: str
    scan_id: str
    progress_percentage: float = Field(ge=0.0, le=100.0, description="進度百分比")
    tests_completed: int = Field(default=0, description="已完成測試數")
    tests_total: int = Field(default=0, description="總測試數")
    vulnerabilities_found: int = Field(default=0, description="已發現漏洞數")
    current_test: str | None = Field(default=None, description="當前測試項目")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class FunctionTaskCompletedPayload(BaseModel):
    """功能測試完成通知 - 測試任務完成時的報告"""

    task_id: str
    scan_id: str
    status: str = Field(description="狀態: success|partial|failed")
    vulnerabilities_found: int = Field(default=0, description="發現的漏洞數")
    tests_executed: int = Field(default=0, description="執行的測試數")
    tests_passed: int = Field(default=0, description="通過的測試數")
    tests_failed: int = Field(default=0, description="失敗的測試數")
    duration_seconds: float = Field(description="執行時間（秒）")
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="測試結果詳情"
    )
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class FunctionTaskFailedPayload(BaseModel):
    """功能測試失敗通知 - 測試過程中發生錯誤"""

    task_id: str
    scan_id: str
    error_type: str = Field(description="錯誤類型: network|timeout|crash|validation")
    error_message: str = Field(description="錯誤訊息")
    traceback: str | None = Field(default=None, description="錯誤追蹤")
    tests_completed: int = Field(default=0, description="已完成測試數")
    tests_failed: int = Field(default=0, description="失敗測試數")
    partial_results: list[dict[str, Any]] = Field(
        default_factory=list, description="部分測試結果"
    )
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class FunctionVulnFoundEvent(BaseModel):
    """漏洞發現事件 - 當功能測試發現漏洞時觸發"""

    task_id: str
    scan_id: str
    vulnerability: Vulnerability
    confidence: Confidence
    severity: Severity
    test_type: str = Field(description="測試類型: xss|sqli|ssrf|idor|etc")
    evidence: FindingEvidence
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== 補充的 Integration 模組合約 ====================


class IntegrationAnalysisStartPayload(BaseModel):
    """整合分析啟動請求 - 啟動漏洞相關性分析和攻擊路徑生成"""

    analysis_id: str = Field(description="分析會話 ID")
    scan_id: str
    analysis_types: list[str] = Field(
        description="分析類型: correlation|attack_path|risk_assessment|root_cause"
    )
    findings: list[FindingPayload] = Field(description="待分析的漏洞發現")
    context: dict[str, Any] = Field(default_factory=dict, description="分析上下文")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationAnalysisProgressPayload(BaseModel):
    """整合分析進度通知 - 定期報告分析進度"""

    analysis_id: str
    scan_id: str
    progress_percentage: float = Field(ge=0.0, le=100.0, description="進度百分比")
    current_analysis_type: str = Field(description="當前分析類型")
    correlations_found: int = Field(default=0, description="已發現關聯數")
    attack_paths_generated: int = Field(default=0, description="已生成攻擊路徑數")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationAnalysisCompletedPayload(BaseModel):
    """整合分析完成通知 - 分析完成時的最終報告"""

    analysis_id: str
    scan_id: str
    status: str = Field(description="狀態: success|partial|failed")
    correlations: list[VulnerabilityCorrelation] = Field(
        default_factory=list, description="漏洞相關性分析結果"
    )
    attack_paths: list[AttackPathPayload] = Field(
        default_factory=list, description="攻擊路徑分析結果"
    )
    risk_assessment: RiskAssessmentResult | None = Field(
        default=None, description="風險評估結果"
    )
    recommendations: list[str] = Field(default_factory=list, description="修復建議")
    duration_seconds: float = Field(description="分析時間（秒）")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationReportGenerateCommand(BaseModel):
    """報告生成命令 - 請求生成掃描報告"""

    report_id: str = Field(description="報告 ID")
    scan_id: str
    report_format: str = Field(description="報告格式: pdf|html|json|sarif|markdown")
    include_sections: list[str] = Field(
        default_factory=lambda: ["summary", "findings", "recommendations"],
        description="包含的章節",
    )
    template: str | None = Field(default=None, description="報告模板")
    language: str = Field(default="zh-TW", description="報告語言")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationReportGeneratedEvent(BaseModel):
    """報告生成完成事件 - 報告生成完成時觸發"""

    report_id: str
    scan_id: str
    report_format: str
    file_path: str | None = Field(default=None, description="報告文件路徑")
    file_size_bytes: int | None = Field(default=None, description="文件大小（字節）")
    download_url: str | None = Field(default=None, description="下載 URL")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
