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
    """漏洞基本資訊 - 用於 Finding 中的漏洞描述"""

    name: VulnerabilityType
    cwe: str | None = None
    severity: Severity
    confidence: Confidence
    description: str | None = None


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
