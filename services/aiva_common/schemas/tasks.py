"""
任務相關 Schema

此模組定義了所有類型的任務負載 (Payload)，包括掃描任務、功能測試任務、
威脅情報查詢任務等。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..enums import (
    Confidence,
    IntelSource,
    IOCType,
    PostExTestType,
    RemediationType,
    Severity,
    TestStatus,
    ThreatLevel,
    VulnerabilityType,
)
from .base import Asset, Authentication, Fingerprints, RateLimit, ScanScope, Summary

# ==================== 掃描任務 ====================


class ScanStartPayload(BaseModel):
    """掃描啟動 Payload"""

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
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        if len(v) < 10:
            raise ValueError("scan_id too short (minimum 10 characters)")
        return v

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if not v:
            raise ValueError("At least one target required")
        if len(v) > 100:
            raise ValueError("Too many targets (maximum 100)")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"quick", "normal", "deep", "full", "custom"}
        if v not in allowed:
            raise ValueError(f"Invalid strategy: {v}. Must be one of {allowed}")
        return v


class ScanCompletedPayload(BaseModel):
    """掃描完成 Payload"""

    scan_id: str
    status: str
    summary: Summary
    assets: list[Asset] = []
    fingerprints: Fingerprints | None = None
    error_info: str | None = None


# ==================== 功能測試任務 ====================


class FunctionTaskTarget(BaseModel):
    """功能測試目標"""

    url: Any  # Accept arbitrary URL-like values
    parameter: str | None = None
    method: str = "GET"
    parameter_location: str = "query"
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    form_data: dict[str, Any] = Field(default_factory=dict)
    json_data: dict[str, Any] | None = None
    body: str | None = None


class FunctionTaskContext(BaseModel):
    """功能測試上下文"""

    db_type_hint: str | None = None
    waf_detected: bool = False
    related_findings: list[str] | None = None


class FunctionTaskTestConfig(BaseModel):
    """功能測試配置"""

    payloads: list[str] = Field(default_factory=lambda: ["basic"])
    custom_payloads: list[str] = Field(default_factory=list)
    blind_xss: bool = False
    dom_testing: bool = False
    timeout: float | None = None


class FunctionTaskPayload(BaseModel):
    """功能測試任務 Payload"""

    task_id: str
    scan_id: str
    priority: int = 5
    target: FunctionTaskTarget
    context: FunctionTaskContext = Field(default_factory=FunctionTaskContext)
    strategy: str = "full"
    custom_payloads: list[str] | None = None
    test_config: FunctionTaskTestConfig = Field(default_factory=FunctionTaskTestConfig)

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

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("priority must be between 1 and 10")
        return v


class FeedbackEventPayload(BaseModel):
    """反饋事件 Payload"""

    task_id: str
    scan_id: str
    event_type: str
    details: dict[str, Any] = {}
    form_url: HttpUrl | None = None


class TaskUpdatePayload(BaseModel):
    """任務更新 Payload"""

    task_id: str
    scan_id: str
    status: str
    worker_id: str
    details: dict[str, Any] | None = None


class ConfigUpdatePayload(BaseModel):
    """配置更新 Payload"""

    update_id: str
    config_items: dict[str, Any] = {}


# ==================== 威脅情報任務 ====================


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


# ==================== 權限檢查任務 ====================


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


# ==================== 修復任務 ====================


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
    status: str
    patch_content: str | None = None
    instructions: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 後滲透測試任務 ====================


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


# ==================== 業務邏輯測試任務 ====================


class BizLogicTestPayload(BaseModel):
    """業務邏輯測試 Payload"""

    task_id: str
    scan_id: str
    test_type: str  # price_manipulation, workflow_bypass, race_condition
    target_urls: dict[str, str]  # 目標 URL 字典
    test_config: dict[str, Any] = Field(default_factory=dict)
    product_id: str | None = None
    workflow_steps: list[dict[str, str]] = Field(default_factory=list)


class BizLogicResultPayload(BaseModel):
    """業務邏輯測試結果 Payload"""

    task_id: str
    scan_id: str
    test_type: str
    status: str  # completed, failed, error
    findings: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== API 測試任務 ====================


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


# ==================== EASM 資產探索任務 ====================


class EASMDiscoveryPayload(BaseModel):
    """EASM 資產探索 Payload"""

    discovery_id: str
    scan_id: str
    discovery_type: str  # "subdomain", "port_scan", "cloud_storage", "certificate"
    targets: list[str]  # 起始目標
    scope: ScanScope = Field(default_factory=ScanScope)
    max_depth: int = 3
    passive_only: bool = False


class EASMDiscoveryResult(BaseModel):
    """EASM 探索結果"""

    discovery_id: str
    scan_id: str
    status: str  # "completed", "in_progress", "failed"
    discovered_assets: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# 測試和漏洞利用
# ============================================================================


class Scenario(BaseModel):
    """訓練場景定義"""

    scenario_id: str
    name: str
    description: str
    difficulty: str = "medium"
    target_url: str
    objectives: list[str] = Field(default_factory=list)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """場景執行結果"""

    scenario_id: str
    session_id: str
    success: bool
    score: float = 0.0
    findings: list[dict[str, Any]] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardScenario(BaseModel):
    """標準靶場場景 - 用於訓練和測試"""

    scenario_id: str
    name: str
    description: str
    vulnerability_type: VulnerabilityType
    difficulty_level: str  # "easy", "medium", "hard", "expert"
    target_config: dict[str, Any]  # 靶場配置
    expected_plan: dict[str, Any]  # 預期的最佳攻擊計畫 (簡化為 dict 避免循環引用)
    success_criteria: dict[str, Any]  # 成功標準
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioTestResult(BaseModel):
    """場景測試結果 - 模型在標準場景上的表現"""

    model_config = {"protected_namespaces": ()}

    test_id: str
    scenario_id: str
    model_version: str
    generated_plan: dict[str, Any]  # 簡化避免循環引用
    execution_result: dict[str, Any]  # 簡化避免循環引用
    score: float  # 綜合評分 (0.0 - 100.0)
    comparison: dict[str, Any]  # 與預期計畫的對比
    passed: bool
    tested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExploitPayload(BaseModel):
    """漏洞利用載荷"""

    payload_id: str = Field(description="載荷ID")
    payload_type: str = Field(
        description="載荷類型"
    )  # "xss", "sqli", "command_injection"
    payload_content: str = Field(description="載荷內容")

    # 載荷屬性
    encoding: str = Field(default="none", description="編碼方式")
    obfuscation: bool = Field(default=False, description="是否混淆")
    bypass_technique: str | None = Field(default=None, description="繞過技術")

    # 適用條件
    target_technology: list[str] = Field(default_factory=list, description="目標技術")
    required_context: dict[str, Any] = Field(
        default_factory=dict, description="所需上下文"
    )

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
    impact_level: str = Field(
        description="影響級別"
    )  # "critical", "high", "medium", "low"

    # 利用細節
    exploit_technique: str = Field(description="利用技術")
    payload_used: str = Field(description="使用的載荷")
    execution_time: float = Field(ge=0.0, description="執行時間(秒)")

    # 獲得的訪問
    access_gained: dict[str, Any] = Field(
        default_factory=dict, description="獲得的訪問權限"
    )
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
    priority_weights: dict[str, float] = Field(
        default_factory=dict, description="優先級權重"
    )
    resource_limits: dict[str, Any] = Field(
        default_factory=dict, description="資源限制"
    )

    # 適應性配置
    learning_enabled: bool = Field(default=True, description="是否啟用學習")
    adaptation_threshold: float = Field(ge=0.0, le=1.0, description="適應閾值")

    # 效果評估
    effectiveness_score: float = Field(ge=0.0, le=10.0, description="效果評分")
    usage_count: int = Field(ge=0, description="使用次數")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
