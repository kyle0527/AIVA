"""AIVA Core Models - 核心業務模組

此文件包含AIVA系統核心業務邏輯相關的所有數據模型。

職責範圍：
1. 風險評估和分析 (RiskAssessment, RiskTrend)
2. 攻擊路徑分析 (AttackPath, AttackPathNode)
3. 漏洞關聯分析 (VulnerabilityCorrelation)
4. 任務管理和編排 (Task, TaskQueue)
5. 測試策略生成 (TestStrategy)
6. 系統協調和監控 (ModuleStatus, SystemOrchestration)
7. AI智能系統 (訓練、推理、RAG)
8. 發現和影響管理 (Finding, Evidence, Impact)
9. 修復建議生成
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aiva_common.enums import (
    AttackPathEdgeType,
    AttackPathNodeType,
    ComplianceFramework,
    Confidence,
    ModuleName,
    RiskLevel,
    Severity,
    TaskStatus,
)
from aiva_common.schemas import (
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    Target,
)

# ==================== 發現和影響管理 ====================
# 注意：以下類別已從 aiva_common.schemas.findings 導入：
# - Target
# - FindingEvidence
# - FindingImpact
# - FindingRecommendation
# - FindingPayload


# ==================== 增強漏洞信息 ====================


class EnhancedVulnerability(BaseModel):
    """增強漏洞信息"""

    vuln_id: str = Field(description="漏洞ID")
    title: str = Field(description="漏洞標題")
    description: str = Field(description="漏洞描述")
    severity: Severity = Field(description="嚴重程度")

    # CVSS 整合
    cvss_metrics: CVSSv3Metrics | None = Field(default=None, description="CVSS評分")

    # 標準引用
    cve_references: list[CVEReference] = Field(
        default_factory=list, description="CVE引用"
    )
    cwe_references: list[CWEReference] = Field(
        default_factory=list, description="CWE引用"
    )

    # 詳細信息
    exploit_available: bool = Field(default=False, description="是否有可用利用")
    exploitability: str | None = Field(default=None, description="可利用性")
    remediation_available: bool = Field(default=False, description="是否有修復方案")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedFindingPayload(BaseModel):
    """增強發現載荷"""

    finding_id: str = Field(description="發現ID")
    vulnerability: EnhancedVulnerability = Field(description="漏洞信息")
    target: Target = Field(description="目標")
    evidence: list[FindingEvidence] = Field(description="證據")
    impact: FindingImpact = Field(description="影響")
    recommendations: list[FindingRecommendation] = Field(description="建議")
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# 注意：FeedbackEventPayload 已從 aiva_common.schemas.tasks 導入


# ==================== 風險評估 ====================


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str = Field(description="風險因子名稱")
    weight: float = Field(ge=0.0, le=1.0, description="權重")
    value: float = Field(ge=0.0, le=10.0, description="因子值")
    description: str | None = Field(default=None, description="因子描述")


class RiskAssessmentContext(BaseModel):
    """風險評估上下文"""

    asset_id: str = Field(description="資產ID")
    asset_type: str = Field(description="資產類型")
    business_criticality: str = Field(description="業務重要性")
    environment: str = Field(description="環境類型")
    compliance_requirements: list[ComplianceFramework] = Field(
        default_factory=list, description="合規要求"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class RiskAssessmentResult(BaseModel):
    """風險評估結果"""

    assessment_id: str = Field(description="評估ID")
    asset_id: str = Field(description="資產ID")
    overall_risk_score: float = Field(ge=0.0, le=10.0, description="總體風險評分")
    risk_level: RiskLevel = Field(description="風險級別")
    contributing_factors: list[dict[str, Any]] = Field(description="貢獻因素")
    mitigation_recommendations: list[str] = Field(description="緩解建議")
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


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
    mitigation_strategies: list[str] = Field(
        default_factory=list, description="緩解策略"
    )
    residual_risk: float = Field(ge=0.0, le=10.0, description="殘餘風險")

    # 時間戳
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = Field(default=None, description="有效期限")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class RiskTrendAnalysis(BaseModel):
    """風險趨勢分析"""

    analysis_id: str = Field(description="分析ID")
    asset_id: str = Field(description="資產ID")
    time_period: str = Field(description="時間周期")
    historical_scores: list[dict[str, Any]] = Field(description="歷史評分")
    trend_direction: str = Field(
        description="趨勢方向"
    )  # "increasing", "decreasing", "stable"
    forecast: dict[str, Any] = Field(default_factory=dict, description="預測")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 攻擊路徑分析 ====================


class AttackPathNode(BaseModel):
    """攻擊路徑節點"""

    node_id: str = Field(description="節點ID")
    node_type: AttackPathNodeType = Field(description="節點類型")
    name: str = Field(description="節點名稱")
    description: str | None = Field(default=None, description="節點描述")
    exploitability: float = Field(ge=0.0, le=10.0, description="可利用性")
    impact: float = Field(ge=0.0, le=10.0, description="影響度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AttackPathEdge(BaseModel):
    """攻擊路徑邊"""

    edge_id: str = Field(description="邊ID")
    edge_type: AttackPathEdgeType = Field(description="邊類型")
    source_node: str = Field(description="源節點ID")
    target_node: str = Field(description="目標節點ID")
    likelihood: float = Field(ge=0.0, le=1.0, description="可能性")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AttackPathPayload(BaseModel):
    """攻擊路徑載荷"""

    path_id: str = Field(description="路徑ID")
    target_asset: str = Field(description="目標資產")
    nodes: list[AttackPathNode] = Field(description="節點列表")
    edges: list[AttackPathEdge] = Field(description="邊列表")
    overall_risk: float = Field(ge=0.0, le=10.0, description="總體風險")
    feasibility: float = Field(ge=0.0, le=1.0, description="可行性")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AttackPathRecommendation(BaseModel):
    """攻擊路徑建議"""

    path_id: str = Field(description="路徑ID")
    priority: int = Field(ge=1, le=10, description="優先級")
    blocking_points: list[str] = Field(description="阻斷點列表")
    remediation_steps: list[str] = Field(description="修復步驟")
    cost_estimate: str | None = Field(default=None, description="成本估算")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class EnhancedAttackPathNode(BaseModel):
    """增強攻擊路徑節點"""

    node_id: str = Field(description="節點ID")
    node_type: str = Field(
        description="節點類型"
    )  # "asset", "vulnerability", "technique"
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


# ==================== 漏洞關聯分析 ====================


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析"""

    correlation_id: str = Field(description="關聯ID")
    primary_vuln_id: str = Field(description="主要漏洞ID")
    related_vuln_ids: list[str] = Field(description="相關漏洞ID列表")
    correlation_type: str = Field(description="關聯類型")
    correlation_strength: float = Field(ge=0.0, le=1.0, description="關聯強度")
    combined_risk: float = Field(ge=0.0, le=10.0, description="組合風險")
    analysis_details: dict[str, Any] = Field(
        default_factory=dict, description="分析詳情"
    )
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    recommended_order: list[str] = Field(
        default_factory=list, description="建議利用順序"
    )

    # 緩解建議
    coordinated_mitigation: list[str] = Field(
        default_factory=list, description="協調緩解措施"
    )
    priority_ranking: list[str] = Field(default_factory=list, description="優先級排序")

    # 時間戳
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class CodeLevelRootCause(BaseModel):
    """代碼級根本原因分析"""

    analysis_id: str = Field(description="分析ID")
    vulnerability_id: str = Field(description="漏洞ID")
    file_path: str = Field(description="文件路徑")
    line_number: int | None = Field(default=None, description="行號")
    function_name: str | None = Field(default=None, description="函數名")
    code_snippet: str | None = Field(default=None, description="代碼片段")
    root_cause: str = Field(description="根本原因")
    fix_suggestion: str | None = Field(default=None, description="修復建議")
    confidence: Confidence = Field(description="置信度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# Note: SASTDASTCorrelation removed - external SAST functionality not needed for Bug Bounty hunting


# ==================== 任務管理和編排 ====================
# 注意：TaskUpdatePayload 已從 aiva_common.schemas.tasks 導入


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str = Field(
        description="依賴類型"
    )  # "prerequisite", "blocker", "input"
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
    dependencies: list[TaskDependency] = Field(
        default_factory=list, description="任務依賴"
    )

    # 執行狀態
    status: TaskStatus = Field(description="執行狀態")
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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 系統協調和監控 ====================
# 注意：ModuleStatus, HeartbeatPayload, ConfigUpdatePayload 已從 aiva_common.schemas 導入


class EnhancedModuleStatus(BaseModel):
    """增強模組狀態"""

    module_name: ModuleName = Field(description="模組名稱")
    version: str = Field(description="模組版本")

    # 狀態信息
    status: str = Field(
        description="運行狀態"
    )  # "running", "stopped", "error", "maintenance"
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


# 注意：HeartbeatPayload 已從 aiva_common.schemas.telemetry 導入
# 注意：ConfigUpdatePayload 已從 aiva_common.schemas.tasks 導入


class SystemOrchestration(BaseModel):
    """系統編排"""

    orchestration_id: str = Field(description="編排ID")
    orchestration_name: str = Field(description="編排名稱")

    # 模組狀態
    module_statuses: list[EnhancedModuleStatus] = Field(description="模組狀態列表")

    # 系統配置
    load_balancing: dict[str, Any] = Field(
        default_factory=dict, description="負載均衡配置"
    )
    failover_rules: dict[str, Any] = Field(
        default_factory=dict, description="故障轉移規則"
    )

    # 整體狀態
    overall_health: float = Field(ge=0.0, le=1.0, description="整體健康度")
    system_load: float = Field(ge=0.0, le=1.0, description="系統負載")

    # 事件處理
    active_incidents: list[str] = Field(default_factory=list, description="活躍事件")
    maintenance_windows: list[dict] = Field(
        default_factory=list, description="維護時段"
    )

    # 時間戳
    status_updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 修復建議 ====================
# 注意：RemediationGeneratePayload, RemediationResultPayload 已從 aiva_common.schemas.tasks 導入


__all__ = [
    # ========== 從 aiva_common 導入（共享標準） ==========
    # 已從 aiva_common.schemas.findings 導入：
    # - Target
    # - FindingEvidence
    # - FindingImpact
    # - FindingRecommendation
    # - FindingPayload
    # 已從 aiva_common.schemas.tasks 導入：
    # - FeedbackEventPayload
    # - TaskUpdatePayload
    # - ConfigUpdatePayload
    # - RemediationGeneratePayload
    # - RemediationResultPayload
    # 已從 aiva_common.schemas.telemetry 導入：
    # - ModuleStatus
    # - HeartbeatPayload
    # ========== Core 模組特定擴展 ==========
    # 漏洞增強
    "EnhancedVulnerability",
    "EnhancedFindingPayload",
    # 風險評估
    "RiskFactor",
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "EnhancedRiskAssessment",
    "RiskTrendAnalysis",
    # 攻擊路徑
    "AttackPathNode",
    "AttackPathEdge",
    "AttackPathPayload",
    "AttackPathRecommendation",
    "EnhancedAttackPathNode",
    "EnhancedAttackPath",
    # 漏洞關聯
    "VulnerabilityCorrelation",
    "EnhancedVulnerabilityCorrelation",
    "CodeLevelRootCause",

    # 任務管理擴展
    "TaskDependency",
    "EnhancedTaskExecution",
    "TaskQueue",
    "TestStrategy",
    # 系統協調擴展
    "EnhancedModuleStatus",
    "SystemOrchestration",
]
