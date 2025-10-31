"""AIVA 核心業務模式定義

包含風險評估、任務管理、策略生成、系統協調等核心業務邏輯相關的數據模式。
屬於 core 模組的業務特定定義。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.aiva_common.enums import ModuleName, Severity, TestStatus
from services.aiva_common.schemas.ai import CVSSv3Metrics

# ==================== 風險評估 ====================


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str = Field(description="風險因子名稱")
    weight: float = Field(ge=0.0, le=1.0, description="權重")
    value: float = Field(ge=0.0, le=10.0, description="因子值")
    description: str | None = Field(default=None, description="因子描述")


class RiskAssessment(BaseModel):
    """風險評估"""

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


class AttackPathNode(BaseModel):
    """攻擊路徑節點"""

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


class AttackPath(BaseModel):
    """攻擊路徑"""

    path_id: str = Field(description="路徑ID")
    target_asset: str = Field(description="目標資產")

    # 路徑信息
    nodes: list[AttackPathNode] = Field(description="路徑節點")
    edges: list[dict[str, str]] = Field(
        description="邊關係"
    )  # {"from": "node1", "to": "node2", "condition": "..."}

    # 路徑評估
    path_feasibility: float = Field(ge=0.0, le=1.0, description="路徑可行性")
    estimated_time: int = Field(ge=0, description="估計時間(分鐘)")
    skill_level_required: str = Field(
        description="所需技能等級"
    )  # "low", "medium", "high", "expert"

    # 風險評估
    success_probability: float = Field(ge=0.0, le=1.0, description="成功概率")
    detection_probability: float = Field(ge=0.0, le=1.0, description="被檢測概率")
    overall_risk: float = Field(ge=0.0, le=10.0, description="總體風險")

    # 緩解措施
    blocking_controls: list[str] = Field(default_factory=list, description="阻斷控制")
    detection_controls: list[str] = Field(default_factory=list, description="檢測控制")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 任務管理 ====================


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str = Field(
        description="依賴類型"
    )  # "prerequisite", "blocker", "input"
    dependent_task_id: str = Field(description="依賴任務ID")
    condition: str | None = Field(default=None, description="依賴條件")
    required: bool = Field(default=True, description="是否必需")


class TaskExecution(BaseModel):
    """任務執行"""

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


# ==================== 策略生成 ====================


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


# ==================== 系統協調 ====================


class ModuleStatus(BaseModel):
    """模組狀態"""

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


class SystemOrchestration(BaseModel):
    """系統編排"""

    orchestration_id: str = Field(description="編排ID")
    orchestration_name: str = Field(description="編排名稱")

    # 模組狀態
    module_statuses: list[ModuleStatus] = Field(description="模組狀態列表")

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


# ==================== 漏洞關聯分析 ====================


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析"""

    correlation_id: str = Field(description="關聯分析ID")
    primary_vulnerability: str = Field(description="主要漏洞ID")

    # 關聯漏洞
    related_vulnerabilities: list[str] = Field(description="相關漏洞列表")
    correlation_strength: float = Field(ge=0.0, le=1.0, description="關聯強度")

    # 關聯類型
    correlation_type: str = Field(description="關聯類型")
    # "chain", "cluster", "prerequisite", "amplification"

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


__all__ = [
    "RiskFactor",
    "RiskAssessment",
    "AttackPathNode",
    "AttackPath",
    "TaskDependency",
    "TaskExecution",
    "TaskQueue",
    "TestStrategy",
    "ModuleStatus",
    "SystemOrchestration",
    "VulnerabilityCorrelation",
]
