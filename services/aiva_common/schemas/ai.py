"""
AI 相關 Schema

此模組定義了所有 AI 驅動功能相關的資料模型，包括訓練、強化學習、
攻擊計劃、執行追蹤等。

注意: 此檔案整合了原 ai_schemas.py 的內容
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..enums import VulnerabilityType

# ==================== CVSS v3.1 標準 ====================


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
    remediation_level: Literal["X", "U", "W", "T", "O"] = Field(
        default="X", description="修復級別"
    )
    report_confidence: Literal["X", "C", "R", "U"] = Field(
        default="X", description="報告置信度"
    )

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
    base_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="基本分數"
    )
    temporal_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="時間分數"
    )
    environmental_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="環境分數"
    )
    vector_string: str | None = Field(default=None, description="CVSS 向量字符串")

    def calculate_base_score(self) -> float:
        """計算 CVSS v3.1 基本分數"""
        av_weights = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
        ac_weights = {"L": 0.77, "H": 0.44}
        if self.scope == "C":
            pr_weights = {"N": 0.85, "L": 0.68, "H": 0.50}
        else:
            pr_weights = {"N": 0.85, "L": 0.62, "H": 0.27}
        ui_weights = {"N": 0.85, "R": 0.62}
        cia_weights = {"N": 0.0, "L": 0.22, "H": 0.56}

        impact = 1 - (1 - cia_weights[self.confidentiality]) * (
            1 - cia_weights[self.integrity]
        ) * (1 - cia_weights[self.availability])

        if self.scope == "C":
            impact_adjusted = 7.52 * (impact - 0.029) - \
                3.25 * pow(impact - 0.02, 15)
        else:
            impact_adjusted = 6.42 * impact

        exploitability = (
            8.22
            * av_weights[self.attack_vector]
            * ac_weights[self.attack_complexity]
            * pr_weights[self.privileges_required]
            * ui_weights[self.user_interaction]
        )

        if impact_adjusted <= 0:
            return 0.0

        if self.scope == "U":
            base = impact_adjusted + exploitability
        else:
            base = 1.08 * (impact_adjusted + exploitability)

        return min(10.0, round(base, 1))


# ==================== 攻擊計劃和步驟 ====================


class AttackStep(BaseModel):
    """攻擊步驟 (整合 MITRE ATT&CK)"""

    step_id: str
    action: str
    tool_type: str
    target: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_result: str | None = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    mitre_technique_id: str | None = Field(
        default=None,
        pattern=r"^T\d{4}(\.\d{3})?$",
    )
    mitre_tactic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPlan(BaseModel):
    """攻擊計畫"""

    plan_id: str
    scan_id: str
    attack_type: VulnerabilityType
    steps: list[AttackStep]
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    target_info: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "ai_planner"
    mitre_techniques: list[str] = Field(default_factory=list)
    mitre_tactics: list[str] = Field(default_factory=list)
    capec_id: str | None = Field(default=None, pattern=r"^CAPEC-\d+$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        if not v.startswith("plan_"):
            raise ValueError("plan_id must start with 'plan_'")
        return v


class TraceRecord(BaseModel):
    """執行追蹤記錄"""

    trace_id: str
    plan_id: str
    step_id: str
    session_id: str
    tool_name: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    status: str
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    environment_response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"success", "failed", "timeout", "skipped", "error"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


class PlanExecutionMetrics(BaseModel):
    """計畫執行指標"""

    plan_id: str
    session_id: str
    expected_steps: int
    executed_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    extra_actions: int
    completion_rate: float
    success_rate: float
    sequence_accuracy: float
    goal_achieved: bool
    reward_score: float
    total_execution_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanExecutionResult(BaseModel):
    """計畫執行結果"""

    result_id: str
    plan_id: str
    session_id: str
    plan: AttackPlan
    trace: list[TraceRecord]
    metrics: PlanExecutionMetrics
    findings: list[dict[str, Any]] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    status: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"completed", "partial", "failed", "aborted"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v


# ==================== AI 訓練相關 ====================


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    config_id: str
    model_type: str
    training_mode: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 3
    reward_function: str = "completion_rate"
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AITrainingStartPayload(BaseModel):
    """AI 訓練啟動請求"""

    training_id: str
    training_type: str
    scenario_id: str | None = None
    target_vulnerability: str | None = None
    config: ModelTrainingConfig
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("training_id")
    @classmethod
    def validate_training_id(cls, v: str) -> str:
        if not v.startswith("training_"):
            raise ValueError("training_id must start with 'training_'")
        return v


class AITrainingProgressPayload(BaseModel):
    """AI 訓練進度報告"""

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


class RAGKnowledgeUpdatePayload(BaseModel):
    """RAG 知識庫更新請求"""

    knowledge_type: str
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
    """RAG 查詢請求"""

    query_id: str
    query_text: str
    top_k: int = Field(default=5, ge=1, le=100)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    knowledge_types: list[str] | None = None
    categories: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 經驗學習
# ============================================================================


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
        default_factory=dict,
        description="獎勵分解 (completion, success, sequence, goal)",
    )

    # 上下文信息
    context: dict[str, Any] = Field(default_factory=dict, description="環境上下文")
    target_info: dict[str, Any] = Field(
        default_factory=dict, description="目標信息")

    # 時間信息
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = Field(default=None, ge=0, description="執行時長")

    # 質量標記
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="樣本質量分數"
    )
    is_positive: bool = Field(description="是否為正樣本")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="樣本置信度")

    # 學習標籤
    learning_tags: list[str] = Field(default_factory=list, description="學習標籤")
    difficulty_level: int = Field(default=1, ge=1, le=5, description="難度級別")


# ============================================================================
# 增強漏洞
# ============================================================================


class EnhancedVulnerability(BaseModel):
    """增強漏洞信息 (整合 AI 分析結果)"""

    vulnerability_id: str = Field(description="漏洞唯一標識")
    title: str = Field(description="漏洞標題")
    description: str = Field(description="漏洞描述")

    # 基本信息
    vulnerability_type: str = Field(description="漏洞類型")
    severity: Literal["low", "medium", "high",
                      "critical"] = Field(description="嚴重性")

    # 位置信息
    url: str = Field(description="漏洞URL")
    parameter: str | None = Field(default=None, description="參數名")
    location: str = Field(description="參數位置")

    # CVSS 評分
    cvss_metrics: CVSSv3Metrics | None = Field(
        default=None, description="CVSS v3.1 指標"
    )

    # AI 分析結果
    ai_confidence: float = Field(ge=0.0, le=1.0, description="AI 置信度")
    ai_risk_assessment: dict[str, Any] = Field(
        default_factory=dict, description="AI 風險評估"
    )
    exploitability_score: float = Field(ge=0.0, le=1.0, description="可利用性分數")

    # 攻擊路徑
    attack_vector: str = Field(description="攻擊向量")
    attack_complexity: str = Field(description="攻擊複雜度")
    prerequisites: list[str] = Field(default_factory=list, description="利用前提")

    # 影響分析
    business_impact: dict[str, Any] = Field(
        default_factory=dict, description="業務影響"
    )
    technical_impact: dict[str, Any] = Field(
        default_factory=dict, description="技術影響"
    )

    # 修復建議
    remediation_effort: str = Field(description="修復難度")
    remediation_priority: int = Field(ge=1, le=5, description="修復優先級")
    fix_recommendations: list[str] = Field(
        default_factory=list, description="修復建議")

    # 驗證信息
    poc_available: bool = Field(default=False, description="是否有概念驗證")
    verified: bool = Field(default=False, description="是否已驗證")
    false_positive_probability: float = Field(
        ge=0.0, le=1.0, description="誤報概率")

    # 時間信息
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_verified_at: datetime | None = Field(
        default=None, description="最後驗證時間")

    # 元數據
    tags: list[str] = Field(default_factory=list, description="標籤")
    references: list[str] = Field(default_factory=list, description="參考資料")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")


# ============================================================================
# SARIF 報告 (v2.1.0)
# ============================================================================


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
    level: Literal["error", "warning", "info",
                   "note"] = Field(description="級別")
    locations: list[SARIFLocation] = Field(description="位置列表")

    # 可選字段
    partial_fingerprints: dict[str, str] = Field(
        default_factory=dict, description="部分指紋"
    )
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFRule(BaseModel):
    """SARIF 規則"""

    id: str = Field(description="規則ID")
    name: str = Field(description="規則名稱")
    short_description: str = Field(description="簡短描述")
    full_description: str | None = Field(default=None, description="完整描述")
    help_uri: str | None = Field(default=None, description="幫助URI")

    # 安全等級
    default_level: Literal["error", "warning", "info", "note"] = Field(
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
    invocations: list[dict[str, Any]] = Field(
        default_factory=list, description="調用信息"
    )
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list, description="工件信息"
    )
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFReport(BaseModel):
    """SARIF v2.1.0 報告"""

    model_config = {
        "protected_namespaces": (),
        "arbitrary_types_allowed": True}

    version: str = Field(default="2.1.0", description="SARIF版本")
    sarif_schema: str = Field(
        default="https://json.schemastore.org/sarif-2.1.0.json",
        description="JSON Schema URL",
        alias="$schema",
    )
    runs: list[SARIFRun] = Field(description="運行列表")

    # 元數據
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


# ============================================================================
# AI 訓練和事件
# ============================================================================


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


# ============================================================================
# RAG 系統
# ============================================================================


class RAGResponsePayload(BaseModel):
    """RAG 查詢響應 - RAG 知識庫查詢的結果"""

    query_id: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_results: int
    avg_similarity: float | None = None
    enhanced_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
