"""
AI 核心系統相關 schemas

包含 BioNeuronRAGAgent 和強化學習系統的數據結構
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# CVSS v3.1 標準
# ============================================================================

class CVSSv3Metrics(BaseModel):
    """CVSS v3.1 標準指標"""

    # Base Metrics
    attack_vector: Literal["N", "A", "L", "P"] = Field(description="攻擊向量 Network/Adjacent/Local/Physical")
    attack_complexity: Literal["L", "H"] = Field(description="攻擊複雜度 Low/High")
    privileges_required: Literal["N", "L", "H"] = Field(description="所需權限 None/Low/High")
    user_interaction: Literal["N", "R"] = Field(description="用戶交互 None/Required")
    scope: Literal["U", "C"] = Field(description="範圍 Unchanged/Changed")
    confidentiality: Literal["N", "L", "H"] = Field(description="機密性影響 None/Low/High")
    integrity: Literal["N", "L", "H"] = Field(description="完整性影響 None/Low/High")
    availability: Literal["N", "L", "H"] = Field(description="可用性影響 None/Low/High")

    # Temporal Metrics (Optional)
    exploit_code_maturity: Literal["X", "H", "F", "P", "U"] | None = Field(
        default=None, description="漏洞利用代碼成熟度"
    )
    remediation_level: Literal["X", "U", "W", "T", "O"] | None = Field(
        default=None, description="修復級別"
    )
    report_confidence: Literal["X", "C", "R", "U"] | None = Field(
        default=None, description="報告置信度"
    )

    # Environmental Metrics (Optional)
    confidentiality_requirement: Literal["X", "L", "M", "H"] | None = Field(
        default=None, description="機密性要求"
    )
    integrity_requirement: Literal["X", "L", "M", "H"] | None = Field(
        default=None, description="完整性要求"
    )
    availability_requirement: Literal["X", "L", "M", "H"] | None = Field(
        default=None, description="可用性要求"
    )

    # Calculated Scores
    base_score: float | None = Field(default=None, ge=0.0, le=10.0, description="基本分數")
    temporal_score: float | None = Field(default=None, ge=0.0, le=10.0, description="時間分數")
    environmental_score: float | None = Field(default=None, ge=0.0, le=10.0, description="環境分數")
    vector_string: str | None = Field(default=None, description="CVSS 向量字符串")


# ============================================================================
# 攻擊計劃和步驟
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
    status: Literal["pending", "running", "completed", "failed", "skipped"] = Field(
        default="pending", description="執行狀態"
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
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="可並行執行的步驟組"
    )

    # 計劃屬性
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    estimated_duration: int = Field(default=300, ge=1, description="預估執行時間(秒)")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", description="風險級別"
    )

    # 執行狀態
    status: Literal["draft", "ready", "running", "completed", "failed", "cancelled"] = Field(
        default="draft", description="計劃狀態"
    )

    # 時間戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 元數據
    tags: list[str] = Field(default_factory=list, description="標籤")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")


# ============================================================================
# 執行追蹤
# ============================================================================

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
    ai_model_name: str = Field(description="模型名稱")
    ai_model_version: str = Field(description="模型版本")

    # 訓練參數
    learning_rate: float = Field(default=0.001, gt=0.0, description="學習率")
    batch_size: int = Field(default=32, ge=1, description="批次大小")
    epochs: int = Field(default=100, ge=1, description="訓練輪數")

    # 網絡結構
    hidden_layers: list[int] = Field(default_factory=lambda: [512, 256, 128], description="隱藏層大小")
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
    severity: Literal["low", "medium", "high", "critical"] = Field(description="嚴重性")

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
    level: Literal["error", "warning", "info", "note"] = Field(description="級別")
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
    invocations: list[dict[str, Any]] = Field(default_factory=list, description="調用信息")
    artifacts: list[dict[str, Any]] = Field(default_factory=list, description="工件信息")
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")


class SARIFReport(BaseModel):
    """SARIF v2.1.0 報告"""

    model_config = {
        "protected_namespaces": (),
        "arbitrary_types_allowed": True
    }

    version: str = Field(default="2.1.0", description="SARIF版本")
    sarif_schema: str = Field(
        default="https://json.schemastore.org/sarif-2.1.0.json",
        description="JSON Schema URL",
        alias="$schema"
    )
    runs: list[SARIFRun] = Field(description="運行列表")

    # 元數據
    properties: dict[str, Any] = Field(default_factory=dict, description="屬性")
