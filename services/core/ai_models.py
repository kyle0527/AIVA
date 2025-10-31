"""
AIVA AI System Models - AI智能系統模組

此文件包含AIVA AI子系統相關的所有數據模型。

職責範圍：
1. AI驗證系統 (AIVerification)
2. AI訓練系統 (AITraining)
3. AI攻擊規劃 (AttackPlan)
4. AI追蹤記錄 (TraceRecord)
5. AI經驗學習 (ExperienceSample)
6. RAG知識庫 (RAG Knowledge Base)
7. AIVA核心接口 (AIVA Core Interface)
8. 模型管理 (Model Management)
9. 場景測試 (Scenario Testing)
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ..aiva_common.enums import (
    Confidence,
    ModuleName,
    RiskLevel,
    Severity,
    TestStatus,
)

# ==================== AI 驗證系統 ====================


class AIVerificationRequest(BaseModel):
    """AI驗證請求"""

    request_id: str = Field(description="請求ID")
    verification_type: str = Field(description="驗證類型")
    target_data: dict[str, Any] = Field(description="目標數據")
    context: dict[str, Any] = Field(default_factory=dict, description="上下文")
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="置信度閾值"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIVerificationResult(BaseModel):
    """AI驗證結果"""

    request_id: str = Field(description="請求ID")
    verified: bool = Field(description="驗證結果")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    details: dict[str, Any] = Field(description="詳細信息")
    recommendations: list[str] = Field(default_factory=list, description="建議")
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AI 訓練系統 ====================


class AITrainingStartPayload(BaseModel):
    """AI訓練開始載荷"""

    model_config = {"protected_namespaces": ()}

    training_id: str = Field(description="訓練ID")
    model_type: str = Field(description="模型類型")
    training_config: dict[str, Any] = Field(description="訓練配置")
    dataset_info: dict[str, Any] = Field(description="數據集信息")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AITrainingProgressPayload(BaseModel):
    """AI訓練進度載荷"""

    model_config = {"protected_namespaces": ()}

    training_id: str = Field(description="訓練ID")
    progress: float = Field(ge=0.0, le=1.0, description="進度")
    current_epoch: int = Field(ge=0, description="當前輪次")
    total_epochs: int = Field(ge=1, description="總輪次")
    metrics: dict[str, Any] = Field(description="指標")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AITrainingCompletedPayload(BaseModel):
    """AI訓練完成載荷"""

    model_config = {"protected_namespaces": ()}

    training_id: str = Field(description="訓練ID")
    success: bool = Field(description="是否成功")
    final_metrics: dict[str, Any] = Field(description="最終指標")
    model_path: str | None = Field(default=None, description="模型路徑")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AI 事件系統 ====================


class AIExperienceCreatedEvent(BaseModel):
    """AI經驗創建事件"""

    experience_id: str = Field(description="經驗ID")
    experience_type: str = Field(description="經驗類型")
    source_task: str = Field(description="來源任務")
    outcome: str = Field(description="結果")
    learned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AITraceCompletedEvent(BaseModel):
    """AI追蹤完成事件"""

    trace_id: str = Field(description="追蹤ID")
    trace_type: str = Field(description="追蹤類型")
    duration_seconds: int = Field(ge=0, description="持續時間(秒)")
    result_summary: dict[str, Any] = Field(description="結果摘要")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIModelUpdatedEvent(BaseModel):
    """AI模型更新事件"""

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(description="模型ID")
    model_type: str = Field(description="模型類型")
    version: str = Field(description="版本")
    update_reason: str = Field(description="更新原因")
    performance_delta: dict[str, Any] = Field(description="性能變化")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIModelDeployCommand(BaseModel):
    """AI模型部署命令"""

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(description="模型ID")
    deployment_target: str = Field(description="部署目標")
    deployment_config: dict[str, Any] = Field(description="部署配置")
    rollback_enabled: bool = Field(default=True, description="是否啟用回滾")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AI 攻擊規劃 ====================


class AttackStep(BaseModel):
    """攻擊步驟"""

    step_id: str = Field(description="步驟ID")
    step_name: str = Field(description="步驟名稱")
    description: str = Field(description="步驟描述")
    technique: str | None = Field(default=None, description="技術")
    expected_outcome: str = Field(description="預期結果")
    success_criteria: list[str] = Field(default_factory=list, description="成功標準")
    fallback_options: list[str] = Field(default_factory=list, description="備用方案")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AttackPlan(BaseModel):
    """攻擊計劃"""

    plan_id: str = Field(description="計劃ID")
    target: str = Field(description="目標")
    objective: str = Field(description="目標")
    steps: list[AttackStep] = Field(description="攻擊步驟")
    estimated_duration: int = Field(ge=0, description="預計持續時間(分鐘)")
    risk_level: RiskLevel = Field(description="風險等級")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class PlanExecutionMetrics(BaseModel):
    """計劃執行指標"""

    execution_id: str = Field(description="執行ID")
    plan_id: str = Field(description="計劃ID")

    # 執行統計
    total_steps: int = Field(ge=0, description="總步驟數")
    completed_steps: int = Field(ge=0, description="完成步驟數")
    failed_steps: int = Field(ge=0, description="失敗步驟數")

    # 時間統計
    started_at: datetime = Field(description="開始時間")
    completed_at: datetime | None = Field(default=None, description="完成時間")
    total_duration: int | None = Field(default=None, ge=0, description="總時長(秒)")

    # 成功率
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class PlanExecutionResult(BaseModel):
    """計劃執行結果"""

    execution_id: str = Field(description="執行ID")
    plan_id: str = Field(description="計劃ID")
    success: bool = Field(description="是否成功")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="發現列表")
    metrics: PlanExecutionMetrics = Field(description="執行指標")
    lessons_learned: list[str] = Field(default_factory=list, description="經驗教訓")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AI 追蹤記錄 ====================


class TraceRecord(BaseModel):
    """追蹤記錄"""

    trace_id: str = Field(description="追蹤ID")
    trace_type: str = Field(description="追蹤類型")
    source_module: ModuleName = Field(description="來源模組")

    # 追蹤數據
    operation: str = Field(description="操作")
    input_data: dict[str, Any] = Field(description="輸入數據")
    output_data: dict[str, Any] = Field(description="輸出數據")

    # 性能數據
    duration_ms: int = Field(ge=0, description="持續時間(毫秒)")
    cpu_time_ms: int | None = Field(default=None, ge=0, description="CPU時間(毫秒)")
    memory_used_mb: float | None = Field(
        default=None, ge=0.0, description="內存使用(MB)"
    )

    # 結果狀態
    status: TestStatus = Field(description="狀態")
    error_message: str | None = Field(default=None, description="錯誤消息")

    # 時間戳
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AI 經驗學習 ====================


class ExperienceSample(BaseModel):
    """經驗樣本"""

    sample_id: str = Field(description="樣本ID")
    experience_type: str = Field(description="經驗類型")

    # 上下文信息
    scenario: str = Field(description="場景")
    context: dict[str, Any] = Field(description="上下文")

    # 行為和結果
    action_taken: str = Field(description="採取的行動")
    outcome: str = Field(description="結果")
    success: bool = Field(description="是否成功")

    # 學習價值
    learning_value: float = Field(ge=0.0, le=1.0, description="學習價值")
    relevance_score: float = Field(ge=0.0, le=1.0, description="相關性評分")

    # 時間信息
    learned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage_count: int = Field(ge=0, description="使用次數")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class SessionState(BaseModel):
    """會話狀態"""

    session_id: str = Field(description="會話ID")
    session_type: str = Field(description="會話類型")

    # 會話信息
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = Field(default=True, description="是否活躍")

    # 會話上下文
    context: dict[str, Any] = Field(default_factory=dict, description="上下文")
    accumulated_data: dict[str, Any] = Field(
        default_factory=dict, description="累積數據"
    )

    # 會話統計
    operation_count: int = Field(ge=0, description="操作次數")
    success_count: int = Field(ge=0, description="成功次數")
    error_count: int = Field(ge=0, description="錯誤次數")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 模型管理 ====================


class ModelTrainingConfig(BaseModel):
    """模型訓練配置"""

    model_config = {"protected_namespaces": ()}

    config_id: str = Field(description="配置ID")
    model_type: str = Field(description="模型類型")

    # 訓練參數
    learning_rate: float = Field(gt=0.0, description="學習率")
    batch_size: int = Field(ge=1, description="批次大小")
    epochs: int = Field(ge=1, description="訓練輪數")

    # 優化器配置
    optimizer: str = Field(description="優化器")
    loss_function: str = Field(description="損失函數")

    # 數據配置
    training_data_path: str = Field(description="訓練數據路徑")
    validation_data_path: str | None = Field(default=None, description="驗證數據路徑")
    test_data_path: str | None = Field(default=None, description="測試數據路徑")

    # 早停配置
    early_stopping: bool = Field(default=True, description="是否使用早停")
    patience: int = Field(default=5, ge=1, description="耐心值")

    # 檢查點配置
    checkpoint_interval: int = Field(default=1, ge=1, description="檢查點間隔")
    save_best_only: bool = Field(default=True, description="只保存最佳模型")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ModelTrainingResult(BaseModel):
    """模型訓練結果"""

    model_config = {"protected_namespaces": ()}

    training_id: str = Field(description="訓練ID")
    model_type: str = Field(description="模型類型")

    # 訓練狀態
    success: bool = Field(description="是否成功")
    epochs_completed: int = Field(ge=0, description="完成輪數")

    # 性能指標
    final_loss: float = Field(description="最終損失")
    final_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="最終準確率"
    )
    best_epoch: int | None = Field(default=None, ge=0, description="最佳輪數")

    # 驗證指標
    validation_loss: float | None = Field(default=None, description="驗證損失")
    validation_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="驗證準確率"
    )

    # 模型信息
    model_path: str | None = Field(default=None, description="模型保存路徑")
    model_size_mb: float | None = Field(
        default=None, ge=0.0, description="模型大小(MB)"
    )

    # 時間信息
    started_at: datetime = Field(description="開始時間")
    completed_at: datetime = Field(description="完成時間")
    total_duration_seconds: int = Field(ge=0, description="總時長(秒)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 場景測試 ====================


class StandardScenario(BaseModel):
    """標準場景"""

    scenario_id: str = Field(description="場景ID")
    scenario_name: str = Field(description="場景名稱")
    description: str = Field(description="場景描述")

    # 場景配置
    target_type: str = Field(description="目標類型")
    vulnerability_types: list[str] = Field(description="漏洞類型列表")
    attack_vectors: list[str] = Field(description="攻擊向量列表")

    # 測試步驟
    test_steps: list[dict[str, Any]] = Field(description="測試步驟")
    success_criteria: list[str] = Field(description="成功標準")

    # 難度和風險
    difficulty_level: int = Field(ge=1, le=10, description="難度等級")
    risk_level: RiskLevel = Field(description="風險等級")

    # 預期結果
    expected_findings: list[str] = Field(default_factory=list, description="預期發現")
    expected_severity: Severity = Field(description="預期嚴重程度")

    # 使用統計
    usage_count: int = Field(ge=0, description="使用次數")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ScenarioTestResult(BaseModel):
    """場景測試結果"""

    model_config = {"protected_namespaces": ()}

    test_id: str = Field(description="測試ID")
    scenario_id: str = Field(description="場景ID")

    # 測試狀態
    status: TestStatus = Field(description="測試狀態")
    success: bool = Field(description="是否成功")

    # 發現結果
    findings_count: int = Field(ge=0, description="發現數量")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="發現列表")

    # 性能指標
    execution_time_seconds: int = Field(ge=0, description="執行時間(秒)")
    steps_completed: int = Field(ge=0, description="完成步驟數")
    total_steps: int = Field(ge=1, description="總步驟數")

    # 評估結果
    accuracy: float = Field(ge=0.0, le=1.0, description="準確率")
    confidence: Confidence = Field(description="置信度")

    # 時間信息
    started_at: datetime = Field(description="開始時間")
    completed_at: datetime = Field(description="完成時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== RAG 知識庫 ====================


class RAGKnowledgeUpdatePayload(BaseModel):
    """RAG知識更新載荷"""

    update_id: str = Field(description="更新ID")
    knowledge_type: str = Field(description="知識類型")
    content: str = Field(description="內容")
    source: str = Field(description="來源")
    tags: list[str] = Field(default_factory=list, description="標籤")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class RAGQueryPayload(BaseModel):
    """RAG查詢載荷"""

    query_id: str = Field(description="查詢ID")
    query_text: str = Field(description="查詢文本")
    context: dict[str, Any] = Field(default_factory=dict, description="上下文")
    max_results: int = Field(default=5, ge=1, le=20, description="最大結果數")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class RAGResponsePayload(BaseModel):
    """RAG響應載荷"""

    query_id: str = Field(description="查詢ID")
    results: list[dict[str, Any]] = Field(description="結果列表")
    total_matches: int = Field(ge=0, description="總匹配數")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== AIVA 核心接口 ====================


class AIVARequest(BaseModel):
    """AIVA請求"""

    request_id: str = Field(description="請求ID")
    request_type: str = Field(description="請求類型")
    payload: dict[str, Any] = Field(description="載荷")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    timeout: int = Field(default=300, ge=10, description="超時時間(秒)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIVAResponse(BaseModel):
    """AIVA響應"""

    request_id: str = Field(description="請求ID")
    success: bool = Field(description="是否成功")
    result: dict[str, Any] = Field(description="結果")
    error_message: str | None = Field(default=None, description="錯誤消息")
    processing_time_ms: int = Field(ge=0, description="處理時間(毫秒)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIVAEvent(BaseModel):
    """AIVA事件"""

    event_id: str = Field(description="事件ID")
    event_type: str = Field(description="事件類型")
    source: str = Field(description="來源")
    payload: dict[str, Any] = Field(description="載荷")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AIVACommand(BaseModel):
    """AIVA命令"""

    command_id: str = Field(description="命令ID")
    command_type: str = Field(description="命令類型")
    target_module: ModuleName = Field(description="目標模組")
    parameters: dict[str, Any] = Field(description="參數")
    execute_at: datetime | None = Field(default=None, description="執行時間")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


__all__ = [
    # AI驗證
    "AIVerificationRequest",
    "AIVerificationResult",
    # AI訓練
    "AITrainingStartPayload",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    # AI事件
    "AIExperienceCreatedEvent",
    "AITraceCompletedEvent",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
    # 攻擊規劃
    "AttackStep",
    "AttackPlan",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    # 追蹤記錄
    "TraceRecord",
    # 經驗學習
    "ExperienceSample",
    "SessionState",
    # 模型管理
    "ModelTrainingConfig",
    "ModelTrainingResult",
    # 場景測試
    "StandardScenario",
    "ScenarioTestResult",
    # RAG知識庫
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
    # AIVA核心
    "AIVARequest",
    "AIVAResponse",
    "AIVAEvent",
    "AIVACommand",
]
