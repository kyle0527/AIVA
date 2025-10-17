"""
AIVA Core - 核心引擎模組

這是 AIVA 的核心處理引擎，負責協調掃描結果處理、
測試策略生成、任務分發和執行狀態監控。

核心功能:
- bio_neuron_master: 生物神經網絡主控制器
- ai_engine: AI 引擎和策略生成
- storage: 數據存儲和持久化
- analysis: 風險分析和評估
- execution: 任務執行和調度
"""

__version__ = "1.0.0"

# 從 aiva_common 導入共享基礎設施
from services.aiva_common.enums import (
    ComplianceFramework,
    Confidence,
    ModuleName,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    Severity,
    TaskStatus,
    Topic,
)
from services.aiva_common.schemas import CVEReference, CVSSv3Metrics, CWEReference

# 從 core.ai_models 導入 AI 系統相關模型
from services.core.ai_models import (
    AIExperienceCreatedEvent,
    AIModelDeployCommand,
    AIModelUpdatedEvent,
    AITraceCompletedEvent,
    AITrainingCompletedPayload,
    AITrainingProgressPayload,
    AITrainingStartPayload,
    AIVACommand,
    AIVAEvent,
    AIVARequest,
    AIVAResponse,
    AIVerificationRequest,
    AIVerificationResult,
    AttackPlan,
    AttackStep,
    ExperienceSample,
    ModelTrainingConfig,
    ModelTrainingResult,
    PlanExecutionMetrics,
    PlanExecutionResult,
    RAGKnowledgeUpdatePayload,
    RAGQueryPayload,
    RAGResponsePayload,
    ScenarioTestResult,
    SessionState,
    StandardScenario,
    TraceRecord,
)

# 從 core.models 導入核心業務邏輯模型
from services.core.models import (
    AttackPathEdge,
    AttackPathNode,
    AttackPathPayload,
    AttackPathRecommendation,
    CodeLevelRootCause,
    ConfigUpdatePayload,
    EnhancedAttackPath,
    EnhancedAttackPathNode,
    EnhancedModuleStatus,
    EnhancedRiskAssessment,
    EnhancedTaskExecution,
    EnhancedVulnerability,
    EnhancedVulnerabilityCorrelation,
    FeedbackEventPayload,
    HeartbeatPayload,
    ModuleStatus,
    RemediationGeneratePayload,
    RemediationResultPayload,
    RiskAssessmentContext,
    RiskAssessmentResult,
    RiskFactor,
    RiskTrendAnalysis,
    SASTDASTCorrelation,
    SystemOrchestration,
    TaskDependency,
    TaskQueue,
    TaskUpdatePayload,
    TestStrategy,
    VulnerabilityCorrelation,
)

# 從 aiva_common.schemas 導入共用 schemas
from services.aiva_common.schemas import (
    EnhancedFindingPayload,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    Target,
)

__all__ = [
    # 來自 aiva_common
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    "ComplianceFramework",
    "Confidence",
    "ModuleName",
    "RemediationStatus",
    "RemediationType",
    "RiskLevel",
    "Severity",
    "TestStatus",
    "Topic",
    # AI 系統模型
    "AIExperienceCreatedEvent",
    "AIModelDeployCommand",
    "AIModelUpdatedEvent",
    "AITraceCompletedEvent",
    "AITrainingCompletedPayload",
    "AITrainingProgressPayload",
    "AITrainingStartPayload",
    "AIVACommand",
    "AIVAEvent",
    "AIVARequest",
    "AIVAResponse",
    "AIVerificationRequest",
    "AIVerificationResult",
    "AttackPlan",
    "AttackStep",
    "ExperienceSample",
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
    "ScenarioTestResult",
    "SessionState",
    "StandardScenario",
    "TraceRecord",
    # 核心業務邏輯模型
    "AttackPathEdge",
    "AttackPathNode",
    "AttackPathPayload",
    "AttackPathRecommendation",
    "CodeLevelRootCause",
    "ConfigUpdatePayload",
    "EnhancedAttackPath",
    "EnhancedAttackPathNode",
    "EnhancedFindingPayload",
    "EnhancedModuleStatus",
    "EnhancedRiskAssessment",
    "EnhancedTaskExecution",
    "EnhancedVulnerability",
    "EnhancedVulnerabilityCorrelation",
    "FeedbackEventPayload",
    "FindingEvidence",
    "FindingImpact",
    "FindingPayload",
    "FindingRecommendation",
    "HeartbeatPayload",
    "ModuleStatus",
    "RemediationGeneratePayload",
    "RemediationResultPayload",
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "RiskFactor",
    "RiskTrendAnalysis",
    "SASTDASTCorrelation",
    "SystemOrchestration",
    "Target",
    "TaskDependency",
    "TaskQueue",
    "TaskUpdatePayload",
    "TestStrategy",
    "VulnerabilityCorrelation",
]

