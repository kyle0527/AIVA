"""
AIVA Core - 核心引擎模組

這是 AIVA 的核心處理引擎，負責協調掃描結果處理、
測試策略生成、任務分發和執行狀態監控。

核心功能:
- bio_neuron_master: 生物神經網絡主控制器
- ai_engine: AI 引擎和策略生成
- attack: 攻擊執行和漏洞利用
- storage: 數據存儲和持久化
- analysis: 風險分析和評估
- execution: 任務執行和調度

新增核心組件 (附件要求實現):
- dialog: 對話助理 - AI 對話層，支援自然語言問答和一鍵執行
- decision: 技能圖 - 能力關係映射和決策支援
- learning: 能力評估器 - 訓練探索和學習反饋機制
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
# 從 aiva_common.schemas 導入共享標準模型
from services.aiva_common.schemas import (
    ConfigUpdatePayload,
    FeedbackEventPayload,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    HeartbeatPayload,
    ModuleStatus,
    RemediationGeneratePayload,
    RemediationResultPayload,
    Target,
    TaskUpdatePayload,
)

# 從 services.core.models 導入核心擴展模型
from services.core.models import (
    AttackPathEdge,
    AttackPathNode,
    AttackPathPayload,
    AttackPathRecommendation,
    CodeLevelRootCause,
    EnhancedAttackPath,
    EnhancedAttackPathNode,
    EnhancedFindingPayload,
    EnhancedModuleStatus,
    EnhancedRiskAssessment,
    EnhancedTaskExecution,
    EnhancedVulnerability,
    EnhancedVulnerabilityCorrelation,
    RiskAssessmentContext,
    RiskAssessmentResult,
    RiskFactor,
    RiskTrendAnalysis,
    SASTDASTCorrelation,
    SystemOrchestration,
    TaskDependency,
    TaskQueue,
    TestStrategy,
    VulnerabilityCorrelation,
)

# 從新遷移的核心服務組件導入 (從 aiva_core_v2 遷移而來)
from .command_router import (
    CommandContext,
    CommandRouter,
    CommandType,
    ExecutionMode,
    ExecutionResult,
    get_command_router,
)
from .context_manager import ContextManager, get_context_manager
from .core_service_coordinator import (
    AIVACoreServiceCoordinator,
    get_core_service_coordinator,
    initialize_core_module,
    process_command,
    shutdown_core_module,
)
from .decision.skill_graph import AIVASkillGraph, skill_graph

# 從新增核心組件導入 (附件要求實現)
from .dialog.assistant import AIVADialogAssistant, dialog_assistant
from .execution_planner import ExecutionPlanner, get_execution_planner

# capability_evaluator 現在使用 aiva_common.ai.capability_evaluator 統一實現






__all__ = [
    # 新增核心組件 (附件要求實現)
    "AIVADialogAssistant",
    "dialog_assistant",
    "AIVASkillGraph",
    "skill_graph",
    # capability_evaluator 已移至 aiva_common.ai
    # 從 aiva_core_v2 遷移的核心服務組件
    "CommandRouter",
    "get_command_router",
    "CommandType",
    "ExecutionMode",
    "CommandContext",
    "ExecutionResult",
    "ContextManager",
    "get_context_manager",
    "ExecutionPlanner",
    "get_execution_planner",
    "AIVACoreServiceCoordinator",
    "get_core_service_coordinator",
    "process_command",
    "initialize_core_module",
    "shutdown_core_module",
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
    "TaskStatus",
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
