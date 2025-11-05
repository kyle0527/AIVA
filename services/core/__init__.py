"""
Core service module for AIVA.

This module contains core business logic, process orchestration,
and AI system models.
"""

# 從共通模組導入
from ..aiva_common.enums import (
    Severity,
    TaskStatus,
)
from ..aiva_common.schemas import (
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

# 從 AI 模型模組導入
from .ai_models import (
    AIModelDeployCommand,
    AIModelUpdatedEvent,
    AITrainingCompletedPayload,
    AITrainingProgressPayload,
    ModelTrainingConfig,
    ModelTrainingResult,
    ScenarioTestResult,
)

# 從本模組導入核心邏輯模型（僅本地特定擴展）
from .models import (
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

    SystemOrchestration,
    TaskDependency,
    TaskQueue,
    TestStrategy,
    VulnerabilityCorrelation,
)
from .aiva_core.execution_tracer import (
    ExecutionContext,
    ExecutionMonitor,
    ExecutionResult,
    TaskExecutor,
    TraceEntry,
    TraceRecorder,
    TraceType,
)

__all__ = [
    # Core logic models
    "ExecutionContext",
    "ExecutionMonitor",
    "ExecutionResult",
    "TaskExecutor",
    "TraceEntry",
    "TraceRecorder",
    "TraceType",
    # AI models
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "ScenarioTestResult",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
]
