"""
Core service module for AIVA.

This module contains core business logic, process orchestration,
and AI system models.
"""

# 注意：共通模組導入已移除，如需要請確認正確路徑

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
from .aiva_core.execution import (
    UnifiedTracer,
    TraceType,
    ExecutionTrace,
    get_global_tracer,
    record_execution_trace,
    # 向後相容性別名
    TraceRecorder,
    TraceLogger,
)

__all__ = [
    # Execution models - 統一追蹤器
    "UnifiedTracer",
    "TraceType", 
    "ExecutionTrace",
    "get_global_tracer",
    "record_execution_trace",
    # 向後相容性
    "TraceRecorder",
    "TraceLogger",
    # AI models
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "ScenarioTestResult",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
]
