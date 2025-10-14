"""
Core service module for AIVA.

This module contains core business logic, process orchestration,
and AI system models.
"""

from __future__ import annotations

# 從共通模組導入
from ..aiva_common.enums import (
    Severity,
    TaskStatus,
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

# 從本模組導入核心邏輯模型
from .models import (
    AttackPathEdge,
    AttackPathNode,
    AttackPathPayload,
    AttackPathRecommendation,
    CodeLevelRootCause,
    ConfigUpdatePayload,
    EnhancedAttackPath,
    EnhancedAttackPathNode,
    EnhancedFindingPayload,
    EnhancedModuleStatus,
    EnhancedRiskAssessment,
    EnhancedTaskExecution,
    EnhancedVulnerability,
    EnhancedVulnerabilityCorrelation,
    FeedbackEventPayload,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
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
    Target,
    TaskDependency,
    TaskQueue,
    TaskUpdatePayload,
    TestStrategy,
    VulnerabilityCorrelation,
)

__all__ = [
    # Core logic models
    "ExecutionContext",
    "ExecutionStep",
    "ExecutionPipeline",
    "ValidationResult",
    "ProcessDefinition",
    "ProcessInstance",
    "ProcessStep",
    "ConditionalStep",
    "ParallelStep",
    "LoopStep",
    "ProcessState",
    "StepState",
    "ServiceInstanceInfo",
    "ServiceHealthInfo",
    "ServiceMetrics",
    "ServiceLogs",
    "ServiceConfiguration",
    "SystemInfo",
    "SystemHealth",
    "HealthCheck",
    "ResourceUsage",
    "SystemMonitoring",
    "SystemMetrics",
    "ServiceStatus",
    "Notification",
    "NotificationHistory",
    # AI models
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "ScenarioTestResult",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
]
