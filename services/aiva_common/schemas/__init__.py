"""
AIVA Common Schemas Package

此套件提供了 AIVA 微服務生態系統中所有資料合約的統一介面。

使用方式:
    from aiva_common.schemas import FindingPayload, ScanStartPayload, MessageHeader

架構說明:
    - base.py: 基礎模型和通用類別
    - messaging.py: 訊息佇列標準信封
    - tasks.py: 各類掃描與功能任務
    - findings.py: 漏洞發現與細節
    - ai.py: AI 相關模型
    - api_testing.py: API 安全測試
    - assets.py: 資產與 EASM
    - risk.py: 風險評估與攻擊路徑
    - telemetry.py: 監控、心跳與遙測
"""

# ==================== 基礎模型 ====================
# ==================== AI 相關 ====================
from .ai import (
    AIExperienceCreatedEvent,
    AIModelDeployCommand,
    AIModelUpdatedEvent,
    AITraceCompletedEvent,
    AITrainingCompletedPayload,
    AITrainingProgressPayload,
    AITrainingStartPayload,
    AttackPlan,
    AttackStep,
    CVSSv3Metrics,
    EnhancedVulnerability,
    ExperienceSample,
    ModelTrainingConfig,
    PlanExecutionMetrics,
    PlanExecutionResult,
    RAGKnowledgeUpdatePayload,
    RAGQueryPayload,
    RAGResponsePayload,
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    SARIFRun,
    SARIFTool,
    TraceRecord,
)

# ==================== 資產管理 ====================
from .assets import (
    AssetInventoryItem,
    AssetLifecyclePayload,
    DiscoveredAsset,
    EASMAsset,
    TechnicalFingerprint,
    VulnerabilityLifecyclePayload,
    VulnerabilityUpdatePayload,
)
from .base import (
    Asset,
    Authentication,
    ExecutionError,
    Fingerprints,
    MessageHeader,
    RateLimit,
    RiskFactor,
    ScanScope,
    Summary,
    TaskDependency,
)

# ==================== Enhanced 版本 ====================
from .enhanced import (
    EnhancedAttackPath,
    EnhancedAttackPathNode,
    EnhancedFindingPayload,
    EnhancedFunctionTaskTarget,
    EnhancedIOCRecord,
    EnhancedRiskAssessment,
    EnhancedScanRequest,
    EnhancedScanScope,
    EnhancedTaskExecution,
    EnhancedVulnerabilityCorrelation,
)

# ==================== 漏洞發現 ====================
from .findings import (
    AIVerificationRequest,
    AIVerificationResult,
    CodeLevelRootCause,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    JavaScriptAnalysisResult,
    SASTDASTCorrelation,
    SensitiveMatch,
    Target,
    Vulnerability,
    VulnerabilityCorrelation,
)

# ==================== 程式語言支援 ====================
from .languages import (
    AILanguageModel,
    CodeQualityReport,
    CrossLanguageAnalysis,
    LanguageDetectionResult,
    LanguageInteroperability,
    LanguageSpecificPayload,
    LanguageSpecificScanConfig,
    LanguageSpecificVulnerability,
    MultiLanguageCodebase,
)

# ==================== 訊息系統 ====================
from .messaging import (
    AIVACommand,
    AIVAEvent,
    AivaMessage,
    AIVARequest,
    AIVAResponse,
)

# ==================== 參考資料 ====================
from .references import (
    CVEReference,
    CWEReference,
    VulnerabilityDiscovery,
)

# ==================== 風險評估 ====================
from .risk import (
    AttackPathEdge,
    AttackPathNode,
    AttackPathPayload,
    AttackPathRecommendation,
    RiskAssessmentContext,
    RiskAssessmentResult,
    RiskTrendAnalysis,
)

# ==================== 系統編排 ====================
from .system import (
    EnhancedModuleStatus,
    ModelTrainingResult,
    SessionState,
    SystemOrchestration,
    TaskQueue,
    WebhookPayload,
)

# ==================== 任務相關 ====================
from .tasks import (
    APISchemaPayload,
    APISecurityTestPayload,
    APITestCase,
    AuthZAnalysisPayload,
    AuthZCheckPayload,
    AuthZResultPayload,
    BizLogicResultPayload,
    BizLogicTestPayload,
    ConfigUpdatePayload,
    EASMDiscoveryPayload,
    EASMDiscoveryResult,
    ExploitPayload,
    ExploitResult,
    FeedbackEventPayload,
    FunctionTaskContext,
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskTestConfig,
    PostExResultPayload,
    PostExTestPayload,
    RemediationGeneratePayload,
    RemediationResultPayload,
    ScanCompletedPayload,
    ScanStartPayload,
    ScenarioTestResult,
    StandardScenario,
    TaskUpdatePayload,
    TestExecution,
    TestStrategy,
    ThreatIntelLookupPayload,
    ThreatIntelResultPayload,
)

# ==================== 遙測與監控 ====================
from .telemetry import (
    FunctionExecutionResult,
    FunctionTelemetry,
    HeartbeatPayload,
    ModuleStatus,
    NotificationPayload,
    OastEvent,
    OastProbe,
    SIEMEvent,
    SIEMEventPayload,
)

# 為了保持向後相容，明確匯出所有公開介面
__all__ = [
    # 基礎模型
    "MessageHeader",
    "Authentication",
    "RateLimit",
    "ScanScope",
    "Asset",
    "Summary",
    "Fingerprints",
    "ExecutionError",
    "RiskFactor",
    "TaskDependency",
    # 訊息系統
    "AivaMessage",
    "AIVARequest",
    "AIVAResponse",
    "AIVAEvent",
    "AIVACommand",
    # 任務相關
    "ScanStartPayload",
    "ScanCompletedPayload",
    "FunctionTaskPayload",
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "FeedbackEventPayload",
    "TaskUpdatePayload",
    "ConfigUpdatePayload",
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "AuthZCheckPayload",
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    "RemediationGeneratePayload",
    "RemediationResultPayload",
    "PostExTestPayload",
    "PostExResultPayload",
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    "EASMDiscoveryPayload",
    "EASMDiscoveryResult",
    "StandardScenario",
    "ScenarioTestResult",
    "ExploitPayload",
    "TestExecution",
    "ExploitResult",
    "TestStrategy",
    # 漏洞發現
    "Vulnerability",
    "Target",
    "FindingTarget",  # 別名
    "FindingEvidence",
    "FindingImpact",
    "FindingRecommendation",
    "FindingPayload",
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
    "VulnerabilityCorrelation",
    "CodeLevelRootCause",
    "SASTDASTCorrelation",
    "AIVerificationRequest",
    "AIVerificationResult",
    # 遙測與監控
    "HeartbeatPayload",
    "ModuleStatus",
    "FunctionTelemetry",
    "FunctionExecutionResult",
    "OastEvent",
    "OastProbe",
    "SIEMEventPayload",
    "SIEMEvent",
    "NotificationPayload",
    # AI 相關
    "CVSSv3Metrics",
    "AttackStep",
    "AttackPlan",
    "TraceRecord",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    "ModelTrainingConfig",
    "AITrainingStartPayload",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    "AIExperienceCreatedEvent",
    "AITraceCompletedEvent",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
    "ExperienceSample",
    "EnhancedVulnerability",
    "SARIFLocation",
    "SARIFResult",
    "SARIFRule",
    "SARIFTool",
    "SARIFRun",
    "SARIFReport",
    # 資產管理
    "AssetLifecyclePayload",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload",
    "DiscoveredAsset",
    "TechnicalFingerprint",
    "AssetInventoryItem",
    "EASMAsset",
    # 風險評估
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "RiskTrendAnalysis",
    "AttackPathNode",
    "AttackPathEdge",
    "AttackPathPayload",
    "AttackPathRecommendation",
    # Enhanced 版本
    "EnhancedFindingPayload",
    "EnhancedScanScope",
    "EnhancedScanRequest",
    "EnhancedFunctionTaskTarget",
    "EnhancedIOCRecord",
    "EnhancedRiskAssessment",
    "EnhancedAttackPathNode",
    "EnhancedAttackPath",
    "EnhancedTaskExecution",
    "EnhancedModuleStatus",
    "EnhancedVulnerabilityCorrelation",
    # 系統編排
    "SessionState",
    "ModelTrainingResult",
    "TaskQueue",
    "SystemOrchestration",
    "WebhookPayload",
    # 參考資料
    "CVEReference",
    "CWEReference",
    "VulnerabilityDiscovery",
    # 程式語言支援
    "LanguageDetectionResult",
    "LanguageSpecificVulnerability",
    "MultiLanguageCodebase",
    "LanguageSpecificScanConfig",
    "CrossLanguageAnalysis",
    "LanguageSpecificPayload",
    "AILanguageModel",
    "CodeQualityReport",
    "LanguageInteroperability",
]

# 版本資訊
__version__ = "2.1.0"
__schema_version__ = "1.0"
