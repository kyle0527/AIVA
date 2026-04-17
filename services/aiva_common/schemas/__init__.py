"""
AIVA Common Schemas Package - Domain-Driven Design (DDD) Architecture

此套件採用領域驅動設計，將schemas按業務功能分組：

🏗️ 架構說明:
    - _base/: 核心基礎設施 (所有領域依賴)
    - analysis/: 分析引擎領域 (代碼分析、AI分析)  
    - security/: 安全檢測領域 (漏洞發現、威脅情報)
    - testing/: 測試執行領域 (API測試、任務執行)
    - infrastructure/: 基礎設施領域 (資產、遙測、系統編排)
    - interfaces/: 外部接口領域 (API標準、CLI、異步工具) 
    - risk/: 風險評估領域 (風險分析、攻擊路徑)

📦 領域依賴關係:
    _base ← domains ← interfaces
    (避免循環依賴，單向依賴流)

[Backward] 向後相容性:
    完全保持原有API，現有代碼無需修改

使用方式:
    from aiva_common.schemas import FindingPayload, ScanStartPayload, MessageHeader
"""

# ==================== 核心基礎設施 ====================
# 注意：_base/common.py 已移除重複，統一使用 base.py 中的定義

# ==================== AI 指令系統 ====================
# ==================== AI 相關 ====================
# AI模組導入已重構為使用TYPE_CHECKING模式，遵循PEP-484標準
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
    AttackTarget,  # 添加缺失的 AttackTarget
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

# ==================== 分析引擎領域 ====================
from .analysis import (
    AILanguageModel,
    AnalysisType,
    BaseAnalysisResult,
    CodeQualityReport,
    CrossLanguageAnalysis,
    DataLeak,
    JavaScriptAnalysisResult,
    LanguageDetectionResult,
    LanguageInteroperability,
    LanguageSpecificPayload,
    LanguageSpecificScanConfig,
    LanguageSpecificVulnerability,
    LegacyJavaScriptAnalysisResultAdapter,
    MultiLanguageCodebase,
)

# ==================== API 標準 (OpenAPI/AsyncAPI/GraphQL) ====================
from .api_standards import (
    APISecurityTest,

    AsyncAPIChannel,
    AsyncAPIDocument,
    AsyncAPIInfo,
    AsyncAPIMessage,
    GraphQLDirectiveDefinition,
    GraphQLFieldDefinition,
    GraphQLSchema,
    GraphQLTypeDefinition,
    OpenAPIComponents,
    OpenAPIDocument,
    OpenAPIInfo,
    OpenAPIOperation,
    OpenAPIParameter,
    OpenAPIPathItem,
    OpenAPISchema,
    OpenAPISecurityScheme,
    OpenAPIServer,
)
from .api_standards import (
    AsyncAPIComponents as AsyncComponents,  # OpenAPI 3.1; AsyncAPI 3.0; GraphQL; API 安全測試
)
from .api_standards import AsyncAPIOperation as AsyncOperation
from .api_standards import AsyncAPIServer as AsyncServer

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

# ==================== 異步工具 ====================
from .async_utils import (
    AsyncBatchConfig,
    AsyncBatchResult,
    AsyncTaskConfig,
    AsyncTaskResult,
    ResourceLimits,
    RetryConfig,
)

# ==================== 基礎模型 ====================
# ==================== 核心基礎設施（統一來源）====================
from .base import (
    APIResponse,
    Asset,
    Authentication,
    ExecutionError,
    Fingerprints,
    MessageHeader,
    RateLimit,
    RiskFactor,
    ScanScope,
    Summary,
    Task,
    TaskDependency,
)

# ==================== 能力管理 ====================
from .capability import (
    ExecutableCapability,
    ExecutableParameter,
    ParameterType,
)

# ==================== CLI 界面 ====================
from .cli import (
    CLICommand,
    CLIConfiguration,
    CLIExecutionResult,
    CLIMetrics,
    CLIParameter,
    CLISession,
)
from .commands import (
    AICommand,
    AICommandBatch,
    AICommandBatchResult,
    AICommandResult,
    CommandContext,
    CommandPriority,
    CommandStatus,
    CommandType,
)

# ==================== 決策數據合約 ====================
from .decision import (
    DecisionConstraints,
    DecisionFeedback,
    DecisionToASTContract,
    HighLevelIntent,
    IntentType,
    TargetInfo,
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
    ExternalToolFinding,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    # SASTDASTCorrelation,  # Removed - external SAST not needed
    SensitiveMatch,
    Target,
    Vulnerability,
    VulnerabilityCorrelation,
    VulnerabilityScorecard,
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

# ==================== 訊息處理 ====================
from .messaging import (
    AIVACommand,
    AIVAEvent,
    AivaMessage,
    AIVARequest,
    AIVAResponse,
)

# ==================== 低價值高概率漏洞 ====================
# 暫時註解掉缺少的模組導入
# from .low_value_vulnerabilities import (  # 低價值漏洞相關模型
#     BountyPrediction,
#     BugBountyStrategy,
#     ClickjackingPattern,
#     CORSMisconfigurationPattern,
#     CSRFJSONBypass,
#     CSRFMissingToken,
#     CSRFPattern,
#     DebugInfoDisclosure,
#     DOMXSSSimple,
#     ErrorMessageDisclosure,
#     HostHeaderInjectionPattern,
#     IDORPattern,
#     IDORSimpleID,
#     IDORUserData,
#     InfoDisclosurePattern,
#     LowValueVulnerabilityResult,
#     LowValueVulnerabilityTest,
#     LowValueVulnerabilityType,
#     OpenRedirectPattern,
#     ReflectedXSSBasic,
#     ROIAnalysis,
#     VulnerabilityPattern,
#     XSSPattern,
# )  # 註解結束
# ==================== 訊息系統 ====================
# ==================== 插件系統 ====================
from .plugins import (
    PluginConfig,
    PluginExecutionContext,
    PluginExecutionResult,
    PluginHealthCheck,
    PluginManifest,
    PluginRegistry,
)

# ==================== 參考資料 ====================
from .references import (
    CAPECReference,
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

# ==================== 安全檢測領域 ====================
from .security import (
    # SASTDASTCorrelation,  # Removed - external SAST not needed
    AIVerificationRequest,
    AIVerificationResult,
    AttackPathEdgeType,
    AttackPathNodeType,
    AttackPattern,
    BaseAttackPath,
    BaseAttackPathEdge,
    BaseAttackPathNode,
    BaseSIEMEvent,
    BugBountyIntelligence,
    # LowValueVulnerabilityPattern,  # 已刪除沒用的定義
    Bundle,
    Campaign,
    CodeLevelRootCause,
    CourseOfAction,
    EnhancedSIEMEvent,
    EventStatus,
    ExternalReference,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    GranularMarking,
    Indicator,
    IntrusionSet,
    IOCEnrichment,
    JavaScriptAnalysisResult,  # 新增缺失的類別
    KillChainPhase,
    LegacyAttackPathAdapter,
    LegacySIEMEventAdapter,
    Malware,
    ObservedData,
    Priority,
    Relationship,
    Report,
    SensitiveMatch,
    Sighting,
    SkillLevel,
    STIXDomainObject,
    STIXRelationshipObject,
    Target,
    TAXIICollection,
    TAXIIErrorMessage,
    TAXIIManifest,
    TAXIIManifestEntry,
    TAXIIStatus,
    ThreatActor,
    ThreatIntelligenceReport,
    Tool,
    Vulnerability,
    VulnerabilityCorrelation,
    VulnerabilityScorecard,
)

# ==================== 系統編排 ====================
from .system import (
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
    Phase0CompletedPayload,
    # Phase0/Phase1 兩階段掃描 Schema
    Phase0StartPayload,
    Phase1CompletedPayload,
    Phase1StartPayload,
    PostExResultPayload,  # [Fixed] 修復: 補充遺漏的導入
    PostExTestPayload,
    RemediationGeneratePayload,
    RemediationResultPayload,
    ScanCompletedPayload,
    ScanStartPayload,
    Scenario,
    ScenarioResult,
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
    AdaptiveBehaviorInfo,
    EarlyStoppingInfo,
    EnhancedFunctionTelemetry,
    ErrorRecord,
    FunctionExecutionResult,
    FunctionTelemetry,
    HeartbeatPayload,
    ModuleStatus,
    NotificationPayload,
    OastCallbackDetail,
    OastEvent,
    OastProbe,
    SIEMEvent,
    SIEMEventPayload,
)

# ==================== 威脅情報 (STIX/TAXII) ====================
from .threat_intelligence import (
    AttackPattern,
    BugBountyIntelligence,
    Bundle,
    Campaign,
    CourseOfAction,
    ExternalReference,
    GranularMarking,
    Indicator,
    IntrusionSet,
    IOCEnrichment,
    KillChainPhase,
    LowValueVulnerabilityPattern,
    Malware,
    ObservedData,
    Relationship,
    Report,
    Sighting,
    STIXDomainObject,
    STIXRelationshipObject,
    TAXIICollection,
    TAXIIErrorMessage,
    TAXIIManifest,
    TAXIIManifestEntry,
    TAXIIStatus,
    ThreatActor,
    ThreatIntelligenceReport,
    Tool,
)
from .threat_intelligence import (
    Vulnerability as STIXVulnerability,  # HackerOne 優化相關
)

# 為了保持向後相容，明確匯出所有公開介面
__all__ = [
    # 基礎模型
    "APIResponse",
    "MessageHeader",
    "Authentication",
    "RateLimit",
    "ScanScope",
    "Asset",
    "Summary",
    "Fingerprints",
    "ExecutionError",
    "RiskFactor",
    "Task",
    "TaskDependency",
    # 能力管理 (v2.0)
    "ExecutableCapability",
    "ExecutableParameter",
    "ParameterType",
    # 訊息系統
    "AivaMessage",
    "AIVARequest",
    "AIVAResponse",
    "AIVAEvent",
    "AIVACommand",
    # 任務相關
    "ScanStartPayload",
    "ScanCompletedPayload",
    # Phase0/Phase1 兩階段掃描
    "Phase0StartPayload",
    "Phase0CompletedPayload",
    "Phase1StartPayload",
    "Phase1CompletedPayload",
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
    "Scenario",
    "ScenarioTestResult",
    "ExploitPayload",
    "TestExecution",
    "ExploitResult",
    "TestStrategy",
    # 漏洞發現
    "Vulnerability",
    "Target",
    "FindingTarget",  # 別名
    "ExternalToolFinding",
    "FindingEvidence",
    "FindingImpact",
    "FindingRecommendation",
    "FindingPayload",
    "SensitiveMatch",
    "VulnerabilityCorrelation",
    "VulnerabilityScorecard",
    "CodeLevelRootCause",
    # "SASTDASTCorrelation",  # Removed - external SAST not needed
    "AIVerificationRequest",
    "AIVerificationResult",
    # 遙測與監控
    "HeartbeatPayload",
    "ModuleStatus",
    "FunctionTelemetry",
    "EnhancedFunctionTelemetry",
    "ErrorRecord",
    "OastCallbackDetail",
    "EarlyStoppingInfo",
    "AdaptiveBehaviorInfo",
    "FunctionExecutionResult",
    "OastEvent",
    "OastProbe",
    "SIEMEventPayload",
    "SIEMEvent",
    "NotificationPayload",
    # AI相關類別已使用TYPE_CHECKING模式重構，符合PEP-484循環導入最佳實踐
    "CVSSv3Metrics",
    "AttackStep",
    "AttackPlan",
    "AttackTarget",  # 添加缺失的 AttackTarget
    "TraceRecord",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    "ModelTrainingConfig",
    "AITrainingStartPayload",
    "AITrainingProgressPayload",
    "AITrainingCompletedPayload",
    "ExperienceSample",
    "AIExperienceCreatedEvent",
    "AITraceCompletedEvent",
    "AIModelUpdatedEvent",
    "AIModelDeployCommand",
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
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
    "EnhancedVulnerabilityCorrelation",
    # 系統編排
    "SessionState",
    "ModelTrainingResult",
    "TaskQueue",
    "SystemOrchestration",
    "WebhookPayload",
    # 參考資料
    "CAPECReference",
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
    # 威脅情報 (STIX/TAXII)
    "STIXDomainObject",
    "STIXRelationshipObject",
    "AttackPattern",
    "Malware",
    "Indicator",
    "ThreatActor",
    "IntrusionSet",
    "Campaign",
    "CourseOfAction",
    "STIXVulnerability",
    "Tool",
    "ObservedData",
    "Report",
    "Relationship",
    "Sighting",
    "Bundle",
    "ExternalReference",
    "GranularMarking",
    "KillChainPhase",
    "TAXIICollection",
    "TAXIIManifest",
    "TAXIIManifestEntry",
    "TAXIIStatus",
    "TAXIIErrorMessage",
    "ThreatIntelligenceReport",
    "IOCEnrichment",
    "BugBountyIntelligence",
    "LowValueVulnerabilityPattern",
    # API 標準 (OpenAPI/AsyncAPI/GraphQL)
    "OpenAPIDocument",
    "OpenAPIInfo",
    "OpenAPIServer",
    "OpenAPIPathItem",
    "OpenAPIOperation",
    "OpenAPIParameter",
    "OpenAPISchema",
    "OpenAPIComponents",
    "OpenAPISecurityScheme",
    "AsyncAPIDocument",
    "AsyncAPIInfo",
    "AsyncServer",
    "AsyncAPIChannel",
    "AsyncAPIMessage",
    "AsyncOperation",
    "AsyncComponents",
    "GraphQLSchema",
    "GraphQLTypeDefinition",
    "GraphQLFieldDefinition",
    "GraphQLDirectiveDefinition",
    "APISecurityTest",

    # 決策數據合約 (問題三修復)
    "IntentType",
    "TargetInfo",
    "DecisionConstraints",
    "HighLevelIntent",
    "DecisionToASTContract",
    "DecisionFeedback",
    # 分析結果統一標準
    "BaseAnalysisResult",
    "JavaScriptAnalysisResult",
    "DataLeak",
    "AnalysisType",
    "LegacyJavaScriptAnalysisResultAdapter",
    # 安全事件統一標準
    "BaseSIEMEvent",
    "BaseAttackPathNode",
    "BaseAttackPathEdge",
    "BaseAttackPath",
    "EnhancedSIEMEvent",
    "EventStatus",
    "SkillLevel",
    "Priority",
    "AttackPathNodeType",
    "AttackPathEdgeType",
    "LegacySIEMEventAdapter",
    "LegacyAttackPathAdapter",
    # 異步工具
    "AsyncTaskConfig",
    "AsyncTaskResult",
    "RetryConfig",
    "ResourceLimits",
    "AsyncBatchConfig",
    "AsyncBatchResult",
    # 插件系統
    "PluginManifest",
    "PluginExecutionContext",
    "PluginExecutionResult",
    "PluginConfig",
    "PluginRegistry",
    "PluginHealthCheck",
    # CLI 界面
    "CLIParameter",
    "CLICommand",
    "CLIExecutionResult",
    "CLISession",
    "CLIConfiguration",
    "CLIMetrics",
]

# 版本資訊
__version__ = "2.1.0"
__schema_version__ = "1.0"
