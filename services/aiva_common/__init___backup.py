"""
AIVA Common - 通用模組

這個包包含了 AIVA 系統中所有服務共享的通用組件，
包括數據結構定義、配置管理和工具函數。

符合官方標準:
- CVSS v3.1: Common Vulnerability Scoring System
- MITRE ATT&CK: 攻擊技術框架
- SARIF v2.1.0: Static Analysis Results Interchange Format
- CVE/CWE/CAPEC: 漏洞和弱點標識標準
"""

__version__ = "1.0.0"

# 從各模組中導出核心類別和枚舉 (明確導入以避免循環依賴)
# 向後兼容：從舊的 schemas.py 重新導出 (如果存在)
import contextlib

from .enums import (
    AssetType,
    AttackPathEdgeType,
    AttackPathNodeType,
    ComplianceFramework,
    Confidence,
    IntelSource,
    ModuleName,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    ScanStatus,
    Severity,
    TaskStatus,
    Topic,
    VulnerabilityType,
)
from .models import (
    # 消息系統
    AivaMessage,
    # 認證授權
    Authentication,
    # 標準引用
    CAPECReference,
    CVEReference,
    # CVSS v3.1
    CVSSv3Metrics,
    CWEReference,
    MessageHeader,
    RateLimit,
    # SARIF v2.1.0
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRun,
)

# 從 schemas.py 中導入所有主要的數據模型類
with contextlib.suppress(ImportError):
    from .schemas import (
        # 基礎消息類型
        MessageHeader,
        AivaMessage,
        Authentication,
        RateLimit,
        # 掃描相關
        ScanScope,
        ScanStartPayload,
        Asset,
        Summary,
        Fingerprints,
        ScanCompletedPayload,
        # 功能測試相關
        FunctionTaskTarget,
        FunctionTaskContext,
        FunctionTaskTestConfig,
        FunctionTaskPayload,
        FeedbackEventPayload,
        # 漏洞和發現相關
        Vulnerability,
        Target,
        FindingEvidence,
        FindingImpact,
        FindingRecommendation,
        FindingPayload,
        # 任務和遙測
        TaskUpdatePayload,
        HeartbeatPayload,
        ConfigUpdatePayload,
        FunctionTelemetry,
        ExecutionError,
        FunctionExecutionResult,
        # OAST 相關
        OastEvent,
        OastProbe,
        # 模組狀態
        ModuleStatus,
        # 威脅情報相關
        ThreatIntelLookupPayload,
        ThreatIntelResultPayload,
        # 授權相關
        AuthZCheckPayload,
        AuthZAnalysisPayload,
        AuthZResultPayload,
        # 修復相關
        RemediationGeneratePayload,
        RemediationResultPayload,
        # 後滲透測試
        PostExTestPayload,
        PostExResultPayload,
        SensitiveMatch,
        JavaScriptAnalysisResult,
        # 業務邏輯測試
        BizLogicTestPayload,
        BizLogicResultPayload,
        # 生命週期管理
        AssetLifecyclePayload,
        VulnerabilityLifecyclePayload,
        VulnerabilityUpdatePayload,
        # 風險評估
        RiskAssessmentContext,
        RiskAssessmentResult,
        RiskTrendAnalysis,
        # 攻擊路徑
        AttackPathNode,
        AttackPathEdge,
        AttackPathPayload,
        AttackPathRecommendation,
        # 漏洞關聯
        VulnerabilityCorrelation,
        CodeLevelRootCause,
        SASTDASTCorrelation,
        # API 安全測試
        APISchemaPayload,
        APITestCase,
        APISecurityTestPayload,
        # AI 驗證
        AIVerificationRequest,
        AIVerificationResult,
        # SIEM 和通知
        SIEMEventPayload,
        NotificationPayload,
        # EASM 相關
        EASMDiscoveryPayload,
        DiscoveredAsset,
        EASMDiscoveryResult,
        EASMAsset,
        # AI 模型和訓練
        AttackStep,
        AttackPlan,
        TraceRecord,
        PlanExecutionMetrics,
        PlanExecutionResult,
        ExperienceSample,
        SessionState,
        ModelTrainingConfig,
        ModelTrainingResult,
        StandardScenario,
        ScenarioTestResult,
        # 增強型漏洞和發現
        EnhancedVulnerability,
        EnhancedFindingPayload,
        # AI 訓練事件
        AITrainingStartPayload,
        AITrainingProgressPayload,
        AITrainingCompletedPayload,
        AIExperienceCreatedEvent,
        AITraceCompletedEvent,
        AIModelUpdatedEvent,
        AIModelDeployCommand,
        # RAG 相關
        RAGKnowledgeUpdatePayload,
        RAGQueryPayload,
        RAGResponsePayload,
        # AIVA 核心
        AIVARequest,
        AIVAResponse,
        AIVAEvent,
        AIVACommand,
        # 增強型掃描
        EnhancedScanScope,
        EnhancedScanRequest,
        TechnicalFingerprint,
        # 資產清單
        AssetInventoryItem,  # 正確的類名
        VulnerabilityDiscovery,
        # 增強型功能任務
        EnhancedFunctionTaskTarget,
        ExploitPayload,
        TestExecution,
        ExploitResult,
        # 增強型 IOC
        EnhancedIOCRecord,
        SIEMEvent,
        # Webhook 和風險
        WebhookPayload,
        RiskFactor,
        EnhancedRiskAssessment,
        # 增強型攻擊路徑
        EnhancedAttackPathNode,
        EnhancedAttackPath,
        # 任務編排
        TaskDependency,
        EnhancedTaskExecution,
        TaskQueue,
        TestStrategy,
        # 系統編排
        EnhancedModuleStatus,
        SystemOrchestration,
        EnhancedVulnerabilityCorrelation,
    )

# 只包含實際存在且可以導入的項目
__all__ = [
    # Enums - 所有從 enums.py 導入的枚舉
    "ModuleName",
    "Topic",
    "Severity",
    "Confidence",
    "VulnerabilityType",
    "TaskStatus",
    "ScanStatus",
    "SensitiveInfoType",
    "Location",
    "ThreatLevel",
    "IntelSource",
    "IOCType",
    "RemediationType",
    "RemediationStatus",
    "Permission",
    "AccessDecision",
    "PostExTestType",
    "PersistenceType",
    "BusinessCriticality",
    "Environment",
    "AssetType",
    "AssetStatus",
    "VulnerabilityStatus",
    "DataSensitivity",
    "AssetExposure",
    "Exploitability",
    "ComplianceFramework",
    "RiskLevel",
    "AttackPathNodeType",
    "AttackPathEdgeType",
    # Models - 從 models.py 導入的基礎模型
    "CVSSv3Metrics",
    "CVEReference",
    "CWEReference",
    "CAPECReference",
    "SARIFLocation",
    "SARIFResult",
    "SARIFRule",
    "SARIFTool",
    "SARIFRun",
    "SARIFReport",
    # Schemas - 從 schemas.py 導入的消息和數據結構（只包含確實存在的）
    "MessageHeader",
    "AivaMessage",
    "Authentication",
    "RateLimit",
    "ScanScope",
    "ScanStartPayload",
    "Asset",
    "Summary",
    "Fingerprints",
    "ScanCompletedPayload",
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "FunctionTaskPayload",
    "FeedbackEventPayload",
    "Vulnerability",
    "Target",
    "FindingEvidence",
    "FindingImpact",
    "FindingRecommendation",
    "FindingPayload",
    "TaskUpdatePayload",
    "HeartbeatPayload",
    "ConfigUpdatePayload",
    "FunctionTelemetry",
    "ExecutionError",
    "FunctionExecutionResult",
    "OastEvent",
    "OastProbe",
    "ModuleStatus",
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "AuthZCheckPayload",
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    "RemediationGeneratePayload",
    "RemediationResultPayload",
    "PostExTestPayload",
    "PostExResultPayload",
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    "AssetLifecyclePayload",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload",
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "RiskTrendAnalysis",
    "AttackPathNode",
    "AttackPathEdge",
    "AttackPathPayload",
    "AttackPathRecommendation",
    "VulnerabilityCorrelation",
    "CodeLevelRootCause",
    "SASTDASTCorrelation",
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    "AIVerificationRequest",
    "AIVerificationResult",
    "SIEMEventPayload",
    "NotificationPayload",
    "EASMDiscoveryPayload",
    "DiscoveredAsset",
    "EASMDiscoveryResult",
    "AttackStep",
    "AttackPlan",
    "TraceRecord",
    "PlanExecutionMetrics",
    "PlanExecutionResult",
    "ExperienceSample",
    "SessionState",
    "ModelTrainingConfig",
    "ModelTrainingResult",
    "StandardScenario",
    "ScenarioTestResult",
    "EnhancedVulnerability",
    "EnhancedFindingPayload",
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
    "AIVARequest",
    "AIVAResponse",
    "AIVAEvent",
    "AIVACommand",
    "EnhancedScanScope",
    "EnhancedScanRequest",
    "TechnicalFingerprint",
    "AssetInventoryItem",  # 注意：這是正確的類名，不是 AssetInventory
    "VulnerabilityDiscovery",
    "EnhancedFunctionTaskTarget",
    "ExploitPayload",
    "TestExecution",
    "ExploitResult",
    "EnhancedIOCRecord",
    "SIEMEvent",
    "EASMAsset",
    "WebhookPayload",
    "RiskFactor",
    "EnhancedRiskAssessment",
    "EnhancedAttackPathNode",
    "EnhancedAttackPath",
    "TaskDependency",
    "EnhancedTaskExecution",
    "TaskQueue",
    "TestStrategy",
    "EnhancedModuleStatus",
    "SystemOrchestration",
    "EnhancedVulnerabilityCorrelation",
]
