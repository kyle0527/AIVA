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

# 從各模組中導出核心類別和枚舉
import contextlib

from .enums import (
    AssetExposure,
    AssetType,
    AttackPathEdgeType,
    AttackPathNodeType,
    BusinessCriticality,
    ComplianceFramework,
    Confidence,
    DataSensitivity,
    Environment,
    Exploitability,
    IntelSource,
    IOCType,
    Location,
    ModuleName,
    Permission,
    AccessDecision,
    PostExTestType,
    PersistenceType,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    ScanStatus,
    SensitiveInfoType,
    Severity,
    TaskStatus,
    ThreatLevel,
    Topic,
    VulnerabilityStatus,
    VulnerabilityType,
)

# 從 models.py 導入基礎模型類（如果存在）
with contextlib.suppress(ImportError):
    from .models import (
        CAPECReference,
        SARIFRule,
        SARIFTool,
        SARIFRun,
    )

# 從 schemas.py 導入修復版本中的所有類
with contextlib.suppress(ImportError):
    from .schemas import (
        # 核心消息系統
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

        # 功能任務相關
        FunctionTaskTarget,
        FunctionTaskContext,
        FunctionTaskTestConfig,
        FunctionTaskPayload,
        FeedbackEventPayload,

        # 漏洞和發現
        Vulnerability,
        Target,
        FindingEvidence,
        FindingImpact,
        FindingRecommendation,
        FindingPayload,

        # 任務管理和遙測
        TaskUpdatePayload,
        HeartbeatPayload,
        ConfigUpdatePayload,
        FunctionTelemetry,
        ExecutionError,
        FunctionExecutionResult,

        # OAST (Out-of-Band Application Security Testing)
        OastEvent,
        OastProbe,

        # 模組狀態
        ModuleStatus,

        # 威脅情報
        ThreatIntelLookupPayload,
        ThreatIntelResultPayload,

        # 授權分析
        AuthZCheckPayload,
        AuthZAnalysisPayload,
        AuthZResultPayload,

        # 修復建議
        RemediationGeneratePayload,
        RemediationResultPayload,

        # 後滲透測試
        PostExTestPayload,
        PostExResultPayload,
        SensitiveMatch,
        JavaScriptAnalysisResult,

        # 安全標準和評分
        CVSSv3Metrics,
        CVEReference,
        CWEReference,

        # SARIF 報告格式
        SARIFLocation,
        SARIFResult,
        SARIFReport,

        # 增強型漏洞處理
        EnhancedVulnerability,
        EnhancedFindingPayload,
    )

# 只包含修復版本中實際存在且可以導入的項目
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
    "AssetExposure",
    "VulnerabilityStatus",
    "DataSensitivity",
    "Exploitability",
    "ComplianceFramework",
    "RiskLevel",
    "AttackPathNodeType",
    "AttackPathEdgeType",

    # Schemas - 從修復版本中導入的類
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
    "CVSSv3Metrics",
    "CVEReference",
    "CWEReference",
    "SARIFLocation",
    "SARIFResult",
    "SARIFReport",
    "EnhancedVulnerability",
    "EnhancedFindingPayload",
]
