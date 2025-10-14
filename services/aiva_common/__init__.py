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
    AccessDecision,
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
    PersistenceType,
    PostExTestType,
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
        SARIFRun,
        SARIFTool,
    )

# 從 schemas.py 導入修復版本中的所有類
with contextlib.suppress(ImportError):
    from .schemas import (
        AivaMessage,
        Asset,
        Authentication,
        AuthZAnalysisPayload,
        # 授權分析
        AuthZCheckPayload,
        AuthZResultPayload,
        ConfigUpdatePayload,
        CVEReference,
        # 安全標準和評分
        CVSSv3Metrics,
        CWEReference,
        EnhancedFindingPayload,
        # 增強型漏洞處理
        EnhancedVulnerability,
        ExecutionError,
        FeedbackEventPayload,
        FindingEvidence,
        FindingImpact,
        FindingPayload,
        FindingRecommendation,
        Fingerprints,
        FunctionExecutionResult,
        FunctionTaskContext,
        FunctionTaskPayload,
        # 功能任務相關
        FunctionTaskTarget,
        FunctionTaskTestConfig,
        FunctionTelemetry,
        HeartbeatPayload,
        JavaScriptAnalysisResult,
        # 核心消息系統
        MessageHeader,
        # 模組狀態
        ModuleStatus,
        # OAST (Out-of-Band Application Security Testing)
        OastEvent,
        OastProbe,
        PostExResultPayload,
        # 後滲透測試
        PostExTestPayload,
        RateLimit,
        # 修復建議
        RemediationGeneratePayload,
        RemediationResultPayload,
        # SARIF 報告格式
        SARIFLocation,
        SARIFReport,
        SARIFResult,
        ScanCompletedPayload,
        # 掃描相關
        ScanScope,
        ScanStartPayload,
        SensitiveMatch,
        Summary,
        Target,
        # 任務管理和遙測
        TaskUpdatePayload,
        # 威脅情報
        ThreatIntelLookupPayload,
        ThreatIntelResultPayload,
        # 漏洞和發現
        Vulnerability,
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
