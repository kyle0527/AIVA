"""
AIVA Common - 現代化 Python 共享庫

這個包包含了 AIVA 系統中所有服務共享的通用組件，
基於 2024-2025 年 Python 最佳實踐實現：

核心功能:
- 統一的數據模型和枚舉定義
- 現代化配置管理 (Pydantic Settings v2)
- 可觀測性和監控 (OpenTelemetry)  
- 異步工具和任務管理
- 插件架構系統
- 命令行工具支援

符合標準:
- CVSS v3.1: Common Vulnerability Scoring System
- MITRE ATT&CK: 攻擊技術框架
- SARIF v2.1.0: Static Analysis Results Interchange Format
- CVE/CWE/CAPEC: 漏洞和弱點標識標準
- PEP 518: Python 包裝標準
"""

# 版本信息從專用模組導入
from .version import __version__, get_version, get_version_info

# 從各模組中導出核心類別和枚舉
import contextlib

# 導入新增的通用模組
from . import enums, utils, schemas as common_schemas

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
    TestStatus,
    ThreatLevel,
    Topic,
    VulnerabilityStatus,
    VulnerabilityType,
)

# 從 schemas.py 導入所有類（統一來源）
with contextlib.suppress(ImportError):
    # 授權分析; 安全標準和評分; 增強型漏洞處理; 功能任務相關; 核心消息系統
    # 模組狀態; OAST (Out-of-Band Application Security Testing); 後滲透測試
    # 修復建議; SARIF 報告格式; 掃描相關; 任務管理和遙測; 威脅情報; 漏洞和發現
    from .schemas import (
        AivaMessage,
        Asset,
        Authentication,
        AuthZAnalysisPayload,
        AuthZCheckPayload,
        AuthZResultPayload,
        CAPECReference,
        ConfigUpdatePayload,
        CVEReference,
        CVSSv3Metrics,
        CWEReference,
        EnhancedFindingPayload,
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
        FunctionTaskTarget,
        FunctionTaskTestConfig,
        FunctionTelemetry,
        HeartbeatPayload,
        JavaScriptAnalysisResult,
        MessageHeader,
        ModuleStatus,
        OastEvent,
        OastProbe,
        PostExResultPayload,
        PostExTestPayload,
        RateLimit,
        RemediationGeneratePayload,
        RemediationResultPayload,
        SARIFLocation,
        SARIFReport,
        SARIFResult,
        SARIFRule,
        SARIFRun,
        SARIFTool,
        ScanCompletedPayload,
        ScanScope,
        ScanStartPayload,
        SensitiveMatch,
        Summary,
        Target,
        TaskUpdatePayload,
        ThreatIntelLookupPayload,
        ThreatIntelResultPayload,
        Vulnerability,
    )

# 現代化模組的條件導入
try:
    from .config.settings import BaseAIVASettings, get_settings
    _has_config = True
except ImportError:
    _has_config = False

try:
    from .observability import AIVALogger, MetricCollector, get_logger
    _has_observability = True
except ImportError:
    _has_observability = False

try:
    from .async_utils import (
        AsyncContext,
        AsyncTaskManager,
        async_timeout,
        async_retry,
        default_task_manager,
    )
    _has_async = True
except ImportError:
    _has_async = False

try:
    from .plugins import (
        BasePlugin,
        PluginManager,
        plugin_hook,
        default_plugin_manager,
    )
    _has_plugins = True
except ImportError:
    _has_plugins = False

try:
    from .cli import CLIContext, create_aiva_cli, default_config_manager
    _has_cli = True
except ImportError:
    _has_cli = False

# 只包含修復版本中實際存在且可以導入的項目
__all__ = [
    # Enums - 所有從 enums.py 導入的枚舉
    "ModuleName",
    "Topic",
    "Severity",
    "Confidence",
    "VulnerabilityType",
    "TaskStatus",
    "TestStatus",
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
    "CAPECReference",
    "SARIFLocation",
    "SARIFResult",
    "SARIFRule",
    "SARIFTool",
    "SARIFRun",
    "SARIFReport",
    "EnhancedVulnerability",
    "EnhancedFindingPayload",
    # 版本信息
    "__version__",
    "get_version", 
    "get_version_info",
]

# 動態添加現代化模組到 __all__
if _has_config:
    __all__.extend(["BaseAIVASettings", "get_settings"])

if _has_observability:
    __all__.extend(["AIVALogger", "MetricCollector", "get_logger"])

if _has_async:
    __all__.extend([
        "AsyncContext",
        "AsyncTaskManager",
        "async_timeout", 
        "async_retry",
        "default_task_manager",
    ])

if _has_plugins:
    __all__.extend([
        "BasePlugin",
        "PluginManager",
        "plugin_hook",
        "default_plugin_manager",
    ])

if _has_cli:
    __all__.extend(["CLIContext", "create_aiva_cli", "default_config_manager"])

# 元數據
__author__ = "AIVA Team"
__email__ = "team@aiva.ai" 
__license__ = "MIT"
__description__ = "AIVA Common - 現代化 Python 共享庫組件"
