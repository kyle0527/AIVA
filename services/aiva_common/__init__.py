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

# 從各模組中導出核心類別和枚舉
import contextlib

# 導入新增的通用模組
from . import enums, utils
from . import schemas as common_schemas
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

# 版本信息從專用模組導入
from .version import __version__, get_version, get_version_info

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
        async_retry,
        async_timeout,
        default_task_manager,
    )

    _has_async = True
except ImportError:
    _has_async = False

try:
    from .plugins import (
        BasePlugin,
        PluginManager,
        default_plugin_manager,
        plugin_hook,
    )

    _has_plugins = True
except ImportError:
    _has_plugins = False

try:
    from .cli import CLIContext, create_aiva_cli, default_config_manager

    _has_cli = True
except ImportError:
    _has_cli = False

# 新增的跨語言架構模組
try:
    from .config_manager import (
        ConfigManager,
        ConfigSchema,
        ConfigScope,
        ConfigType,
        get_config_manager,
    )

    _has_config_manager = True
except ImportError:
    _has_config_manager = False

try:
    from .service_discovery import (
        HealthCheck,
        HealthCheckType,
        ServiceDiscoveryManager,
        ServiceEndpoint,
        ServiceMetadata,
        ServiceRegistry,
        ServiceStatus,
        get_service_discovery_manager,
    )

    _has_service_discovery = True
except ImportError:
    _has_service_discovery = False

# 數據處理管道
try:
    from .data_pipeline import (
        DataPipeline,
        FileDataSink,
        FileDataSource,
        MemoryDataSink,
        MemoryDataSource,
        PipelineManager,
        ProcessingMode,
        QueueDataSource,
        get_pipeline_manager,
    )

    _has_data_pipeline = True
except ImportError:
    _has_data_pipeline = False

# 流處理器
try:
    from .stream_processor import (
        StreamAggregations,
        StreamProcessor,
        WindowType,
        create_stream_processor,
    )

    _has_stream_processor = True
except ImportError:
    _has_stream_processor = False

# 監控和日誌系統
try:
    from .monitoring import (
        Alert,
        AlertManager,
        AlertSeverity,
        LogAggregator,
        LogEntry,
        LogLevel,
        MetricData,
        MetricsCollector,
        MetricType,
        MonitoringService,
        SystemMonitor,
        TimerContext,
        TraceContext,
        TraceManager,
        TraceSpan,
        TraceStatus,
        create_monitoring_service,
        get_monitoring_service,
        timer_metric,
        trace_operation,
    )

    _has_monitoring = True
except ImportError:
    _has_monitoring = False

# 監控日誌處理器
try:
    from .monitoring_log_handler import (
        MonitoringLogHandler,
        TraceLoggerAdapter,
        create_trace_logger,
        get_logger_with_monitoring,
        setup_monitoring_logging,
    )

    _has_monitoring_handler = True
except ImportError:
    _has_monitoring_handler = False

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
    __all__.extend(
        [
            "AsyncContext",
            "AsyncTaskManager",
            "async_timeout",
            "async_retry",
            "default_task_manager",
        ]
    )

if _has_plugins:
    __all__.extend(
        [
            "BasePlugin",
            "PluginManager",
            "plugin_hook",
            "default_plugin_manager",
        ]
    )

if _has_cli:
    __all__.extend(["CLIContext", "create_aiva_cli", "default_config_manager"])

# 新增跨語言架構模組到 __all__
if _has_config_manager:
    __all__.extend(
        [
            "ConfigManager",
            "get_config_manager",
            "ConfigScope",
            "ConfigType",
            "ConfigSchema",
        ]
    )

if _has_service_discovery:
    __all__.extend(
        [
            "ServiceDiscoveryManager",
            "get_service_discovery_manager",
            "ServiceRegistry",
            "ServiceEndpoint",
            "ServiceMetadata",
            "ServiceStatus",
            "HealthCheck",
            "HealthCheckType",
        ]
    )

# 數據處理管道模組
if _has_data_pipeline:
    __all__.extend(
        [
            "DataPipeline",
            "PipelineManager",
            "ProcessingMode",
            "MemoryDataSource",
            "FileDataSource",
            "QueueDataSource",
            "MemoryDataSink",
            "FileDataSink",
            "get_pipeline_manager",
        ]
    )

# 流處理器模組
if _has_stream_processor:
    __all__.extend(
        [
            "StreamProcessor",
            "WindowType",
            "StreamAggregations",
            "create_stream_processor",
        ]
    )

# 監控系統模組
if _has_monitoring:
    __all__.extend(
        [
            "MonitoringService",
            "MetricsCollector",
            "SystemMonitor",
            "TraceManager",
            "LogAggregator",
            "AlertManager",
            "MetricType",
            "LogLevel",
            "AlertSeverity",
            "TraceStatus",
            "MetricData",
            "LogEntry",
            "TraceSpan",
            "Alert",
            "get_monitoring_service",
            "create_monitoring_service",
            "TraceContext",
            "TimerContext",
            "trace_operation",
            "timer_metric",
        ]
    )

# 監控日誌處理器模組
if _has_monitoring_handler:
    __all__.extend(
        [
            "MonitoringLogHandler",
            "TraceLoggerAdapter",
            "setup_monitoring_logging",
            "get_logger_with_monitoring",
            "create_trace_logger",
        ]
    )

# 安全系統模組 - 先嘗試導入以定義變數
try:
    from .security import (
        AuthenticationService,
        AuthenticationType,
        AuthorizationService,
        CryptographyService,
        SecurityAuditService,
        SecurityEvent,
        SecurityEventType,
        SecurityManager,
        TokenService,
        create_security_manager,
        get_security_manager,
        require_authentication,
        require_authorization,
        secure_endpoint,
    )

    _has_security = True
except ImportError:
    _has_security = False

# 安全中間件模組 - 先嘗試導入以定義變數
try:
    from .security_middleware import (
        CORSHandler,
        RateLimiter,
        RateLimitRule,
        SecurityHeaders,
        SecurityMiddleware,
        SecurityValidator,
        create_security_middleware,
        secure_api_endpoint,
        validate_request_data,
    )

    _has_security_middleware = True
except ImportError:
    _has_security_middleware = False

# 安全系統模組 - 現在可以安全使用變數
if _has_security:
    __all__.extend(
        [
            "SecurityManager",
            "CryptographyService",
            "TokenService",
            "AuthenticationService",
            "AuthorizationService",
            "SecurityAuditService",
            "AuthenticationType",
            "SecurityEventType",
            "SecurityEvent",
            "get_security_manager",
            "create_security_manager",
            "require_authentication",
            "require_authorization",
            "secure_endpoint",
        ]
    )

# 安全中間件模組 - 現在可以安全使用變數
if _has_security_middleware:
    __all__.extend(
        [
            "SecurityMiddleware",
            "RateLimiter",
            "CORSHandler",
            "SecurityHeaders",
            "SecurityValidator",
            "RateLimitRule",
            "create_security_middleware",
            "secure_api_endpoint",
            "validate_request_data",
        ]
    )

# 元數據
__author__ = "AIVA Team"
__email__ = "team@aiva.ai"
__license__ = "MIT"
__description__ = "AIVA Common - 現代化 Python 共享庫組件"
