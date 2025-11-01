"""
AIVA Infrastructure Domain Schemas
==================================

基礎設施領域模型，包含：
- 資產管理
- 插件系統  
- 遙測監控
- 系統編排

此領域專注於平台基礎設施和運維相關功能。
"""

from .assets import *
from .plugins import *
from .telemetry import *
from .system import *

__all__ = [
    # 資產管理 (assets.py)
    "AssetLifecyclePayload",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload", 
    "DiscoveredAsset",
    "TechnicalFingerprint",
    "AssetInventoryItem",
    "EASMAsset",
    # 插件系統 (plugins.py)
    "PluginManifest",
    "PluginExecutionContext",
    "PluginExecutionResult",
    "PluginConfig", 
    "PluginRegistry",
    "PluginHealthCheck",
    # 遙測監控 (telemetry.py)  
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
    # 系統編排 (system.py)
    "SessionState",
    "ModelTrainingResult", 
    "TaskQueue",
    "SystemOrchestration",
    "WebhookPayload",
]