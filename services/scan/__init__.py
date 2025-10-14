"""
AIVA Scan - 掃描模組

這是 AIVA 的掃描模組包，負責目標發現、指紋識別和漏洞掃描。

模組包含:
- aiva_scan: 主要掃描功能
  - core_crawling_engine: 核心爬取引擎
  - info_gatherer: 信息收集器
  - authentication_manager: 認證管理
  - scope_manager: 範圍管理
  - javascript_analyzer: JavaScript 分析器
"""

__version__ = "1.0.0"

# 從 aiva_common 導入共享基礎設施
from ..aiva_common.enums import (
    AssetType,
    ScanStatus,
    Severity,
)
from ..aiva_common.models import CVEReference, CVSSv3Metrics, CWEReference

# 從本模組導入掃描相關模型
from .models import (
    Asset,
    AssetInventoryItem,
    AssetLifecyclePayload,
    DiscoveredAsset,
    EASMAsset,
    EASMDiscoveryPayload,
    EASMDiscoveryResult,
    EnhancedScanRequest,
    EnhancedScanScope,
    Fingerprints,
    JavaScriptAnalysisResult,
    ScanCompletedPayload,
    ScanScope,
    ScanStartPayload,
    Summary,
    TechnicalFingerprint,
    Vulnerability,
    VulnerabilityDiscovery,
    VulnerabilityLifecyclePayload,
    VulnerabilityUpdatePayload,
)

__all__ = [
    # 來自 aiva_common
    "AssetType",
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    "ScanStatus",
    "Severity",
    # 來自本模組
    "Asset",
    "AssetInventoryItem",
    "AssetLifecyclePayload",
    "DiscoveredAsset",
    "EASMAsset",
    "EASMDiscoveryPayload",
    "EASMDiscoveryResult",
    "EnhancedScanScope",
    "EnhancedScanRequest",
    "Fingerprints",
    "JavaScriptAnalysisResult",
    "ScanCompletedPayload",
    "ScanScope",
    "ScanStartPayload",
    "Summary",
    "TechnicalFingerprint",
    "Vulnerability",
    "VulnerabilityDiscovery",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload",
]
