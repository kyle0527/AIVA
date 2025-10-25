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
from ..aiva_common.schemas import (
    Asset,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    Fingerprints,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Vulnerability,
)

# 從本模組導入掃描相關模型（僅本地特定擴展）
from .models import (
    AssetInventoryItem,
    AssetLifecyclePayload,
    DiscoveredAsset,
    EASMAsset,
    EASMDiscoveryPayload,
    EASMDiscoveryResult,
    EnhancedScanRequest,
    EnhancedScanScope,
    JavaScriptAnalysisResult,
    TechnicalFingerprint,
    VulnerabilityDiscovery,
    VulnerabilityLifecyclePayload,
    VulnerabilityUpdatePayload,
)

# 導入 SARIF 轉換器
from .sarif_converter import SARIFConverter

__all__ = [
    # ========== 從 aiva_common 導入（共享標準） ==========
    # 枚舉
    "AssetType",
    "ScanStatus",
    "Severity",
    # Schema - 基礎模型
    "Asset",
    "Fingerprints",
    "Summary",
    # Schema - 掃描任務
    "ScanStartPayload",
    "ScanCompletedPayload",
    # Schema - 漏洞
    "Vulnerability",
    # Schema - 標準引用
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    
    # ========== 本模組特定擴展 ==========
    # 掃描配置擴展
    "EnhancedScanScope",
    "EnhancedScanRequest",
    # 資產管理擴展
    "AssetInventoryItem",
    "AssetLifecyclePayload",
    "DiscoveredAsset",
    "TechnicalFingerprint",
    # 漏洞管理擴展
    "VulnerabilityDiscovery",
    "VulnerabilityLifecyclePayload",
    "VulnerabilityUpdatePayload",
    # EASM 功能
    "EASMAsset",
    "EASMDiscoveryPayload",
    "EASMDiscoveryResult",
    # 分析功能
    "JavaScriptAnalysisResult",
    # SARIF 支援
    "SARIFConverter",
]
