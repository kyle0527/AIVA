"""
AIVA Scan Module

核心掃描引擎模組，提供安全掃描功能。
"""

__version__ = "1.0.0"

# 模組初始化日誌
import logging
logger = logging.getLogger(__name__)

# 導入核心掃描組件
try:
    from .scan_orchestrator import ScanOrchestrator
    from .vulnerability_scanner import VulnerabilityScanner
    from .network_scanner import NetworkScanner
    from .service_detector import ServiceDetector
    from .target_environment_detector import TargetEnvironmentDetector
except ImportError as e:
    # 向後兼容
    logger.error(f"部分掃描組件導入失敗: {e}")
    ScanOrchestrator = None
    VulnerabilityScanner = None
    NetworkScanner = None
    ServiceDetector = None

__all__ = [
    "ScanOrchestrator",
    "VulnerabilityScanner", 
    "NetworkScanner",
    "ServiceDetector",
    "TargetEnvironmentDetector",
]

logger.debug("AIVA 掃描模組已初始化")
