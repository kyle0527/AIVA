"""
Python 掃描引擎 - 重構後版本

包含從 aiva_scan 移動過來的所有 Python 掃描組件：
- core_crawling_engine/: 核心爬蟲引擎
- dynamic_engine/: 動態掃描引擎
- info_gatherer/: 資訊收集器
- authentication_manager.py: 認證管理器
- vulnerability_scanner.py: 漏洞掃描器
- network_scanner.py: 網路掃描器
- 其他 Python 掃描模組

注意：路徑已從 services.scan.aiva_scan 更新為 services.scan.engines.python_engine
"""

__version__ = "1.1.0"  # 重構版本

# 模組初始化日誌
import logging
logger = logging.getLogger(__name__)

# 導入核心掃描組件 (更新後的導入路徑)
try:
    from .authentication_manager import *
    from .fingerprint_manager import *
    from .network_scanner import NetworkScanner
    from .vulnerability_scanner import VulnerabilityScanner
    from .service_detector import ServiceDetector
    from .target_environment_detector import TargetEnvironmentDetector
    from .worker import *
    logger.info("Python 掃描引擎組件載入成功")
except ImportError as e:
    # 向後兼容
    logger.error(f"部分 Python 掃描組件導入失敗: {e}")
    NetworkScanner = None
    VulnerabilityScanner = None
    ServiceDetector = None

__all__ = [
    "NetworkScanner",
    "VulnerabilityScanner", 
    "ServiceDetector",
    "TargetEnvironmentDetector",
    "authentication_manager",
    "fingerprint_manager",
    "worker",
    "core_crawling_engine",
    "dynamic_engine", 
    "info_gatherer",
    "examples"
]

logger.debug("AIVA 掃描模組已初始化")
