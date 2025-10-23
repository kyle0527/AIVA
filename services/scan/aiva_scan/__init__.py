"""
AIVA Scan Module

核心掃描引擎模組，提供安全掃描功能。
"""

__version__ = "1.0.0"

# 導入核心掃描組件
try:
    from .scan_orchestrator import ScanOrchestrator
except ImportError:
    # 向後兼容
    ScanOrchestrator = None

__all__ = [
    "ScanOrchestrator",
]

# 模組初始化日誌
import logging
logger = logging.getLogger(__name__)
logger.debug("AIVA 掃描模組已初始化")
