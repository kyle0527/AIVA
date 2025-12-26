"""
AIVA SSRF Function Module
========================

Enhanced SSRF detection capabilities with safe mode support.
Integrated with AIVA five-module architecture.
"""

from .detector.ssrf_detector import SSRFDetector
from .engine.ssrf_engine import SSRFEngine, SSRFIssue
from .config.ssrf_config import SsrfConfig

__all__ = ["SSRFDetector", "SSRFEngine", "SSRFIssue", "SsrfConfig"]

# 整合狀態 (2025-12-17)
# ✅ 核心檢測器已完成 (safe mode support)
# ⏳ 需要確認同步接口
# ⏳ CommandHandler 待確認
__version__ = "1.0.0"
__status__ = "ready_for_integration"
__last_updated__ = "2025-12-17"