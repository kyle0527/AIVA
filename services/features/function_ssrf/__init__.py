"""
AIVA SSRF Function Module
========================

Enhanced SSRF detection capabilities with safe mode support.
Integrated with AIVA five-module architecture.
"""

from .config.ssrf_config import SsrfConfig
from .detector.ssrf_detector import SSRFDetector
from .engine.ssrf_engine import SSRFEngine, SSRFIssue

__all__ = ["SSRFDetector", "SSRFEngine", "SSRFIssue", "SsrfConfig"]

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "detector-based"
__last_updated__ = "2026-01-23"

# 架構說明 (2026-01-23)
# ✅ Detector-based 架構（直接調用檢測器）
# ✅ 支援 safe mode SSRF 檢測
# ❌ Worker 層已移除（不需要）