"""
AIVA IDOR Function Module
========================

Enhanced IDOR detection capabilities with horizontal and vertical testing.
Integrated with AIVA five-module architecture.
"""

from .config.idor_config import IdorConfig
from .detector.idor_detector import IDORDetector
from .engine.idor_engine import IDOREngine, IDORIssue

__all__ = ["IDORDetector", "IDOREngine", "IDORIssue", "IdorConfig"]

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "detector-based"
__last_updated__ = "2026-01-23"

# 架構說明 (2026-01-23)
# ✅ Detector-based 架構（直接調用檢測器）
# ✅ 支援 horizontal/vertical IDOR 檢測
# ❌ Worker 層已移除（不需要）