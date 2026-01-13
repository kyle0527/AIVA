"""
AIVA IDOR Function Module
========================

Enhanced IDOR detection capabilities with horizontal and vertical testing.
Integrated with AIVA five-module architecture.
"""

from .detector.idor_detector import IDORDetector
from .engine.idor_engine import IDOREngine, IDORIssue
from .config.idor_config import IdorConfig

__all__ = ["IDORDetector", "IDOREngine", "IDORIssue", "IdorConfig"]

# 整合狀態 (2025-12-17)
# ✅ 核心檢測器已完成 (horizontal/vertical testing)
# ⏳ 需要確認同步接口
# ⏳ CommandHandler 待確認
__version__ = "1.0.0"
__status__ = "ready_for_integration"
__last_updated__ = "2025-12-17"