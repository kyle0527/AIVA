"""
Cross-Site Scripting (XSS) Detection Module

跨站腳本攻擊檢測功能模組。

遵循 README 規範：
- 移除 ImportError fallback 機制
- 確保依賴可用，導入失敗時明確報錯
"""

__version__ = "1.0.0"

# 導入核心組件 - 遵循 README 規範，不使用 try/except fallback
from .dom_xss_detector import DomXssDetector
from .payload_generator import XssPayloadGenerator
from .result_publisher import XssResultPublisher

__all__ = [
    "DomXssDetector",
    "XssPayloadGenerator", 
    "XssResultPublisher",
]
