"""
Server-Side Request Forgery (SSRF) Detection Module

服務端請求偽造攻擊檢測功能模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .result_publisher import ResultPublisher
    __all__ = ["ResultPublisher"]
except ImportError:
    __all__ = []
