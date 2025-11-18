"""
AIVA Information Gatherer

資訊收集和指紋識別模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .passive_fingerprinter import PassiveFingerprinter
    __all__ = ["PassiveFingerprinter"]
except ImportError:
    __all__ = []
