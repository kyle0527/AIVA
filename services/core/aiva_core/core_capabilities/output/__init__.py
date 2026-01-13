"""
AIVA Output Module

輸出處理和格式化模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .to_functions import to_function_message

    __all__ = ["to_function_message"]
except ImportError:
    __all__ = []
