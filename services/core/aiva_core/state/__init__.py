"""
AIVA State Management Module

狀態管理和會話模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .session_state_manager import SessionStateManager

    __all__ = ["SessionStateManager"]
except ImportError:
    __all__ = []
