"""
Dialog Module - 對話與用戶交互模組

提供 AI 助理對話、智能選單等用戶交互功能
"""

from .assistant import AIVADialogAssistant, AIVACommandProcessor
from .ai_menu import AIVAIntelligentMenu

__all__ = [
    "AIVADialogAssistant",
    "AIVACommandProcessor",
    "AIVAIntelligentMenu",
]
