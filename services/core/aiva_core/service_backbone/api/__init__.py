"""
API Module - API 服務模組

提供 FastAPI 伺服器和 AI 持續運行服務
"""

# app.py 提供 FastAPI 應用和服務器功能（替代 fastapi_server.py）
from .app import app as AIVAServer  # FastAPI 應用實例
from .ai_service import AIService

__all__ = [
    "AIVAServer",
    "AIService",
]
