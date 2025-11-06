"""
AIVA Integration Module

整合各種外部服務和系統的核心模組。
"""

__version__ = "1.0.0"

# 延遲導入，避免循環依賴和資源初始化問題
from .settings import IntegrationSettings

def get_app():
    """延遲加載 FastAPI 應用"""
    from .app import app
    return app

__all__ = [
    "get_app",
    "IntegrationSettings",
]
