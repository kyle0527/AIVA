"""
AIVA Integration Module

整合各種外部服務和系統的核心模組。
"""

__version__ = "1.0.0"

# 導入主要組件
from .app import app
from .settings import IntegrationSettings

__all__ = [
    "app",
    "IntegrationSettings",
]
