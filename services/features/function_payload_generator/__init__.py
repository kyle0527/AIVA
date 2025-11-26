"""
AIVA Payload Generator Module

此模組提供完整的 Payload 生成能力,包括:
- MSFVenom 全平台封裝
- Reverse Shell 生成器 (8種語言)
- Web Shell 生成器 (PHP/ASPX/JSP)
- PoC 自動生成框架

風險等級: L2-L3 (High-Critical Risk)
授權要求: RiskGuard L2+ 或 Authorization Token
"""

from .manager import PayloadGeneratorManager
from .models import (
    PayloadType,
    PayloadPlatform,
    PayloadFormat,
    PayloadConfig,
    PayloadResult,
)

__all__ = [
    "PayloadGeneratorManager",
    "PayloadType",
    "PayloadPlatform",
    "PayloadFormat",
    "PayloadConfig",
    "PayloadResult",
]

__version__ = "1.0.0"
__risk_level__ = "L2"
