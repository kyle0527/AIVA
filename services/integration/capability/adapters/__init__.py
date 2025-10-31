"""
AIVA Capability Adapters
========================

工具適配器模組，提供不同安全工具格式到 AIVA CapabilityRecord 的轉換功能。

支援的適配器:
- HackingToolAdapter: 整合 HackingTool 專案的工具
"""

from .hackingtool_adapter import (
    AIVAToolAdapter,
    HackingToolDefinition,
    create_adapter_factory,
    HACKINGTOOL_CATEGORIES
)

__all__ = [
    "AIVAToolAdapter",
    "HackingToolDefinition", 
    "create_adapter_factory",
    "HACKINGTOOL_CATEGORIES"
]

__version__ = "1.0.0"
__author__ = "AIVA Development Team"