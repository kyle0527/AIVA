"""
AIVA Plugin System - AI 模組插件化架構

此包提供統一的插件接口和管理機制，支援動態註冊、權重管理和生命週期控制。

設計參考:
- Kubernetes Device Plugin Pattern
- Ray Serve Model Management
- FastAPI Lifespan Management

核心組件:
- base_plugin: AIModulePlugin 插件基礎接口
- module_registry: 模組註冊和發現機制
- weight_manager: 權重版本管理和完整性驗證
"""

from .base_plugin import (
    AIModulePlugin,
    AITask,
    AIResult,
    AITaskType,
)

from .module_registry import ModuleRegistry

from .weight_manager import WeightManager

__all__ = [
    "AIModulePlugin",
    "AITask",
    "AIResult",
    "AITaskType",
    "ModuleRegistry",
    "WeightManager",
]

__version__ = "2.0.0"
