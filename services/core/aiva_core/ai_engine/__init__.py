"""
AI Engine - AIVA 的 AI 決策引擎
整合生物啟發式神經網路、RAG 知識檢索與工具執行系統
"""

from .ai_model_manager import AIModelManager
from .real_neural_core import (
    RealAICore,
    RealDecisionEngine,
)
from .anti_hallucination_module import AntiHallucinationModule
from .knowledge_base import KnowledgeBase
from .performance_enhancements import (
    ComponentPool,
    OptimizedBioSpikingLayer,
    OptimizedScalableBioNet,
    PerformanceConfig,
)
# 記憶體管理從統一模組導入
from ..performance.unified_memory_manager import MemoryManager
# 工具系統從新的統一結構導入
from .tools import (
    Tool,
    ToolManager,
    CodeReader,
    CodeWriter,
    CodeAnalyzer,
    CommandExecutor,
    ShellCommandTool,
    SystemStatusTool,
)

__all__ = [
    # AI Core Components
    "RealAICore",
    "RealDecisionEngine", 
    "AntiHallucinationModule",
    # AI Model Management
    "AIModelManager",
    # Performance Enhancements
    "OptimizedScalableBioNet",
    "OptimizedBioSpikingLayer",
    "PerformanceConfig",
    "MemoryManager",
    "ComponentPool",
    # Knowledge Base
    "KnowledgeBase",
    # Tools - 統一工具系統
    "Tool",
    "ToolManager",
    "CodeReader",
    "CodeWriter",
    "CodeAnalyzer",
    "CommandExecutor",
    "ShellCommandTool",
    "SystemStatusTool",
]
