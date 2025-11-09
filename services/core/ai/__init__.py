"""
AIVA AI 組件包
===============

這個包包含 AIVA 系統的所有 AI 智能化組件，提供對基礎五大模組的 AI 增強能力。

結構:
- core/: AI 核心組件（事件系統、協議、編排、控制器）
- modules/: AI 功能模組（感知、認知、知識、自我改進）

設計理念:
- 可插拔設計，可獨立於基礎系統運行
- 當載入時，為基礎模組提供智能化增強
- 支援優雅降級，移除後系統仍可正常運作
"""

__version__ = "2.0.0"
__author__ = "AIVA Team"

# AI 核心組件
from .core.event_system.event_bus import AIEventBus, AIEvent, EventPriority
from .core.mcp_protocol.mcp_protocol import MCPManager, AIVAMCPAdapter, MCPResource, MCPTool
from .core.orchestration.agentic_orchestration import AgenticOrchestrator
from .core.controller.strangler_fig_controller import StranglerFigController

# AI 功能模組
from .modules.perception.perception_module import PerceptionModule
from .modules.cognition.cognition_module import CognitionModule
from .modules.knowledge.knowledge_module import KnowledgeModule
from .modules.self_improvement.self_improving_mechanism import SelfImprovingMechanism

__all__ = [
    # 核心組件
    'AIEventBus', 'AIEvent', 'EventPriority',
    'MCPManager', 'AIVAMCPAdapter', 'MCPResource', 'MCPTool',
    'AgenticOrchestrator',
    'StranglerFigController',
    
    # 功能模組
    'PerceptionModule',
    'CognitionModule', 
    'KnowledgeModule',
    'SelfImprovingMechanism'
]