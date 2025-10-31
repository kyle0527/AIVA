"""
AI Engine - AIVA 的 AI 決策引擎
整合生物啟發式神經網路、RAG 知識檢索與工具執行系統
"""

from .ai_model_manager import AIModelManager
from .bio_neuron_core import (
    AntiHallucinationModule,
    BiologicalSpikingLayer,
    BioNeuronRAGAgent,
    ScalableBioNet,
)
from .knowledge_base import KnowledgeBase
from .performance_enhancements import (
    ComponentPool,
    MemoryManager,
    OptimizedBioSpikingLayer,
    OptimizedScalableBioNet,
    PerformanceConfig,
)
from .tools import (
    CodeAnalyzer,
    CodeReader,
    CodeWriter,
    CommandExecutor,
    ScanTrigger,
    Tool,
    VulnerabilityDetector,
)

__all__ = [
    # Bio Neuron Core
    "BiologicalSpikingLayer",
    "AntiHallucinationModule",
    "ScalableBioNet",
    "BioNeuronRAGAgent",
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
    # Tools
    "Tool",
    "CodeReader",
    "CodeWriter",
    "CodeAnalyzer",
    "CommandExecutor",
    "ScanTrigger",
    "VulnerabilityDetector",
]
