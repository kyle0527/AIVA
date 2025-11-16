"""
Cognitive Core - AI 認知核心

本模組是 AIVA 的「大腦」,負責思考和決策。整合神經網路推理、RAG 知識增強、
決策支援和反幻覺機制,實現 AI 自我優化雙重閉環的核心決策功能。

主要組件:
- neural: 神經網路核心 (5M 參數 BioNeuron)
- rag: RAG 增強系統 (知識檢索與上下文增強)
- decision: 決策支援系統
- anti_hallucination: 反幻覺機制
- InternalLoopConnector: 內部閉環連接器 (探索結果 → RAG)
- ExternalLoopConnector: 外部閉環連接器 (偏差報告 → 學習系統)

使用範例:
    >>> from aiva_core.cognitive_core import CognitiveCoreOrchestrator
    >>> core = CognitiveCoreOrchestrator()
    >>> result = await core.reason_and_decide(task, context)

對應設計理念:
    - 內部閉環: 探索(對內) + 分析 + RAG → 了解自身能力
    - 外部閉環: 掃描(對外) + 攻擊 → 收集優化方向
"""

__version__ = "3.0.0-alpha"
__status__ = "架構搭建中"

# TODO: 在模組遷移完成後添加以下導入
# from .neural import RealNeuralCore, RealBioNetAdapter
# from .rag import RAGEngine, KnowledgeBase, UnifiedVectorStore
# from .decision import EnhancedDecisionAgent
# from .anti_hallucination import AntiHallucinationModule

# ✅ 內部閉環連接器已實現
from .internal_loop_connector import InternalLoopConnector

# ✅ 外部閉環連接器已實現
from .external_loop_connector import ExternalLoopConnector

__all__ = [
    # 將在遷移完成後導出
    # "RealNeuralCore",
    # "RealBioNetAdapter",
    # "RAGEngine",
    # "KnowledgeBase",
    # "UnifiedVectorStore",
    # "EnhancedDecisionAgent",
    # "AntiHallucinationModule",
    
    # ✅ 雙閉環連接器已導出
    "InternalLoopConnector",
    "ExternalLoopConnector",
]
