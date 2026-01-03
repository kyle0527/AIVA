"""
Cognitive Core - AI 認知核心

本模組是 AIVA 的「大腦」,負責思考和決策。整合神經網路推理、RAG 知識增強、
決策支援和反幻覺機制,實現 AI 自我優化雙重閉環的核心決策功能。

主要組件:
- neural: 5M 參數特化決策神經網路 (RealAICore, RealDecisionEngine)
- rag: RAG 增強系統 (知識檢索與上下文增強)
- decision: 決策支援系統
- anti_hallucination: 反幻覺機制
- InternalLoopConnector: 內部閉環連接器 (探索結果 → RAG)
- ExternalLoopConnector: 外部閉環連接器 (偏差報告 → 學習系統)

⚠️ 設計限制:
- 無 NLU（自然語言理解）能力
- 無 LLM（大型語言模型）依賴
- 僅支援結構化輸入/輸出的決策任務

使用範例:
    >>> from aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
    >>> engine = RealDecisionEngine(use_5m_model=True)
    >>> result = engine.generate_decision(target_info, context)

對應設計理念:
    - 內部閉環: 探索(對內) + 分析 + RAG → 了解自身能力
    - 外部閉環: 掃描(對外) + 攻擊 → 收集優化方向
"""

__version__ = "3.0.0-alpha"
__status__ = "架構搭建中"

# NOTE: 其他認知核心模組將在 v3.1 版本中遷移完成後導入
# 計劃包含: neural (RealNeuralCore), rag (RAGEngine), decision (EnhancedDecisionAgent), anti_hallucination

# ✅ 內部閉環連接器已實現
from .internal_loop_connector import InternalLoopConnector

# ✅ 外部閉環連接器已實現
from .external_loop_connector import ExternalLoopConnector

__all__ = [
    # ✅ 雙閉環連接器已導出
    "InternalLoopConnector",
    "ExternalLoopConnector",
]
