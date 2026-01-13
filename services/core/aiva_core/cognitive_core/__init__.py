"""
Cognitive Core - AI 認知核心

本模組是 AIVA 的「大腦」,負責思考和決策。整合神經網路推理、RAG 知識增強、
決策支援和反幻覺機制,實現 AI 自我優化雙重閉環的核心決策功能。

主要組件:
- neural: 5M 參數特化決策神經網路 (RealAICore, RealDecisionEngine)
- rag: RAG 增強系統 (知識檢索與上下文增強)
- decision: 決策支援系統 (包含 Bug Bounty 特化決策)
- anti_hallucination: 反幻覺機制
- InternalLoopConnector: 內部閉環連接器 (探索結果 → RAG)
- ExternalLoopConnector: 外部閉環連接器 (偏差報告 → 學習系統)

✅ 特化 AI 設計:
- 5M Decision Engine 專注於安全測試決策
- 結構化輸入/輸出，高效任務處理
- Bug Bounty 特化決策 (ROI 評估、WAF 繞過、目標優先級)

使用範例:
    >>> from aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
    >>> engine = RealDecisionEngine(use_5m_model=True)
    >>> result = engine.generate_decision(target_info, context)
    
    >>> from aiva_core.cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
    >>> agent = EnhancedDecisionAgent()
    >>> decision = agent.decide_phase1_strategy(phase0_result, target_value=1000)

對應設計理念:
    - 內部閉環: 探索(對內) + 分析 + RAG → 了解自身能力
    - 外部閉環: 掃描(對外) + 攻擊 → 收集優化方向
    - Bug Bounty 閉環: Phase0 → AI 決策 → Phase1 → 價值評估
"""

__version__ = "3.1.0-beta"
__status__ = "Bug Bounty 特化完成"

# ✅ 內部閉環連接器已實現
from .internal_loop_connector import InternalLoopConnector

# ✅ 外部閉環連接器已實現
from .external_loop_connector import ExternalLoopConnector

# ✅ Bug Bounty 決策代理已導出
from .decision.enhanced_decision_agent import EnhancedDecisionAgent

__all__ = [
    # ✅ 雙閉環連接器
    "InternalLoopConnector",
    "ExternalLoopConnector",
    # ✅ Bug Bounty AI 決策
    "EnhancedDecisionAgent",
]
