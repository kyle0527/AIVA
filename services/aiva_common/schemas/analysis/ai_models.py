"""
AI Analysis Models - 分析相關的AI模型

從原ai.py中分離出與分析相關的模型，主要包括：
- 訓練相關模型  
- RAG知識庫模型
- 經驗學習模型
- 追蹤記錄模型
"""

# 這裡會包含從ai.py分離出的分析相關模型
# 由於ai.py較大，暫時先保持原導入關係
# 後續會進一步重構

# 暫時從原ai.py導入分析相關模型
from ...ai import (
    AITrainingStartPayload,
    AITrainingProgressPayload, 
    AITrainingCompletedPayload,
    ModelTrainingConfig,
    ExperienceSample,
    TraceRecord,
    RAGKnowledgeUpdatePayload,
    RAGQueryPayload,
    RAGResponsePayload,
)

__all__ = [
    "AITrainingStartPayload",
    "AITrainingProgressPayload", 
    "AITrainingCompletedPayload",
    "ModelTrainingConfig",
    "ExperienceSample",
    "TraceRecord",
    "RAGKnowledgeUpdatePayload",
    "RAGQueryPayload",
    "RAGResponsePayload",
]