"""
AIVA Analysis Domain Schemas
============================

分析引擎領域模型，包含：
- 分析結果統一標準
- 語言支持相關模型  
- AI分析模型

此領域專注於代碼分析、漏洞發現的分析過程。
"""

from .results import *
from .language_support import *
# ai_models導入使用TYPE_CHECKING模式，遵循PEP-484循環導入最佳實踐
# from .ai_models import *  # 使用TYPE_CHECKING避免循環導入

__all__ = [
    # 分析結果標準 (analysis.py)
    "BaseAnalysisResult", 
    "JavaScriptAnalysisResult",
    "DataLeak",
    "AnalysisType",
    "LegacyJavaScriptAnalysisResultAdapter",
    # 語言支持 (languages.py)
    "LanguageDetectionResult",
    "LanguageSpecificVulnerability", 
    "MultiLanguageCodebase",
    "LanguageSpecificScanConfig",
    "CrossLanguageAnalysis",
    "LanguageSpecificPayload",
    "AILanguageModel",
    "CodeQualityReport",
    "LanguageInteroperability",
    # AI分析模型 (ai.py部分)
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