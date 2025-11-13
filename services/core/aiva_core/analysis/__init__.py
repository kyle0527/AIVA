"""
Analysis Module - 分析和評估模組

提供 AST/Trace 對比分析、計畫評估、AI增強分析等功能
"""

from .plan_comparator import PlanComparator
# Import AI analysis components from ai_analysis module
try:
    from ..ai_analysis import AIAnalysisEngine, AnalysisType, AIAnalysisResult
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False
    AIAnalysisEngine = None
    AnalysisType = None
    AIAnalysisResult = None

__all__ = ["PlanComparator"]

# Add AI analysis exports if available
if AI_ANALYSIS_AVAILABLE:
    __all__.extend(["AIAnalysisEngine", "AnalysisType", "AIAnalysisResult"])
