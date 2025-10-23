"""
AIVA Performance Feedback Module

性能反饋和優化建議模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .improvement_suggestion_generator import ImprovementSuggestionGenerator
    from .scan_metadata_analyzer import ScanMetadataAnalyzer
    
    __all__ = [
        "ImprovementSuggestionGenerator",
        "ScanMetadataAnalyzer"
    ]
except ImportError:
    __all__ = []
