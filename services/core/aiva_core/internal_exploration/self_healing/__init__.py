"""
Self-Healing - 自我修復子模組
================================

本子模組專注於代碼自我診斷和修復能力。

核心組件:

🔬 診斷分析器:
    - CoreAnalyzer: 統一分析入口
    - DataFlowBreakpointAnalyzer: 數據流斷點檢測
    - MissingConnectionAnalyzer: 缺失連接分析
    - PracticalAnalyzer: 智能過濾和分級

使用方式:

    from internal_exploration.self_healing import CoreAnalyzer
    
    analyzer = CoreAnalyzer("path/to/source")
    report = analyzer.full_analysis()

或直接從主模組導入:

    from internal_exploration import CoreAnalyzer
"""

__version__ = "10.0.0"

# 核心分析器
# 各個獨立分析器
from .analyze_dataflow_breakpoints import DataFlowBreakpointAnalyzer
from .analyze_missing_function_connections import MissingConnectionAnalyzer
from .core_analyzer import AnalysisReport, CoreAnalyzer
from .practical_analyzer import PracticalAnalyzer

__all__ = [
    'CoreAnalyzer',
    'AnalysisReport',
    'DataFlowBreakpointAnalyzer',
    'MissingConnectionAnalyzer',
    'PracticalAnalyzer',
]
