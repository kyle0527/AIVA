"""
Internal Exploration - 對內探索與自我修復模組

本模組負責 AIVA 系統的自我認知與自我修復能力。

核心能力體系:

🔍 階段 1: 自我認知 (原有功能)
    ✅ AST 靜態分析 - 掃描並解析 Python 代碼結構
    ✅ 函數調用追蹤 - 建立完整的調用圖 (Call Graph)
    ✅ 數據流分類 - 分類數據流，按模組/長度/起終點
    ✅ 動態執行 - 自動導入模組、實例化類別、執行方法
    ✅ 知識圖譜生成 - classification_data.json

🩺 階段 2: 自我診斷 (v10.0.0 新增)
    ✅ 數據流斷點檢測 - 發現架構瓶頸和死胡同
    ✅ 未連接函數識別 - 發現孤立和未整合的功能
    ✅ 連接關係分析 - 識別缺失的函數連接
    ✅ 智能錯誤過濾 - 過濾噪音，聚焦真實問題
    ✅ 影響評估 - 評估問題的嚴重程度
    ✅ 核心分析器 - 統一入口，整合所有工具

🔧 階段 3: 自我修復 (規劃中)
    🚧 自動修復建議 - 生成具體的修復代碼
    🚧 類型錯誤修復 - 自動修正類型註解
    🚧 導入錯誤修復 - 自動添加缺失的導入
    🚧 連接修復 - 自動連接孤立的函數

使用方式:

    # 推薦：使用核心分析器（統一入口）
    from services.core.aiva_core.internal_exploration import CoreAnalyzer
    
    analyzer = CoreAnalyzer("path/to/source")
    
    # 完整深度分析（慢但全面）
    report = analyzer.full_analysis()
    
    # 快速掃描（只檢測關鍵問題）
    report = analyzer.quick_scan()
    
    # 只診斷 CRITICAL
    report = analyzer.diagnose_critical_only()
    
    # 單獨使用各分析器
    from services.core.aiva_core.internal_exploration import (
        AIVAFlowAnalyzer,           # 流程分析
        DataFlowBreakpointAnalyzer, # 數據流斷點
        MissingConnectionAnalyzer,  # 缺失連接
        PracticalAnalyzer,          # 智能過濾
    )
    
    # 命令行使用
    python -m internal_exploration.core_analyzer path/to/source --mode full
    python -m internal_exploration.core_analyzer path/to/source --mode quick
    python -m internal_exploration.core_analyzer path/to/source --mode critical

版本歷史:
    - v10.0.0 (2024-01): 添加自我修復能力，核心分析器統一入口
    - v9.0.0 (2024-01): 完善自我診斷工具鏈
    - v8.0.0: 添加數據流斷點檢測
    - v7.0.0: 基礎流程分析功能
"""

__version__ = "10.0.0"
__status__ = "Production"

# 核心分析器（推薦使用）- 從 self_healing 子模組導入
from .self_healing import CoreAnalyzer, AnalysisReport

# 自我修復分析器
try:
    from .self_healing import (
        DataFlowBreakpointAnalyzer,
        MissingConnectionAnalyzer,
        PracticalAnalyzer,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"自我修復分析器導入失敗: {e}")

# 基礎流程分析器（保留在主模組）
try:
    from .python_tools.aiva_flow_analyzer import AIVAFlowAnalyzer
except ImportError as e:
    import warnings
    warnings.warn(f"流程分析器導入失敗: {e}")

__all__ = [
    # 核心（推薦使用）
    'CoreAnalyzer',
    'AnalysisReport',
    
    # 自我修復分析器
    'DataFlowBreakpointAnalyzer',
    'MissingConnectionAnalyzer',
    'PracticalAnalyzer',
    
    # 基礎分析器
    'AIVAFlowAnalyzer',
]
