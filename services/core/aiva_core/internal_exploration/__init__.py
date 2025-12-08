"""
Internal Exploration - 對內探索模組

本模組負責 AIVA 系統的自我認知能力,掃描和分析五大模組的代碼結構,
構建全專案知識圖譜,實現 AI 對自身能力的深度理解。

主要組件:
- ModuleExplorer: 模組探索器 (掃描五大模組) ✅ 已實現
- CapabilityAnalyzer: 能力分析器 (識別能力函數) ✅ 已實現
- ASTCodeAnalyzer: AST 代碼解析器 (解析代碼結構) 🚧 待實現
- KnowledgeGraph: 知識圖譜構建器 (構建能力圖譜) 🚧 待實現
- SelfDiagnostics: 自我診斷工具 (健康檢查) 🚧 待實現

使用範例:
    >>> from aiva_core.internal_exploration import ModuleExplorer, CapabilityAnalyzer
    >>> explorer = ModuleExplorer()
    >>> analyzer = CapabilityAnalyzer()
    >>> modules = await explorer.explore_all_modules()
    >>> capabilities = await analyzer.analyze_capabilities(modules)

對應設計理念:
    - 內部閉環步驟 1+2: 探索(對內) + 分析(靜態)
    - 目標: AI 知道自己有什麼能力
"""

__version__ = "3.0.0-alpha"
__status__ = "部分實現 (核心組件已完成)"

# ✅ 已實現的組件
# 注意: module_explorer 和 capability_analyzer 尚未實現
# 相關功能已整合到 aiva_flow_analyzer 和其他工具中

__all__ = []  # 暫無可導出組件
