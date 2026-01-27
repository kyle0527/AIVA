"""
AIVA SQL Injection Detection Module

SQL 注入檢測模組，支援六種檢測引擎:
- Error-based: 錯誤型注入檢測
- Boolean-based: 布林型注入檢測
- Time-based: 時間型注入檢測
- Union-based: 聯合查詢注入檢測
- Out-of-band: 帶外注入檢測
- HackingTool: 外部工具集成 (sqlmap, NoSQLMap)

架構: Detector-based (直接調用核心檢測器)
風險等級: L2 (需要授權)
模組版本: 3.0.0
"""

# 安全導入：避免缺失檔案導致模組無法使用
try:
    from .command_handler import SQLiCommandHandler
except (ImportError, NameError):
    SQLiCommandHandler = None

try:
    from .detector.sqli_detector import SqliDetector
except (ImportError, NameError):
    SqliDetector = None

__all__ = [
    "SQLiCommandHandler",
    "SqliDetector",
]

__version__ = "3.0.0"
__architecture__ = "detector-based"
__risk_level__ = "L2"
__status__ = "production_ready"
__last_updated__ = "2026-01-23"

# 整合狀態 (2026-01-23)
# ✅ CommandHandler 已完成 (SQLiCommandHandler)
# ✅ 核心檢測器架構 (SqliDetector)
# ✅ 6種檢測引擎已完成
# ✅ AI Commander 直接整合
# ❌ Worker 層已移除（不需要）
