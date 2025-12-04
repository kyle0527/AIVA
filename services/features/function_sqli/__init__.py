"""
AIVA SQL Injection Detection Module

SQL 注入檢測模組，支援六種檢測引擎:
- Error-based: 錯誤型注入檢測
- Boolean-based: 布林型注入檢測
- Time-based: 時間型注入檢測
- Union-based: 聯合查詢注入檢測
- Out-of-band: 帶外注入檢測
- HackingTool: 外部工具集成 (sqlmap, NoSQLMap)

架構: Worker-based (高並發)
風險等級: L2 (需要授權)
模組版本: 2.0.0
"""

from .command_handler import SQLiCommandHandler
from .worker import SqliWorkerService

__all__ = [
    "SQLiCommandHandler",
    "SqliWorkerService",
]

__version__ = "2.0.0"
__architecture__ = "worker-based"
__risk_level__ = "L2"
