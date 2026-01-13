"""SQLI Integration Tools - 整合工具模組

提供與其他工具（Sqlmap、NoSQLMap等）的整合功能
"""

from .sql_tools import (
    SQLTarget,
    SQLInjectionResult,
    SqlmapIntegration,
    SQLInjectionManager,
)

__all__ = [
    "SQLTarget",
    "SQLInjectionResult",
    "SqlmapIntegration",
    "SQLInjectionManager",
]
