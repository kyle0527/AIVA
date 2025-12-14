"""
Post-Exploitation Module

後滲透測試模組

⚠️ 警告: 僅用於授權測試
架構: 統一管理器 + 多引擎
模組版本: 1.1.0

使用方式:
    from services.features.function_postex import PostExManager
    manager = PostExManager()
    result = manager.scan("privilege_escalation", safe_mode=True)
"""

from services.features.function_postex.postex_manager import (
    PostExManager,
    scan_target
)

__all__ = ["PostExManager", "scan_target"]

__version__ = "1.1.0"
__status__ = "ready"
