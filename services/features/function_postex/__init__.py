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

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "manager-based"
__last_updated__ = "2026-01-23"

# 架構說明 (2026-01-23)
# ✅ Manager-based 架構（統一管理器）
# ✅ PostExManager 與 scan_target 可直接調用
# ❌ Worker 層已移除（不需要）
