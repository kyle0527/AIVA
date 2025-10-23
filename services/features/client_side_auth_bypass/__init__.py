# services/features/client_side_auth_bypass/__init__.py
"""
客戶端授權繞過檢測功能模組 (Client-Side Authorization Bypass)

檢測前端 JavaScript 中實現的授權邏輯是否存在缺陷，
例如：
- 僅在客戶端隱藏按鈕或鏈接，但後端 API 未做權限檢查。
- 將用戶角色或權限信息明文存儲在 LocalStorage 或 Cookie 中，且易於篡改。
- 前端路由配置不當，允許未授權用戶訪問管理頁面（即使數據無法加載）。
- JavaScript 代碼中硬編碼了敏感操作的觸發邏輯，可以被繞過。
"""

from .client_side_auth_bypass_worker import ClientSideAuthBypassWorker
from .js_analysis_engine import JavaScriptAnalysisEngine

__all__ = [
    "ClientSideAuthBypassWorker",
    "JavaScriptAnalysisEngine",
]