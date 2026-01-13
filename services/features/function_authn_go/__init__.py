"""
Authentication Testing Module (Go)

認證測試模組（Go 高性能實現 + Python 接口）

架構: Go 引擎 + Python 包裝器
模組版本: 1.2.0

⚠️ 重要: 需要先編譯 Go 引擎
   cd services/features/function_authn_go
   go build -o bin/authn-worker.exe cmd/worker/main.go
   
   詳見: BUILD_GUIDE.md

使用方式:
    from services.features.function_authn_go.authn_wrapper import scan_authentication
    result = scan_authentication("https://example.com/login", {})

整合狀態 (2025-12-17):
✅ Python 包裝器: authn_wrapper.py
✅ 編譯指南: BUILD_GUIDE.md
🔴 需編譯 Go 引擎
⏳ AI Commander: 準備整合
"""

# 導入 Python 包裝器（推薦）
from .authn_wrapper import scan_authentication, get_engine_info

# 導入舊的 Manager（已廢棄，請使用 authn_wrapper）
# from .authn_manager import AuthnManager, scan_target

__all__ = ["scan_authentication", "get_engine_info"]

__version__ = "1.2.0"
__status__ = "needs_compilation"
__last_updated__ = "2025-12-17"
