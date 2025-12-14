"""
Authentication Testing Module (Go)

認證測試模組（Go 高性能實現 + Python 接口）

架構: Go 引擎 + Python 接口層
模組版本: 1.1.0

使用方式:
    from services.features.function_authn_go import AuthnManager
    manager = AuthnManager()
    result = manager.scan("https://example.com/login")
"""

from services.features.function_authn_go.authn_manager import (
    AuthnManager,
    scan_target
)

__all__ = ["AuthnManager", "scan_target"]

__version__ = "1.1.0"
__status__ = "ready"
