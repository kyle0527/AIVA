"""
AIVA Cross-Language Adapters
跨語言適配器模組

此模組提供各種程式語言的適配器：
- Python 適配器：原生支持
- Rust 適配器：高性能計算和系統級操作
- Go 適配器：微服務和並發處理
- JavaScript/TypeScript 適配器：前端和 Node.js 支持

每個適配器提供統一的接口：
1. 數據序列化/反序列化
2. 錯誤處理和映射
3. 異步執行支持
4. 資源管理
"""

from .go_adapter import GoAdapter, GoConfig, create_go_adapter
from .rust_adapter import RustAdapter, RustConfig, create_rust_adapter

__all__ = [
    "RustAdapter",
    "RustConfig",
    "create_rust_adapter",
    "GoAdapter",
    "GoConfig",
    "create_go_adapter",
]
