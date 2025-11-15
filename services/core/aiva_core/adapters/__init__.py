"""協議適配器模組 - 實現 Gang of Four Adapter 設計模式

此模組提供各種協議適配器，遵循軟件工程最佳實踐。
"""

from .protocol_adapter import ProtocolAdapter, HttpProtocolAdapter, create_http_adapter

__all__ = ["ProtocolAdapter", "HttpProtocolAdapter", "create_http_adapter"]