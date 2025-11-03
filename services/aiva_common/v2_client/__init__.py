"""
AIVA V2 易用層客戶端
統一的跨語言通信客戶端，讓 V2 框架比 V1 更容易使用

此模組的核心理念：
- 讓 V2 的調用和 V1 一樣簡單
- 封裝 gRPC 的複雜性
- 提供統一的錯誤處理和重試機制
- 支持自動服務發現和負載均衡
"""

from .aiva_client import AivaClient, aiva_client

__all__ = ["AivaClient", "aiva_client"]