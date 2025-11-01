"""
AIVA Common Schemas - 核心基礎設施
=====================================

此模組提供所有其他schema模組依賴的核心基礎設施：
- 通用基礎模型和類型
- 訊息系統基礎架構
- 共享枚舉和常量

設計原則：
- 最小依賴：只依賴標準庫和Pydantic
- 穩定接口：避免頻繁變更
- 向後相容：維護API穩定性
"""

from .common import *
from .messaging import *

__all__ = [
    # 從 common.py 匯出
    "APIResponse",
    "MessageHeader", 
    "Authentication",
    "RateLimit",
    "ScanScope",
    "Asset",
    "Summary",
    "Fingerprints",
    "ExecutionError",
    "RiskFactor",
    "Task",
    "TaskDependency",
    # 從 messaging.py 匯出
    "AivaMessage",
    "AIVARequest",
    "AIVAResponse", 
    "AIVAEvent",
    "AIVACommand",
]