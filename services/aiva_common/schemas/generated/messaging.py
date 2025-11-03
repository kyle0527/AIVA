"""
AIVA Messaging Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .base_types import *


class AivaMessage(BaseModel):
    """AIVA統一訊息格式 - 所有跨服務通訊的標準信封 - V2增強版"""

    header: MessageHeader
    """訊息標頭"""

    topic: str = Field(values=['tasks.scan.start', 'tasks.function.start', 'tasks.ai.training.start', 'results.scan.completed', 'results.function.completed', 'findings.detected', 'events.ai.experience.created', 'commands.task.cancel', 'responses.task.status'])
    """訊息主題（枚舉化Topic管理）"""

    schema_version: str = Field(default="1.1")
    """Schema版本（V2統一架構）"""

    source_module: str
    """來源模組識別（發送者）"""

    target_module: Optional[str] = None
    """目標模組識別（接收者，廣播時可為空）"""

    trace_id: str
    """分散式追蹤識別碼"""

    correlation_id: Optional[str] = None
    """關聯識別碼（用於請求響應配對）"""

    routing_strategy: str = Field(values=['broadcast', 'direct', 'fanout', 'round_robin'], default="broadcast")
    """路由策略"""

    priority: int = Field(default=5)
    """訊息優先級（1-10，10最高）"""

    ttl_seconds: Optional[int] = None
    """訊息存活時間（秒）"""

    payload: Dict[str, Any]
    """訊息載荷"""

    metadata: Optional[Dict[str, Any]] = None
    """額外中繼資料"""


class AIVARequest(BaseModel):
    """統一請求格式 - 模組間請求通訊"""

    request_id: str
    """請求識別碼"""

    source_module: str
    """來源模組"""

    target_module: str
    """目標模組"""

    request_type: str
    """請求類型"""

    payload: Dict[str, Any]
    """請求載荷"""

    trace_id: Optional[str] = None
    """追蹤識別碼"""

    timeout_seconds: int = Field(ge=1, le=300, default=30)
    """逾時秒數"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """中繼資料"""

    timestamp: str
    """時間戳"""


class AIVAResponse(BaseModel):
    """統一響應格式 - 模組間響應通訊"""

    request_id: str
    """對應的請求識別碼"""

    response_type: str
    """響應類型"""

    success: bool
    """執行是否成功"""

    payload: Optional[Dict[str, Any]] = None
    """響應載荷"""

    error_code: Optional[str] = None
    """錯誤代碼"""

    error_message: Optional[str] = None
    """錯誤訊息"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """中繼資料"""

    timestamp: str
    """時間戳"""

