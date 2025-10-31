"""
訊息佇列相關 Schema

此模組定義了訊息佇列系統使用的標準信封格式。
"""

from typing import Any

from pydantic import BaseModel, Field

from ..enums import Topic
from .base import MessageHeader


class AivaMessage(BaseModel):
    """AIVA 統一訊息格式 - 訊息佇列的標準信封"""

    header: MessageHeader
    topic: Topic
    schema_version: str = "1.0"
    payload: dict[str, Any]


# ==================== 模組間通訊統一包裝 ====================


class AIVARequest(BaseModel):
    """統一的請求包裝器 - 用於模組間的請求消息"""

    request_id: str
    source_module: str
    target_module: str
    request_type: str
    payload: dict[str, Any]
    trace_id: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class AIVAResponse(BaseModel):
    """統一的響應包裝器 - 用於模組間的響應消息"""

    request_id: str
    response_type: str
    success: bool
    payload: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class AIVAEvent(BaseModel):
    """統一的事件包裝器 - 用於模組間的事件通知"""

    event_id: str
    event_type: str
    source_module: str
    payload: dict[str, Any]
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class AIVACommand(BaseModel):
    """統一的命令包裝器 - 用於模組間的命令消息"""

    command_id: str
    command_type: str
    source_module: str
    target_module: str
    payload: dict[str, Any]
    priority: int = Field(default=0, ge=0, le=10)
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
