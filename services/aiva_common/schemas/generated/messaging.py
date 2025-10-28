"""
AIVA 訊息傳遞 Schema - 自動生成 (相容版本)
====================================

此檔案基於手動維護的 Schema 定義自動生成，確保完全相容

⚠️  此檔案由 core_schema_sot.yaml 自動生成，請勿手動修改
📅 最後更新: 2025-10-28T10:55:40.860463
🔄 Schema 版本: 1.0.0
🎯 相容性: 完全相容手動維護版本
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from pydantic import BaseModel, Field

# 導入基礎類型以保持相容性
try:
    from .base_types import MessageHeader
except ImportError:
from services.aiva_common.schemas.base import MessageHeader

class MessagePayload(BaseModel):
    """訊息負載 - 統一的訊息傳遞格式"""
    
    header: MessageHeader
    payload_type: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class MessageResponse(BaseModel):
    """訊息回應格式"""
    
    response_id: str
    original_message_id: str
    status: str = "success"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
