"""
AIVA Messaging Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: 2025-10-23T00:00:00Z
🔄 Schema 版本: 1.0.0
"""

from typing import Any

from pydantic import BaseModel, Field


class AivaMessage(BaseModel):
    """AIVA統一訊息格式 - 所有跨服務通訊的標準信封"""

    header: MessageHeader
    """訊息標頭"""

    topic: str = Field(values=["tasks", "findings", "events", "commands", "responses"])
    """訊息主題"""

    schema_version: str = Field(default="1.0")
    """Schema版本"""

    payload: dict[str, Any]
    """訊息載荷"""


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

    payload: dict[str, Any]
    """請求載荷"""

    trace_id: str | None = None
    """追蹤識別碼"""

    timeout_seconds: int = Field(ge=1, le=300, default=30)
    """逾時秒數"""

    metadata: dict[str, Any] = Field(default_factory=dict)
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

    payload: dict[str, Any] | None = None
    """響應載荷"""

    error_code: str | None = None
    """錯誤代碼"""

    error_message: str | None = None
    """錯誤訊息"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """中繼資料"""

    timestamp: str
    """時間戳"""
