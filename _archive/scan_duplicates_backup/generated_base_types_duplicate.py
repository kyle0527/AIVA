"""
AIVA 基礎類型 Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: 2025-10-23T00:00:00Z
🔄 Schema 版本: 1.0.0
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MessageHeader(BaseModel):
    """統一訊息標頭 - 所有跨服務通訊的基礎"""

    message_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", max_length=128)
    """唯一訊息識別碼"""

    trace_id: str = Field(pattern=r"^[a-fA-F0-9-]+$")
    """分散式追蹤識別碼"""

    correlation_id: str | None = None
    """關聯識別碼 - 用於請求-響應配對"""

    source_module: str = Field(
        values=[
            "ai_engine",
            "attack_engine",
            "scan_engine",
            "integration_services",
            "feature_detection",
        ]
    )
    """來源模組名稱"""

    timestamp: datetime
    """訊息時間戳"""

    version: str = Field(default="1.0")
    """Schema版本號"""


class Target(BaseModel):
    """掃描/攻擊目標定義"""

    url: str = Field(url=True)
    """目標URL"""

    parameter: str | None = None
    """目標參數名稱"""

    method: str | None = Field(
        values=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        default="GET",
    )
    """HTTP方法"""

    headers: dict[str, str] = Field(default_factory=dict)
    """HTTP標頭"""

    params: dict[str, Any] = Field(default_factory=dict)
    """HTTP參數"""

    body: str | None = None
    """HTTP請求體"""


class Vulnerability(BaseModel):
    """漏洞資訊定義"""

    name: str = Field(max_length=255)
    """漏洞名稱"""

    cwe: str | None = Field(pattern=r"^CWE-[0-9]+$", default=None)
    """CWE編號"""

    severity: str = Field(values=["critical", "high", "medium", "low", "info"])
    """嚴重程度"""

    confidence: str = Field(values=["confirmed", "firm", "tentative"])
    """信心度"""

    description: str | None = None
    """漏洞描述"""
