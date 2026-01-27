"""
AIVA Async_Utils Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from .base_types import *
from ...enums import AsyncTaskStatus


class RetryConfig(BaseModel):
    """重試配置"""

    max_attempts: int = Field(ge=1, le=10, default=3)
    """最大重試次數"""

    backoff_base: float = Field(ge=0.1, default=1.0)
    """退避基礎時間(秒)"""

    backoff_factor: float = Field(ge=1.0, default=2.0)
    """退避倍數"""

    max_backoff: float = Field(ge=1.0, default=60.0)
    """最大退避時間(秒)"""

    exponential_backoff: bool = Field(default=True)
    """是否使用指數退避"""


class ResourceLimits(BaseModel):
    """資源限制配置"""

    max_memory_mb: Optional[int] = Field(ge=1, default=None)
    """最大內存限制(MB)"""

    max_cpu_percent: Optional[float] = Field(ge=0.1, le=100.0, default=None)
    """最大CPU使用率(%)"""

    max_execution_time: Optional[int] = Field(ge=1, default=None)
    """最大執行時間(秒)"""

    max_concurrent_tasks: int = Field(ge=1, le=100, default=10)
    """最大並發任務數"""


class AsyncTaskConfig(BaseModel):
    """異步任務配置"""

    task_name: str
    """任務名稱"""

    timeout_seconds: int = Field(ge=1, le=3600, default=30)
    """超時時間(秒)"""

    retry_config: RetryConfig
    """重試配置"""

    priority: int = Field(ge=1, le=10, default=5)
    """任務優先級"""

    resource_limits: ResourceLimits
    """資源限制"""

    tags: List[str] = Field(default_factory=list)
    """任務標籤"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """任務元數據"""


class AsyncTaskResult(BaseModel):
    """異步任務結果"""

    task_id: str
    """任務ID"""

    task_name: str
    """任務名稱"""

    status: AsyncTaskStatus
    """任務狀態"""

    result: Optional[Dict[str, Any]] = None
    """執行結果"""

    error_message: Optional[str] = None
    """錯誤信息"""

    execution_time_ms: float = Field(ge=0)
    """執行時間(毫秒)"""

    start_time: datetime
    """開始時間"""

    end_time: Optional[datetime] = None
    """結束時間"""

    retry_count: int = Field(ge=0, default=0)
    """重試次數"""

    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    """資源使用情況"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """結果元數據"""


class AsyncBatchConfig(BaseModel):
    """異步批次任務配置"""

    batch_id: str
    """批次ID"""

    batch_name: str
    """批次名稱"""

    tasks: List[AsyncTaskConfig]
    """任務列表"""

    max_concurrent: int = Field(ge=1, le=50, default=5)
    """最大並發數"""

    stop_on_first_error: bool = Field(default=False)
    """遇到第一個錯誤時停止"""

    batch_timeout_seconds: int = Field(ge=1, default=3600)
    """批次超時時間(秒)"""


class AsyncBatchResult(BaseModel):
    """異步批次任務結果"""

    batch_id: str
    """批次ID"""

    batch_name: str
    """批次名稱"""

    total_tasks: int = Field(ge=0)
    """總任務數"""

    completed_tasks: int = Field(ge=0, default=0)
    """已完成任務數"""

    failed_tasks: int = Field(ge=0, default=0)
    """失敗任務數"""

    task_results: List[AsyncTaskResult] = Field(default_factory=list)
    """任務結果列表"""

    batch_status: Literal['pending', 'running', 'completed', 'failed', 'cancelled', 'partial'] = Field(default='pending')
    """批次狀態"""

    start_time: datetime
    """開始時間"""

    end_time: Optional[datetime] = None
    """結束時間"""

    total_execution_time_ms: float = Field(ge=0, default=0)
    """總執行時間(毫秒)"""

