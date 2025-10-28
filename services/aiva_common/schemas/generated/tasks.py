"""
AIVA 任務管理 Schema - 自動生成 (相容版本)
====================================

此檔案基於手動維護的 Schema 定義自動生成，確保完全相容

⚠️  此檔案由 core_schema_sot.yaml 自動生成，請勿手動修改
📅 最後更新: 2025-10-28T10:55:40.862313
🔄 Schema 版本: 1.0.0
🎯 相容性: 完全相容手動維護版本
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from pydantic import BaseModel, Field
from enum import Enum

class TaskStatus(str, Enum):
    """任務狀態枚舉"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """任務優先級枚舉"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskPayload(BaseModel):
    """任務負載 - 統一的任務定義格式"""
    
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    target: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None

class TaskResult(BaseModel):
    """任務結果格式"""
    
    task_id: str
    status: TaskStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
