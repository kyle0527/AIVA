"""
AIVA 數據存儲模組

提供統一的數據存儲接口，支持多種後端：
- SQLite（默認）
- PostgreSQL
- JSONL 文件
- 混合存儲
"""

from .backends import (
    HybridBackend,
    JSONLBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)
from .models import (
    ExperienceSampleModel,
    KnowledgeEntryModel,
    ModelCheckpointModel,
    TraceRecordModel,
    TrainingSessionModel,
)
from .storage_manager import StorageManager

__all__ = [
    "StorageManager",
    "ExperienceSampleModel",
    "TraceRecordModel",
    "TrainingSessionModel",
    "ModelCheckpointModel",
    "KnowledgeEntryModel",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "JSONLBackend",
    "HybridBackend",
]
