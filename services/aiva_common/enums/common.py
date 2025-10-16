"""
通用枚舉 - 狀態、級別、類型等基礎枚舉
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFORMATIONAL = "Informational"


class Confidence(str, Enum):
    CERTAIN = "Certain"
    FIRM = "Firm"
    POSSIBLE = "Possible"


class TaskStatus(str, Enum):
    """任務狀態枚舉 - 用於追蹤任務執行狀態"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TestStatus(str, Enum):
    """測試狀態枚舉 - 用於追蹤測試執行狀態"""

    DRAFT = "draft"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScanStatus(str, Enum):
    """掃描狀態枚舉 - 用於追蹤掃描執行狀態"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThreatLevel(str, Enum):
    """威脅等級枚舉 - 用於威脅情報"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """風險等級 - 用於風險評估"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ==================== 攻擊路徑分析 ====================


class RemediationStatus(str, Enum):
    """修復狀態枚舉"""

    PENDING = "pending"
    GENERATED = "generated"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"
