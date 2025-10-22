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


class TaskType(str, Enum):
    """任務類型枚舉 - 定義不同的功能掃描類型"""
    
    FUNCTION_SSRF = "function_ssrf"
    FUNCTION_SQLI = "function_sqli"
    FUNCTION_XSS = "function_xss"
    FUNCTION_IDOR = "function_idor"
    FUNCTION_GRAPHQL_AUTHZ = "function_graphql_authz"
    FUNCTION_API_TESTING = "function_api_testing"
    FUNCTION_BUSINESS_LOGIC = "function_business_logic"
    FUNCTION_POST_EXPLOITATION = "function_post_exploitation"
    FUNCTION_EASM_DISCOVERY = "function_easm_discovery"
    FUNCTION_THREAT_INTEL = "function_threat_intel"


class ScanStrategy(str, Enum):
    """掃描策略枚舉 - 定義掃描的深度和方法"""
    
    FAST = "fast"
    NORMAL = "normal"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"
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


class ErrorCategory(str, Enum):
    """錯誤分類 - 用於統計和分析"""

    NETWORK = "network"  # 網絡錯誤
    TIMEOUT = "timeout"  # 超時錯誤
    RATE_LIMIT = "rate_limit"  # 速率限制
    VALIDATION = "validation"  # 驗證錯誤
    PROTECTION = "protection"  # 保護機制檢測到
    PARSING = "parsing"  # 解析錯誤
    AUTHENTICATION = "authentication"  # 認證錯誤
    AUTHORIZATION = "authorization"  # 授權錯誤
    UNKNOWN = "unknown"  # 未知錯誤


class StoppingReason(str, Enum):
    """Early Stopping 原因 - 用於記錄檢測提前終止的原因"""

    MAX_VULNERABILITIES = "max_vulnerabilities_reached"  # 達到最大漏洞數
    TIME_LIMIT = "time_limit_exceeded"  # 超過時間限制
    PROTECTION_DETECTED = "protection_detected"  # 檢測到防護
    ERROR_THRESHOLD = "error_threshold_exceeded"  # 錯誤率過高
    RATE_LIMITED = "rate_limited"  # 被速率限制
    NO_RESPONSE = "no_response_timeout"  # 無響應超時
    MANUAL_STOP = "manual_stop"  # 手動停止
    RESOURCE_EXHAUSTED = "resource_exhausted"  # 資源耗盡
