from __future__ import annotations

from enum import Enum


class ModuleName(str, Enum):
    API_GATEWAY = "ApiGateway"
    CORE = "CoreModule"
    SCAN = "ScanModule"
    INTEGRATION = "IntegrationModule"
    FUNC_XSS = "FunctionXSS"
    FUNC_SQLI = "FunctionSQLI"
    FUNC_SSRF = "FunctionSSRF"
    FUNC_IDOR = "FunctionIDOR"
    OAST = "OASTService"


class Topic(str, Enum):
    TASK_SCAN_START = "tasks.scan.start"
    TASK_FUNCTION_XSS = "tasks.function.xss"
    TASK_FUNCTION_SQLI = "tasks.function.sqli"
    TASK_FUNCTION_SSRF = "tasks.function.ssrf"
    FUNCTION_IDOR_TASK = "tasks.function.idor"

    RESULTS_SCAN_COMPLETED = "results.scan.completed"
    FINDING_DETECTED = "findings.detected"
    LOG_RESULTS_ALL = "log.results.all"
    STATUS_TASK_UPDATE = "status.task.update"

    FEEDBACK_CORE_STRATEGY = "feedback.core.strategy"
    MODULE_HEARTBEAT = "module.heartbeat"

    COMMAND_TASK_CANCEL = "command.task.cancel"
    CONFIG_GLOBAL_UPDATE = "config.global.update"


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


class VulnerabilityType(str, Enum):
    XSS = "XSS"
    SQLI = "SQL Injection"
    SSRF = "SSRF"
    IDOR = "IDOR"
    BOLA = "BOLA"
    INFO_LEAK = "Information Leak"
    WEAK_AUTH = "Weak Authentication"


class TaskStatus(str, Enum):
    """任務狀態枚舉 - 用於追蹤任務執行狀態"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ScanStatus(str, Enum):
    """掃描狀態枚舉 - 用於追蹤掃描執行狀態"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
