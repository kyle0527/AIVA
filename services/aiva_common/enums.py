from __future__ import annotations

from enum import Enum


class ModuleName(str, Enum):
    API_GATEWAY = "ApiGateway"
    CORE = "CoreModule"
    SCAN = "ScanModule"
    INTEGRATION = "IntegrationModule"
    FUNCTION = "FunctionModule"
    FUNC_XSS = "FunctionXSS"
    FUNC_SQLI = "FunctionSQLI"
    FUNC_SSRF = "FunctionSSRF"
    FUNC_IDOR = "FunctionIDOR"
    OAST = "OASTService"
    THREAT_INTEL = "ThreatIntelModule"
    AUTHZ = "AuthZModule"
    POSTEX = "PostExModule"
    REMEDIATION = "RemediationModule"
    BIZLOGIC = "BizLogicModule"


class Topic(str, Enum):
    # Scan Topics
    TASK_SCAN_START = "tasks.scan.start"
    RESULTS_SCAN_COMPLETED = "results.scan.completed"

    # Function Topics
    TASK_FUNCTION_START = "tasks.function.start"
    TASK_FUNCTION_XSS = "tasks.function.xss"
    TASK_FUNCTION_SQLI = "tasks.function.sqli"
    TASK_FUNCTION_SSRF = "tasks.function.ssrf"
    FUNCTION_IDOR_TASK = "tasks.function.idor"
    RESULTS_FUNCTION_COMPLETED = "results.function.completed"

    # AI Training Topics
    TASK_AI_TRAINING_START = "tasks.ai.training.start"
    TASK_AI_TRAINING_EPISODE = "tasks.ai.training.episode"
    TASK_AI_TRAINING_STOP = "tasks.ai.training.stop"
    RESULTS_AI_TRAINING_PROGRESS = "results.ai.training.progress"
    RESULTS_AI_TRAINING_COMPLETED = "results.ai.training.completed"
    RESULTS_AI_TRAINING_FAILED = "results.ai.training.failed"

    # AI Experience & Learning Topics
    EVENT_AI_EXPERIENCE_CREATED = "events.ai.experience.created"
    EVENT_AI_TRACE_COMPLETED = "events.ai.trace.completed"
    EVENT_AI_MODEL_UPDATED = "events.ai.model.updated"
    COMMAND_AI_MODEL_DEPLOY = "commands.ai.model.deploy"

    # RAG Knowledge Topics
    TASK_RAG_KNOWLEDGE_UPDATE = "tasks.rag.knowledge.update"
    TASK_RAG_QUERY = "tasks.rag.query"
    RESULTS_RAG_RESPONSE = "results.rag.response"

    # General Topics
    FINDING_DETECTED = "findings.detected"
    LOG_RESULTS_ALL = "log.results.all"
    STATUS_TASK_UPDATE = "status.task.update"
    FEEDBACK_CORE_STRATEGY = "feedback.core.strategy"
    MODULE_HEARTBEAT = "module.heartbeat"
    COMMAND_TASK_CANCEL = "command.task.cancel"
    CONFIG_GLOBAL_UPDATE = "config.global.update"

    # ThreatIntel Topics
    TASK_THREAT_INTEL_LOOKUP = "tasks.threat_intel.lookup"
    TASK_IOC_ENRICHMENT = "tasks.threat_intel.ioc_enrichment"
    TASK_MITRE_MAPPING = "tasks.threat_intel.mitre_mapping"
    RESULTS_THREAT_INTEL = "results.threat_intel"

    # AuthZ Topics
    TASK_AUTHZ_CHECK = "tasks.authz.check"
    TASK_AUTHZ_ANALYZE = "tasks.authz.analyze"
    RESULTS_AUTHZ = "results.authz"

    # PostEx Topics (僅用於授權測試環境)
    TASK_POSTEX_TEST = "tasks.postex.test"
    TASK_POSTEX_PRIVILEGE_ESCALATION = "tasks.postex.privilege_escalation"
    TASK_POSTEX_LATERAL_MOVEMENT = "tasks.postex.lateral_movement"
    TASK_POSTEX_DATA_EXFILTRATION = "tasks.postex.data_exfiltration"
    TASK_POSTEX_PERSISTENCE = "tasks.postex.persistence"
    RESULTS_POSTEX = "results.postex"

    # Remediation Topics
    TASK_REMEDIATION_GENERATE = "tasks.remediation.generate"
    RESULTS_REMEDIATION = "results.remediation"


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
    RCE = "Remote Code Execution"
    AUTHENTICATION_BYPASS = "Authentication Bypass"
    # BizLogic Vulnerabilities
    PRICE_MANIPULATION = "Price Manipulation"
    WORKFLOW_BYPASS = "Workflow Bypass"
    RACE_CONDITION = "Race Condition"
    FORCED_BROWSING = "Forced Browsing"
    STATE_MANIPULATION = "State Manipulation"


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


class SensitiveInfoType(str, Enum):
    """敏感信息類型枚舉"""

    # 認證憑證
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    SECRET_KEY = "secret_key"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    JWT_TOKEN = "jwt_token"
    SESSION_ID = "session_id"
    AUTH_COOKIE = "auth_cookie"

    # 個人識別信息 (PII)
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # Social Security Number
    CREDIT_CARD = "credit_card"
    ID_CARD = "id_card"
    PASSPORT = "passport"

    # 系統信息
    DATABASE_CONNECTION = "database_connection"
    INTERNAL_IP = "internal_ip"
    AWS_KEY = "aws_key"
    GCP_KEY = "gcp_key"
    AZURE_KEY = "azure_key"
    GITHUB_TOKEN = "github_token"
    SLACK_TOKEN = "slack_token"

    # 路徑和配置
    FILE_PATH = "file_path"
    BACKUP_FILE = "backup_file"
    DEBUG_INFO = "debug_info"
    ERROR_MESSAGE = "error_message"
    STACK_TRACE = "stack_trace"
    SOURCE_CODE = "source_code"
    COMMENT = "comment"


class Location(str, Enum):
    """信息位置枚舉"""

    HTML_BODY = "html_body"
    HTML_COMMENT = "html_comment"
    JAVASCRIPT = "javascript"
    RESPONSE_HEADER = "response_header"
    RESPONSE_BODY = "response_body"
    URL = "url"
    COOKIE = "cookie"
    META_TAG = "meta_tag"
    INLINE_SCRIPT = "inline_script"
    EXTERNAL_SCRIPT = "external_script"


class ThreatLevel(str, Enum):
    """威脅等級枚舉 - 用於威脅情報"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


class IntelSource(str, Enum):
    """威脅情報來源枚舉"""

    VIRUSTOTAL = "virustotal"
    ABUSEIPDB = "abuseipdb"
    SHODAN = "shodan"
    ALIENVAULT_OTX = "alienvault_otx"
    MITRE_ATTACK = "mitre_attack"
    INTERNAL = "internal"


class IOCType(str, Enum):
    """IOC (Indicator of Compromise) 類型枚舉"""

    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    CVE = "cve"


class RemediationType(str, Enum):
    """修復類型枚舉"""

    CODE_FIX = "code_fix"
    CONFIG_CHANGE = "config_change"
    PATCH = "patch"
    UPGRADE = "upgrade"
    WORKAROUND = "workaround"
    MITIGATION = "mitigation"


class RemediationStatus(str, Enum):
    """修復狀態枚舉"""

    PENDING = "pending"
    GENERATED = "generated"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


class Permission(str, Enum):
    """權限枚舉 - 用於授權檢查"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"
    LIST = "list"
    MANAGE = "manage"


class AccessDecision(str, Enum):
    """訪問決策枚舉 - 用於授權結果"""

    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"
    NOT_APPLICABLE = "not_applicable"


class PostExTestType(str, Enum):
    """後滲透測試類型枚舉 - 僅用於授權測試環境"""

    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    PERSISTENCE = "persistence"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    COMMAND_EXECUTION = "command_execution"


class PersistenceType(str, Enum):
    """持久化類型枚舉 - 用於持久化機制檢測"""

    REGISTRY = "registry"
    SCHEDULED_TASK = "scheduled_task"
    SERVICE = "service"
    STARTUP = "startup"
    CRON = "cron"
    SYSTEMD = "systemd"
    AUTOSTART = "autostart"


# ==================== 資產與漏洞生命週期管理 ====================


class BusinessCriticality(str, Enum):
    """業務重要性等級"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Environment(str, Enum):
    """環境類型"""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


class AssetType(str, Enum):
    """資產類型"""

    URL = "url"
    REPOSITORY = "repository"
    HOST = "host"
    CONTAINER = "container"
    API_ENDPOINT = "api_endpoint"
    MOBILE_APP = "mobile_app"
    WEB_APPLICATION = "web_application"
    DATABASE = "database"
    API_SERVICE = "api_service"


class AssetStatus(str, Enum):
    """資產狀態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class VulnerabilityStatus(str, Enum):
    """漏洞狀態 - 用於漏洞生命週期管理"""

    NEW = "new"
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    FIXED = "fixed"
    VERIFIED = "verified"
    RISK_ACCEPTED = "risk_accepted"
    FALSE_POSITIVE = "false_positive"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"


class DataSensitivity(str, Enum):
    """資料敏感度等級"""

    HIGHLY_SENSITIVE = "highly_sensitive"  # 信用卡、健康資料、密碼
    SENSITIVE = "sensitive"  # PII（個人識別信息）
    INTERNAL = "internal"  # 內部資料
    PUBLIC = "public"  # 公開資料


class AssetExposure(str, Enum):
    """資產網路暴露度"""

    INTERNET_FACING = "internet_facing"  # 直接暴露於互聯網
    DMZ = "dmz"  # DMZ 區域
    INTERNAL_NETWORK = "internal_network"  # 內部網路
    ISOLATED = "isolated"  # 隔離網路


class Exploitability(str, Enum):
    """漏洞可利用性評估"""

    PROVEN = "proven"  # 已有公開 exploit
    HIGH = "high"  # 高度可利用
    MEDIUM = "medium"  # 中等可利用性
    LOW = "low"  # 低可利用性
    THEORETICAL = "theoretical"  # 理論上可利用


class ComplianceFramework(str, Enum):
    """合規框架標籤"""

    PCI_DSS = "pci-dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"
    CIS = "cis"


class RiskLevel(str, Enum):
    """風險等級 - 用於風險評估"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ==================== 攻擊路徑分析 ====================


class AttackPathNodeType(str, Enum):
    """攻擊路徑節點類型"""

    ATTACKER = "attacker"
    ASSET = "asset"
    VULNERABILITY = "vulnerability"
    CREDENTIAL = "credential"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    INTERNAL_NETWORK = "internal_network"


class AttackPathEdgeType(str, Enum):
    """攻擊路徑邊類型"""

    EXPLOITS = "exploits"
    LEADS_TO = "leads_to"
    GRANTS_ACCESS = "grants_access"
    EXPOSES = "exposes"
    CONTAINS = "contains"
    CAN_ACCESS = "can_access"
