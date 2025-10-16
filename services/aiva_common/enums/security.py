"""
安全測試相關枚舉 - 漏洞、攻擊、權限等
"""

from __future__ import annotations

from enum import Enum


class VulnerabilityByLanguage(str, Enum):
    """按程式語言分類的漏洞類型"""

    # 記憶體安全（C/C++）
    BUFFER_OVERFLOW = "buffer_overflow"
    USE_AFTER_FREE = "use_after_free"
    MEMORY_LEAK = "memory_leak"
    NULL_POINTER_DEREFERENCE = "null_pointer_dereference"

    # 型別安全（動態語言）
    TYPE_CONFUSION = "type_confusion"
    PROTOTYPE_POLLUTION = "prototype_pollution"  # JavaScript

    # 並發安全（Go, Rust, Java）
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"
    DATA_RACE = "data_race"

    # 反序列化（Java, Python, C#）
    DESERIALIZATION = "deserialization"
    PICKLE_INJECTION = "pickle_injection"  # Python

    # 程式碼注入
    CODE_INJECTION = "code_injection"
    EVAL_INJECTION = "eval_injection"
    TEMPLATE_INJECTION = "template_injection"

    # 特定語言漏洞
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    XXE = "xxe"
    LDAP_INJECTION = "ldap_injection"

    # 配置和依賴
    DEPENDENCY_CONFUSION = "dependency_confusion"
    SUPPLY_CHAIN = "supply_chain"
    MISCONFIGURATION = "misconfiguration"


class SecurityPattern(str, Enum):
    """安全模式枚舉 - 針對不同程式語言的安全實踐"""

    # 輸入驗證
    INPUT_VALIDATION = "input_validation"
    SANITIZATION = "sanitization"
    ENCODING = "encoding"

    # 認證授權
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    JWT_VALIDATION = "jwt_validation"

    # 加密
    ENCRYPTION = "encryption"
    HASHING = "hashing"
    DIGITAL_SIGNATURE = "digital_signature"

    # 安全編碼
    SECURE_RANDOM = "secure_random"
    SAFE_DESERIALIZATION = "safe_deserialization"
    BOUNDS_CHECKING = "bounds_checking"

    # 並發安全
    THREAD_SAFETY = "thread_safety"
    ATOMIC_OPERATIONS = "atomic_operations"
    LOCK_FREE = "lock_free"

    # 錯誤處理
    ERROR_HANDLING = "error_handling"
    FAIL_SECURE = "fail_secure"
    LOGGING = "logging"


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


class Exploitability(str, Enum):
    """漏洞可利用性評估"""

    PROVEN = "proven"  # 已有公開 exploit
    HIGH = "high"  # 高度可利用
    MEDIUM = "medium"  # 中等可利用性
    LOW = "low"  # 低可利用性
    THEORETICAL = "theoretical"  # 理論上可利用


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
