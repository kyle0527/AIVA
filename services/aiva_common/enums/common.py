"""
通用枚舉 - 狀態、級別、類型等基礎枚舉

遵循官方標準：
- CVSS v4.0: Common Vulnerability Scoring System
- RFC 5424: Syslog Protocol  
- RFC 7231: HTTP/1.1 Semantics and Content
"""

from enum import Enum

# 常用 MIME 類型常量
HTML_MIME_TYPE = "text/html"


class CVSSSeverity(str, Enum):
    """CVSS v4.0 官方嚴重程度等級
    
    基於 CVSS v4.0 規範: https://www.first.org/cvss/v4.0/specification-document
    嚴重程度按照官方分數範圍定義
    """
    NONE = "None"        # 0.0 - 無影響
    LOW = "Low"          # 0.1-3.9 - 低風險  
    MEDIUM = "Medium"    # 4.0-6.9 - 中等風險
    HIGH = "High"        # 7.0-8.9 - 高風險
    CRITICAL = "Critical" # 9.0-10.0 - 極高風險


# 向後相容別名 (將於 v6.0 移除)
Severity = CVSSSeverity


class Confidence(str, Enum):
    CERTAIN = "certain"
    FIRM = "firm"
    POSSIBLE = "possible"


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


class VulnerabilityRiskLevel(str, Enum):
    """漏洞風險等級 - 用於整體風險評估和攻擊路徑分析
    
    與 Severity 的區別：
    - Severity: 單個漏洞的嚴重程度 (技術層面)
    - VulnerabilityRiskLevel: 整體業務風險等級 (業務層面)
    
    遵循 AIVA Common 開發規範 - 明確語義區分原則
    """

    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# 向後相容別名，將於下一版本移除
RiskLevel = VulnerabilityRiskLevel


# ============================================================================
# 系統和基礎設施枚舉
# ============================================================================


class OperatingSystem(str, Enum):
    """操作系統類型"""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNIX = "unix"
    BSD = "bsd"
    SOLARIS = "solaris"
    AIX = "aix"
    ANDROID = "android"
    IOS = "ios"
    EMBEDDED = "embedded"
    CONTAINER = "container"
    VIRTUAL = "virtual"


class Architecture(str, Enum):
    """系統架構"""

    X86 = "x86"
    X64 = "x64"
    ARM = "arm"
    ARM64 = "arm64"
    MIPS = "mips"
    SPARC = "sparc"
    POWER = "power"
    RISC_V = "risc_v"


class ServiceType(str, Enum):
    """服務類型"""

    WEB_SERVICE = "web_service"
    DATABASE_SERVICE = "database_service"
    FILE_SERVICE = "file_service"
    EMAIL_SERVICE = "email_service"
    DNS_SERVICE = "dns_service"
    AUTHENTICATION_SERVICE = "authentication_service"
    LOGGING_SERVICE = "logging_service"
    MONITORING_SERVICE = "monitoring_service"
    BACKUP_SERVICE = "backup_service"
    PROXY_SERVICE = "proxy_service"
    LOAD_BALANCER = "load_balancer"
    FIREWALL = "firewall"
    VPN = "vpn"
    API_GATEWAY = "api_gateway"
    MESSAGE_QUEUE = "message_queue"
    CACHE_SERVICE = "cache_service"
    SEARCH_SERVICE = "search_service"
    NOTIFICATION_SERVICE = "notification_service"
    SCHEDULER = "scheduler"
    CONTAINER_ORCHESTRATOR = "container_orchestrator"


class DatabaseType(str, Enum):
    """數據庫類型"""

    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    SQL_SERVER = "sql_server"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    CASSANDRA = "cassandra"
    DYNAMODB = "dynamodb"
    COUCHDB = "couchdb"
    NEO4J = "neo4j"
    INFLUXDB = "influxdb"
    CLICKHOUSE = "clickhouse"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"


class ContainerTechnology(str, Enum):
    """容器技術"""

    DOCKER = "docker"
    CONTAINERD = "containerd"
    PODMAN = "podman"
    LXC = "lxc"
    RUNC = "runc"
    CRIO = "crio"


class OrchestrationPlatform(str, Enum):
    """編排平台"""

    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    NOMAD = "nomad"
    MESOS = "mesos"
    OPENSHIFT = "openshift"
    RANCHER = "rancher"
    ECS = "ecs"
    AKS = "aks"
    GKE = "gke"
    EKS = "eks"


# ============================================================================
# 網絡和通信枚舉
# ============================================================================


class NetworkLayer(str, Enum):
    """網絡層級 (OSI Model)"""

    PHYSICAL = "layer_1_physical"
    DATA_LINK = "layer_2_data_link"
    NETWORK = "layer_3_network"
    TRANSPORT = "layer_4_transport"
    SESSION = "layer_5_session"
    PRESENTATION = "layer_6_presentation"
    APPLICATION = "layer_7_application"


class NetworkZone(str, Enum):
    """網絡區域"""

    DMZ = "dmz"
    INTERNAL = "internal"
    EXTERNAL = "external"
    MANAGEMENT = "management"
    GUEST = "guest"
    QUARANTINE = "quarantine"
    HIGH_SECURITY = "high_security"
    LOW_SECURITY = "low_security"


class CommunicationProtocol(str, Enum):
    """通信協議"""

    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    FTPS = "ftps"
    SFTP = "sftp"
    SSH = "ssh"
    TELNET = "telnet"
    SMTP = "smtp"
    SMTPS = "smtps"
    POP3 = "pop3"
    POP3S = "pop3s"
    IMAP = "imap"
    IMAPS = "imaps"
    DNS = "dns"
    DHCP = "dhcp"
    SNMP = "snmp"
    LDAP = "ldap"
    LDAPS = "ldaps"
    SMB = "smb"
    CIFS = "cifs"
    NFS = "nfs"
    RDP = "rdp"
    VNC = "vnc"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    AMQP = "amqp"
    KAFKA = "kafka"


class EncryptionAlgorithm(str, Enum):
    """加密算法"""

    AES_128 = "aes_128"
    AES_192 = "aes_192"
    AES_256 = "aes_256"
    RSA_1024 = "rsa_1024"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECC_P256 = "ecc_p256"
    ECC_P384 = "ecc_p384"
    ECC_P521 = "ecc_p521"
    CHACHA20 = "chacha20"
    BLOWFISH = "blowfish"
    TWOFISH = "twofish"
    DES = "des"
    TRIPLE_DES = "triple_des"


# ============================================================================
# 數據和存儲枚舉
# ============================================================================


class DataClassification(str, Enum):
    """數據分類"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class DataState(str, Enum):
    """數據狀態"""

    AT_REST = "at_rest"
    IN_TRANSIT = "in_transit"
    IN_USE = "in_use"
    IN_MEMORY = "in_memory"
    ARCHIVED = "archived"
    DELETED = "deleted"


class StorageType(str, Enum):
    """存儲類型"""

    LOCAL_DISK = "local_disk"
    NETWORK_ATTACHED = "network_attached"
    STORAGE_AREA_NETWORK = "storage_area_network"
    CLOUD_STORAGE = "cloud_storage"
    OBJECT_STORAGE = "object_storage"
    BLOCK_STORAGE = "block_storage"
    FILE_STORAGE = "file_storage"
    TAPE_STORAGE = "tape_storage"
    OPTICAL_STORAGE = "optical_storage"
    SOLID_STATE = "solid_state"
    MAGNETIC = "magnetic"


class BackupType(str, Enum):
    """備份類型"""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    MIRROR = "mirror"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"
    COMPRESSED = "compressed"
    ENCRYPTED = "encrypted"


# ============================================================================
# 性能和監控枚舉
# ============================================================================


class MetricCategory(str, Enum):
    """指標類別"""

    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    COST = "cost"
    QUALITY = "quality"
    USAGE = "usage"
    CAPACITY = "capacity"
    EFFICIENCY = "efficiency"


class AlertSeverity(str, Enum):
    """告警嚴重程度"""

    EMERGENCY = "emergency"
    ALERT = "alert"
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFORMATIONAL = "informational"
    DEBUG = "debug"


class MonitoringScope(str, Enum):
    """監控範圍"""

    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    BUSINESS = "business"
    USER_EXPERIENCE = "user_experience"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"


class ThresholdType(str, Enum):
    """閾值類型"""

    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    RANGE = "range"
    DELTA = "delta"
    PERCENTAGE = "percentage"
    RATE = "rate"
    COUNT = "count"
    BOOLEAN = "boolean"


# ============================================================================
# 集成和API枚舉
# ============================================================================


class IntegrationType(str, Enum):
    """集成類型"""

    REST_API = "rest_api"
    GRAPHQL = "graphql"
    SOAP = "soap"
    WEBHOOK = "webhook"
    MESSAGE_QUEUE = "message_queue"
    DATABASE_SYNC = "database_sync"
    FILE_SYNC = "file_sync"
    EVENT_DRIVEN = "event_driven"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time"
    ETL = "etl"
    STREAMING = "streaming"


class APIVersion(str, Enum):
    """API 版本管理"""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    BETA = "beta"
    ALPHA = "alpha"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class ContentType(str, Enum):
    """內容類型"""

    JSON = "application/json"
    XML = "application/xml"
    HTML = HTML_MIME_TYPE
    PLAIN_TEXT = "text/plain"
    CSV = "text/csv"
    YAML = "application/yaml"
    BINARY = "application/octet-stream"
    FORM_DATA = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    PDF = "application/pdf"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    VIDEO_MP4 = "video/mp4"
    AUDIO_MP3 = "audio/mpeg"


# ============================================================================
# 用戶和權限枚舉
# ============================================================================


class UserRole(str, Enum):
    """用戶角色"""

    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"
    USER = "user"
    GUEST = "guest"
    SERVICE_ACCOUNT = "service_account"
    API_USER = "api_user"


class PermissionType(str, Enum):
    """權限類型"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    ADMIN = "admin"
    OWNER = "owner"
    SHARE = "share"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REVIEW = "review"


class AccessControlModel(str, Enum):
    """訪問控制模型"""

    RBAC = "rbac"  # Role-Based Access Control
    ABAC = "abac"  # Attribute-Based Access Control
    MAC = "mac"  # Mandatory Access Control
    DAC = "dac"  # Discretionary Access Control
    PBAC = "pbac"  # Policy-Based Access Control
    OAUTH = "oauth"
    SAML = "saml"
    JWT = "jwt"


# ============================================================================
# 業務流程枚舉
# ============================================================================


class WorkflowStatus(str, Enum):
    """工作流狀態"""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class ApprovalStatus(str, Enum):
    """審批狀態"""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    ESCALATED = "escalated"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class BusinessImpact(str, Enum):
    """業務影響"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    NONE = "none"


class ChangeType(str, Enum):
    """變更類型"""

    EMERGENCY = "emergency"
    STANDARD = "standard"
    NORMAL = "normal"
    MINOR = "minor"
    MAJOR = "major"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    MAINTENANCE = "maintenance"


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


# ==================== 插件管理 (新增) ====================


class PluginStatus(str, Enum):
    """插件狀態枚舉 - 用於插件生命週期管理"""

    LOADED = "loaded"  # 已載入但未啟用
    ENABLED = "enabled"  # 已啟用並運行
    DISABLED = "disabled"  # 已禁用
    ERROR = "error"  # 錯誤狀態
    UNLOADED = "unloaded"  # 已卸載


class PluginType(str, Enum):
    """插件類型枚舉 - 定義插件的功能分類"""

    SECURITY = "security"  # 安全功能插件
    ANALYSIS = "analysis"  # 分析功能插件
    REPORTING = "reporting"  # 報告生成插件
    INTEGRATION = "integration"  # 整合服務插件
    MONITORING = "monitoring"  # 監控功能插件
    AUTOMATION = "automation"  # 自動化插件
    EXTENSION = "extension"  # 擴展功能插件
    UTILITY = "utility"  # 工具類插件


class AsyncTaskStatus(str, Enum):
    """異步任務狀態 - async_utils 專用增強版"""

    SUBMITTED = "submitted"  # 已提交但未開始
    SCHEDULED = "scheduled"  # 已排程等待執行
    RUNNING = "running"  # 正在執行
    PAUSED = "paused"  # 暫停狀態
    COMPLETED = "completed"  # 執行完成
    FAILED = "failed"  # 執行失敗
    CANCELLED = "cancelled"  # 已取消
    TIMEOUT = "timeout"  # 執行超時


# ==================== 數據響應類型 (中需求) ====================


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


# ==================== HTTP 相關枚舉 (高需求) ====================
# 根據截圖建議新增，統一 16+ 檔案中的硬編碼驗證邏輯


class HttpMethod(str, Enum):
    """HTTP 方法枚舉 - 統一 HTTP 請求方法定義"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    CONNECT = "CONNECT"


class HTTPStatusClass(str, Enum):
    """HTTP 狀態碼分類 - RFC 7231 官方術語
    
    參考: https://tools.ietf.org/html/rfc7231#section-6
    """

    INFORMATIONAL = "1xx"  # 100-199: RFC 7231 Section 6.2 - 信息響應
    SUCCESSFUL = "2xx"     # 200-299: RFC 7231 Section 6.3 - 成功響應  
    REDIRECTION = "3xx"    # 300-399: RFC 7231 Section 6.4 - 重定向響應
    CLIENT_ERROR = "4xx"   # 400-499: RFC 7231 Section 6.5 - 客戶端錯誤
    SERVER_ERROR = "5xx"   # 500-599: RFC 7231 Section 6.6 - 服務器錯誤


# 向後相容別名 (將於 v6.0 移除)  
HttpStatusCodeRange = HTTPStatusClass


class ResponseDataType(str, Enum):
    """回應資料類型 - 用於響應內容分類"""

    JSON = "json"
    XML = "xml"
    HTML = "html"
    TEXT = "text"
    BINARY = "binary"
    FORM_DATA = "form_data"


# ==================== 操作結果枚舉 (高需求) ====================


class OperationResult(str, Enum):
    """操作結果狀態 - 統一操作執行結果"""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ValidationResult(str, Enum):
    """驗證結果 - 用於數據驗證狀態"""

    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    SKIPPED = "skipped"


# ==================== 日誌與監控枚舉 (中需求) ====================


class LogLevel(str, Enum):
    """日誌等級 - 遵循 RFC 5424 Syslog 和 Python logging 官方標準"""

    # RFC 5424 Syslog 標準等級 + Python logging 官方等級
    NOTSET = "notset"  # Python logging 官方標準 (0)
    DEBUG = "debug"  # RFC 5424 Syslog Level 7 / Python logging 10
    INFO = "info"  # RFC 5424 Syslog Level 6 / Python logging 20
    WARNING = "warning"  # RFC 5424 Syslog Level 4 / Python logging 30 (官方是 WARNING 不是 WARN)
    ERROR = "error"  # RFC 5424 Syslog Level 3 / Python logging 40
    CRITICAL = "critical"  # RFC 5424 Syslog Level 2 / Python logging 50 (官方是 CRITICAL 不是 FATAL)


class AlertType(str, Enum):
    """告警類型 - 用於系統告警分類"""

    SECURITY_INCIDENT = "security_incident"
    PERFORMANCE_ISSUE = "performance_issue"
    SYSTEM_ERROR = "system_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMPLIANCE_VIOLATION = "compliance_violation"


# ==================== 資料處理枚舉 (低需求) ====================


class DataFormat(str, Enum):
    """資料格式 - 遵循 IANA MIME Types 官方標準"""

    # IANA 官方註冊的 MIME types
    JSON = "application/json"  # RFC 7159 官方標準
    XML = "application/xml"  # RFC 7303 官方標準
    YAML = "application/yaml"  # IANA 註冊的官方 MIME type
    CSV = "text/csv"  # RFC 4180 官方標準
    HTML = HTML_MIME_TYPE  # RFC 2854 官方標準
    PLAIN_TEXT = "text/plain"  # RFC 2046 官方標準

    # 專業格式（使用官方定義的擴展）
    SARIF = "application/sarif+json"  # SARIF 2.1.0 官方格式標準


class EncodingType(str, Enum):
    """編碼類型枚舉 - 使用官方標準編碼"""

    # 字符編碼 (IANA Character Sets 官方標準)
    UTF8 = "utf-8"  # RFC 3629
    UTF16 = "utf-16"  # RFC 2781
    ASCII = "ascii"  # ANSI X3.4-1986
    ISO_8859_1 = "iso-8859-1"  # ISO 8859-1

    # 傳輸編碼 (HTTP 標準)
    BASE64 = "base64"  # RFC 4648
    URL_ENCODED = "application/x-www-form-urlencoded"  # HTML 4.01
    HTML_ENCODED = HTML_MIME_TYPE  # RFC 2854


# ==================== 網路協議相關枚舉 ====================


class NetworkProtocol(str, Enum):
    """網路協議枚舉 - 使用 IANA Protocol Numbers 官方標準"""

    # 傳輸層協議 (RFC 標準)
    TCP = "tcp"  # Transmission Control Protocol - RFC 793
    UDP = "udp"  # User Datagram Protocol - RFC 768
    SCTP = "sctp"  # Stream Control Transmission Protocol - RFC 4960

    # 應用層協議 (各種 RFC 標準)
    HTTP = "http"  # HyperText Transfer Protocol - RFC 7230
    HTTPS = "https"  # HTTP over TLS - RFC 2818
    FTP = "ftp"  # File Transfer Protocol - RFC 959
    FTPS = "ftps"  # FTP over TLS - RFC 4217
    SSH = "ssh"  # Secure Shell - RFC 4251
    TELNET = "telnet"  # Telnet Protocol - RFC 854
    SMTP = "smtp"  # Simple Mail Transfer Protocol - RFC 5321
    POP3 = "pop3"  # Post Office Protocol v3 - RFC 1939
    IMAP = "imap"  # Internet Message Access Protocol - RFC 3501
    DNS = "dns"  # Domain Name System - RFC 1035
    DHCP = "dhcp"  # Dynamic Host Configuration Protocol - RFC 2131
    SNMP = "snmp"  # Simple Network Management Protocol - RFC 1157
    LDAP = "ldap"  # Lightweight Directory Access Protocol - RFC 4511
    LDAPS = "ldaps"  # LDAP over TLS - RFC 4513

    # 網路層協議
    ICMP = "icmp"  # Internet Control Message Protocol - RFC 792
    ARP = "arp"  # Address Resolution Protocol - RFC 826
    NTP = "ntp"  # Network Time Protocol - RFC 5905
