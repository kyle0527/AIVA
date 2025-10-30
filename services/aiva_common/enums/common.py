"""
通用枚舉 - 狀態、級別、類型等基礎枚舉
"""



from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "info"


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


# ==================== 插件管理 (新增) ====================


class PluginStatus(str, Enum):
    """插件狀態枚舉 - 用於插件生命週期管理"""

    LOADED = "loaded"      # 已載入但未啟用
    ENABLED = "enabled"    # 已啟用並運行
    DISABLED = "disabled"  # 已禁用
    ERROR = "error"        # 錯誤狀態
    UNLOADED = "unloaded"  # 已卸載


class PluginType(str, Enum):
    """插件類型枚舉 - 定義插件的功能分類"""

    SECURITY = "security"          # 安全功能插件
    ANALYSIS = "analysis"          # 分析功能插件
    REPORTING = "reporting"        # 報告生成插件
    INTEGRATION = "integration"    # 整合服務插件
    MONITORING = "monitoring"      # 監控功能插件
    AUTOMATION = "automation"      # 自動化插件
    EXTENSION = "extension"        # 擴展功能插件
    UTILITY = "utility"            # 工具類插件


class AsyncTaskStatus(str, Enum):
    """異步任務狀態 - async_utils 專用增強版"""

    SUBMITTED = "submitted"    # 已提交但未開始
    SCHEDULED = "scheduled"    # 已排程等待執行
    RUNNING = "running"        # 正在執行
    PAUSED = "paused"          # 暫停狀態
    COMPLETED = "completed"    # 執行完成
    FAILED = "failed"          # 執行失敗
    CANCELLED = "cancelled"    # 已取消
    TIMEOUT = "timeout"        # 執行超時


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


class HttpStatusCodeRange(str, Enum):
    """HTTP 狀態碼範圍 - 用於狀態碼分類"""
    
    INFORMATIONAL = "1xx"     # 100-199 信息性響應
    SUCCESS = "2xx"           # 200-299 成功響應
    REDIRECT = "3xx"          # 300-399 重定向響應
    CLIENT_ERROR = "4xx"      # 400-499 客戶端錯誤
    SERVER_ERROR = "5xx"      # 500-599 服務器錯誤


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
    NOTSET = "notset"      # Python logging 官方標準 (0)
    DEBUG = "debug"        # RFC 5424 Syslog Level 7 / Python logging 10
    INFO = "info"          # RFC 5424 Syslog Level 6 / Python logging 20  
    WARNING = "warning"    # RFC 5424 Syslog Level 4 / Python logging 30 (官方是 WARNING 不是 WARN)
    ERROR = "error"        # RFC 5424 Syslog Level 3 / Python logging 40
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
    JSON = "application/json"        # RFC 7159 官方標準
    XML = "application/xml"          # RFC 7303 官方標準  
    YAML = "application/yaml"        # IANA 註冊的官方 MIME type
    CSV = "text/csv"                 # RFC 4180 官方標準
    HTML = "text/html"               # RFC 2854 官方標準
    PLAIN_TEXT = "text/plain"        # RFC 2046 官方標準
    
    # 專業格式（使用官方定義的擴展）
    SARIF = "application/sarif+json" # SARIF 2.1.0 官方格式標準


class EncodingType(str, Enum):
    """編碼類型枚舉 - 使用官方標準編碼"""
    # 字符編碼 (IANA Character Sets 官方標準)
    UTF8 = "utf-8"                   # RFC 3629
    UTF16 = "utf-16"                 # RFC 2781
    ASCII = "ascii"                  # ANSI X3.4-1986
    ISO_8859_1 = "iso-8859-1"       # ISO 8859-1
    
    # 傳輸編碼 (HTTP 標準)
    BASE64 = "base64"                # RFC 4648
    URL_ENCODED = "application/x-www-form-urlencoded"  # HTML 4.01
    HTML_ENCODED = "text/html"       # RFC 2854


# ==================== 網路協議相關枚舉 ====================

class NetworkProtocol(str, Enum):
    """網路協議枚舉 - 使用 IANA Protocol Numbers 官方標準"""
    # 傳輸層協議 (RFC 標準)
    TCP = "tcp"              # Transmission Control Protocol - RFC 793
    UDP = "udp"              # User Datagram Protocol - RFC 768
    SCTP = "sctp"            # Stream Control Transmission Protocol - RFC 4960
    
    # 應用層協議 (各種 RFC 標準)
    HTTP = "http"            # HyperText Transfer Protocol - RFC 7230
    HTTPS = "https"          # HTTP over TLS - RFC 2818
    FTP = "ftp"              # File Transfer Protocol - RFC 959
    FTPS = "ftps"            # FTP over TLS - RFC 4217
    SSH = "ssh"              # Secure Shell - RFC 4251
    TELNET = "telnet"        # Telnet Protocol - RFC 854
    SMTP = "smtp"            # Simple Mail Transfer Protocol - RFC 5321
    POP3 = "pop3"            # Post Office Protocol v3 - RFC 1939
    IMAP = "imap"            # Internet Message Access Protocol - RFC 3501
    DNS = "dns"              # Domain Name System - RFC 1035
    DHCP = "dhcp"            # Dynamic Host Configuration Protocol - RFC 2131
    SNMP = "snmp"            # Simple Network Management Protocol - RFC 1157
    LDAP = "ldap"            # Lightweight Directory Access Protocol - RFC 4511
    LDAPS = "ldaps"          # LDAP over TLS - RFC 4513
    
    # 網路層協議
    ICMP = "icmp"            # Internet Control Message Protocol - RFC 792
    ARP = "arp"              # Address Resolution Protocol - RFC 826
    NTP = "ntp"              # Network Time Protocol - RFC 5905
