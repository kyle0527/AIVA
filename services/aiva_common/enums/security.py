"""
安全測試相關枚舉 - 漏洞、攻擊、權限等
"""

from enum import Enum


class ExploitType(str, Enum):
    """漏洞利用類型枚舉"""

    IDOR = "idor"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    AUTH_BYPASS = "auth_bypass"
    JWT_ATTACK = "jwt_attack"
    GRAPHQL_INJECTION = "graphql_injection"
    CSRF = "csrf"
    XXE = "xxe"
    SSRF = "ssrf"
    LFI = "lfi"
    RFI = "rfi"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    DESERIALIZATION = "deserialization"
    LDAP_INJECTION = "ldap_injection"


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


# ==================== 官方安全標準擴展 ====================


class CVSSMetric(str, Enum):
    """CVSS v4.0 評分指標枚舉（基於 First.org 官方標準）"""

    # Base Metrics (基本指標)
    ATTACK_VECTOR = "attack_vector"  # AV
    ATTACK_COMPLEXITY = "attack_complexity"  # AC
    ATTACK_REQUIREMENTS = "attack_requirements"  # AT (v4.0新增)
    PRIVILEGES_REQUIRED = "privileges_required"  # PR
    USER_INTERACTION = "user_interaction"  # UI
    VULNERABLE_SYSTEM_CONFIDENTIALITY = "vulnerable_system_confidentiality"  # VC
    VULNERABLE_SYSTEM_INTEGRITY = "vulnerable_system_integrity"  # VI
    VULNERABLE_SYSTEM_AVAILABILITY = "vulnerable_system_availability"  # VA
    SUBSEQUENT_SYSTEM_CONFIDENTIALITY = (
        "subsequent_system_confidentiality"  # SC (v4.0新增)
    )
    SUBSEQUENT_SYSTEM_INTEGRITY = "subsequent_system_integrity"  # SI (v4.0新增)
    SUBSEQUENT_SYSTEM_AVAILABILITY = "subsequent_system_availability"  # SA (v4.0新增)

    # Threat Metrics (威脅指標)
    EXPLOIT_MATURITY = "exploit_maturity"  # E

    # Environmental Metrics (環境指標)
    CONFIDENTIALITY_REQUIREMENT = "confidentiality_requirement"  # CR
    INTEGRITY_REQUIREMENT = "integrity_requirement"  # IR
    AVAILABILITY_REQUIREMENT = "availability_requirement"  # AR

    # Supplemental Metrics (補充指標，v4.0新增)
    SAFETY = "safety"  # S
    AUTOMATABLE = "automatable"  # AU
    RECOVERY = "recovery"  # R
    VALUE_DENSITY = "value_density"  # V
    VULNERABILITY_RESPONSE_EFFORT = "vulnerability_response_effort"  # RE
    PROVIDER_URGENCY = "provider_urgency"  # U


class AttackTactic(str, Enum):
    """MITRE ATT&CK 戰術枚舉（基於 MITRE v18 官方矩陣）"""

    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0011"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


class AttackTechnique(str, Enum):
    """MITRE ATT&CK 技術枚舉（核心技術選集）"""

    # Reconnaissance
    ACTIVE_SCANNING = "T1595"
    GATHER_VICTIM_HOST_INFORMATION = "T1592"
    GATHER_VICTIM_IDENTITY_INFORMATION = "T1589"
    GATHER_VICTIM_NETWORK_INFORMATION = "T1590"
    GATHER_VICTIM_ORG_INFORMATION = "T1591"
    PHISHING_FOR_INFORMATION = "T1598"
    SEARCH_CLOSED_SOURCES = "T1597"
    SEARCH_OPEN_TECHNICAL_DATABASES = "T1596"
    SEARCH_OPEN_WEBSITES = "T1593"
    SEARCH_VICTIM_OWNED_WEBSITES = "T1594"

    # Initial Access
    DRIVE_BY_COMPROMISE = "T1189"
    EXPLOIT_PUBLIC_FACING_APPLICATION = "T1190"
    EXTERNAL_REMOTE_SERVICES = "T1133"
    HARDWARE_ADDITIONS = "T1200"
    PHISHING = "T1566"
    REPLICATION_THROUGH_REMOVABLE_MEDIA = "T1091"
    SUPPLY_CHAIN_COMPROMISE = "T1195"
    TRUSTED_RELATIONSHIP = "T1199"
    VALID_ACCOUNTS = "T1078"

    # Execution
    COMMAND_AND_SCRIPTING_INTERPRETER = "T1059"
    CONTAINER_ADMINISTRATION_COMMAND = "T1609"
    DEPLOY_CONTAINER = "T1610"
    EXPLOITATION_FOR_CLIENT_EXECUTION = "T1203"
    INTER_PROCESS_COMMUNICATION = "T1559"
    NATIVE_API = "T1106"
    SCHEDULED_TASK_JOB = "T1053"
    SHARED_MODULES = "T1129"
    SOFTWARE_DEPLOYMENT_TOOLS = "T1072"
    SYSTEM_SERVICES = "T1569"
    USER_EXECUTION = "T1204"

    # Persistence
    ACCOUNT_MANIPULATION = "T1098"
    BOOT_OR_LOGON_AUTOSTART_EXECUTION = "T1547"
    BOOT_OR_LOGON_INITIALIZATION_SCRIPTS = "T1037"
    BROWSER_EXTENSIONS = "T1176"
    COMPROMISE_CLIENT_SOFTWARE_BINARY = "T1554"
    CREATE_ACCOUNT = "T1136"
    CREATE_OR_MODIFY_SYSTEM_PROCESS = "T1543"
    EVENT_TRIGGERED_EXECUTION = "T1546"
    EXTERNAL_REMOTE_SERVICES_PERSIST = "T1133"  # 重用但標注持久化用途
    HIJACK_EXECUTION_FLOW = "T1574"
    IMPLANT_INTERNAL_IMAGE = "T1525"
    MODIFY_AUTHENTICATION_PROCESS = "T1556"
    OFFICE_APPLICATION_STARTUP = "T1137"
    PRE_OS_BOOT = "T1542"
    SCHEDULED_TASK_JOB_PERSIST = "T1053"  # 重用但標注持久化用途
    SERVER_SOFTWARE_COMPONENT = "T1505"
    TRAFFIC_SIGNALING = "T1205"
    VALID_ACCOUNTS_PERSIST = "T1078"  # 重用但標注持久化用途

    # Privilege Escalation
    ABUSE_ELEVATION_CONTROL_MECHANISM = "T1548"
    ACCESS_TOKEN_MANIPULATION = "T1134"
    BOOT_OR_LOGON_AUTOSTART_EXECUTION_PRIV = "T1547"  # 重用但標注權限提升用途
    BOOT_OR_LOGON_INITIALIZATION_SCRIPTS_PRIV = "T1037"  # 重用但標注權限提升用途
    CREATE_OR_MODIFY_SYSTEM_PROCESS_PRIV = "T1543"  # 重用但標注權限提升用途
    DOMAIN_POLICY_MODIFICATION = "T1484"
    ESCAPE_TO_HOST = "T1611"
    EVENT_TRIGGERED_EXECUTION_PRIV = "T1546"  # 重用但標注權限提升用途
    EXPLOITATION_FOR_PRIVILEGE_ESCALATION = "T1068"
    HIJACK_EXECUTION_FLOW_PRIV = "T1574"  # 重用但標注權限提升用途
    PROCESS_INJECTION = "T1055"
    SCHEDULED_TASK_JOB_PRIV = "T1053"  # 重用但標注權限提升用途
    VALID_ACCOUNTS_PRIV = "T1078"  # 重用但標注權限提升用途

    # Defense Evasion (部分核心技術)
    ABUSE_ELEVATION_CONTROL_MECHANISM_DEF = "T1548"  # 重用但標注防禦逃避用途
    ACCESS_TOKEN_MANIPULATION_DEF = "T1134"  # 重用但標注防禦逃避用途
    BINARY_PADDING = "T1027.001"
    BUILD_IMAGE_ON_HOST = "T1612"
    DEBUGGER_EVASION = "T1622"
    DEOBFUSCATE_DECODE_FILES_OR_INFORMATION = "T1140"
    DEPLOY_CONTAINER_DEF = "T1610"  # 重用但標注防禦逃避用途
    DIRECT_VOLUME_ACCESS = "T1006"
    DOMAIN_POLICY_MODIFICATION_DEF = "T1484"  # 重用但標注防禦逃避用途
    EXECUTION_GUARDRAILS = "T1480"
    EXPLOITATION_FOR_DEFENSE_EVASION = "T1211"
    FILE_AND_DIRECTORY_PERMISSIONS_MODIFICATION = "T1222"
    HIDE_ARTIFACTS = "T1564"
    HIJACK_EXECUTION_FLOW_DEF = "T1574"  # 重用但標注防禦逃避用途
    IMPAIR_DEFENSES = "T1562"
    INDICATOR_REMOVAL = "T1070"
    INDIRECT_COMMAND_EXECUTION = "T1202"
    MASQUERADING = "T1036"
    MODIFY_AUTHENTICATION_PROCESS_DEF = "T1556"  # 重用但標注防禦逃避用途
    MODIFY_CLOUD_COMPUTE_INFRASTRUCTURE = "T1578"
    MODIFY_REGISTRY = "T1112"
    MODIFY_SYSTEM_IMAGE = "T1601"
    NETWORK_BOUNDARY_BRIDGING = "T1599"
    OBFUSCATED_FILES_OR_INFORMATION = "T1027"
    PLIST_FILE_MODIFICATION = "T1647"
    PROCESS_INJECTION_DEF = "T1055"  # 重用但標注防禦逃避用途
    REFLECTIVE_CODE_LOADING = "T1620"
    ROGUE_DOMAIN_CONTROLLER = "T1207"
    ROOTKIT = "T1014"
    SUBVERT_TRUST_CONTROLS = "T1553"
    SYSTEM_BINARY_PROXY_EXECUTION = "T1218"
    TEMPLATE_INJECTION = "T1221"
    TRAFFIC_SIGNALING_DEF = "T1205"  # 重用但標注防禦逃避用途
    TRUSTED_DEVELOPER_UTILITIES_PROXY_EXECUTION = "T1127"
    UNUSED_UNSUPPORTED_CLOUD_REGIONS = "T1535"
    USE_ALTERNATE_AUTHENTICATION_MATERIAL = "T1606"
    VALID_ACCOUNTS_DEF = "T1078"  # 重用但標注防禦逃避用途
    VIRTUALIZATION_SANDBOX_EVASION = "T1497"
    WEAKEN_ENCRYPTION = "T1600"
    XSL_SCRIPT_PROCESSING = "T1220"


class CWECategory(str, Enum):
    """CWE（Common Weakness Enumeration）主要類別（基於 MITRE CWE v4.18）"""

    # Top 25 Most Dangerous Software Weaknesses (2024)
    OUT_OF_BOUNDS_WRITE = "CWE-787"
    IMPROPER_NEUTRALIZATION_OF_SPECIAL_ELEMENTS_USED_IN_SQL_COMMAND = "CWE-89"
    IMPROPER_NEUTRALIZATION_OF_INPUT_DURING_WEB_PAGE_GENERATION = "CWE-79"
    IMPROPER_INPUT_VALIDATION = "CWE-20"
    OUT_OF_BOUNDS_READ = "CWE-125"
    IMPROPER_NEUTRALIZATION_OF_SPECIAL_ELEMENTS_USED_IN_OS_COMMAND = "CWE-78"
    USE_AFTER_FREE = "CWE-416"
    IMPROPER_LIMITATION_OF_PATHNAME_TO_RESTRICTED_DIRECTORY = "CWE-22"
    CROSS_SITE_REQUEST_FORGERY = "CWE-352"
    MISSING_AUTHORIZATION = "CWE-862"
    NULL_POINTER_DEREFERENCE = "CWE-476"
    IMPROPER_AUTHENTICATION = "CWE-287"
    INTEGER_OVERFLOW_OR_WRAPAROUND = "CWE-190"
    DESERIALIZATION_OF_UNTRUSTED_DATA = "CWE-502"
    IMPROPER_NEUTRALIZATION_OF_DIRECTIVES_IN_DYNAMICALLY_EVALUATED_CODE = "CWE-95"
    IMPROPER_CONTROL_OF_GENERATION_OF_CODE = "CWE-94"
    IMPROPER_RESTRICTION_OF_OPERATIONS_WITHIN_THE_BOUNDS_OF_A_MEMORY_BUFFER = "CWE-119"
    MISSING_AUTHENTICATION_FOR_CRITICAL_FUNCTION = "CWE-306"
    IMPROPER_PRIVILEGE_MANAGEMENT = "CWE-269"
    INCORRECT_AUTHORIZATION = "CWE-863"
    EXPOSURE_OF_SENSITIVE_INFORMATION_TO_AN_UNAUTHORIZED_ACTOR = "CWE-200"
    INCORRECT_DEFAULT_PERMISSIONS = "CWE-276"
    EXPRESSION_LANGUAGE_INJECTION = "CWE-917"
    IMPROPER_CERTIFICATE_VALIDATION = "CWE-295"
    SERVER_SIDE_REQUEST_FORGERY = "CWE-918"

    # Additional important CWEs
    BUFFER_OVERFLOW = "CWE-120"
    RACE_CONDITION = "CWE-362"
    CRYPTOGRAPHIC_ISSUES = "CWE-310"
    HARDCODED_CREDENTIALS = "CWE-798"
    INSECURE_RANDOMNESS = "CWE-330"
    INSUFFICIENT_CRYPTOGRAPHY = "CWE-326"


# ==================== STIX 威脅情報標準擴展 ====================


class STIXObjectType(str, Enum):
    """STIX v2.1 物件類型枚舉（基於 OASIS 官方標準）"""

    # Domain Objects
    ATTACK_PATTERN = "attack-pattern"
    CAMPAIGN = "campaign"
    COURSE_OF_ACTION = "course-of-action"
    GROUPING = "grouping"
    IDENTITY = "identity"
    INDICATOR = "indicator"
    INFRASTRUCTURE = "infrastructure"
    INTRUSION_SET = "intrusion-set"
    LOCATION = "location"
    MALWARE = "malware"
    MALWARE_ANALYSIS = "malware-analysis"
    NOTE = "note"
    OBSERVED_DATA = "observed-data"
    OPINION = "opinion"
    REPORT = "report"
    THREAT_ACTOR = "threat-actor"
    TOOL = "tool"
    VULNERABILITY = "vulnerability"

    # Relationship Objects
    RELATIONSHIP = "relationship"
    SIGHTING = "sighting"

    # Meta Objects
    BUNDLE = "bundle"
    LANGUAGE_CONTENT = "language-content"
    MARKING_DEFINITION = "marking-definition"


class STIXRelationshipType(str, Enum):
    """STIX 關係類型枚舉（基於 STIX v2.1 官方標準）"""

    ATTRIBUTED_TO = "attributed-to"
    BASED_ON = "based-on"
    COMPROMISES = "compromises"
    CONSISTS_OF = "consists-of"
    CONTROLS = "controls"
    DELIVERS = "delivers"
    DERIVED_FROM = "derived-from"
    DOWNLOADS = "downloads"
    DROPS = "drops"
    DUPLICATE_OF = "duplicate-of"
    EXFILTRATES_TO = "exfiltrates-to"
    EXPLOITS = "exploits"
    HAS = "has"
    HOSTS = "hosts"
    INDICATES = "indicates"
    INVESTIGATES = "investigates"
    LOCATED_AT = "located-at"
    MITIGATES = "mitigates"
    OWNS = "owns"
    ORIGINATES_FROM = "originates-from"
    PARTICIPATES_IN = "participates-in"
    RELATED_TO = "related-to"
    REMEDIATES = "remediates"
    REVOKED_BY = "revoked-by"
    TARGETS = "targets"
    USES = "uses"
    VARIANT_OF = "variant-of"


class TLPMarking(str, Enum):
    """Traffic Light Protocol (TLP) 標記枚舉（基於 FIRST 官方標準）"""

    TLP_RED = "TLP:RED"  # 極度敏感：僅限指定收件人
    TLP_AMBER = "TLP:AMBER"  # 敏感：僅限組織內部或相關組織
    TLP_AMBER_STRICT = "TLP:AMBER+STRICT"  # 嚴格琥珀：僅限指定組織
    TLP_GREEN = "TLP:GREEN"  # 社群：可在社群內分享
    TLP_WHITE = "TLP:WHITE"  # 公開：無限制分享
    TLP_CLEAR = "TLP:CLEAR"  # 透明：無限制分享（TLP:WHITE 的新名稱）


class MITREPlatform(str, Enum):
    """MITRE ATT&CK 平台枚舉（基於 MITRE v18 官方矩陣）"""

    WINDOWS = "Windows"
    MACOS = "macOS"
    LINUX = "Linux"
    PRE = "PRE"
    AZURE_AD = "Azure AD"
    OFFICE_365 = "Office 365"
    GOOGLE_WORKSPACE = "Google Workspace"
    SAAS = "SaaS"
    IAS = "IaaS"
    CONTAINERS = "Containers"
    NETWORK = "Network"
    ICS = "ICS"
    MOBILE = "Mobile"


class TAXIIMediaType(str, Enum):
    """TAXII v2.1 媒體類型枚舉（基於 OASIS 官方標準）"""

    STIX_JSON = "application/stix+json;version=2.1"
    TAXII_JSON = "application/taxii+json;version=2.1"
    JSON = "application/json"


class CVSSVersion(str, Enum):
    """CVSS 版本枚舉（基於 First.org 官方標準）"""

    CVSS_2_0 = "2.0"
    CVSS_3_0 = "3.0"
    CVSS_3_1 = "3.1"
    CVSS_4_0 = "4.0"


class CVSSAttackVector(str, Enum):
    """CVSS v4.0 攻擊向量枚舉（基於 First.org 官方標準）"""

    NETWORK = "N"  # Network
    ADJACENT = "A"  # Adjacent Network
    LOCAL = "L"  # Local
    PHYSICAL = "P"  # Physical


class CVSSAttackComplexity(str, Enum):
    """CVSS v4.0 攻擊複雜度枚舉"""

    LOW = "L"  # Low
    HIGH = "H"  # High


class CVSSPrivilegesRequired(str, Enum):
    """CVSS v4.0 所需權限枚舉"""

    NONE = "N"  # None
    LOW = "L"  # Low
    HIGH = "H"  # High


class CVSSUserInteraction(str, Enum):
    """CVSS v4.0 使用者互動枚舉"""

    NONE = "N"  # None
    REQUIRED = "R"  # Required


# CVSS v3.1 Impact metrics - separate enums per official specification
class CVSSConfidentialityImpact(str, Enum):
    """CVSS v3.1 機密性影響（基於 First.org 官方 Table 6）"""

    HIGH = "H"  # High
    LOW = "L"  # Low
    NONE = "N"  # None


class CVSSIntegrityImpact(str, Enum):
    """CVSS v3.1 完整性影響（基於 First.org 官方 Table 7）"""

    HIGH = "H"  # High
    LOW = "L"  # Low
    NONE = "N"  # None


class CVSSAvailabilityImpact(str, Enum):
    """CVSS v3.1 可用性影響（基於 First.org 官方 Table 8）"""

    HIGH = "H"  # High
    LOW = "L"  # Low
    NONE = "N"  # None


class VulnerabilityDisclosure(str, Enum):
    """漏洞披露狀態枚舉"""

    PRIVATE = "private"  # 私有披露
    PUBLIC = "public"  # 公開披露
    COORDINATED = "coordinated"  # 協調披露
    FULL_DISCLOSURE = "full_disclosure"  # 完全披露
    ZERO_DAY = "zero_day"  # 零日漏洞
    PARTIAL = "partial"  # 部分披露


class ThreatActorType(str, Enum):
    """威脅行為者類型枚舉（基於 STIX v2.1 官方標準）"""

    ACTIVIST = "activist"
    COMPETITOR = "competitor"
    CRIME_SYNDICATE = "crime-syndicate"
    CRIMINAL = "criminal"
    HACKER = "hacker"
    INSIDER_ACCIDENTAL = "insider-accidental"
    INSIDER_DISGRUNTLED = "insider-disgruntled"
    NATION_STATE = "nation-state"
    SENSATIONALIST = "sensationalist"
    SPY = "spy"
    TERRORIST = "terrorist"
    UNKNOWN = "unknown"


class MalwareType(str, Enum):
    """惡意軟體類型枚舉（基於 STIX v2.1 官方標準）"""

    ADWARE = "adware"
    BACKDOOR = "backdoor"
    BOT = "bot"
    BOOTKIT = "bootkit"
    DDOS = "ddos"
    DOWNLOADER = "downloader"
    DROPPER = "dropper"
    EXPLOIT_KIT = "exploit-kit"
    KEYLOGGER = "keylogger"
    RANSOMWARE = "ransomware"
    REMOTE_ACCESS_TROJAN = "remote-access-trojan"
    RESOURCE_EXPLOITATION = "resource-exploitation"
    ROGUE_SECURITY_SOFTWARE = "rogue-security-software"
    ROOTKIT = "rootkit"
    SCREEN_CAPTURE = "screen-capture"
    SPYWARE = "spyware"
    TROJAN = "trojan"
    UNKNOWN = "unknown"
    VIRUS = "virus"
    WEBSHELL = "webshell"
    WIPER = "wiper"
    WORM = "worm"


class ToolType(str, Enum):
    """工具類型枚舉（基於 STIX v2.1 官方標準）"""

    DENIAL_OF_SERVICE = "denial-of-service"
    EXPLOITATION = "exploitation"
    INFORMATION_GATHERING = "information-gathering"
    NETWORK_CAPTURE = "network-capture"
    CREDENTIAL_EXPLOITATION = "credential-exploitation"
    REMOTE_ACCESS = "remote-access"
    VULNERABILITY_SCANNING = "vulnerability-scanning"
    UNKNOWN = "unknown"


# ==================== HackerOne 優化相關枚舉 ====================


class BugBountyCategory(str, Enum):
    """Bug Bounty 漏洞類別枚舉（基於 HackerOne 分類）"""

    # 高價值類別 ($1000+)
    BUSINESS_LOGIC = "business_logic"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RCE = "remote_code_execution"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PAYMENT_MANIPULATION = "payment_manipulation"

    # 中等價值類別 ($200-$1000)
    IDOR = "idor"
    SQL_INJECTION = "sql_injection"
    STORED_XSS = "stored_xss"
    SSRF = "ssrf"
    XXE = "xxe"

    # 低價值高概率類別 ($50-$500) - 穩定收入目標
    REFLECTED_XSS = "reflected_xss"
    CSRF = "csrf"
    INFORMATION_DISCLOSURE = "information_disclosure"
    OPEN_REDIRECT = "open_redirect"
    CLICKJACKING = "clickjacking"
    HOST_HEADER_INJECTION = "host_header_injection"
    CORS_MISCONFIGURATION = "cors_misconfiguration"

    # 新興類別
    GRAPHQL_INJECTION = "graphql_injection"
    JWT_VULNERABILITIES = "jwt_vulnerabilities"
    API_ABUSE = "api_abuse"
    CONTAINER_ESCAPE = "container_escape"
    CLOUD_MISCONFIG = "cloud_misconfiguration"


class BountyPriorityTier(str, Enum):
    """獎金優先級層級枚舉"""

    CRITICAL = "critical"  # $5000+ - 20% 資源分配
    HIGH = "high"  # $1000-$5000 - 0% 資源分配（避免競爭激烈）
    MEDIUM = "medium"  # $200-$1000 - 20% 資源分配
    LOW_STABLE = "low_stable"  # $50-$500 - 60% 資源分配（主要目標）


class VulnerabilityDifficulty(str, Enum):
    """漏洞發現難度枚舉"""

    TRIVIAL = "trivial"  # 自動化工具即可發現
    EASY = "easy"  # 基本手動測試
    MEDIUM = "medium"  # 需要中等技巧和時間
    HARD = "hard"  # 需要深度分析和創意
    EXPERT = "expert"  # 需要專家級技能


class TestingApproach(str, Enum):
    """測試方法枚舉"""

    AUTOMATED = "automated"  # 完全自動化
    SEMI_AUTOMATED = "semi_automated"  # 半自動化
    MANUAL = "manual"  # 手動測試
    HYBRID = "hybrid"  # 混合方法


class ProgramType(str, Enum):
    """Bug Bounty 程式類型枚舉"""

    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    API = "api"
    DESKTOP_APPLICATION = "desktop_application"
    CLOUD_SERVICES = "cloud_services"
    IOT_DEVICE = "iot_device"
    BLOCKCHAIN = "blockchain"
    HARDWARE = "hardware"
    SOURCE_CODE = "source_code"


class ResponseTimeCategory(str, Enum):
    """回應時間類別枚舉"""

    FAST = "fast"  # < 3 天
    NORMAL = "normal"  # 3-14 天
    SLOW = "slow"  # 14-30 天
    VERY_SLOW = "very_slow"  # > 30 天


class LowValueVulnerabilityType(str, Enum):
    """低價值高概率漏洞類型枚舉（穩定收入策略）"""

    # 最常見的低風險漏洞（$50-$200）
    REFLECTED_XSS_BASIC = "reflected_xss_basic"
    INFO_DISCLOSURE_ERROR_MESSAGES = "info_disclosure_error_messages"
    INFO_DISCLOSURE_VERSION = "info_disclosure_version"
    CORS_MISCONFIGURATION_BASIC = "cors_misconfiguration_basic"
    OPEN_REDIRECT_BASIC = "open_redirect_basic"
    CLICKJACKING_BASIC = "clickjacking_basic"

    # 常見的CSRF變體（$100-$300）
    CSRF_STATE_CHANGING = "csrf_state_changing"
    CSRF_GET_BASED = "csrf_get_based"

    # 信息洩露變體（$50-$250）
    INFO_DISCLOSURE_DIRECTORY_LISTING = "info_disclosure_directory_listing"
    INFO_DISCLOSURE_BACKUP_FILES = "info_disclosure_backup_files"
    INFO_DISCLOSURE_DEBUG_INFO = "info_disclosure_debug_info"
    INFO_DISCLOSURE_STACK_TRACE = "info_disclosure_stack_trace"

    # 配置問題（$100-$400）
    SECURITY_HEADERS_MISSING = "security_headers_missing"
    HTTP_SECURITY_HEADERS = "http_security_headers"
    CACHE_POISONING_BASIC = "cache_poisoning_basic"
    HOST_HEADER_INJECTION_BASIC = "host_header_injection_basic"

    # 認證繞過（低影響）（$200-$500）
    AUTH_BYPASS_WEAK_TOKEN = "auth_bypass_weak_token"
    SESSION_FIXATION_BASIC = "session_fixation_basic"
    PASSWORD_POLICY_BYPASS = "password_policy_bypass"
