"""
AIVA 預定義能力枚舉 - 選單式操作系統

設計原則：
1. 所有能力都是預先定義好的選項，不需要自然語言解析
2. 只有目標(target)和參數(parameters)需要使用者輸入
3. 避免 AI 或人類輸入錯誤的情況
4. 無 NLU/LLM 依賴 - 使用 5M 決策引擎

使用方式：
>>> from aiva_common.enums.capabilities import AttackCapability, ScanCapability
>>> # 選擇能力
>>> capability = AttackCapability.SQL_INJECTION
>>> # 只需輸入目標
>>> target = "https://example.com/api"
>>> # 執行
>>> result = execute_capability(capability, target)
"""

from enum import Enum
from typing import Any


# =============================================================================
# 攻擊能力枚舉 - 滲透測試攻擊類型
# =============================================================================


class AttackCapability(str, Enum):
    """攻擊能力選單 - 所有可執行的攻擊類型
    
    選擇後只需提供：
    - target: 目標 URL/IP
    - parameters: 可選的額外參數
    """
    
    # === Web 應用攻擊 ===
    SQL_INJECTION = "sql_injection"
    SQL_INJECTION_BLIND = "sql_injection_blind"
    SQL_INJECTION_TIME_BASED = "sql_injection_time_based"
    SQL_INJECTION_UNION = "sql_injection_union"
    
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    XSS_DOM = "xss_dom"
    
    SSRF_BASIC = "ssrf_basic"
    SSRF_BLIND = "ssrf_blind"
    
    XXE_BASIC = "xxe_basic"
    XXE_BLIND = "xxe_blind"
    XXE_OOB = "xxe_oob"
    
    IDOR = "idor"
    CSRF = "csrf"
    
    # === 注入攻擊 ===
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    XPATH_INJECTION = "xpath_injection"
    TEMPLATE_INJECTION = "template_injection"
    HEADER_INJECTION = "header_injection"
    
    # === 路徑攻擊 ===
    PATH_TRAVERSAL = "path_traversal"
    LFI = "local_file_inclusion"
    RFI = "remote_file_inclusion"
    
    # === 認證攻擊 ===
    AUTH_BYPASS = "auth_bypass"
    JWT_ATTACK = "jwt_attack"
    JWT_NONE_ALGORITHM = "jwt_none_algorithm"
    JWT_WEAK_SECRET = "jwt_weak_secret"
    SESSION_FIXATION = "session_fixation"
    PASSWORD_BRUTE_FORCE = "password_brute_force"
    
    # === API 攻擊 ===
    GRAPHQL_INJECTION = "graphql_injection"
    GRAPHQL_INTROSPECTION = "graphql_introspection"
    GRAPHQL_BATCHING = "graphql_batching"
    REST_API_ABUSE = "rest_api_abuse"
    
    # === 反序列化攻擊 ===
    DESERIALIZATION_JAVA = "deserialization_java"
    DESERIALIZATION_PHP = "deserialization_php"
    DESERIALIZATION_PYTHON = "deserialization_python"
    DESERIALIZATION_DOTNET = "deserialization_dotnet"
    
    # === 其他攻擊 ===
    OPEN_REDIRECT = "open_redirect"
    CORS_MISCONFIGURATION = "cors_misconfiguration"
    CLICKJACKING = "clickjacking"
    SUBDOMAIN_TAKEOVER = "subdomain_takeover"
    
    # === 阻斷服務攻擊 ===
    DOS_ATTACK = "dos_attack"
    DDOS_ATTACK = "ddos_attack"
    
    # === NoSQL 攻擊 ===
    NOSQL_INJECTION = "nosql_injection"


class ScanCapability(str, Enum):
    """掃描能力選單 - 所有可執行的掃描類型"""
    
    # === 主動掃描 ===
    PORT_SCAN_TCP = "port_scan_tcp"
    PORT_SCAN_UDP = "port_scan_udp"
    PORT_SCAN_SYN = "port_scan_syn"
    PORT_SCAN_FULL = "port_scan_full"
    
    SERVICE_DETECTION = "service_detection"
    VERSION_DETECTION = "version_detection"
    OS_DETECTION = "os_detection"
    
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_VULNERABILITY_SCAN = "web_vulnerability_scan"
    
    # === 被動掃描 ===
    PASSIVE_RECON = "passive_recon"
    DNS_ENUMERATION = "dns_enumeration"
    SUBDOMAIN_ENUMERATION = "subdomain_enumeration"
    
    # === Web 掃描 ===
    DIRECTORY_BRUTEFORCE = "directory_bruteforce"
    WEB_CRAWLER = "web_crawler"
    PARAMETER_DISCOVERY = "parameter_discovery"
    TECHNOLOGY_DETECTION = "technology_detection"
    
    # === 資產發現 ===
    ASSET_DISCOVERY = "asset_discovery"
    CLOUD_ASSET_SCAN = "cloud_asset_scan"
    CERTIFICATE_TRANSPARENCY = "certificate_transparency"


class ReconCapability(str, Enum):
    """偵察能力選單 - 資訊收集類型"""
    
    # === OSINT ===
    WHOIS_LOOKUP = "whois_lookup"
    DNS_LOOKUP = "dns_lookup"
    REVERSE_DNS = "reverse_dns"
    IP_GEOLOCATION = "ip_geolocation"
    
    # === 子域名 ===
    SUBDOMAIN_ENUMERATION = "subdomain_enumeration"
    
    # === 社交工程偵察 ===
    EMAIL_HARVESTING = "email_harvesting"
    LINKEDIN_RECON = "linkedin_recon"
    GITHUB_RECON = "github_recon"
    
    # === 技術偵察 ===
    TECHNOLOGY_FINGERPRINT = "technology_fingerprint"
    WAF_DETECTION = "waf_detection"
    CDN_DETECTION = "cdn_detection"
    CMS_DETECTION = "cms_detection"
    
    # === 網路偵察 ===
    ASN_LOOKUP = "asn_lookup"
    CIDR_EXPANSION = "cidr_expansion"
    SHODAN_SEARCH = "shodan_search"
    CENSYS_SEARCH = "censys_search"


class AnalysisCapability(str, Enum):
    """分析能力選單 - 程式碼和安全分析"""
    
    # === 靜態分析 ===
    SAST_SCAN = "sast_scan"
    CODE_REVIEW = "code_review"
    DEPENDENCY_CHECK = "dependency_check"
    SECRET_DETECTION = "secret_detection"
    
    # === 動態分析 ===
    DAST_SCAN = "dast_scan"
    FUZZING = "fuzzing"
    API_FUZZING = "api_fuzzing"
    
    # === 配置分析 ===
    CONFIG_AUDIT = "config_audit"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_HEADERS_CHECK = "security_headers_check"
    SSL_TLS_ANALYSIS = "ssl_tls_analysis"
    
    # === 二進制分析 ===
    BINARY_ANALYSIS = "binary_analysis"
    MALWARE_ANALYSIS = "malware_analysis"
    REVERSE_ENGINEERING = "reverse_engineering"


class ForensicCapability(str, Enum):
    """數位鑑識能力選單"""
    
    # === 記憶體鑑識 ===
    MEMORY_DUMP = "memory_dump"
    MEMORY_ANALYSIS = "memory_analysis"
    PROCESS_LIST = "process_list"
    NETWORK_CONNECTIONS = "network_connections"
    
    # === 磁碟鑑識 ===
    DISK_IMAGE = "disk_image"
    FILE_RECOVERY = "file_recovery"
    TIMELINE_ANALYSIS = "timeline_analysis"
    
    # === 網路鑑識 ===
    PCAP_ANALYSIS = "pcap_analysis"
    TRAFFIC_ANALYSIS = "traffic_analysis"
    LOG_ANALYSIS = "log_analysis"
    
    # === 隱寫術 ===
    STEGANOGRAPHY_DETECT = "steganography_detect"
    STEGANOGRAPHY_EXTRACT = "steganography_extract"


class ExploitCapability(str, Enum):
    """漏洞利用能力選單 - 後滲透操作"""
    
    # === 權限提升 ===
    PRIVILEGE_ESCALATION_LINUX = "privilege_escalation_linux"
    PRIVILEGE_ESCALATION_WINDOWS = "privilege_escalation_windows"
    SUDO_EXPLOIT = "sudo_exploit"
    KERNEL_EXPLOIT = "kernel_exploit"
    
    # === 橫向移動 ===
    LATERAL_MOVEMENT = "lateral_movement"
    PASS_THE_HASH = "pass_the_hash"
    PASS_THE_TICKET = "pass_the_ticket"
    
    # === 持久化 ===
    PERSISTENCE_INSTALL = "persistence_install"
    BACKDOOR_INSTALL = "backdoor_install"
    
    # === 資料外洩 ===
    DATA_EXFILTRATION = "data_exfiltration"
    CREDENTIAL_DUMPING = "credential_dumping"


class ReportCapability(str, Enum):
    """報告能力選單"""
    
    VULNERABILITY_REPORT = "vulnerability_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_REPORT = "technical_report"
    COMPLIANCE_REPORT = "compliance_report"
    RISK_ASSESSMENT_REPORT = "risk_assessment_report"


# =============================================================================
# 能力參數定義 - 每個能力的必需和可選參數
# =============================================================================


class CapabilityParameter(str, Enum):
    """能力參數類型 - 定義每個能力需要的輸入參數類型"""
    
    # === 通用參數 ===
    TARGET_URL = "target_url"           # URL 目標
    TARGET_IP = "target_ip"             # IP 目標
    TARGET_HOST = "target_host"         # 主機名
    TARGET_DOMAIN = "target_domain"     # 域名
    TARGET_CIDR = "target_cidr"         # CIDR 範圍
    TARGET_PORT = "target_port"         # 端口
    TARGET_PORT_RANGE = "target_port_range"  # 端口範圍
    
    # === 認證參數 ===
    USERNAME = "username"
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    COOKIE = "cookie"
    
    # === 配置參數 ===
    TIMEOUT = "timeout"                 # 超時時間 (秒)
    THREADS = "threads"                 # 並發線程數
    DEPTH = "depth"                     # 掃描深度
    WORDLIST = "wordlist"               # 字典檔路徑
    
    # === 輸出參數 ===
    OUTPUT_FORMAT = "output_format"     # 輸出格式 (json/html/pdf)
    OUTPUT_PATH = "output_path"         # 輸出路徑


# =============================================================================
# 能力配置定義 - 描述每個能力的參數需求
# =============================================================================


CAPABILITY_CONFIGS: dict[str, dict[str, Any]] = {
    # SQL Injection
    AttackCapability.SQL_INJECTION.value: {
        "name": "SQL 注入攻擊",
        "description": "檢測並利用 SQL 注入漏洞",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.COOKIE, CapabilityParameter.TIMEOUT],
        "default_timeout": 30,
        "risk_level": "high",
    },
    AttackCapability.SQL_INJECTION_BLIND.value: {
        "name": "盲注 SQL 注入",
        "description": "基於時間或布林的盲注攻擊",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.COOKIE, CapabilityParameter.TIMEOUT],
        "default_timeout": 60,
        "risk_level": "high",
    },
    AttackCapability.NOSQL_INJECTION.value: {
        "name": "NoSQL 注入攻擊",
        "description": "檢測並利用 NoSQL 注入漏洞",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.COOKIE, CapabilityParameter.TIMEOUT],
        "default_timeout": 60,
        "risk_level": "high",
    },
    
    # DDOS / DOS
    AttackCapability.DDOS_ATTACK.value: {
        "name": "DDoS 阻斷服務壓力測試",
        "description": "分散式阻斷服務壓力測試",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.THREADS, CapabilityParameter.TIMEOUT],
        "default_timeout": 300,
        "risk_level": "high",
    },
    
    # XSS
    AttackCapability.XSS_REFLECTED.value: {
        "name": "反射型 XSS",
        "description": "檢測反射型跨站腳本漏洞",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.COOKIE],
        "default_timeout": 30,
        "risk_level": "medium",
    },
    AttackCapability.XSS_STORED.value: {
        "name": "存儲型 XSS",
        "description": "檢測存儲型跨站腳本漏洞",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.COOKIE, CapabilityParameter.TOKEN],
        "default_timeout": 60,
        "risk_level": "high",
    },
    
    # Port Scan
    ScanCapability.PORT_SCAN_TCP.value: {
        "name": "TCP 端口掃描",
        "description": "掃描目標的 TCP 端口",
        "required_params": [CapabilityParameter.TARGET_IP],
        "optional_params": [CapabilityParameter.TARGET_PORT_RANGE, CapabilityParameter.THREADS],
        "default_timeout": 120,
        "risk_level": "low",
    },
    ScanCapability.PORT_SCAN_FULL.value: {
        "name": "完整端口掃描",
        "description": "掃描所有 65535 個端口",
        "required_params": [CapabilityParameter.TARGET_IP],
        "optional_params": [CapabilityParameter.THREADS],
        "default_timeout": 600,
        "risk_level": "low",
    },
    
    # Vulnerability Scan
    ScanCapability.VULNERABILITY_SCAN.value: {
        "name": "漏洞掃描",
        "description": "全面漏洞掃描",
        "required_params": [CapabilityParameter.TARGET_URL],
        "optional_params": [CapabilityParameter.DEPTH, CapabilityParameter.TIMEOUT],
        "default_timeout": 300,
        "risk_level": "medium",
    },
    
    # Recon
    ReconCapability.WHOIS_LOOKUP.value: {
        "name": "WHOIS 查詢",
        "description": "查詢域名註冊資訊",
        "required_params": [CapabilityParameter.TARGET_DOMAIN],
        "optional_params": [],
        "default_timeout": 30,
        "risk_level": "info",
    },
    ReconCapability.DNS_LOOKUP.value: {
        "name": "DNS 查詢",
        "description": "查詢 DNS 記錄",
        "required_params": [CapabilityParameter.TARGET_DOMAIN],
        "optional_params": [],
        "default_timeout": 30,
        "risk_level": "info",
    },
    ReconCapability.SUBDOMAIN_ENUMERATION.value: {
        "name": "子域名枚舉",
        "description": "發現目標的子域名",
        "required_params": [CapabilityParameter.TARGET_DOMAIN],
        "optional_params": [CapabilityParameter.WORDLIST, CapabilityParameter.THREADS],
        "default_timeout": 300,
        "risk_level": "info",
    },
    
    # Analysis
    AnalysisCapability.SAST_SCAN.value: {
        "name": "靜態應用安全測試",
        "description": "原始碼安全分析",
        "required_params": [CapabilityParameter.OUTPUT_PATH],  # 原始碼路徑
        "optional_params": [],
        "default_timeout": 600,
        "risk_level": "info",
    },
    AnalysisCapability.SECRET_DETECTION.value: {
        "name": "密鑰檢測",
        "description": "檢測原始碼中的密鑰和敏感資訊",
        "required_params": [CapabilityParameter.OUTPUT_PATH],
        "optional_params": [],
        "default_timeout": 120,
        "risk_level": "high",
    },
    
    # Forensic
    ForensicCapability.MEMORY_ANALYSIS.value: {
        "name": "記憶體分析",
        "description": "分析記憶體傾印檔案",
        "required_params": [CapabilityParameter.OUTPUT_PATH],  # dump 檔案路徑
        "optional_params": [],
        "default_timeout": 300,
        "risk_level": "info",
    },
    ForensicCapability.STEGANOGRAPHY_DETECT.value: {
        "name": "隱寫術檢測",
        "description": "檢測圖片中的隱藏資訊",
        "required_params": [CapabilityParameter.OUTPUT_PATH],  # 圖片路徑
        "optional_params": [],
        "default_timeout": 60,
        "risk_level": "info",
    },
    ForensicCapability.STEGANOGRAPHY_EXTRACT.value: {
        "name": "隱寫術提取",
        "description": "提取圖片中的隱藏資訊",
        "required_params": [CapabilityParameter.OUTPUT_PATH],  # 圖片路徑
        "optional_params": [CapabilityParameter.PASSWORD],
        "default_timeout": 60,
        "risk_level": "info",
    },
}


# =============================================================================
# 輔助函數 - 獲取能力資訊
# =============================================================================


def get_capability_config(capability: str) -> dict[str, Any] | None:
    """獲取能力配置
    
    Args:
        capability: 能力名稱（枚舉值）
        
    Returns:
        能力配置字典，若不存在則返回 None
    """
    return CAPABILITY_CONFIGS.get(capability)


def get_required_params(capability: str) -> list[CapabilityParameter]:
    """獲取能力的必需參數
    
    Args:
        capability: 能力名稱
        
    Returns:
        必需參數列表
    """
    config = get_capability_config(capability)
    if config:
        return config.get("required_params", [])
    return []


def get_all_capabilities() -> dict[str, list[str]]:
    """獲取所有可用能力，按類別分組
    
    Returns:
        按類別分組的能力字典
    """
    return {
        "attack": [cap.value for cap in AttackCapability],
        "scan": [cap.value for cap in ScanCapability],
        "recon": [cap.value for cap in ReconCapability],
        "analysis": [cap.value for cap in AnalysisCapability],
        "forensic": [cap.value for cap in ForensicCapability],
        "exploit": [cap.value for cap in ExploitCapability],
        "report": [cap.value for cap in ReportCapability],
    }


def validate_capability_params(
    capability: str, 
    params: dict[str, Any]
) -> tuple[bool, list[str]]:
    """驗證能力參數是否完整
    
    Args:
        capability: 能力名稱
        params: 提供的參數字典
        
    Returns:
        (是否有效, 缺少的參數列表)
    """
    config = get_capability_config(capability)
    if not config:
        return False, [f"未知能力: {capability}"]
    
    required = config.get("required_params", [])
    missing = []
    
    for param in required:
        param_name = param.value if isinstance(param, CapabilityParameter) else param
        if param_name not in params:
            missing.append(param_name)
    
    return len(missing) == 0, missing
