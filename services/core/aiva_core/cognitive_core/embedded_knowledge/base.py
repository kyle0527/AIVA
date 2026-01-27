"""
內嵌知識基礎模組 - 定義通用數據類型和基礎結構

此模組提供所有內嵌知識模組共用的基礎類型定義，
確保 AI 決策系統能夠統一解析和使用檢測結果。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeAlias
from datetime import datetime, timezone


# ==================== 枚舉類型 ====================

class ConfidenceLevel(Enum):
    """置信度等級 - 用於 AI 決策判斷"""
    ABSOLUTE = auto()      # 100% 確定 (如：明確的錯誤指紋)
    HIGH = auto()          # 高置信度 (>85%)
    MEDIUM = auto()        # 中等置信度 (50-85%)
    LOW = auto()           # 低置信度 (<50%)
    UNCERTAIN = auto()     # 不確定 (需要更多驗證)
    
    def to_score(self) -> float:
        """轉換為數值分數"""
        mapping = {
            ConfidenceLevel.ABSOLUTE: 1.0,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MEDIUM: 0.65,
            ConfidenceLevel.LOW: 0.35,
            ConfidenceLevel.UNCERTAIN: 0.1,
        }
        return mapping[self]


class VulnerabilityType(Enum):
    """漏洞類型枚舉"""
    # 注入類
    SQLI = "sql_injection"
    SQLI_ERROR = "sql_injection_error_based"
    SQLI_BLIND_TIME = "sql_injection_blind_time"
    SQLI_BLIND_BOOL = "sql_injection_blind_boolean"
    SQLI_UNION = "sql_injection_union"
    
    # XSS 類
    XSS = "cross_site_scripting"
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    XSS_DOM = "xss_dom_based"
    
    # SSRF 類
    SSRF = "server_side_request_forgery"
    SSRF_CLOUD_METADATA = "ssrf_cloud_metadata"
    SSRF_INTERNAL = "ssrf_internal_service"
    
    # 授權類
    IDOR = "insecure_direct_object_reference"
    BOLA = "broken_object_level_authorization"
    
    # 現代架構
    GRAPHQL_INTROSPECTION = "graphql_introspection"
    JWT_NONE_ALG = "jwt_none_algorithm"
    JWT_WEAK_SECRET = "jwt_weak_secret"
    MASS_ASSIGNMENT = "mass_assignment"
    WEBSOCKET_HIJACKING = "websocket_hijacking"
    
    # RCE 類
    RCE = "remote_code_execution"
    COMMAND_INJECTION = "command_injection"
    
    # 信息泄露
    INFO_DISCLOSURE = "information_disclosure"
    WEBSOCKET_HIJACK = "websocket_hijack"
    
    # 其他
    PATH_TRAVERSAL = "path_traversal"
    LFI = "local_file_inclusion"
    UNKNOWN = "unknown"


class DatabaseType(Enum):
    """資料庫類型"""
    MYSQL = "mysql"
    MARIADB = "mariadb"
    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    ORACLE = "oracle"
    SQLITE = "sqlite"
    UNKNOWN = "unknown"


class WAFVendor(Enum):
    """WAF 廠商"""
    CLOUDFLARE = "cloudflare"
    AWS_WAF = "aws_waf"
    IMPERVA = "imperva"
    AKAMAI = "akamai"
    MODSECURITY = "modsecurity"
    F5 = "f5"
    FORTINET = "fortinet"
    UNKNOWN = "unknown"
    NONE = "none"


class CloudProvider(Enum):
    """雲端供應商"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ORACLE_CLOUD = "oracle_cloud"
    ALIBABA_CLOUD = "alibaba_cloud"
    UNKNOWN = "unknown"


# ==================== 數據類 ====================

@dataclass
class DetectionResult:
    """
    漏洞檢測結果 - AI 決策系統的標準輸入格式
    
    Attributes:
        detected: 是否檢測到漏洞
        vulnerability_type: 漏洞類型
        confidence: 置信度等級
        confidence_score: 置信度數值 (0.0-1.0)
        evidence: 證據列表
        indicators: 匹配的指標
        false_positive_risk: 誤報風險 (0.0-1.0)
        recommendations: AI 建議的後續動作
        raw_data: 原始檢測數據
    """
    detected: bool
    vulnerability_type: VulnerabilityType
    confidence: ConfidenceLevel
    confidence_score: float
    evidence: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    false_positive_risk: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    technical_details: dict[str, Any] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典格式，便於 JSON 序列化"""
        return {
            "detected": self.detected,
            "vulnerability_type": self.vulnerability_type.value,
            "confidence": self.confidence.name,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "indicators": self.indicators,
            "false_positive_risk": self.false_positive_risk,
            "recommendations": self.recommendations,
            "technical_details": self.technical_details,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def should_exploit(self, risk_threshold: float = 0.7) -> bool:
        """
        判斷是否應該進行利用
        
        AI 決策系統可直接調用此方法獲取建議
        """
        return (
            self.detected 
            and self.confidence_score >= risk_threshold
            and self.false_positive_risk < 0.3
        )


@dataclass
class AttackContext:
    """
    攻擊上下文 - 描述當前攻擊場景
    
    用於為檢測器提供環境信息，提高判斷準確度
    """
    target_url: str
    injection_point: str  # header, parameter, body, path
    parameter_name: str | None = None
    http_method: str = "GET"
    content_type: str | None = None
    detected_waf: WAFVendor = WAFVendor.NONE
    detected_db: DatabaseType = DatabaseType.UNKNOWN
    cloud_provider: CloudProvider = CloudProvider.UNKNOWN
    baseline_response_time: float = 0.0  # 基準響應時間 (秒)
    baseline_response_length: int = 0    # 基準響應長度


@dataclass
class ResponseAnalysis:
    """
    HTTP 響應分析結果
    
    統一的響應分析格式，供各檢測器使用
    """
    status_code: int
    response_time: float  # 秒
    content_length: int
    content_type: str
    body: str
    headers: dict[str, str] = field(default_factory=dict)
    
    # 預處理標記
    contains_error: bool = False
    error_patterns: list[str] = field(default_factory=list)
    reflected_input: bool = False
    dom_changes: list[str] = field(default_factory=list)


@dataclass 
class PayloadResult:
    """
    Payload 執行結果
    
    記錄單個 payload 的執行情況和檢測結果
    """
    payload: str
    success: bool
    detection_result: DetectionResult | None = None
    response: ResponseAnalysis | None = None
    execution_time: float = 0.0
    error: str | None = None


# ==================== 知識註冊表 ====================

class KnowledgeRegistry:
    """
    知識註冊表 - 支援動態擴展內嵌知識
    
    允許在運行時添加新的檢測規則、payload 或指紋
    """
    
    _instance: "KnowledgeRegistry | None" = None
    
    def __new__(cls) -> "KnowledgeRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """初始化註冊表"""
        self._sqli_fingerprints: dict[DatabaseType, list[str]] = {}
        self._xss_payloads: list[str] = []
        self._waf_signatures: dict[WAFVendor, list[str]] = {}
        self._cve_patterns: dict[str, dict[str, Any]] = {}
        self._custom_detectors: dict[str, Callable] = {}
    
    # === SQLi 指紋擴展 ===
    def register_sqli_fingerprint(
        self, 
        db_type: DatabaseType, 
        fingerprints: list[str]
    ) -> None:
        """註冊新的 SQLi 錯誤指紋"""
        if db_type not in self._sqli_fingerprints:
            self._sqli_fingerprints[db_type] = []
        self._sqli_fingerprints[db_type].extend(fingerprints)
    
    def get_sqli_fingerprints(self, db_type: DatabaseType) -> list[str]:
        """獲取指定資料庫的 SQLi 指紋"""
        return self._sqli_fingerprints.get(db_type, [])
    
    # === XSS Payload 擴展 ===
    def register_xss_payload(self, payloads: list[str]) -> None:
        """註冊新的 XSS payload"""
        self._xss_payloads.extend(payloads)
    
    def get_xss_payloads(self) -> list[str]:
        """獲取所有 XSS payload"""
        return self._xss_payloads.copy()
    
    # === WAF 簽名擴展 ===
    def register_waf_signature(
        self, 
        vendor: WAFVendor, 
        signatures: list[str]
    ) -> None:
        """註冊新的 WAF 識別簽名"""
        if vendor not in self._waf_signatures:
            self._waf_signatures[vendor] = []
        self._waf_signatures[vendor].extend(signatures)
    
    # === CVE 模式擴展 ===
    def register_cve_pattern(
        self, 
        cve_id: str, 
        pattern: dict[str, Any]
    ) -> None:
        """註冊新的 CVE 檢測模式"""
        self._cve_patterns[cve_id] = pattern
    
    def get_cve_pattern(self, cve_id: str) -> dict[str, Any] | None:
        """獲取 CVE 檢測模式"""
        return self._cve_patterns.get(cve_id)
    
    # === 自定義檢測器 ===
    def register_detector(
        self, 
        name: str, 
        detector: Callable[[ResponseAnalysis, AttackContext], DetectionResult]
    ) -> None:
        """註冊自定義檢測器"""
        self._custom_detectors[name] = detector
    
    def run_custom_detector(
        self, 
        name: str, 
        response: ResponseAnalysis, 
        context: AttackContext
    ) -> DetectionResult | None:
        """執行自定義檢測器"""
        detector = self._custom_detectors.get(name)
        if detector:
            return detector(response, context)
        return None


# 類型別名 - 便於類型提示
DetectionCallback: TypeAlias = Callable[[ResponseAnalysis, AttackContext], DetectionResult]
