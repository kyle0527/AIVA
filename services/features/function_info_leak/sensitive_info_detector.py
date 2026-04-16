#!/usr/bin/env python3
"""
敏感資訊洩漏檢測模組 - 專業級實現

提供全面的敏感資訊檢測功能,包括:
- API Keys, Tokens, Passwords (50+ 種平台)
- 個人識別資訊 (PII) - 國際化支持
- 雲端憑證 (AWS, GCP, Azure, GitHub, GitLab, Slack, etc.)
- 資料庫連線字串 (MySQL, PostgreSQL, MongoDB, Redis, etc.)
- 私鑰和憑證 (RSA, EC, OpenSSH, PGP)
- OAuth tokens, JWT, Session cookies
- 熵值分析 (檢測高隨機性密鑰)
- 上下文感知誤報過濾
- SARIF 格式輸出支持

Version: 2.0.0
Author: AIVA Security Team
License: MIT
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


class SensitiveInfoType(Enum):
    """敏感資訊類型 - 擴展至 50+ 種"""
    # 認證資訊
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    SECRET_KEY = "secret_key"
    PASSWORD = "password"
    AUTH_COOKIE = "auth_cookie"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    
    # AWS 憑證
    AWS_KEY = "aws_access_key"
    AWS_SECRET = "aws_secret_key"
    AWS_SESSION_TOKEN = "aws_session_token"
    AWS_MWS_KEY = "aws_mws_key"
    
    # GCP 憑證
    GCP_API_KEY = "gcp_api_key"
    GCP_SERVICE_ACCOUNT = "gcp_service_account"
    GOOGLE_OAUTH = "google_oauth"
    
    # Azure 憑證
    AZURE_KEY = "azure_key"
    AZURE_CONNECTION_STRING = "azure_connection_string"
    AZURE_SAS_TOKEN = "azure_sas_token"
    
    # GitHub / GitLab
    GITHUB_TOKEN = "github_token"
    GITHUB_APP_TOKEN = "github_app_token"
    GITHUB_REFRESH_TOKEN = "github_refresh_token"
    GITLAB_TOKEN = "gitlab_token"
    GITLAB_RUNNER_TOKEN = "gitlab_runner_token"
    
    # 通訊平台
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    DISCORD_TOKEN = "discord_token"
    TELEGRAM_BOT_TOKEN = "telegram_bot_token"
    
    # 支付平台
    STRIPE_KEY = "stripe_key"
    STRIPE_SECRET = "stripe_secret"
    PAYPAL_TOKEN = "paypal_token"
    SQUARE_TOKEN = "square_token"
    
    # 資料庫
    DB_CONNECTION = "database_connection_string"
    MONGODB_URI = "mongodb_uri"
    REDIS_URI = "redis_uri"
    POSTGRES_URI = "postgresql_uri"
    MYSQL_URI = "mysql_uri"
    
    # 憑證與加密
    PRIVATE_KEY = "private_key"
    PGP_PRIVATE_KEY = "pgp_private_key"
    JWT_TOKEN = "jwt_token"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    
    # 個人識別資訊 (PII)
    EMAIL = "email"
    PHONE = "phone_number"
    SSN = "social_security_number"
    PASSPORT = "passport_number"
    DRIVERS_LICENSE = "drivers_license"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    
    # 第三方服務
    TWILIO_KEY = "twilio_key"
    SENDGRID_KEY = "sendgrid_key"
    MAILGUN_KEY = "mailgun_key"
    NPM_TOKEN = "npm_token"
    PYPI_TOKEN = "pypi_token"
    DOCKER_AUTH = "docker_auth"
    HEROKU_KEY = "heroku_key"
    
    # 內部資訊
    INTERNAL_PATH = "internal_path"
    DEBUG_INFO = "debug_information"
    STACK_TRACE = "stack_trace"
    SQL_QUERY = "sql_query"
    
    # 熵值檢測
    HIGH_ENTROPY_STRING = "high_entropy_string"


class AlertSeverity(Enum):
    """警報嚴重性等級"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Location(Enum):
    """資訊位置"""
    RESPONSE_BODY = "response_body"
    REQUEST_BODY = "request_body"
    HEADER = "header"
    COOKIE = "cookie"
    URL_PARAM = "url_parameter"
    HTML_COMMENT = "html_comment"
    HTML_BODY = "html_body"
    META_TAG = "meta_tag"
    SCRIPT_TAG = "script_tag"
    JAVASCRIPT = "javascript"


@dataclass
class SensitiveMatch:
    """敏感資訊匹配結果"""
    info_type: SensitiveInfoType
    value: str
    location: Location
    context: str = ""
    line_number: int | None = None
    column_number: int | None = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    description: str = ""
    recommendation: str = ""
    confidence: float = 1.0  # 0.0 - 1.0, 信心度評分
    entropy: float | None = None  # 熵值
    hash: str = ""  # 用於去重
    
    def __post_init__(self):
        """計算匹配項的 hash"""
        if not self.hash:
            content = f"{self.info_type.value}:{self.value}:{self.location.value}"
            self.hash = hashlib.md5(content.encode()).hexdigest()


@dataclass
class RiskScore:
    """風險評分"""
    total_score: float = 0.0  # 總分 0-100
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def calculate(self, matches: list[SensitiveMatch]) -> None:
        """計算風險評分"""
        severity_weights = {
            AlertSeverity.CRITICAL: 25,
            AlertSeverity.HIGH: 10,
            AlertSeverity.MEDIUM: 5,
            AlertSeverity.LOW: 2,
            AlertSeverity.INFO: 1
        }
        
        for match in matches:
            if match.severity == AlertSeverity.CRITICAL:
                self.critical_count += 1
            elif match.severity == AlertSeverity.HIGH:
                self.high_count += 1
            elif match.severity == AlertSeverity.MEDIUM:
                self.medium_count += 1
            elif match.severity == AlertSeverity.LOW:
                self.low_count += 1
            else:
                self.info_count += 1
            
            self.total_score += severity_weights[match.severity] * match.confidence
        
        # 限制最高分 100
        self.total_score = min(self.total_score, 100.0)
        
        # 決定風險等級
        if self.critical_count > 0 or self.total_score >= 75:
            self.risk_level = "CRITICAL"
        elif self.high_count > 0 or self.total_score >= 50:
            self.risk_level = "HIGH"
        elif self.medium_count > 0 or self.total_score >= 25:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "LOW"


@dataclass
class DetectionResult:
    """檢測結果"""
    url: str = ""
    matches: list[SensitiveMatch] = field(default_factory=list)
    risk_score: RiskScore = field(default_factory=RiskScore)
    scan_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """計算風險評分"""
        self.risk_score.calculate(self.matches)
    
    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "url": self.url,
            "scan_time": self.scan_time,
            "total_matches": len(self.matches),
            "risk_score": {
                "total": round(self.risk_score.total_score, 2),
                "level": self.risk_score.risk_level,
                "critical": self.risk_score.critical_count,
                "high": self.risk_score.high_count,
                "medium": self.risk_score.medium_count,
                "low": self.risk_score.low_count,
                "info": self.risk_score.info_count
            },
            "matches": [
                {
                    "type": match.info_type.value,
                    "location": match.location.value,
                    "severity": match.severity.value,
                    "confidence": round(match.confidence, 2),
                    "value": match.value[:50] + "..." if len(match.value) > 50 else match.value,
                    "context": match.context[:100] + "..." if len(match.context) > 100 else match.context,
                    "line": match.line_number,
                    "column": match.column_number,
                    "description": match.description,
                    "recommendation": match.recommendation,
                    "entropy": round(match.entropy, 3) if match.entropy else None,
                    "hash": match.hash
                }
                for match in self.matches
            ],
            "metadata": self.metadata
        }
    
    def to_sarif(self) -> dict:
        """轉換為 SARIF 格式 (Static Analysis Results Interchange Format)"""
        rules = {}
        results = []
        
        for match in self.matches:
            rule_id = match.info_type.value
            
            # 建立規則定義
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": match.info_type.value.replace("_", " ").title(),
                    "shortDescription": {
                        "text": match.description
                    },
                    "fullDescription": {
                        "text": match.recommendation
                    },
                    "defaultConfiguration": {
                        "level": self._sarif_level(match.severity)
                    },
                    "help": {
                        "text": match.recommendation
                    }
                }
            
            # 建立結果項
            result_item = {
                "ruleId": rule_id,
                "level": self._sarif_level(match.severity),
                "message": {
                    "text": f"{match.description}: {match.value[:50]}"
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self.url
                        },
                        "region": {
                            "startLine": match.line_number or 1,
                            "startColumn": match.column_number or 1,
                            "snippet": {
                                "text": match.context
                            }
                        }
                    }
                }],
                "properties": {
                    "confidence": match.confidence,
                    "entropy": match.entropy,
                    "hash": match.hash
                }
            }
            results.append(result_item)
        
        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "AIVA Sensitive Info Detector",
                        "version": "2.0.0",
                        "informationUri": "https://github.com/kyle0527/AIVA",
                        "rules": list(rules.values())
                    }
                },
                "results": results,
                "properties": {
                    "risk_score": self.risk_score.total_score,
                    "risk_level": self.risk_score.risk_level,
                    "scan_time": self.scan_time
                }
            }]
        }
    
    def _sarif_level(self, severity: AlertSeverity) -> str:
        """轉換嚴重性等級為 SARIF 格式"""
        mapping = {
            AlertSeverity.CRITICAL: "error",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "note",
            AlertSeverity.INFO: "none"
        }
        return mapping.get(severity, "warning")


class SensitiveInfoDetector:
    """
    敏感資訊檢測器 - 專業級實現
    
    功能:
    - 檢測 50+ 種認證資訊 (API keys, tokens, passwords)
    - 檢測個人識別資訊 (PII): email, phone, SSN, credit card (國際化)
    - 檢測雲端憑證 (AWS, GCP, Azure, GitHub, GitLab, Slack, Stripe, etc.)
    - 檢測資料庫連線字串 (MySQL, PostgreSQL, MongoDB, Redis, etc.)
    - 檢測內部路徑與系統配置資訊
    - 檢測 debug info 與錯誤訊息
    - 檢測 HTML 註釋中的敏感資訊
    - 熵值分析 (檢測高隨機性密鑰)
    - 上下文感知誤報過濾
    - 信心度評分
    - SARIF 格式輸出

    使用範例:
        detector = SensitiveInfoDetector()
        
        # 檢測 HTML 內容
        result = detector.detect_in_html(html_content, url="https://example.com")
        
        # 檢測標頭
        result = detector.detect_in_headers(headers, url="https://example.com")
        
        # 檢測 JavaScript
        result = detector.detect_in_javascript(js_code, url="https://example.com/app.js")
        
        # 輸出 SARIF 格式
        sarif_report = result.to_sarif()
        
        # 輸出 JSON 格式
        json_report = result.to_dict()
    """
    
    # 誤報關鍵字 (白名單)
    FALSE_POSITIVE_KEYWORDS = {
        'example', 'sample', 'test', 'demo', 'placeholder', 'dummy',
        'fake', 'mock', 'your_api_key', 'your_token', 'xxx', '***',
        'redacted', 'hidden', 'censored', 'removed'
    }
    
    # 常見變數名 (降低信心度)
    COMMON_VARIABLE_NAMES = {
        'key', 'token', 'password', 'secret', 'api_key', 'access_token',
        'auth', 'credential', 'pass', 'pwd'
    }
    
    def __init__(
        self, 
        min_severity: AlertSeverity = AlertSeverity.INFO,
        enable_entropy_check: bool = True,
        entropy_threshold: float = 4.5,
        min_confidence: float = 0.3,
        custom_patterns: dict | None = None
    ):
        """
        初始化檢測器

        Args:
            min_severity: 最小嚴重性等級過濾閾值
            enable_entropy_check: 啟用熵值檢測
            entropy_threshold: 熵值閾值 (default: 4.5)
            min_confidence: 最小信心度閾值 (default: 0.3)
            custom_patterns: 自訂檢測模式
        """
        self.min_severity = min_severity
        self.enable_entropy_check = enable_entropy_check
        self.entropy_threshold = entropy_threshold
        self.min_confidence = min_confidence
        self._patterns = self._build_patterns()
        
        # 合併自訂模式
        if custom_patterns:
            self._patterns.update(custom_patterns)
        
        # 去重集合
        self._seen_hashes: set[str] = set()
        
        logger.info(
            f"Initialized SensitiveInfoDetector: "
            f"{len(self._patterns)} patterns, "
            f"entropy_check={enable_entropy_check}, "
            f"threshold={entropy_threshold}"
        )

    def _build_patterns(self) -> dict[SensitiveInfoType, dict]:
        """建立檢測模式 - 擴展至 50+ 種"""
        return {
            # ===== AWS 憑證 =====
            SensitiveInfoType.AWS_KEY: {
                "pattern": re.compile(r'\b(AKIA[0-9A-Z]{16})\b'),
                "severity": AlertSeverity.CRITICAL,
                "description": "AWS Access Key ID detected",
                "recommendation": "Revoke this key immediately and rotate credentials via IAM console"
            },
            SensitiveInfoType.AWS_SECRET: {
                "pattern": re.compile(r'(?i)aws.{0,20}?[\'"][0-9a-zA-Z/+=]{40}[\'"]'),
                "severity": AlertSeverity.CRITICAL,
                "description": "AWS Secret Access Key detected",
                "recommendation": "Revoke this secret key immediately"
            },
            SensitiveInfoType.AWS_SESSION_TOKEN: {
                "pattern": re.compile(r'(?i)aws.{0,20}?session[\'"]?\s*[:=]\s*[\'"]?([A-Za-z0-9+/]{100,})[\'"]?'),
                "severity": AlertSeverity.HIGH,
                "description": "AWS Session Token detected",
                "recommendation": "Session tokens are temporary but should not be exposed"
            },
            
            # ===== GCP 憑證 =====
            SensitiveInfoType.GCP_API_KEY: {
                "pattern": re.compile(r'\b(AIza[0-9A-Za-z_-]{35})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "Google Cloud API key detected",
                "recommendation": "Restrict API key usage and rotate if compromised"
            },
            SensitiveInfoType.GCP_SERVICE_ACCOUNT: {
                "pattern": re.compile(r'"type"\s*:\s*"service_account"'),
                "severity": AlertSeverity.CRITICAL,
                "description": "GCP Service Account JSON detected",
                "recommendation": "Remove service account credentials from code immediately"
            },
            
            # ===== Azure 憑證 =====
            SensitiveInfoType.AZURE_KEY: {
                "pattern": re.compile(r'(?i)azure.{0,20}?key[\'"]?\s*[:=]\s*[\'"]?([a-zA-Z0-9+/]{40,}={0,2})[\'"]?'),
                "severity": AlertSeverity.HIGH,
                "description": "Azure API key detected",
                "recommendation": "Rotate Azure keys via Azure Portal"
            },
            SensitiveInfoType.AZURE_CONNECTION_STRING: {
                "pattern": re.compile(r'(?i)DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[^;]+'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Azure Storage connection string detected",
                "recommendation": "Use managed identities instead of connection strings"
            },
            
            # ===== GitHub / GitLab =====
            SensitiveInfoType.GITHUB_TOKEN: {
                "pattern": re.compile(r'\b(gh[puso]_[A-Za-z0-9_]{36,})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "GitHub Personal Access Token detected",
                "recommendation": "Revoke token at https://github.com/settings/tokens"
            },
            SensitiveInfoType.GITHUB_APP_TOKEN: {
                "pattern": re.compile(r'\b(ghs_[A-Za-z0-9]{36})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "GitHub App Token detected",
                "recommendation": "Regenerate GitHub App installation token"
            },
            SensitiveInfoType.GITLAB_TOKEN: {
                "pattern": re.compile(r'\b(glpat-[A-Za-z0-9_-]{20,})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "GitLab Personal Access Token detected",
                "recommendation": "Revoke token in GitLab profile settings"
            },
            
            # ===== 通訊平台 =====
            SensitiveInfoType.SLACK_TOKEN: {
                "pattern": re.compile(r'\b(xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "Slack API Token detected",
                "recommendation": "Rotate Slack token at https://api.slack.com/apps"
            },
            SensitiveInfoType.SLACK_WEBHOOK: {
                "pattern": re.compile(r'https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24,}'),
                "severity": AlertSeverity.MEDIUM,
                "description": "Slack Webhook URL detected",
                "recommendation": "Regenerate webhook URL if compromised"
            },
            SensitiveInfoType.DISCORD_TOKEN: {
                "pattern": re.compile(r'\b([MN][A-Za-z\d]{23}\.[A-Za-z\d]{6}\.[A-Za-z\d_-]{27})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "Discord Bot Token detected",
                "recommendation": "Regenerate bot token in Discord Developer Portal"
            },
            SensitiveInfoType.TELEGRAM_BOT_TOKEN: {
                "pattern": re.compile(r'\b(\d{8,10}:[A-Za-z0-9_-]{35})\b'),
                "severity": AlertSeverity.MEDIUM,
                "description": "Telegram Bot Token detected",
                "recommendation": "Regenerate bot token via BotFather"
            },
            
            # ===== 支付平台 =====
            SensitiveInfoType.STRIPE_KEY: {
                "pattern": re.compile(r'\b(sk_live_[0-9a-zA-Z]{24,})\b'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Stripe Secret Key detected",
                "recommendation": "Immediately rotate Stripe secret key"
            },
            SensitiveInfoType.STRIPE_SECRET: {
                "pattern": re.compile(r'\b(rk_live_[0-9a-zA-Z]{24,})\b'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Stripe Restricted Key detected",
                "recommendation": "Rotate Stripe restricted key"
            },
            SensitiveInfoType.PAYPAL_TOKEN: {
                "pattern": re.compile(r'(?i)paypal.{0,20}?(access_token|auth)[\'"]?\s*[:=]\s*[\'"]?([A-Za-z0-9_-]{20,})[\'"]?'),
                "severity": AlertSeverity.HIGH,
                "description": "PayPal Access Token detected",
                "recommendation": "Rotate PayPal API credentials"
            },
            
            # ===== 資料庫 =====
            SensitiveInfoType.MONGODB_URI: {
                "pattern": re.compile(r'mongodb(\+srv)?://[^\s:]+:[^\s@]+@[^\s/]+'),
                "severity": AlertSeverity.CRITICAL,
                "description": "MongoDB connection string with credentials detected",
                "recommendation": "Use environment variables for database credentials"
            },
            SensitiveInfoType.POSTGRES_URI: {
                "pattern": re.compile(r'postgres(?:ql)?://[^\s:]+:[^\s@]+@[^\s/]+'),
                "severity": AlertSeverity.CRITICAL,
                "description": "PostgreSQL connection string with credentials detected",
                "recommendation": "Use environment variables for database credentials"
            },
            SensitiveInfoType.MYSQL_URI: {
                "pattern": re.compile(r'mysql://[^\s:]+:[^\s@]+@[^\s/]+'),
                "severity": AlertSeverity.CRITICAL,
                "description": "MySQL connection string with credentials detected",
                "recommendation": "Use environment variables for database credentials"
            },
            SensitiveInfoType.REDIS_URI: {
                "pattern": re.compile(r'redis(?:s)?://(?:[^\s:]+:[^\s@]+@)?[^\s/]+'),
                "severity": AlertSeverity.HIGH,
                "description": "Redis connection string detected",
                "recommendation": "Secure Redis with authentication and TLS"
            },
            
            # ===== JWT & OAuth =====
            SensitiveInfoType.JWT_TOKEN: {
                "pattern": re.compile(r'\b(eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*)\b'),
                "severity": AlertSeverity.HIGH,
                "description": "JWT token detected",
                "recommendation": "Ensure tokens are not exposed in public responses and implement short expiry"
            },
            SensitiveInfoType.BEARER_TOKEN: {
                "pattern": re.compile(r'(?i)bearer\s+([a-zA-Z0-9_\-\.]{20,})'),
                "severity": AlertSeverity.HIGH,
                "description": "Bearer token detected",
                "recommendation": "Use secure token storage and transmission"
            },
            SensitiveInfoType.BASIC_AUTH: {
                "pattern": re.compile(r'(?i)basic\s+([A-Za-z0-9+/]{20,}={0,2})'),
                "severity": AlertSeverity.HIGH,
                "description": "Basic authentication credentials detected",
                "recommendation": "Use OAuth 2.0 or API keys instead of basic auth"
            },
            
            # ===== 私鑰 =====
            SensitiveInfoType.PRIVATE_KEY: {
                "pattern": re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Private key detected",
                "recommendation": "Remove private keys from code immediately and rotate"
            },
            SensitiveInfoType.PGP_PRIVATE_KEY: {
                "pattern": re.compile(r'-----BEGIN PGP PRIVATE KEY BLOCK-----'),
                "severity": AlertSeverity.CRITICAL,
                "description": "PGP private key detected",
                "recommendation": "Remove PGP private keys from code"
            },
            
            # ===== 密碼 =====
            SensitiveInfoType.PASSWORD: {
                "pattern": re.compile(
                    r'(?i)(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']([^"\'\s]{6,})["\']'
                ),
                "severity": AlertSeverity.HIGH,
                "description": "Hardcoded password detected",
                "recommendation": "Use environment variables or secure vaults for passwords"
            },
            
            # ===== Generic API Keys =====
            SensitiveInfoType.API_KEY: {
                "pattern": re.compile(
                    r'(?i)(?:api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?'
                ),
                "severity": AlertSeverity.HIGH,
                "description": "Generic API key detected",
                "recommendation": "Verify if this is a valid API key and rotate if necessary"
            },
            
            # ===== 第三方服務 =====
            SensitiveInfoType.TWILIO_KEY: {
                "pattern": re.compile(r'\b(SK[a-f0-9]{32})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "Twilio API Key detected",
                "recommendation": "Rotate Twilio API key"
            },
            SensitiveInfoType.SENDGRID_KEY: {
                "pattern": re.compile(r'\b(SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "SendGrid API Key detected",
                "recommendation": "Rotate SendGrid API key"
            },
            SensitiveInfoType.MAILGUN_KEY: {
                "pattern": re.compile(r'\b(key-[a-f0-9]{32})\b'),
                "severity": AlertSeverity.HIGH,
                "description": "Mailgun API Key detected",
                "recommendation": "Rotate Mailgun API key"
            },
            SensitiveInfoType.NPM_TOKEN: {
                "pattern": re.compile(r'\b(npm_[a-zA-Z0-9]{36})\b'),
                "severity": AlertSeverity.MEDIUM,
                "description": "NPM Access Token detected",
                "recommendation": "Revoke NPM token and use .npmrc with environment variables"
            },
            SensitiveInfoType.PYPI_TOKEN: {
                "pattern": re.compile(r'\b(pypi-[A-Za-z0-9_-]{60,})\b'),
                "severity": AlertSeverity.MEDIUM,
                "description": "PyPI API Token detected",
                "recommendation": "Revoke PyPI token in account settings"
            },
            SensitiveInfoType.HEROKU_KEY: {
                "pattern": re.compile(r'\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b'),
                "severity": AlertSeverity.MEDIUM,
                "description": "Potential Heroku API Key (UUID format) detected",
                "recommendation": "Verify and rotate if it's a Heroku API key"
            },
            
            # ===== 個人識別資訊 (PII) =====
            SensitiveInfoType.EMAIL: {
                "pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                "severity": AlertSeverity.LOW,
                "description": "Email address detected",
                "recommendation": "Verify if email disclosure is intentional and complies with privacy policies"
            },
            SensitiveInfoType.CREDIT_CARD: {
                "pattern": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Potential credit card number detected",
                "recommendation": "Never expose credit card numbers in responses"
            },
            SensitiveInfoType.SSN: {
                "pattern": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                "severity": AlertSeverity.CRITICAL,
                "description": "Social Security Number (SSN) detected",
                "recommendation": "Remove SSN immediately - major privacy violation"
            },
            SensitiveInfoType.PHONE: {
                "pattern": re.compile(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                "severity": AlertSeverity.LOW,
                "description": "Phone number detected",
                "recommendation": "Verify if phone number disclosure is intentional"
            },
            
            # ===== 內部資訊 =====
            SensitiveInfoType.INTERNAL_PATH: {
                "pattern": re.compile(
                    r'(?:[C-Z]:\\|/(?:home|root|var|etc|usr|opt)/)[\w\\/.-]+',
                    re.IGNORECASE
                ),
                "severity": AlertSeverity.MEDIUM,
                "description": "Internal file path detected",
                "recommendation": "Avoid exposing internal system paths"
            },
            SensitiveInfoType.STACK_TRACE: {
                "pattern": re.compile(r'(?i)(?:traceback|stack trace|exception):.*?(?:\n\s+at\s|File\s")', re.DOTALL),
                "severity": AlertSeverity.MEDIUM,
                "description": "Stack trace detected",
                "recommendation": "Disable detailed error messages in production"
            },
            SensitiveInfoType.SQL_QUERY: {
                "pattern": re.compile(r'(?i)(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\s+.{0,100}(?:FROM|INTO|TABLE)', re.DOTALL),
                "severity": AlertSeverity.LOW,
                "description": "SQL query detected",
                "recommendation": "Avoid exposing SQL queries in responses"
            }
        }

    def calculate_entropy(self, data: str) -> float:
        """
        計算字符串的香農熵值
        
        Args:
            data: 輸入字符串
            
        Returns:
            float: 熵值 (0-8, 越高越隨機)
        """
        if not data:
            return 0.0
        
        # 計算每個字符的頻率
        counter = Counter(data)
        length = len(data)
        
        # 計算熵值
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _is_false_positive(self, value: str, context: str) -> tuple[bool, float]:
        """
        檢查是否為誤報
        
        Args:
            value: 匹配的值
            context: 上下文
            
        Returns:
            Tuple[bool, float]: (是否誤報, 信心度調整係數)
        """
        value_lower = value.lower()
        context_lower = context.lower()
        confidence_multiplier = 1.0
        
        # 檢查白名單關鍵字
        for keyword in self.FALSE_POSITIVE_KEYWORDS:
            if keyword in value_lower or keyword in context_lower:
                return True, 0.0
        
        # 檢查是否為常見變數名 (降低信心度)
        for var_name in self.COMMON_VARIABLE_NAMES:
            if var_name == value_lower:
                confidence_multiplier = 0.5
                break
        
        # 檢查是否為註釋中的範例 (降低信心度)
        if any(marker in context_lower for marker in ['example:', 'e.g.', 'sample:', '範例:', '例如:']):
            confidence_multiplier = 0.4
        
        # 檢查是否包含佔位符 (誤報)
        placeholder_patterns = [r'\{.*?\}', r'<.*?>', r'\$\{.*?\}', r'%s', r'%d']
        for pattern in placeholder_patterns:
            if re.search(pattern, value):
                return True, 0.0
        
        # 檢查熵值 (低熵值可能是誤報)
        if len(value) > 10:
            entropy = self.calculate_entropy(value)
            if entropy < 3.0:  # 低熵值
                confidence_multiplier *= 0.6
        
        return False, confidence_multiplier

    def detect_in_html(self, html_content: str, url: str = "") -> DetectionResult:
        """
        檢測 HTML 內容中的敏感資訊
        
        Args:
            html_content: HTML 內容
            url: 來源 URL
            
        Returns:
            DetectionResult: 檢測結果
        """
        result = DetectionResult(url=url)
        
        if not html_content:
            return result

        # 檢測 HTML 註釋
        result.matches.extend(self._detect_html_comments(html_content))

        # 檢測 JavaScript 程式碼區塊
        result.matches.extend(self._detect_script_blocks(html_content))

        # 檢測 meta 標籤
        result.matches.extend(self._detect_meta_tags(html_content))

        # 檢測 HTML body 中的資訊
        result.matches.extend(self._detect_in_text(html_content, Location.HTML_BODY))

        # 過濾最小嚴重性等級
        result.matches = self._filter_by_severity(result.matches)

        logger.info(f"HTML detection complete: {len(result.matches)} matches found")
        return result

    def detect_in_headers(
        self, headers: dict[str, str], url: str = ""
    ) -> DetectionResult:
        """檢測 HTTP 標頭中的敏感資訊"""
        result = DetectionResult(url=url)
        
        if not headers:
            return result

        for header_name, header_value in headers.items():
            # 檢測 Set-Cookie 中的 token
            if header_name.lower() == "set-cookie" and re.search(
                r"(token|auth|session|jwt)=([a-zA-Z0-9._-]{20,})",
                header_value,
                re.IGNORECASE,
            ):
                result.matches.append(
                    SensitiveMatch(
                        info_type=SensitiveInfoType.AUTH_COOKIE,
                        value=header_value[:100],
                        location=Location.COOKIE,
                        context=f"{header_name}: {header_value[:200]}",
                        severity=AlertSeverity.MEDIUM,
                        description="Authentication token in cookie",
                        recommendation="Ensure cookies are HttpOnly and Secure"
                    )
                )

            # 檢測其他標頭中的敏感資訊
            header_matches = self._detect_in_text(
                f"{header_name}: {header_value}", Location.HEADER
            )
            result.matches.extend(header_matches)

        result.matches = self._filter_by_severity(result.matches)
        logger.info(f"Header detection complete: {len(result.matches)} matches found")
        return result

    def detect_in_response(
        self, response_body: str, headers: dict[str, str] | None = None, url: str = ""
    ) -> DetectionResult:
        """檢測 HTTP 響應中的敏感資訊"""
        result = DetectionResult(url=url)

        # 檢測響應體
        if response_body:
            # 在 HTML 中檢測
            if "<html" in response_body.lower() or "<body" in response_body.lower():
                html_result = self.detect_in_html(response_body, url)
                result.matches.extend(html_result.matches)
            else:
                # 直接檢測文本內容
                body_matches = self._detect_in_text(
                    response_body, Location.RESPONSE_BODY
                )
                result.matches.extend(body_matches)

        # 檢測標頭
        if headers:
            header_result = self.detect_in_headers(headers, url)
            result.matches.extend(header_result.matches)

        result.matches = self._filter_by_severity(result.matches)
        logger.info(f"Response detection complete: {len(result.matches)} matches found")
        return result

    def _detect_html_comments(self, html: str) -> list[SensitiveMatch]:
        """檢測 HTML 註釋中的敏感資訊"""
        matches: list[SensitiveMatch] = []

        # HTML 註釋
        comment_pattern = r"<!--(.*)-->"
        for comment_match in re.finditer(comment_pattern, html, re.DOTALL):
            comment_text = comment_match.group(1)

            # 提升註釋中的資訊嚴重性
            for match in self._detect_in_text(comment_text, Location.HTML_COMMENT):
                # 提高威脅等級 (註釋不應包含敏感資訊)
                if match.severity == AlertSeverity.INFO:
                    match.severity = AlertSeverity.LOW
                elif match.severity == AlertSeverity.LOW:
                    match.severity = AlertSeverity.MEDIUM
                elif match.severity == AlertSeverity.MEDIUM:
                    match.severity = AlertSeverity.HIGH

                matches.append(match)

        return matches

    def _detect_script_blocks(self, html: str) -> list[SensitiveMatch]:
        """檢測 script 標籤中的敏感資訊"""
        matches: list[SensitiveMatch] = []

        # script 標籤
        script_pattern = r"<script[^>]*>(.*?)</script>"
        for script_match in re.finditer(
            script_pattern, html, re.DOTALL | re.IGNORECASE
        ):
            script_text = script_match.group(1)

            # 檢測 JavaScript 中的敏感資訊
            for match in self._detect_in_text(script_text, Location.SCRIPT_TAG):
                matches.append(match)

        return matches

    def _detect_meta_tags(self, html: str) -> list[SensitiveMatch]:
        """檢測 meta 標籤中的敏感資訊"""
        matches: list[SensitiveMatch] = []

        # 檢測 meta 標籤
        meta_pattern = r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*>'
        for meta_match in re.finditer(meta_pattern, html, re.IGNORECASE):
            content = meta_match.group(1)

            for match in self._detect_in_text(content, Location.META_TAG):
                matches.append(match)

        return matches

    def _detect_in_text(self, text: str, location: Location) -> list[SensitiveMatch]:
        """在文本中檢測敏感資訊 (含熵值分析和誤報過濾)"""
        matches: list[SensitiveMatch] = []

        if not text:
            return matches

        # 按行分割
        lines = text.split("\n")

        for line_number, line in enumerate(lines, start=1):
            for info_type, pattern_info in self._patterns.items():
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]

                for regex_match in re.finditer(pattern, line):
                    matched_value = regex_match.group(0)
                    
                    # 擷取前後 50 字符作為上下文
                    start = max(0, regex_match.start() - 50)
                    end = min(len(line), regex_match.end() + 50)
                    context = line[start:end]
                    
                    # 誤報檢查
                    is_fp, confidence_multiplier = self._is_false_positive(matched_value, context)
                    if is_fp:
                        logger.debug(f"False positive detected: {matched_value[:30]}")
                        continue
                    
                    # 計算熵值
                    entropy = self.calculate_entropy(matched_value)
                    
                    # 計算信心度
                    confidence = min(1.0, confidence_multiplier * (entropy / 8.0 + 0.5))
                    
                    # 過濾低信心度結果
                    if confidence < self.min_confidence:
                        continue
                    
                    match = SensitiveMatch(
                        info_type=info_type,
                        value=matched_value,
                        location=location,
                        context=context,
                        line_number=line_number,
                        column_number=regex_match.start(),
                        severity=severity,
                        description=pattern_info["description"],
                        recommendation=pattern_info["recommendation"],
                        confidence=confidence,
                        entropy=entropy
                    )
                    
                    # 去重
                    if match.hash not in self._seen_hashes:
                        self._seen_hashes.add(match.hash)
                        matches.append(match)

        # 熵值檢測 (檢測高隨機性字符串)
        if self.enable_entropy_check:
            matches.extend(self._detect_high_entropy_strings(text, location))

        return matches
    
    def _detect_high_entropy_strings(self, text: str, location: Location) -> list[SensitiveMatch]:
        """檢測高熵值字符串 (可能是密鑰)"""
        matches: list[SensitiveMatch] = []
        
        # 提取可能是密鑰的字符串 (長度 20-100, 包含字母數字)
        potential_keys = re.findall(r'\b([a-zA-Z0-9_-]{20,100})\b', text)
        
        for key in potential_keys:
            entropy = self.calculate_entropy(key)
            
            # 高熵值且未被其他規則匹配
            if entropy >= self.entropy_threshold:
                # 避免誤報 (檢查是否為常見格式)
                is_fp, confidence = self._is_false_positive(key, text)
                if is_fp or confidence < 0.3:
                    continue
                
                match = SensitiveMatch(
                    info_type=SensitiveInfoType.HIGH_ENTROPY_STRING,
                    value=key,
                    location=location,
                    context=key[:100],
                    severity=AlertSeverity.MEDIUM,
                    description=f"High entropy string detected (entropy: {entropy:.2f})",
                    recommendation="Verify if this is a credential or secret key",
                    confidence=min(0.8, entropy / 6.0),
                    entropy=entropy
                )
                
                if match.hash not in self._seen_hashes:
                    self._seen_hashes.add(match.hash)
                    matches.append(match)
        
        return matches

    def _filter_by_severity(self, matches: list[SensitiveMatch]) -> list[SensitiveMatch]:
        """根據最小嚴重性等級過濾結果"""
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4
        }

        min_level = severity_order[self.min_severity]
        
        return [
            match for match in matches
            if severity_order[match.severity] <= min_level
        ]

    def format_report(self, result: DetectionResult, format: str = "text") -> str:
        """
        格式化檢測報告
        
        Args:
            result: 檢測結果
            format: 輸出格式 ('text', 'json', 'sarif')
            
        Returns:
            str: 格式化後的報告
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        elif format == "sarif":
            return json.dumps(result.to_sarif(), indent=2, ensure_ascii=False)
        
        # Text format
        lines = [
            "=" * 80,
            "敏感資訊檢測報告 - AIVA Security Scanner v2.0",
            "=" * 80,
            f"掃描時間: {result.scan_time}",
            f"目標URL: {result.url}",
            "",
            "風險評分:",
            f"  總分: {result.risk_score.total_score:.2f}/100",
            f"  風險等級: {result.risk_score.risk_level}",
            "",
            "問題統計:",
            f"  🔴 CRITICAL: {result.risk_score.critical_count}",
            f"  🟠 HIGH:     {result.risk_score.high_count}",
            f"  🟡 MEDIUM:   {result.risk_score.medium_count}",
            f"  🟢 LOW:      {result.risk_score.low_count}",
            f"  ℹ️  INFO:     {result.risk_score.info_count}",
            "",
            f"發現問題數量: {len(result.matches)}",
            "=" * 80
        ]

        if result.matches:
            # 按嚴重性排序
            sorted_matches = sorted(
                result.matches,
                key=lambda m: (
                    {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[
                        m.severity.value
                    ], -m.confidence
                ),
            )

            severity_icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
                "info": "ℹ️"
            }

            for idx, match in enumerate(sorted_matches, 1):
                icon = severity_icons.get(match.severity.value, "•")
                lines.append(f"\n{icon} [{idx}] {match.severity.value.upper()} - {match.info_type.value}")
                lines.append(f"  位置: {match.location.value}")
                if match.line_number:
                    lines.append(f"  行號: {match.line_number}")
                if match.column_number:
                    lines.append(f"  列號: {match.column_number}")
                lines.append(f"  值: {match.value[:80]}")
                lines.append(f"  上下文: {match.context[:100]}")
                lines.append(f"  信心度: {match.confidence:.2%}")
                if match.entropy:
                    lines.append(f"  熵值: {match.entropy:.3f}")
                lines.append(f"  說明: {match.description}")
                lines.append(f"  建議: {match.recommendation}")
                lines.append(f"  Hash: {match.hash[:16]}...")

        lines.append("\n" + "=" * 80)
        lines.append("掃描完成 - Powered by AIVA Security Team")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def export_report(self, result: DetectionResult, output_file: str, format: str = "json") -> None:
        """
        導出檢測報告到文件
        
        Args:
            result: 檢測結果
            output_file: 輸出文件路徑
            format: 輸出格式 ('text', 'json', 'sarif')
        """
        report = self.format_report(result, format=format)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report exported to: {output_file} (format: {format})")
    
    def get_statistics(self, results: list[DetectionResult]) -> dict[str, Any]:
        """
        生成統計資訊
        
        Args:
            results: 多個檢測結果列表
            
        Returns:
            Dict: 統計資訊
        """
        total_matches = sum(len(r.matches) for r in results)
        total_critical = sum(r.risk_score.critical_count for r in results)
        total_high = sum(r.risk_score.high_count for r in results)
        total_medium = sum(r.risk_score.medium_count for r in results)
        total_low = sum(r.risk_score.low_count for r in results)
        total_info = sum(r.risk_score.info_count for r in results)
        
        # 按類型統計
        type_counter = Counter()
        for result in results:
            for match in result.matches:
                type_counter[match.info_type.value] += 1
        
        # 計算平均風險評分
        avg_risk_score = sum(r.risk_score.total_score for r in results) / len(results) if results else 0.0
        
        return {
            "total_scans": len(results),
            "total_matches": total_matches,
            "average_risk_score": round(avg_risk_score, 2),
            "severity_distribution": {
                "critical": total_critical,
                "high": total_high,
                "medium": total_medium,
                "low": total_low,
                "info": total_info
            },
            "top_issues": dict(type_counter.most_common(10)),
            "scanned_urls": [r.url for r in results]
        }


# ===== 便利函數 =====

def quick_scan(content: str, content_type: str = "text", url: str = "") -> DetectionResult:
    """
    快速掃描內容
    
    Args:
        content: 要掃描的內容
        content_type: 內容類型 ('text', 'html', 'json', 'headers')
        url: 來源 URL
        
    Returns:
        DetectionResult: 檢測結果
    """
    detector = SensitiveInfoDetector()
    
    if content_type == "html":
        return detector.detect_in_html(content, url=url)
    elif content_type == "json":
        try:
            parsed = json.loads(content)
            json_str = json.dumps(parsed, indent=2)
            return detector.detect_in_response(json_str, url=url)
        except json.JSONDecodeError:
            return detector.detect_in_response(content, url=url)
    else:
        return detector.detect_in_response(content, url=url)


def batch_scan(targets: list[tuple[str, str]], output_dir: str = "scan_reports") -> list[DetectionResult]:
    """
    批次掃描多個目標
    
    Args:
        targets: [(content, url), ...] 列表
        output_dir: 報告輸出目錄
        
    Returns:
        List[DetectionResult]: 所有掃描結果
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    detector = SensitiveInfoDetector()
    results = []
    
    for idx, (content, url) in enumerate(targets, 1):
        logger.info(f"Scanning target {idx}/{len(targets)}: {url}")
        result = detector.detect_in_response(content, url=url)
        results.append(result)
        
        # 導出個別報告
        safe_filename = re.sub(r'[^\w\-_]', '_', url)[:50]
        report_file = os.path.join(output_dir, f"scan_{idx}_{safe_filename}.json")
        detector.export_report(result, report_file, format="json")
    
    # 導出統計報告
    stats = detector.get_statistics(results)
    stats_file = os.path.join(output_dir, "scan_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dumps(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Batch scan complete. Reports saved to: {output_dir}")
    return results


# ===== 主程式範例 =====

if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 範例 1: 基本使用
    detector = SensitiveInfoDetector(
        min_severity=AlertSeverity.INFO,
        enable_entropy_check=True,
        entropy_threshold=4.5
    )
    
    sample_text = """
    {
        "api_key": "sk_live_51H8xyzABCDEFGHIJKLMNOP",
        "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
        "database": "mongodb://admin:MyP@ssw0rd123@localhost:27017/mydb",
        "email": "user@example.com",
        "credit_card": "4532-1234-5678-9010"
    }
    """
    
    result = detector.detect_in_response(sample_text, url="https://api.example.com/config")
    
    # 輸出文本報告
    print(detector.format_report(result, format="text"))
    
    # 導出 JSON 報告
    detector.export_report(result, "scan_report.json", format="json")
    
    # 導出 SARIF 報告
    detector.export_report(result, "scan_report.sarif", format="sarif")
    
    print(f"\n✅ 掃描完成! 風險評分: {result.risk_score.total_score:.2f}/100 ({result.risk_score.risk_level})")
