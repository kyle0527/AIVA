"""
Social Engineering Module Data Models

定義社交工程模組的資料模型與枚舉。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ==================== Enums ====================

class PhishingType(str, Enum):
    """釣魚攻擊類型"""
    EMAIL_PHISHING = "email_phishing"           # 電子郵件釣魚
    SPEAR_PHISHING = "spear_phishing"           # 針對性釣魚
    WHALING = "whaling"                         # 捕鯨攻擊 (高階主管)
    VISHING = "vishing"                         # 語音釣魚
    SMISHING = "smishing"                       # 簡訊釣魚
    QR_JACKING = "qr_jacking"                   # QR Code 劫持
    ANGLER_PHISHING = "angler_phishing"         # 社交媒體釣魚


class TargetPlatform(str, Enum):
    """目標平台"""
    OFFICE365 = "office365"                     # Microsoft Office 365
    GOOGLE_WORKSPACE = "google_workspace"       # Google Workspace
    GITHUB = "github"                           # GitHub
    AWS = "aws"                                 # Amazon Web Services
    AZURE = "azure"                             # Microsoft Azure
    SALESFORCE = "salesforce"                   # Salesforce
    SLACK = "slack"                             # Slack
    TEAMS = "teams"                             # Microsoft Teams
    GENERIC = "generic"                         # 通用平台


class DeliveryMethod(str, Enum):
    """傳遞方式"""
    NGROK = "ngrok"                             # Ngrok 隧道
    LOCALTUNNEL = "localtunnel"                 # LocalTunnel
    SERVEO = "serveo"                           # Serveo
    CLOUDFLARE_TUNNEL = "cloudflare_tunnel"     # Cloudflare Tunnel
    DIRECT = "direct"                           # 直接連線 (需公網 IP)
    AWS_LAMBDA = "aws_lambda"                   # AWS Lambda
    AZURE_FUNCTIONS = "azure_functions"         # Azure Functions


class CredentialType(str, Enum):
    """憑證類型"""
    USERNAME_PASSWORD = "username_password"      # 使用者名稱/密碼
    API_KEY = "api_key"                         # API 金鑰
    ACCESS_TOKEN = "access_token"               # 存取權杖
    SESSION_COOKIE = "session_cookie"           # Session Cookie
    SSH_KEY = "ssh_key"                         # SSH 金鑰
    MFA_TOKEN = "mfa_token"                     # 多因素驗證 Token
    CREDIT_CARD = "credit_card"                 # 信用卡資訊


class TemplateCategory(str, Enum):
    """模板分類"""
    SECURITY_ALERT = "security_alert"           # 安全警告
    PASSWORD_RESET = "password_reset"           # 密碼重設
    INVOICE = "invoice"                         # 發票/帳單
    DOCUMENT_SHARE = "document_share"           # 文件分享
    MEETING_INVITE = "meeting_invite"           # 會議邀請
    HR_NOTIFICATION = "hr_notification"         # HR 通知
    IT_SUPPORT = "it_support"                   # IT 支援
    ACCOUNT_VERIFICATION = "account_verification" # 帳號驗證
    DELIVERY_NOTIFICATION = "delivery_notification" # 配送通知


class CampaignStatus(str, Enum):
    """活動狀態"""
    DRAFT = "draft"                             # 草稿
    SCHEDULED = "scheduled"                     # 已排程
    RUNNING = "running"                         # 執行中
    PAUSED = "paused"                           # 已暫停
    COMPLETED = "completed"                     # 已完成
    FAILED = "failed"                           # 失敗
    CANCELLED = "cancelled"                     # 已取消


class AnalyticsMetric(str, Enum):
    """分析指標"""
    EMAIL_SENT = "email_sent"                   # 郵件發送
    EMAIL_DELIVERED = "email_delivered"         # 郵件送達
    EMAIL_OPENED = "email_opened"               # 郵件開啟
    LINK_CLICKED = "link_clicked"               # 連結點擊
    CREDENTIAL_SUBMITTED = "credential_submitted" # 憑證提交
    ATTACHMENT_DOWNLOADED = "attachment_downloaded" # 附件下載
    FORM_COMPLETED = "form_completed"           # 表單完成


# ==================== Data Models ====================

@dataclass
class PhishingConfig:
    """釣魚攻擊配置"""
    phishing_type: PhishingType
    target_platform: TargetPlatform
    target_emails: list[str]
    sender_email: str
    subject: str
    template_name: str
    callback_url: str
    
    # SMTP Configuration
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    
    # Advanced Options
    personalization: bool = False
    use_url_shortener: bool = False
    tracking_pixel: bool = True
    schedule_time: datetime | None = None
    
    # Rate Limiting
    send_rate: int = 10  # Emails per minute
    batch_size: int = 50
    
    # Target Filtering
    target_filters: dict[str, Any] = field(default_factory=dict)
    
    # Template Variables
    template_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class PhishingResult:
    """釣魚攻擊結果"""
    success: bool
    campaign_id: str
    emails_sent: int = 0
    emails_failed: int = 0
    public_url: str | None = None
    log_file: str = ""
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Delivery Details
    delivery_method: DeliveryMethod | None = None
    server_port: int | None = None
    
    # Campaign Info
    status: CampaignStatus = CampaignStatus.RUNNING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignConfig:
    """活動配置"""
    campaign_name: str
    campaign_type: PhishingType
    target_list: list[str]
    
    # Campaign Settings
    start_time: datetime
    end_time: datetime | None = None
    auto_stop: bool = True
    
    # Tracking Settings
    track_opens: bool = True
    track_clicks: bool = True
    track_credentials: bool = True
    track_geo: bool = True
    track_browser: bool = True
    
    # Notification Settings
    notify_on_credential: bool = True
    notification_webhook: str | None = None
    notification_email: str | None = None
    
    # Campaign Metadata
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class TargetInfo:
    """目標資訊"""
    target_id: str
    email: str | None = None
    name: str | None = None
    company: str | None = None
    position: str | None = None
    department: str | None = None
    
    # OSINT Data
    social_profiles: dict[str, str] = field(default_factory=dict)  # {platform: url}
    phone_numbers: list[str] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    
    # Professional Info
    linkedin_url: str | None = None
    github_url: str | None = None
    work_history: list[dict[str, str]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0  # 0.0 - 1.0
    data_sources: list[str] = field(default_factory=list)


@dataclass
class CredentialData:
    """憑證數據"""
    credential_id: str
    campaign_id: str
    credential_type: CredentialType
    
    # Credential Details
    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    token: str | None = None
    
    # Submission Context
    ip_address: str = ""
    user_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Geo Data
    country: str | None = None
    city: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    
    # Browser Fingerprint
    browser: str | None = None
    os: str | None = None
    device: str | None = None
    screen_resolution: str | None = None
    
    # Additional Data
    form_data: dict[str, Any] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalyticsData:
    """分析數據"""
    campaign_id: str
    
    # Email Metrics
    emails_sent: int = 0
    emails_delivered: int = 0
    emails_opened: int = 0
    emails_bounced: int = 0
    
    # Engagement Metrics
    links_clicked: int = 0
    attachments_downloaded: int = 0
    forms_completed: int = 0
    credentials_submitted: int = 0
    
    # Calculated Rates
    delivery_rate: float = 0.0      # (delivered / sent) * 100
    open_rate: float = 0.0          # (opened / delivered) * 100
    click_rate: float = 0.0         # (clicked / opened) * 100
    success_rate: float = 0.0       # (credentials / clicked) * 100
    
    # Geographic Distribution
    geo_distribution: dict[str, int] = field(default_factory=dict)  # {country: count}
    city_distribution: dict[str, int] = field(default_factory=dict) # {city: count}
    
    # Browser/Device Stats
    browser_stats: dict[str, int] = field(default_factory=dict)     # {browser: count}
    os_stats: dict[str, int] = field(default_factory=dict)          # {os: count}
    device_stats: dict[str, int] = field(default_factory=dict)      # {device: count}
    
    # Temporal Analysis
    hourly_activity: dict[int, int] = field(default_factory=dict)   # {hour: count}
    daily_activity: dict[str, int] = field(default_factory=dict)    # {date: count}
    
    # Timeline
    start_time: datetime | None = None
    end_time: datetime | None = None
    last_activity: datetime | None = None
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EmailTemplate:
    """郵件模板"""
    template_id: str
    template_name: str
    category: TemplateCategory
    
    # Content
    subject: str
    html_body: str
    text_body: str | None = None
    
    # Template Variables
    variables: list[str] = field(default_factory=list)  # {{variable_name}}
    
    # Sender Info
    default_sender_name: str | None = None
    default_sender_email: str | None = None
    
    # Attachments
    attachments: list[str] = field(default_factory=list)
    
    # Metadata
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    use_count: int = 0


@dataclass
class LandingPageConfig:
    """登入頁配置"""
    page_id: str
    platform: TargetPlatform
    template_path: str
    
    # Server Settings
    port: int = 8080
    delivery_method: DeliveryMethod = DeliveryMethod.NGROK
    
    # Customization
    logo_url: str | None = None
    company_name: str | None = None
    custom_css: str | None = None
    
    # Tracking
    log_credentials: bool = True
    log_file: str = ""
    redirect_url: str | None = None  # Redirect after submission
    
    # Security
    ssl_enabled: bool = False
    custom_domain: str | None = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
