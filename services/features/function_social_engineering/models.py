"""
Social Engineering Module Data Models

定義社交工程模組的資料模型與枚舉。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


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
    target_emails: List[str]
    sender_email: str
    subject: str
    template_name: str
    callback_url: str
    
    # SMTP Configuration
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Advanced Options
    personalization: bool = False
    use_url_shortener: bool = False
    tracking_pixel: bool = True
    schedule_time: Optional[datetime] = None
    
    # Rate Limiting
    send_rate: int = 10  # Emails per minute
    batch_size: int = 50
    
    # Target Filtering
    target_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Template Variables
    template_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class PhishingResult:
    """釣魚攻擊結果"""
    success: bool
    campaign_id: str
    emails_sent: int = 0
    emails_failed: int = 0
    public_url: Optional[str] = None
    log_file: str = ""
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Delivery Details
    delivery_method: Optional[DeliveryMethod] = None
    server_port: Optional[int] = None
    
    # Campaign Info
    status: CampaignStatus = CampaignStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignConfig:
    """活動配置"""
    campaign_name: str
    campaign_type: PhishingType
    target_list: List[str]
    
    # Campaign Settings
    start_time: datetime
    end_time: Optional[datetime] = None
    auto_stop: bool = True
    
    # Tracking Settings
    track_opens: bool = True
    track_clicks: bool = True
    track_credentials: bool = True
    track_geo: bool = True
    track_browser: bool = True
    
    # Notification Settings
    notify_on_credential: bool = True
    notification_webhook: Optional[str] = None
    notification_email: Optional[str] = None
    
    # Campaign Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class TargetInfo:
    """目標資訊"""
    target_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    company: Optional[str] = None
    position: Optional[str] = None
    department: Optional[str] = None
    
    # OSINT Data
    social_profiles: Dict[str, str] = field(default_factory=dict)  # {platform: url}
    phone_numbers: List[str] = field(default_factory=list)
    addresses: List[str] = field(default_factory=list)
    
    # Professional Info
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    work_history: List[Dict[str, str]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0  # 0.0 - 1.0
    data_sources: List[str] = field(default_factory=list)


@dataclass
class CredentialData:
    """憑證數據"""
    credential_id: str
    campaign_id: str
    credential_type: CredentialType
    
    # Credential Details
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None
    
    # Submission Context
    ip_address: str = ""
    user_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Geo Data
    country: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Browser Fingerprint
    browser: Optional[str] = None
    os: Optional[str] = None
    device: Optional[str] = None
    screen_resolution: Optional[str] = None
    
    # Additional Data
    form_data: Dict[str, Any] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


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
    geo_distribution: Dict[str, int] = field(default_factory=dict)  # {country: count}
    city_distribution: Dict[str, int] = field(default_factory=dict) # {city: count}
    
    # Browser/Device Stats
    browser_stats: Dict[str, int] = field(default_factory=dict)     # {browser: count}
    os_stats: Dict[str, int] = field(default_factory=dict)          # {os: count}
    device_stats: Dict[str, int] = field(default_factory=dict)      # {device: count}
    
    # Temporal Analysis
    hourly_activity: Dict[int, int] = field(default_factory=dict)   # {hour: count}
    daily_activity: Dict[str, int] = field(default_factory=dict)    # {date: count}
    
    # Timeline
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
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
    text_body: Optional[str] = None
    
    # Template Variables
    variables: List[str] = field(default_factory=list)  # {{variable_name}}
    
    # Sender Info
    default_sender_name: Optional[str] = None
    default_sender_email: Optional[str] = None
    
    # Attachments
    attachments: List[str] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
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
    logo_url: Optional[str] = None
    company_name: Optional[str] = None
    custom_css: Optional[str] = None
    
    # Tracking
    log_credentials: bool = True
    log_file: str = ""
    redirect_url: Optional[str] = None  # Redirect after submission
    
    # Security
    ssl_enabled: bool = False
    custom_domain: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
