# 🌐 SSRF 檢測模組

**什麼是 SSRF？**  
Server-Side Request Forgery（伺服器端請求偽造）是一種攻擊技術，攻擊者誘使伺服器對內部或第三方系統發起非預期的HTTP請求。本模組整合了內網探測、雲端元數據訪問、協議測試和帶外（OAST）驗證等多重檢測技術。

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   智能 SSRF 檢測架構                          │
├─────────────────────────────────────────────────────────────┤
│ AI Command      │command_handler │ SmartSSRFDetector│ OAST    │
│ Interface       │               │                  │ Service │
│       ↓         │       ↓       │        ↓         │    ↓    │
│ FEATURE_SSRF_   │ FunctionTask  │ ParamSemantics   │ oast_   │
│ TEST            │ Payload       │ Analyzer         │ dispatcher│
│       │         │               │internal_address_ │         │
│       └─────────┼───────────────┼─ detector        │    ↓    │
│                 │               │        ↓         │ Callback│
│                 ↓               │ 多協議測試 +      │ 驗證    │
│         SsrfTestVector          │ 雲端探測         │         │
│         (param_semantics)       │                  │         │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **參數語義分析** - 識別可能的 SSRF 注入點（URL、檔案路徑參數）
2. **測試向量生成** - 根據參數特性生成對應的測試 payload
3. **多層級檢測** - 執行分層檢測策略：
   - **內網探測**: 掃描 127.0.0.1、169.254.169.254（AWS 元數據）等
   - **協議測試**: file://, gopher://, dict://, ldap:// 等協議
   - **帶外驗證**: 使用 OAST 服務確認盲 SSRF
4. **結果整合** - 綜合直接響應和帶外回調確認漏洞

## 🚀 支援指令

### 實際使用方式
```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# SSRF 檢測命令
command = AICommand(
    command_id="ssrf_test_001",
    command_type=CommandType.FEATURE_SSRF_TEST,
    target_module="features.ssrf",
    payload={
        "target_url": "https://app.com/api/fetch",
        "test_parameters": {
            "url": "https://example.com",
            "callback": "https://webhook.site/xyz",
            "file_path": "/etc/passwd"
        },
        "internal_scan": True,
        "cloud_metadata": ["aws", "gcp", "azure"],
        "protocols": ["http", "https", "file", "gopher"],
        "oast_callback": True,
        "timeout": 30
    }
)

# 執行檢測
result = await command_center.execute(command)
```

### 何時使用？
- ✅ **適用場景**:
  - **URL 參數功能**: 圖片獲取、網頁預覽、文件下載
  - **Webhook 回調**: API 通知、第三方整合
  - **代理服務**: 內容代理、API 轉發
  - **雲端服務**: 檔案處理、數據同步
  
- ⚠️ **使用注意**:
  - 避免掃描生產環境的關鍵內部服務
  - 注意可能觸發的安全警報
  - 謹慎測試雲端元數據端點
  - 確保 OAST 回調域名的安全性

### 如何使用？
```python
# 1. 基本內網探測
internal_scan = {
    "target_url": "https://webapp.com/proxy",
    "test_parameters": {"target_url": "http://example.com"},
    "internal_scan": True,
    "targets": [
        "127.0.0.1:22",      # SSH
        "127.0.0.1:3306",    # MySQL
        "127.0.0.1:6379",    # Redis
        "192.168.1.1"        # 內網閘道
    ]
}

# 2. 雲端元數據檢測
cloud_metadata = {
    "target_url": "https://cloud-app.com/fetch-image",
    "test_parameters": {"image_url": "https://example.com/pic.jpg"},
    "cloud_metadata": ["aws", "gcp", "azure"],
    "metadata_endpoints": [
        "http://169.254.169.254/latest/meta-data/",  # AWS
        "http://metadata.google.internal/",          # GCP
        "http://169.254.169.254/metadata/instance"   # Azure
    ]
}

# 3. 協議測試
protocol_test = {
    "target_url": "https://app.com/load-config",
    "test_parameters": {"config_path": "/app/config.json"},
    "protocols": ["file", "gopher", "dict", "ldap"],
    "payloads": [
        "file:///etc/passwd",
        "file:///c:/windows/win.ini",
        "gopher://127.0.0.1:25/",
        "dict://127.0.0.1:11211/stat"
    ]
}

# 4. 帶外（Blind SSRF）檢測
blind_ssrf = {
    "target_url": "https://app.com/webhook",
    "test_parameters": {"callback_url": "https://legitimate.com"},
    "oast_callback": True,
    "oast_service_url": "http://localhost:8083",
    "blind_payloads": [
        "http://{oast_token}.your-domain.com",
        "https://{oast_token}.your-domain.com/ssrf-test"
    ]
}

# 5. 綜合檢測（推薦）
comprehensive_ssrf = {
    "target_url": "https://complex-app.com/api/process",
    "test_parameters": {
        "data_source": "https://external-api.com/data",
        "backup_url": "https://backup.com/save",
        "template_path": "/templates/default.html"
    },
    "internal_scan": True,
    "cloud_metadata": ["aws", "gcp", "azure"],
    "protocols": ["http", "https", "file", "gopher", "dict"],
    "oast_callback": True,
    "follow_redirects": False,  # 避免重定向干擾
    "custom_headers": {
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "169.254.169.254"
    }
}
```

## 🔧 核心能力
- **多協議支援**: HTTP/HTTPS/File/Gopher/Dict/LDAP 全覆蓋
- **雲端元數據檢測**: AWS/GCP/Azure 元數據服務探測
- **帶外驗證**: OAST 服務實現 blind SSRF 確認
- **智能參數分析**: 根據參數名稱和語義選擇測試策略
- **內網掃描**: 自動化內部服務發現和端口探測
- **繞過技術**: IP 編碼、重定向利用、DNS 重綁定

## 🎯 後續發展方向
- [ ] **容器環境檢測** - Docker socket、Kubernetes API 探測
- [ ] **IPv6 支援** - 擴展到 IPv6 網路環境檢測
- [ ] **DNS 重綁定** - 高級 DNS 攻擊技術實現
- [ ] **機器學習參數識別** - AI 自動識別潛在 SSRF 參數