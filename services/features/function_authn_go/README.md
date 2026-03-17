# function_authn_go - 認證繞過與漏洞檢測

> **版本**: v2.0.0 | **狀態**: 🔧 開發中 | **語言**: Go + Python Wrapper | **更新**: 2026-03-17

## 🎯 模組概述

Go語言實現的高性能認證安全測試引擎，專注於Web應用認證機制的漏洞檢測。

### 核心功能

✅ **弱密碼檢測** - 真實 HTTP POST 登入測試，支援帳號鎖定偵測與速率限制
✅ **Session 安全分析** - Cookie 安全屬性檢查（HttpOnly/Secure/SameSite/熵值）
✅ **2FA 繞過測試** - 直接存取繞過、空碼/零碼測試、API 狀態洩漏
⏳ **JWT 令牌分析** - 待實作
⏳ **OAuth/SSO 漏洞** - 待實作
⏳ **密碼噴灑攻擊** - 待實作

## 📐 架構設計

```
function_authn_go/
├── cmd/
│   └── worker/
│       └── main.go          # Go AMQP Worker 入口
├── internal/
│   ├── engine.go            # 核心測試引擎（弱密碼/Session/2FA）
│   ├── config.go            # 測試配置
│   └── amqp.go              # RabbitMQ 訊息中介
├── authn_wrapper.py         # Python 包裝器（CLI 調用入口）
├── __init__.py              # 模組入口
├── Dockerfile               # 容器化構建
└── README.md
```

## 🚀 快速開始

### 1. 編譯 Go 引擎

```bash
cd services/features/function_authn_go

# Windows
go build -o bin/authn-worker.exe cmd/worker/main.go

# Linux/Mac
go build -o bin/authn-worker cmd/worker/main.go
```

### 2. Python 調用

```python
from services.features.function_authn_go.authn_wrapper import scan_authentication

# 弱密碼測試
result = scan_authentication(
    target="https://example.com/login",
    options={
        "username": "admin",
        "test_types": ["weak_password", "default_credentials"]
    }
)

print(f"發現 {len(result['vulnerabilities'])} 個漏洞")
```

### 3. CLI 直接調用

```bash
# external_executor 調用示例
python aiva_external_executor.py --lang go --module authn_go --target https://example.com/login
```

## 🔧 功能詳解

### 弱密碼檢測

測試常見弱密碼組合：
- `admin/admin123`
- `root/toor`
- `test/test123`
- Top 1000 常見密碼

```python
result = scan_authentication(
    target="https://example.com/login",
    options={
        "test_types": ["weak_password"],
        "wordlist": "rockyou.txt"  # 可選自定義字典
    }
)
```

### Session 安全分析

檢測 Session Cookie 安全問題：
- **Missing HttpOnly** - Cookie 可被 JavaScript 讀取（XSS 風險）
- **Missing Secure Flag** - HTTPS 站點 Cookie 可能透過 HTTP 傳輸
- **Missing SameSite** - 缺少 CSRF 防護屬性
- **Short Session ID** - Session ID 長度不足（可預測性風險）

```python
result = scan_authentication(
    target="https://example.com",
    options={"test_types": ["session_security"]}
)
```

### 2FA 繞過測試

測試常見的雙因素驗證繞過：
- **直接存取繞過** - 跳過 2FA 步驟直接存取受保護頁面
- **空碼/零碼測試** - 測試空值、000000、null 等 OTP 碼
- **API 狀態洩漏** - 檢查 API 端點是否暴露 2FA 配置

```python
result = scan_authentication(
    target="https://example.com/login",
    options={"test_types": ["2fa_bypass"]}
)
```

## 📊 輸出格式

```json
{
  "success": true,
  "target": "https://example.com/login",
  "test_types": ["weak_password", "jwt_analysis"],
  "vulnerabilities": [
    {
      "type": "weak_password",
      "severity": "high",
      "username": "admin",
      "password": "admin123",
      "evidence": "Successfully logged in with default credentials",
      "recommendation": "Enforce strong password policy and change default credentials"
    },
    {
      "type": "jwt_algorithm_confusion",
      "severity": "critical",
      "detail": "JWT accepts 'none' algorithm",
      "recommendation": "Reject JWT tokens with 'none' algorithm"
    }
  ],
  "summary": {
    "total_tests": 15,
    "vulnerabilities_found": 2,
    "execution_time": "2.3s"
  }
}
```

## 🔒 安全模式

所有測試都在**安全模式**下運行：
- ✅ 僅檢測漏洞，不執行破壞性操作
- ✅ 速率限制（避免觸發 WAF/IDS）
- ✅ 自動停止（檢測到帳號鎖定時）

```python
# 安全模式（默認啟用）
result = scan_authentication(
    target="https://example.com",
    options={
        "safe_mode": True,  # 默認值
        "rate_limit": 10    # 每秒最多10個請求
    }
)
```

## 🎯 適用場景

✅ **Bug Bounty 計劃** - 符合各大平台測試範圍  
✅ **滲透測試** - 認證機制安全評估  
✅ **DevSecOps** - CI/CD 流程中的自動化安全測試  
✅ **Red Team** - 初始訪問階段的憑證獲取  

## 📈 性能特點

- ⚡ **高性能**: Go 併發處理，比 Python 快 10x
- 🎯 **智能速率控制**: 自動適應目標響應速度
- 🛡️ **防護繞過**: 內建 User-Agent 輪換、Proxy 支持
- 📊 **詳細報告**: 每個測試的完整證據和建議

## 🔗 相關資源

- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [JWT Security Best Practices](https://datatracker.ietf.org/doc/html/rfc8725)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)

## 📝 更新日誌

### v2.0.0 (2026-03-17)
- ✅ 實作真實 HTTP 弱密碼測試（取代 stub 佔位邏輯）
- ✅ 實作 Session Cookie 安全分析（HttpOnly/Secure/SameSite/熵值）
- ✅ 實作 2FA 繞過測試（直接存取/空碼/API 洩漏）
- ✅ 增加帳號鎖定偵測與速率限制
- ✅ 擴充預設密碼字典至 20 組
- ⏳ JWT 令牌分析（待實作）
- ⏳ OAuth/SSO 漏洞（待實作）

### v1.3.0 (2026-01-20)
- ✅ 移除不必要的 worker.py 包裝層
- ✅ 直接使用 Go 引擎，無 Python 回退

### v1.2.0 (2025-12-17)
- ✅ 創建 BUILD_GUIDE.md
- ✅ authn_wrapper.py 完成
- ✅ 架構符合 SIMPLE_ARCHITECTURE.md

---

**維護者**: AIVA Team | **授權**: MIT License
