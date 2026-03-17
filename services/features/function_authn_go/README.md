# function_authn_go - 認證繞過與漏洞檢測

> **版本**: v2.0.0 | **狀態**: 🔧 開發中 (65%) | **語言**: Go + Python Wrapper | **更新**: 2026-03-17

## 🎯 模組概述

Go 語言實現的認證安全測試引擎，專注於 Web 應用認證機制的漏洞檢測。透過真實 HTTP 請求進行弱密碼、Session 安全、2FA 繞過等測試。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 弱密碼檢測 | ✅ 完成 | 真實 HTTP POST 登入測試，含帳號鎖定偵測 |
| Session 安全分析 | ✅ 完成 | Cookie HttpOnly/Secure/SameSite/熵值檢查 |
| 2FA 繞過測試 | ✅ 完成 | 直接存取繞過、空碼測試、API 狀態洩漏 |
| JWT 令牌分析 | ⏳ 待實作 | alg:none、RS256→HS256 混淆、弱密鑰 |
| OAuth/SSO 漏洞 | ⏳ 待實作 | OAuth 2.0 流程繞過、CSRF |
| 密碼噴灑攻擊 | ⏳ 待實作 | 單密碼多帳號，避免帳號鎖定 |

## 📐 實際架構

```
function_authn_go/
├── cmd/
│   └── worker/
│       └── main.go          # Go AMQP Worker 入口（RabbitMQ）
├── internal/
│   ├── engine.go            # 核心測試引擎（弱密碼/Session/2FA）
│   ├── config.go            # 測試配置（AuthnConfig）
│   └── amqp.go              # RabbitMQ 訊息中介
├── authn_wrapper.py         # Python 包裝器（尋找並執行 Go 二進位）
├── __init__.py              # 模組入口
├── Dockerfile               # 容器化構建（Linux 目標）
├── go.mod                   # Go 模組定義
└── README.md
```

> ⚠️ **注意**：`bin/` 目錄中的二進位為 Windows (.exe) 版本，Docker 環境需重新編譯 Linux 版本。

## 🚀 快速開始

### 1. 編譯 Go 引擎

```bash
cd services/features/function_authn_go

# Linux/Mac（Docker 環境請使用此指令）
go build -o bin/authn-worker ./cmd/worker/

# Windows
go build -o bin/authn-worker.exe ./cmd/worker/
```

### 2. Go 引擎直接呼叫

```go
package main

import (
    "fmt"
    "aiva/function_authn_go/internal"
)

func main() {
    cfg := internal.DefaultConfig()
    cfg.TargetURL = "https://example.com/login"
    cfg.WeakPasswordTest = true
    cfg.SessionHijackTest = true
    cfg.Bypass2FATest = true

    engine := internal.NewAuthnEngine(cfg)
    findings, err := engine.RunTests(internal.AuthnTask{
        Username: "admin",
    })
    if err != nil {
        panic(err)
    }
    for _, f := range findings {
        fmt.Printf("[%s] %s: %s\n", f.Severity, f.Name, f.Description)
    }
}
```

### 3. Python Wrapper 呼叫

```python
from services.features.function_authn_go.authn_wrapper import AuthnWrapper

wrapper = AuthnWrapper()
result = wrapper.run(
    target_url="https://example.com/login",
    username="admin",
    weak_password=True,
    session_hijack=True,
    bypass_2fa=True,
)
print(f"發現 {len(result.get('findings', []))} 個問題")
```

## 🔧 功能詳解

### 弱密碼檢測

對目標登入端點發送真實 HTTP POST 請求，比對基線回應判定是否登入成功。

**預設密碼字典（20組）**：`admin`、`password`、`123456`、`admin123`、`qwerty`、`root`、`test123`、`letmein`、`welcome`、`passw0rd` 等。

**安全機制**：
- 偵測帳號鎖定訊息（`too many attempts`、`account locked` 等），鎖定時自動停止
- 每次請求之間 500ms 速率限制（可透過 `RateLimitMs` 設定）
- 最多嘗試次數限制（`MaxLoginAttempts`，預設 5）

**設定範例**：
```go
cfg := internal.DefaultConfig()
cfg.TargetURL = "https://example.com/login"
cfg.UsernameField = "email"      // 表單欄位名稱
cfg.PasswordField = "pass"       // 表單欄位名稱
cfg.MaxLoginAttempts = 10
cfg.RateLimitMs = 1000           // 1 秒間隔
```

### Session 安全分析

對目標 URL 發送 GET 請求，分析回應的 Set-Cookie 標頭。

**偵測項目**：
- **Missing HttpOnly** — Cookie 可被 JavaScript 讀取（XSS 竊取風險）
- **Missing Secure Flag** — HTTPS 站點 Cookie 可能透過 HTTP 傳輸
- **Missing SameSite** — 缺少 CSRF 防護屬性
- **Short Session ID** — Session ID 長度 < 16 字元（可預測性風險）

**觸發條件**：Cookie 名稱包含 `session`、`sid`、`token`、`auth`、`jwt`、`phpsessid`、`jsessionid` 等關鍵字。

### 2FA 繞過測試

測試三種常見的雙因素驗證繞過手法：

**手法一：直接存取繞過**
測試 `/dashboard`、`/home`、`/account`、`/admin` 等受保護頁面，確認是否可在未完成 2FA 的情況下存取。

**手法二：空碼/零碼測試**
向 `/verify`、`/2fa`、`/otp`、`/mfa` 等端點提交 `""`、`000000`、`null`、`0`，檢查是否被接受。

**手法三：API 狀態洩漏**
檢查 `/api/auth/status`、`/api/user/me`、`/api/session` 等端點是否在 JSON 回應中暴露 2FA 相關欄位（`2fa`、`mfa`、`otp` 等）。

## 📊 輸出格式

`RunTests()` 回傳 `[]AuthnFinding`，每筆結構如下：

```json
[
  {
    "name": "Weak Password Detected",
    "description": "Account 'admin' uses weak password 'admin123'",
    "severity": "Critical",
    "evidence": "HTTP 302 response indicated successful authentication"
  },
  {
    "name": "Missing HttpOnly Flag",
    "description": "Session cookie 'PHPSESSID' lacks HttpOnly flag, vulnerable to XSS cookie theft",
    "severity": "Medium",
    "evidence": "Set-Cookie: PHPSESSID (HttpOnly=false)"
  },
  {
    "name": "2FA Bypass Test Completed",
    "description": "No obvious 2FA bypass vulnerabilities detected via common techniques",
    "severity": "Info",
    "evidence": "Tested 6 post-auth paths and 6 2FA endpoints"
  }
]
```

**Severity 等級**：`Critical` / `High` / `Medium` / `Low` / `Info` / `Error`

## 🔒 安全模式說明

- 弱密碼測試：偵測到帳號鎖定訊息時**自動停止**，不繼續嘗試
- 速率限制：預設 500ms 間隔，可調整
- 2FA 測試：只發送不具破壞性的探測請求（空碼、GET 請求）
- 不會儲存或外傳任何測試中取得的認證資訊

## ⏳ 待實作功能

以下功能尚未實作，不應在 README 或文件中標記為完成：

- **JWT 令牌分析**：alg:none 攻擊、RS256→HS256 混淆、弱密鑰字典爆破
- **OAuth/SSO 漏洞**：OAuth 2.0 流程繞過、state 參數 CSRF
- **密碼噴灑**：多帳號 × 單密碼策略
- **默認憑證庫**：廠商特定的默認帳號密碼清單

## 🔗 相關標準

- [OWASP Testing Guide - Authentication Testing](https://owasp.org/www-project-web-security-testing-guide/stable/4-Web_Application_Security_Testing/04-Authentication_Testing/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Password Guidelines SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)

## 📝 更新日誌

### v2.0.0 (2026-03-17)
- ✅ 實作真實 HTTP 弱密碼測試（取代 stub 佔位邏輯）
- ✅ 實作 Session Cookie 安全分析（HttpOnly/Secure/SameSite/熵值）
- ✅ 實作 2FA 繞過測試（直接存取/空碼/API 洩漏）
- ✅ 增加帳號鎖定偵測與速率限制
- ✅ 擴充 `AuthnConfig`（TargetURL/UsernameField/PasswordField/Timeout/RateLimit）
- ✅ 更正 README 狀態（移除錯誤的「完成」標記）

### v1.3.0 (2026-01-20)
- ✅ 移除 worker.py 包裝層
- ✅ 整合 RabbitMQ AMQP 訊息中介

### v1.2.0 (2025-12-17)
- ✅ `authn_wrapper.py` 完成
- ✅ Dockerfile 建立

---

**維護者**: AIVA Team | **授權**: MIT License
