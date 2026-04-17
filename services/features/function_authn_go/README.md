# function_authn_go - 認證繞過與漏洞檢測

> **版本**: v2.0.0 | **狀態**: 🔧 開發中 (65%) | **語言**: Go + Python Wrapper

## 🎯 模組概述

這是一個用 Go 語言實現的認證安全測試引擎，專注於 Web 應用認證機制的漏洞檢測。透過真實 HTTP 請求進行弱密碼、Session 安全、2FA 繞過等測試，並透過 Python 包裝器 (`authn_wrapper.py`) 提供給 AIVA 其他 Python 模組叫用。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 弱密碼檢測 | ✅ 完成 | 真實 HTTP POST 登入測試，含帳號鎖定偵測 |
| Session 安全分析 | ✅ 完成 | Cookie HttpOnly/Secure/SameSite/熵值檢查 |
| 2FA 繞過測試 | ✅ 完成 | 直接存取繞過、空碼測試、API 狀態洩漏 |
| JWT 令牌分析 | ⏳ 待實作 | alg:none、RS256→HS256 混淆、弱密鑰 |
| OAuth/SSO 漏洞 | ⏳ 待實作 | OAuth 2.0 流程繞過、CSRF |
| 密碼噴灑攻擊 | ⏳ 待實作 | 單密碼多帳號，避免帳號鎖定 |

## 📐 架構設計

```
function_authn_go/
├── __init__.py              # 模組入口，對外提供 scan_authentication
├── authn_wrapper.py         # Python 包裝器（負責尋找並執行 Go 二進位，解析 JSON 結果）
├── cmd/
│   └── worker/
│       └── main.go          # Go 入口點
├── internal/
│   ├── engine.go            # Go 核心測試引擎（弱密碼/Session/2FA）
│   ├── config.go            # Go 測試配置（AuthnConfig）
│   └── amqp.go              # RabbitMQ 訊息中介
├── Dockerfile               # 容器化構建
└── go.mod                   # Go 模組定義
```

> ⚠️ **注意**：模組預期 `bin/` 目錄中存在編譯好的二進位檔案。Windows (.exe) 與 Linux 皆可。

## 🚀 快速開始

### 1. 編譯 Go 引擎

```bash
cd services/features/function_authn_go

# Linux/Mac
go build -o bin/authn-worker ./cmd/worker/

# Windows
go build -o bin/authn-worker.exe ./cmd/worker/
```

### 2. Python Wrapper 呼叫 (推薦)

```python
from services.features.function_authn_go import scan_authentication, get_engine_info

# 檢查引擎狀態
info = get_engine_info()
if info['available']:
    # 執行掃描
    result = scan_authentication(
        target="https://example.com/login",
        options={
            "username": "admin",
            "test_types": ["weak_password", "default_credentials"]
        }
    )
    print(f"發現 {result['total_vulnerabilities']} 個問題")
```

## 🔧 內部實作細節

### 弱密碼檢測
對目標登入端點發送真實 HTTP POST 請求，比對基線回應判定是否登入成功。
包含**安全機制**：
- 偵測帳號鎖定訊息（`too many attempts`、`account locked` 等），鎖定時自動停止。
- 每次請求之間的速率限制。

### Session 安全分析
對目標 URL 發送 GET 請求，分析回應的 Set-Cookie 標頭 (HttpOnly, Secure, SameSite, Short Session ID 等風險)。

### 2FA 繞過測試
測試：直接存取受保護路徑、空碼/零碼提交、API 狀態洩漏等。

## 🔒 安全模式說明

- 弱密碼測試偵測到帳號鎖定訊息時**自動停止**。
- 2FA 測試只發送不具破壞性的探測請求。
- 測試過程中**不會**儲存或外傳任何認證資訊。
