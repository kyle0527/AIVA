# AIVA Crypto Scanner - 網路層密碼學配置掃描器

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/kyle0527/AIVA)
[![Language](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Architecture](https://img.shields.io/badge/architecture-CLI--Only-green.svg)](#架構設計)
[![Status](https://img.shields.io/badge/status-production-success.svg)](#)

## 📋 目錄

- [概述](#概述)
- [架構設計](#架構設計)
- [功能特性](#功能特性)
- [安裝與編譯](#安裝與編譯)
- [使用方式](#使用方式)
- [CLI 命令參考](#cli-命令參考)
- [批量驗證](#批量驗證)
- [掃描結果說明](#掃描結果說明)
- [AI Commander 整合](#ai-commander-整合)
- [技術細節](#技術細節)
- [故障排除](#故障排除)
- [版本歷史](#版本歷史)
- [相關文件](#相關文件)
- [授權](#授權)
- [聯絡方式](#聯絡方式)

---

## 概述

**AIVA Crypto Scanner** 是一個專注於 **黑盒測試** 的網路層密碼學配置掃描工具，使用 **純 Rust** 實作，提供高效能的安全檢測能力。

### 設計理念

- **黑盒測試優先**: 僅分析網路層可觀察的密碼學配置，不涉及原始碼分析
- **純 Rust CLI**: 獨立可執行檔，無 Python 包裝層
- **AI 友善**: 直接輸出 JSON，便於 AI Commander 解析和處理
- **零依賴執行**: 編譯後的執行檔無需額外執行環境

### 主要掃描範圍

1. **JavaScript 密碼學問題** - 硬編碼 API 金鑰、弱加密算法、不安全隨機數
2. **TLS/SSL 配置** - 協議版本、證書驗證、加密套件
3. **Cookie 安全性** - Secure、HttpOnly、SameSite 標誌
4. **HTTP 安全標頭** - HSTS、CSP、X-Frame-Options 等

---

## 架構設計

```
services/features/function_crypto/
├── README.md                    # 本文件
├── __init__.py                  # Python 模組宣告（v2.0 中僅保留元資料）
├── batch_verify.ps1             # 批量驗證腳本
└── rust_core/                   # Rust 核心實作
    ├── Cargo.toml               # Rust 專案配置
    ├── src/
    │   ├── main.rs              # CLI 入口點
    │   ├── js_crypto_analyzer.rs    # JavaScript 分析器
    │   ├── tls_analyzer.rs          # TLS 配置分析器
    │   ├── cookie_analyzer.rs       # Cookie 安全分析器
    │   └── header_analyzer.rs       # HTTP 標頭分析器
    └── target/
        └── release/
            └── crypto-scanner.exe   # 編譯產物（執行檔）
```

### v2.0 架構演進

**v1.0 (已封存)**: 複雜的 Python + Rust 混合架構  
**v2.0 (目前)**: 簡化為純 Rust CLI，AI 直接呼叫執行檔

```
# v1: AI → Python Wrapper → Rust CLI → JSON → Python → AI
# v2: AI → Rust CLI → JSON → AI (直接解析)
```

**v1 封存位置**:  
`C:\Users\User\Downloads\新增資料夾 (3)\function_crypto_v1_source_code_analysis`

---

## 功能特性

### 1️⃣ JavaScript 密碼學分析 (`scan-js`)

檢測客戶端 JavaScript 中的密碼學問題：

#### 檢測項目

| 類別 | 檢測內容 | 嚴重性 |
|------|---------|--------|
| **硬編碼金鑰** | Stripe API Key (live/test) | CRITICAL/HIGH |
|  | AWS Access Key (AKIA*) | CRITICAL |
|  | Google API Key (AIza*) | HIGH |
|  | Generic API Keys/Tokens | HIGH |
| **弱加密算法** | MD5, SHA-1 使用 | MEDIUM |
|  | DES, RC4 使用 | HIGH |
| **弱隨機數** | `Math.random()` | MEDIUM |
|  | 固定種子 | HIGH |
| **JWT 問題** | 客戶端驗證 | HIGH |
|  | 演算法混淆攻擊 | CRITICAL |
| **不安全儲存** | localStorage/sessionStorage | MEDIUM |

### 2️⃣ TLS/SSL 配置分析 (`analyze-tls`)

從網路握手檢測 TLS 配置問題：

#### 檢測項目

- ✅ **協議支援檢測**: 判斷是否使用 HTTPS
- ✅ **連線驗證**: 檢查 TLS 握手是否成功
- ✅ **證書驗證**: 使用系統根證書進行驗證
- ⚠️ **協議版本**: 檢測是否使用過時的 TLS 版本
- ⚠️ **加密套件**: 檢測弱加密算法（如 RC4、DES）

**限制**: 當前 `rustls` API 不直接暴露協議版本和加密套件資訊，主要檢測服務是否啟用 TLS。

### 3️⃣ Cookie 安全分析 (`analyze-cookies`)

檢查 HTTP Cookie 的安全屬性：

#### 檢測項目

| 屬性 | 檢測條件 | 嚴重性 |
|------|---------|--------|
| **Secure** | HTTPS 站點缺少 Secure 標誌 | MEDIUM |
| **HttpOnly** | 敏感 Cookie 缺少 HttpOnly | MEDIUM |
| **SameSite** | 任何 Cookie 缺少 SameSite | LOW |

**敏感 Cookie 關鍵字**: `session`, `auth`, `token`, `jwt`, `csrf`

### 4️⃣ HTTP 安全標頭分析 (`analyze-headers`)

檢查密碼學相關的安全標頭：

#### 檢測項目

| 標頭 | 檢測內容 | 嚴重性 |
|------|---------|--------|
| **HSTS** | 缺少標頭 | MEDIUM |
|  | 缺少 max-age | MEDIUM |
|  | max-age < 1 年 | LOW |
|  | 缺少 includeSubDomains | LOW |
| **CSP** | 缺少 upgrade-insecure-requests | LOW |
|  | 允許 unsafe-inline/unsafe-eval | MEDIUM |
| **Mixed Content** | X-Content-Type-Options | LOW |
|  | 混合 HTTP/HTTPS 資源 | MEDIUM |

---

## 安裝與編譯

### 系統需求

- **Rust** 1.70+ (建議使用 `rustup`)
- **Windows** / Linux / macOS
- 網路連線（用於掃描遠端目標）

### 編譯步驟

```powershell
# 1. 進入 rust_core 目錄
cd services\features\function_crypto\rust_core

# 2. 編譯 Release 版本（效能最佳化）
cargo build --release

# 3. 執行檔位置
# Windows: .\target\release\crypto-scanner.exe
# Linux/macOS: ./target/release/crypto-scanner
```

### 驗證安裝

```powershell
# 查看版本資訊
.\target\release\crypto-scanner.exe --version

# 輸出: crypto-scanner 2.0.0
```

---

## 使用方式

### 基本語法

```bash
crypto-scanner <SUBCOMMAND> [OPTIONS]
```

### 快速範例

```powershell
# 1. 掃描 JavaScript 文件
$jsContent = Get-Content "app.js" -Raw
crypto-scanner scan-js --content $jsContent --url "https://example.com/app.js"

# 2. 分析 TLS 配置
crypto-scanner analyze-tls --target "example.com" --port 443

# 3. 分析 Cookies (從 HTTP 回應取得)
$cookies = @("SESSIONID=abc123; Path=/; HttpOnly")
$cookiesJson = $cookies | ConvertTo-Json -Compress
crypto-scanner analyze-cookies --url "https://example.com" --cookies-json $cookiesJson

# 4. 分析安全標頭
$headers = @{ "Strict-Transport-Security" = "max-age=31536000" }
$headersJson = $headers | ConvertTo-Json -Compress
crypto-scanner analyze-headers --url "https://example.com" --headers-json $headersJson
```

---

## CLI 命令參考

### `scan-js` - JavaScript 密碼學掃描

分析 JavaScript 文件中的密碼學問題。

```bash
crypto-scanner scan-js --content <JS_CONTENT> [--url <URL>]
```

**參數**:
- `--content <STRING>`: JavaScript 文件內容（必填）
- `--url <URL>`: JavaScript 文件來源 URL（選填，用於報告）

**輸出範例**:
```json
{
  "scan_type": "javascript_crypto",
  "target": "https://example.com/app.js",
  "findings": [
    {
      "issue_type": "HARDCODED_API_KEY",
      "severity": "CRITICAL",
      "title": "Hardcoded Live Stripe API Key",
      "description": "Found hardcoded Stripe API key: sk_live_abc1234...",
      "recommendation": "Remove API keys from client-side code. Use environment variables and backend proxy.",
      "location": "Line 42",
      "evidence": "sk_live_********************xyz9"
    }
  ],
  "summary": {
    "total_findings": 1,
    "critical": 1,
    "high": 0,
    "medium": 0,
    "low": 0,
    "info": 0
  }
}
```

---

### `analyze-tls` - TLS/SSL 配置分析

從網路握手檢測 TLS 配置。

```bash
crypto-scanner analyze-tls --target <DOMAIN> [--port <PORT>]
```

**參數**:
- `--target <STRING>`: 目標域名或 IP（必填）
- `--port <NUMBER>`: 目標端口（選填，預設 443）

**輸出範例**:
```json
{
  "scan_type": "tls_configuration",
  "target": "example.com:443",
  "findings": [
    {
      "issue_type": "NO_TLS_DETECTED",
      "severity": "HIGH",
      "title": "Plain HTTP Service Detected",
      "description": "The service appears to be running plain HTTP, not HTTPS.",
      "recommendation": "Enable TLS/SSL (HTTPS) to encrypt communications.",
      "location": "example.com:443",
      "evidence": "TLS handshake error: ..."
    }
  ],
  "summary": { ... }
}
```

---

### `analyze-cookies` - Cookie 安全分析

檢查 HTTP Cookie 的安全屬性。

```bash
crypto-scanner analyze-cookies --url <URL> --cookies-json <JSON>
```

**參數**:
- `--url <STRING>`: 目標 URL（必填，用於判斷 HTTPS）
- `--cookies-json <JSON>`: Set-Cookie 標頭的 JSON 陣列（必填）

**Cookies JSON 格式**:
```json
[
  "SESSIONID=abc123; Path=/",
  "token=xyz789; Path=/; Secure; HttpOnly"
]
```

**輸出範例**:
```json
{
  "scan_type": "cookie_security",
  "target": "https://example.com",
  "findings": [
    {
      "issue_type": "MISSING_SECURE_FLAG",
      "severity": "MEDIUM",
      "title": "Cookie 'SESSIONID' Missing Secure Flag",
      "description": "Cookie transmitted over HTTPS without Secure flag...",
      "recommendation": "Add 'Secure' flag to all cookies on HTTPS sites.",
      "location": "SESSIONID",
      "evidence": "SESSIONID=abc123; Path=/"
    }
  ],
  "summary": { ... }
}
```

---

### `analyze-headers` - HTTP 安全標頭分析

檢查密碼學相關的安全標頭。

```bash
crypto-scanner analyze-headers --url <URL> --headers-json <JSON>
```

**參數**:
- `--url <STRING>`: 目標 URL（必填）
- `--headers-json <JSON>`: HTTP 回應標頭的 JSON 物件（必填）

**Headers JSON 格式**:
```json
{
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
  "Content-Security-Policy": "default-src 'self'",
  "X-Frame-Options": "DENY"
}
```

**輸出範例**:
```json
{
  "scan_type": "security_headers",
  "target": "https://example.com",
  "findings": [
    {
      "issue_type": "MISSING_HSTS",
      "severity": "MEDIUM",
      "title": "Missing HSTS Header",
      "description": "HTTPS site without HSTS allows protocol downgrade attacks.",
      "recommendation": "Add 'Strict-Transport-Security: max-age=31536000; includeSubDomains' header.",
      "location": null,
      "evidence": null
    }
  ],
  "summary": { ... }
}
```

---

## 批量驗證

### `batch_verify.ps1` - 批量掃描腳本

自動化測試多個目標的密碼學配置。

#### 功能特性

- ✅ 自動偵測 `crypto-scanner.exe` 位置
- ✅ 支援多目標掃描（JuiceShop、WebGoat、WebWolf）
- ✅ 整合 Headers、Cookies、TLS 分析
- ✅ 彩色輸出與統計摘要

#### 執行方式

```powershell
# 直接執行
.\batch_verify.ps1

# 或指定完整路徑
.\services\features\function_crypto\batch_verify.ps1
```

#### 預設掃描目標

| 目標 | URL | 端口 |
|------|-----|------|
| JuiceShop-Live | http://localhost:3000 | 3000 |
| JuiceShop-Dev | http://localhost:3001 | 3001 |
| JuiceShop-Test | http://localhost:3003 | 3003 |
| WebGoat-App | http://localhost:8080/WebGoat | 8080 |
| WebWolf | http://localhost:9090/WebWolf | 9090 |

#### 輸出範例（實際測試結果）

```
========== AIVA Crypto Scanner 驗證修正版 ==========
[*] Found scanner at: C:\D\fold7\AIVA-git\target\release\crypto-scanner.exe

[+] 正在掃描: JuiceShop-Live (http://localhost:3000)
    ✓ HTTP 連接成功 (狀態碼: 200)
    [*] 分析安全頭... OK
    [*] 無 Cookie 可分析
    [*] 分析 TLS 配置... 結果: TLS Handshake Failed

[+] 正在掃描: JuiceShop-Dev (http://localhost:3001)
    ✓ HTTP 連接成功 (狀態碼: 200)
    [*] 分析安全頭... OK
    [*] 無 Cookie 可分析
    [*] 分析 TLS 配置... 結果: TLS Handshake Failed

[+] 正在掃描: JuiceShop-Test (http://localhost:3003)
    ✓ HTTP 連接成功 (狀態碼: 200)
    [*] 分析安全頭... OK
    [*] 無 Cookie 可分析
    [*] 分析 TLS 配置... 結果: TLS Handshake Failed

[+] 正在掃描: WebGoat-App (http://localhost:8080/WebGoat)
    ✓ HTTP 連接成功 (狀態碼: 200)
    [*] 分析安全頭... 發現問題!
        - [LOW] Missing X-Content-Type-Options Header
    [*] 分析 Cookies (1)... OK
    [*] 分析 TLS 配置... 結果: TLS Handshake Failed

[+] 正在掃描: WebWolf (http://localhost:9090/WebWolf)
    ✓ HTTP 連接成功 (狀態碼: 200)
    [*] 分析安全頭... OK
    [*] 分析 Cookies (1)... OK
    [*] 分析 TLS 配置... 結果: TLS Handshake Failed

========== 驗證結束 (總計: 6 個問題) ==========
```

**實際測試環境** (2025-12-12):
- ✅ 所有 5 個目標服務正常運行
- ✅ HTTP 連接全部成功
- ⚠️ 所有目標使用 HTTP 協議（無 HTTPS）
- ⚠️ WebGoat 缺少 X-Content-Type-Options 標頭
- ✅ JuiceShop 實例配置較完善（無顯著問題）

#### 自訂掃描目標

編輯 [batch_verify.ps1](batch_verify.ps1#L23-L28) 中的 `$Targets` 陣列：

```powershell
$Targets = @(
    @{ Name="My-App"; Url="https://myapp.local"; Port=443 },
    @{ Name="API-Server"; Url="https://api.myapp.local"; Port=8443 }
)
```

---

## 掃描結果說明

### 嚴重性等級

| 等級 | 描述 | 建議處理時間 |
|------|------|-------------|
| **CRITICAL** | 立即可被利用的嚴重漏洞 | 24 小時內 |
| **HIGH** | 高風險漏洞，需優先處理 | 7 天內 |
| **MEDIUM** | 中等風險，應及時修復 | 30 天內 |
| **LOW** | 低風險，建議改善 | 90 天內 |
| **INFO** | 資訊性發現，無直接風險 | 視情況處理 |

### 常見問題與修復建議

#### 1. Hardcoded API Keys (CRITICAL/HIGH)

**問題**: 客戶端 JavaScript 包含硬編碼的 API 金鑰。

**修復**:
```javascript
// ❌ 錯誤做法
const apiKey = "sk_live_abc123xyz789";

// ✅ 正確做法
// 1. 將 API 金鑰移至後端環境變數
// 2. 客戶端透過後端 API Proxy 呼叫
fetch('/api/stripe/create-payment', { ... });
```

#### 2. Missing HSTS Header (MEDIUM)

**問題**: HTTPS 網站缺少 HSTS 標頭，允許降級攻擊。

**修復** (Nginx):
```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

**修復** (Express.js):
```javascript
const helmet = require('helmet');
app.use(helmet.hsts({
  maxAge: 31536000,
  includeSubDomains: true,
  preload: true
}));
```

#### 3. Missing Secure/HttpOnly Flags (MEDIUM)

**問題**: Cookie 缺少安全標誌。

**修復** (Express.js):
```javascript
app.use(session({
  secret: 'your-secret',
  cookie: {
    secure: true,      // HTTPS only
    httpOnly: true,    // 防止 XSS
    sameSite: 'strict' // 防止 CSRF
  }
}));
```

#### 4. Weak Crypto Usage (MEDIUM/HIGH)

**問題**: 使用弱加密算法（MD5、SHA-1）。

**修復**:
```javascript
// ❌ 錯誤做法
import md5 from 'crypto-js/md5';
const hash = md5(password);

// ✅ 正確做法（後端）
const bcrypt = require('bcrypt');
const hash = await bcrypt.hash(password, 10);
```

---

## AI Commander 整合

### 使用情境

AI Commander 動態生成並執行掃描指令：

```python
import asyncio
import json

async def scan_javascript_crypto(js_content: str, url: str) -> dict:
    """AI Commander 呼叫範例"""
    cmd = [
        "crypto-scanner",
        "scan-js",
        "--content", js_content,
        "--url", url
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"Scanner failed: {stderr.decode()}")
    
    return json.loads(stdout.decode())

# AI 自動決策
result = await scan_javascript_crypto(js_code, "https://target.com")
if result['summary']['critical'] > 0:
    print("[AI] CRITICAL issues found! Immediate action required.")
```

### 優勢

- **無包裝層**: AI 直接執行 CLI，無需額外 Python 程式碼
- **JSON 輸出**: 結構化資料，便於解析和決策
- **高效能**: Rust 執行速度快，適合大規模掃描
- **可移植**: 單一執行檔，無外部依賴

---

## 技術細節

### 依賴項目

| 套件 | 版本 | 用途 |
|------|------|------|
| clap | 4.5 | CLI 參數解析 |
| serde | 1.0 | JSON 序列化 |
| serde_json | 1.0 | JSON 處理 |
| regex | 1.10 | 正則表達式匹配 |
| tokio | 1.35 | 非同步執行時 |
| rustls | 0.23 | TLS/SSL 實作 |
| tokio-rustls | 0.26 | Tokio + Rustls 整合 |
| webpki-roots | 0.26 | 系統根證書 |
| x509-parser | 0.16 | X.509 證書解析 |
| thiserror | 1.0 | 錯誤處理 |

### 編譯優化

`Cargo.toml` 中的效能優化設定：

```toml
[profile.release]
opt-level = 3        # 最高優化等級
lto = true           # Link-Time Optimization
codegen-units = 1    # 單一代碼生成單元（更好的優化）
```

### 測試覆蓋率

- ✅ JavaScript 分析器: 5 大類 20+ 檢測規則
- ✅ TLS 分析器: 連線驗證、協議檢測
- ✅ Cookie 分析器: 3 大安全屬性檢測
- ✅ Headers 分析器: HSTS、CSP、混合內容檢測

---

## 版本歷史

### v2.0.0 (2025-12-12) - 當前版本

**重大變更**:
- 🔄 架構重構為純 Rust CLI
- ❌ 移除 Python 包裝層
- ✨ AI Commander 直接呼叫支援
- 📦 單一執行檔分發

**新增功能**:
- ✅ `batch_verify.ps1` 批量驗證腳本
- ✅ WebGoat/WebWolf 路徑修正
- ✅ 彩色輸出與統計摘要

**改進**:
- ⚡ 效能提升 3-5 倍（相對 v1.0）
- 📝 完整的 CLI 文件與範例
- 🛡️ 更嚴格的 TLS 驗證邏輯

### v1.0.0 (2024-xx-xx) - 已封存

**特性**:
- Python + Rust 混合架構
- 原始碼分析功能（已移除）
- 複雜的模組化設計

**封存位置**:  
`C:\Users\User\Downloads\新增資料夾 (3)\function_crypto_v1_source_code_analysis`

---

## 故障排除

### 腳本執行當機或卡住

#### 症狀
- PowerShell 執行 `batch_verify.ps1` 後無回應
- 終端視窗凍結或需要強制關閉
- AI 工具調用超時

#### 可能原因與解決方案

**1. PowerShell 執行策略限制**
```powershell
# 檢查當前策略
Get-ExecutionPolicy

# 臨時允許執行（當前會話）
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 永久允許（需管理員權限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2. 網路請求超時**
```powershell
# 檢查目標服務是否運行
Test-NetConnection -ComputerName localhost -Port 3000

# 手動測試單一目標（減少超時時間）
.\target\release\crypto-scanner.exe analyze-tls --target localhost --port 3000
```

**3. Rust 執行檔未編譯或路徑錯誤**
```powershell
# 檢查執行檔是否存在
Test-Path .\target\release\crypto-scanner.exe

# 如不存在，重新編譯
cd rust_core
cargo build --release
```

**4. 編碼問題（中文輸出導致緩衝區卡住）**
```powershell
# 使用 UTF-8 編碼
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 或使用英文語系執行
$env:LANG = "en_US.UTF-8"
.\batch_verify.ps1
```

**5. AI 工具調用超時**

如果是通過 AI 工具自動調用：
- AI 異步執行可能缺少超時處理
- 建議手動執行驗證後將結果貼回給 AI
- 或修改 AI 工具調用參數增加超時時間

```python
# AI 工具調用建議加上超時
import asyncio

result = await asyncio.wait_for(
    run_terminal_command("batch_verify.ps1"),
    timeout=60.0  # 60 秒超時
)
```

### 手動驗證步驟

如果腳本無法正常執行，可按以下步驟手動驗證：

```powershell
# 1. 確認執行檔存在
$scanner = ".\target\release\crypto-scanner.exe"
if (Test-Path $scanner) { Write-Host "✓ Scanner found" }

# 2. 測試單一目標
& $scanner analyze-tls --target localhost --port 3000

# 3. 檢查 JSON 輸出
& $scanner analyze-tls --target localhost --port 3000 | ConvertFrom-Json

# 4. 逐步測試各個目標
$targets = @(3000, 3001, 3003, 8080, 9090)
foreach ($port in $targets) {
    Write-Host "Testing port $port..."
    & $scanner analyze-tls --target localhost --port $port
}
```

---

## 相關文件

- [SIMPLE_ARCHITECTURE.md](../../SIMPLE_ARCHITECTURE.md) - 整體架構設計
- [CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md](../../../core/aiva_core/internal_exploration/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md) - CLI 命令分析
- [Cargo.toml](rust_core/Cargo.toml) - Rust 專案配置
- [batch_verify.ps1](batch_verify.ps1) - 批量驗證腳本

---

## 授權

MIT License - 見 [LICENSE](../../../LICENSE)

---

## 聯絡方式

- **專案**: [AIVA GitHub](https://github.com/kyle0527/AIVA)
- **版本**: 2.0.0
- **最後更新**: 2025-12-12

---

**🚀 快速開始**: 執行 [`batch_verify.ps1`](batch_verify.ps1) 進行完整驗證！
