# function_crypto - 密碼學配置掃描器 (純 Rust CLI)

> **版本**: v3.0.0 | **狀態**: ✅ Rust 核心完成 | **語言**: Rust

## 🎯 模組概述

本模組為基於 Rust 實作的密碼學配置掃描器，透過編譯出的 CLI 執行檔 (`crypto-scanner`) 進行掃描。專注於網路層可觀察的密碼學配置，包含 TLS 分析、Cookie 安全、HTTP 安全標頭與 JS 內的密碼學配置問題。

> **設計理念**: 本模組沒有 Python 包裝層，由 AI Commander (或其他呼叫端) 直接使用 `subprocess` 執行 CLI，並自行解析標準輸出之 JSON 結構。

### 功能清單

| 功能 | 說明 |
|------|------|
| JavaScript 分析 | `scan-js` 檢查 JS 檔案中的寫死金鑰、不安全演算法與弱亂數。 |
| TLS 配置分析 | `analyze-tls` 分析目標伺服器的 TLS 憑證與協議配置。 |
| Cookie 安全分析 | `analyze-cookies` 分析 Session 與敏感 Cookie 的安全屬性。 |
| 安全標頭分析 | `analyze-headers` 分析 HSTS、CSP 等安全標頭。 |

## 📐 架構設計

```
function_crypto/
├── __init__.py           # Python 模組入口 (僅提供文件說明，無實體類別匯出)
├── rust_core/            # Rust 核心實作
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── tls_analyzer.rs
│       ├── cookie_analyzer.rs
│       ├── header_analyzer.rs
│       └── js_crypto_analyzer.rs
├── cli_commands.sh       # 開發時期輔助指令紀錄
└── clap_analysis.json    # CLI 解析輔助紀錄
```

## 🚀 執行方式

### 1. 編譯 Rust 專案

本模組必須先編譯 Rust 原始碼：

```bash
cd services/features/function_crypto/rust_core
cargo build --release
```

### 2. 透過 CLI 直接呼叫

編譯後會產生 `crypto-scanner` 執行檔，呼叫方式如下：

```bash
# JavaScript 分析
./rust_core/target/release/crypto-scanner scan-js --content "const key = '123456';" --url "https://target.com/app.js"

# TLS 配置分析
./rust_core/target/release/crypto-scanner analyze-tls --target example.com --port 443

# Cookie 安全分析
./rust_core/target/release/crypto-scanner analyze-cookies --cookies-json '{"session": "xyz"}' --url "https://target.com"

# 安全標頭分析
./rust_core/target/release/crypto-scanner analyze-headers --headers-json '{"Server": "nginx"}' --url "https://target.com"
```

### 3. Python 中呼叫 (範例)

模組內不提供 wrapper，必須自行使用 `subprocess` 呼叫：

```python
import subprocess
import json

cmd = [
    "services/features/function_crypto/rust_core/target/release/crypto-scanner",
    "analyze-tls",
    "--target", "example.com",
    "--port", "443"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    data = json.loads(result.stdout)
    print(data)
```

## 注意事項

- 不需使用 Python 呼叫或尋找 Python `CommandHandler`。
- 回傳一律為標準化 JSON。
- 專注黑盒網路層與前端原始碼掃描，並不包含針對後端原始碼的靜態白盒分析 (SAST)。
