# AIVA Features 模組技術手冊

**版本**: v7.2 | **狀態**: ✅ Architecture Complete | **路徑**: `services/features/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [子模組完成度一覽](#2-子模組完成度一覽)
3. [架構設計](#3-架構設計)
   - 3.1 [核心執行檔案](#31-核心執行檔案)
   - 3.2 [資料輸出格式](#32-資料輸出格式)
   - 3.3 [多語言引擎分工](#33-多語言引擎分工)
4. [關鍵子模組技術細節](#4-關鍵子模組技術細節)
5. [完成狀態](#5-完成狀態)
   - 5.1 [已完成功能](#51-已完成功能-)
   - 5.2 [待完成 / 目標功能](#52-待完成--目標功能-)
6. [與其他模組的整合](#6-與其他模組的整合)
7. [CLI 執行範例](#7-cli-執行範例)
8. [搭配閱讀](#8-搭配閱讀)

---

## 1. 模組概述

Features 模組是 AIVA 的多語言安全檢測引擎層，包含 17 個獨立安全測試功能模組。每個子模組可獨立呼叫，也可由 Core 模組編排並行執行。

**設計原則**："Function determines architecture"（功能決定架構）——每個模組採用最適合的內部結構，不強制統一。

**整體統計**：628 個流程已分類可呼叫，支援 Python / Rust / Go / TypeScript 四種語言。

---

## 2. 子模組完成度一覽

### 高完成度（可直接整合）

| 模組 | 完成度 | 版本 | 功能 |
|---|---|---|---|
| `function_info_leak` | **100%** | v1.0.0 | API Keys, JWT, 憑證偵測 |
| `function_sqli` | 95% | v2.1.0 | SQL 注入（6 引擎，400+ 指紋）|
| `function_xss` | 90% | v2.1.0 | XSS（4 種偵測器 + 外部工具）|
| `function_ssrf` | 85% | v2.0.0 | SSRF（OAST 技術）|
| `function_idor` | 80% | v1.0.0 | IDOR（權限矩陣分析）|
| `function_bizlogic` | 70% | v1.1.0 | 業務邏輯（價格操縱、競態條件）|
| `function_crypto` | 50% | v2.0.0 | 密碼學分析（Rust CLI，需編譯）|

### 中等完成度（開發中）

| 模組 | 完成度 | 說明 |
|---|---|---|
| `function_authn_go` | 50% | Go 引擎，需編譯與測試 |
| `function_postex` | 50% | 架構完成，偵測邏輯需加強 |
| `function_web_scanner` | 35% | 架構完成，缺少 README 與邏輯 |

### 歸檔（低完成度）

| 模組 | 說明 |
|---|---|
| `function_exploit_framework` | 25%，僅為 PoC 工具，非主要模組 |
| `function_payload_generator` | 輔助工具 |

### 需手動操作（無法自動化）

Social Engineering、Forensic、Reverse Engineering、Steganography、Wordlist Generator（共 5 個）

---

## 3. 架構設計

### 3.1 核心執行檔案

| 檔案 | 大小 | 功能 |
|---|---|---|
| `feature_step_executor.py` | 11.7KB | 步驟執行器，協調模組呼叫 |
| `__init__.py` | 7.1KB | 模組匯出 |

### 3.2 資料輸出格式

所有 Features 子模組遵循標準化輸出格式（OWASP/SARIF/CVE 標準）：

```json
{
  "module": "function_sqli",
  "vulnerability_type": "SQL_INJECTION",
  "severity": "HIGH",
  "confidence": 0.92,
  "cve_id": "CVE-XXXX-XXXX",
  "cvss_score": 8.5,
  "evidence": { ... },
  "payload_used": "...",
  "remediation": "..."
}
```

### 3.3 多語言引擎分工

| 語言 | 使用模組 | 適用場景 |
|---|---|---|
| Python | 主要邏輯 | SQLi, XSS, SSRF, IDOR, Info Leak |
| Go | `function_authn_go` | 高效能認證測試 |
| Rust | `function_crypto` | 密碼學分析，高效能運算 |
| TypeScript | 部分 XSS | DOM-based 漏洞 |

---

## 4. 關鍵子模組技術細節

### 4.1 function_sqli（SQL 注入）

- **6 個引擎**：time-based, error-based, boolean-based, union-based, OOB, stacked
- **指紋庫**：400+ SQLi 指紋
- **資料庫覆蓋**：MySQL, PostgreSQL, MSSQL, Oracle, SQLite

### 4.2 function_xss（跨站腳本）

- **4 種偵測器**：Reflected, Stored, DOM-based, mXSS
- **外部工具整合**：dalfox, kxss
- **繞過技術**：WAF bypass payload 庫

### 4.3 function_ssrf（服務端請求偽造）

- **OAST 技術**：Out-of-band application security testing
- **協定覆蓋**：http, https, ftp, file, gopher, dict
- **雲端服務偵測**：AWS metadata, GCP, Azure 端點

### 4.4 function_info_leak（敏感資訊，完成度 100%）

- **API Keys**：AWS, GitHub, Stripe, Slack 等主流服務
- **JWT Token**：解析與安全性驗證（alg:none, weak secret 等）
- **密碼/私鑰**：正則匹配庫
- **敏感路徑暴露**：.env, .git, backup 檔案

### 4.5 function_bizlogic（業務邏輯）

- **價格操縱**：負數金額、整數溢位
- **競態條件**：並發請求漏洞
- **流程繞過**：跳過驗證步驟

---

## 5. 完成狀態

### 5.1 已完成功能 ✅

| 功能 | 說明 |
|---|---|
| 統一執行器（Unified Executor） | 628 個流程已分類可呼叫 |
| function_info_leak | 100% 完成，生產就緒 |
| function_sqli v2.1.0 | 6 引擎，全整合 |
| function_xss v2.1.0 | 多偵測器 + 外部工具 |
| function_ssrf v2.0.0 | OAST + 雲端端點 |
| function_idor v1.0.0 | 權限矩陣分析 |
| function_bizlogic v1.1.0 | 價格操縱、競態條件 |
| CLI 整合 | 4 語言完整支援 |

### 5.2 待完成 / 目標功能 🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| function_crypto 編譯整合 | P1 | Rust binary 編譯，CI/CD 整合 |
| function_authn_go 完成 | P1 | Go binary 編譯 + 認證測試邏輯完成 |
| function_postex 偵測邏輯 | P1 | 後滲透偵測算法強化 |
| function_web_scanner | P2 | 補齊 README、完成核心邏輯（目前 35%）|
| XXE 功能模組 | P2 | 新增 XML External Entity 注入模組 |
| File Upload 功能模組 | P2 | 惡意檔案上傳漏洞偵測 |
| 攻擊鏈組合（Chain Attack）| P2 | SQLi → 取得憑證 → IDOR 等鏈式攻擊 |
| Social Engineering 自動化 | P3 | 部分自動化釣魚/社交工程測試 |
| function_exploit_framework | P3 | PoC 框架完整化（目前 25%）|
| 統一漏洞資料庫整合 | P3 | 連接 NVD/CVE API 即時更新漏洞指紋 |

---

## 6. 與其他模組的整合

```
Core Module
    │ 呼叫
    ▼
feature_step_executor.py
    │
    ├── function_sqli
    ├── function_xss
    ├── function_ssrf
    ├── function_idor
    ├── function_info_leak
    └── ... （其他子模組）
    │
    ▼
結果 → integration/（收集、分析、報告）
```

**與 aiva_common 整合**：
- 使用 `aiva_common.enums.Severity`
- 使用 `aiva_common.enums.Confidence`
- 使用 `aiva_common.schemas` 資料合約

---

## 7. CLI 執行範例

```bash
# 直接執行單一功能模組
python -m services.features.function_sqli --url https://target.com/api?id=1

# 透過 Core 編排並行執行
python -m services.core --target https://target.com --modules sqli,xss,ssrf

# Go 引擎（需先編譯）
./function_authn_go --url https://target.com/login

# Rust 引擎（需先編譯）
./function_crypto --target https://target.com --mode tls
```

---

## 8. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第4冊_功能模組操作.md`
- **技術手冊**：`docs/technical_manuals/01_CORE_MODULE_TECHNICAL_MANUAL.md`（呼叫方）
- **技術手冊**：`docs/technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md`（共用 schemas）
