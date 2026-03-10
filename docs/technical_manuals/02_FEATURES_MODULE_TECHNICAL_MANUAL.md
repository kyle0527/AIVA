# AIVA Features 模組技術手冊

**版本**: v7.2
**狀態**: Architecture Integration Complete
**路徑**: `services/features/`

---

## 1. 模組概述

Features 模組是 AIVA 的多語言安全檢測引擎層，包含 17 個獨立安全測試功能模組。每個子模組可獨立呼叫，也可由 Core 模組編排並行執行。

**設計原則**："Function determines architecture"（功能決定架構）——每個模組採用最適合的內部結構，不強制統一。

---

## 2. 模組完成度一覽

### 高完成度模組（可直接整合）

| 模組 | 完成度 | 功能 | 技術特性 |
|---|---|---|---|
| `function_sqli` | 95% | SQL 注入檢測 | 6 個引擎，400+ 指紋 |
| `function_xss` | 90% | XSS 檢測 | 4 種偵測器，外部工具整合 |
| `function_ssrf` | 85% | SSRF 檢測 | OAST 技術 |
| `function_idor` | 80% | IDOR 檢測 | 權限矩陣分析 |
| `function_bizlogic` | 70% | 業務邏輯測試 | — |
| `function_crypto` | 50% | 密碼學分析 | Rust CLI |
| `function_info_leak` | 100% | 敏感資訊偵測 | API Keys, JWT |

### 中等完成度模組（開發中）

| 模組 | 完成度 | 功能 |
|---|---|---|
| `function_authn_go` | 50% | Go 認證測試 |
| `function_postex` | 50% | 後滲透模組 |
| `function_web_scanner` | 35% | Web 漏洞掃描 |

---

## 3. 架構設計

### 3.1 核心執行檔案

| 檔案 | 功能 |
|---|---|
| `feature_step_executor.py` | 步驟執行器（11.7KB），協調模組呼叫 |
| `__init__.py` | 模組匯出（7.1KB） |

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
| Python | 主要邏輯 | SQLi, XSS, SSRF, IDOR |
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
- **繞過技術**：WAF bypass payloads 庫

### 4.3 function_ssrf（服務端請求偽造）

- **OAST 技術**：Out-of-band application security testing
- **協定覆蓋**：http, https, ftp, file, gopher, dict
- **雲端服務偵測**：AWS metadata, GCP, Azure 端點

### 4.4 function_info_leak（敏感資訊）

- **完成度最高**：100%
- **偵測項目**：
  - API Keys（AWS, GitHub, Stripe, Slack 等）
  - JWT Token 解析與安全性驗證
  - 密碼/私鑰正則匹配
  - 敏感路徑暴露

---

## 5. 與其他模組的整合

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
    └── ... (其他子模組)
    │
    ▼
結果 → integration/ (收集、分析、報告)
```

**與 aiva_common 整合**：
- 使用 `aiva_common.enums.Severity`
- 使用 `aiva_common.enums.Confidence`
- 使用 `aiva_common.schemas` 資料合約

---

## 6. CLI 執行範例

```bash
# 直接執行單一功能模組
python -m services.features.function_sqli --url https://target.com/api?id=1

# 透過 Core 編排並行執行
python -m services.core --target https://target.com --modules sqli,xss,ssrf
```

---

## 7. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第4冊_功能模組操作.md`
- **技術手冊**：`docs/technical_manuals/01_CORE_MODULE_TECHNICAL_MANUAL.md`（呼叫方）
- **技術手冊**：`docs/technical_manuals/05_AIVA_COMMON_TECHNICAL_MANUAL.md`（共用 schemas）
