# 外部模組整合分類報告

生成時間: 2026-01-13 23:04:16

---

## 總體統計

- **總模組數**: 10
- **總流程數**: 222
- **支援語言**: Unknown, Rust, Go, Typescript

---

## 發現的模組

| 模組名稱 | 類別 | 描述 | 攻擊類型 | 流程數 | 語言 |
|---------|------|------|---------|--------|------|
| function_xss | features | XSS 漏洞檢測 | injection | 109 | Unknown |
| function_ssrf | features | SSRF 漏洞檢測 | ssrf | 35 | Unknown |
| function_sqli | features | SQL 注入檢測 | injection | 32 | Unknown |
| function_idor | features | IDOR 漏洞檢測 | access_control | 19 | Unknown |
| function_bizlogic | features | 業務邏輯漏洞 | business_logic | 8 | Unknown |
| function_authn_go | features | 身份驗證 | authentication | 4 | Go |
| function_crypto | features | 加密相關漏洞 | cryptographic | 4 | Rust |
| rust_engine | scan | Rust 分析引擎 | language_engine | 4 | Rust |
| typescript_engine | scan | TypeScript 分析引擎 | language_engine | 3 | Typescript |
| scan_engine | scan | 掃描引擎 | scanner | 0 |  |

---

## 模組類別分布

| 類別 | 模組數 | 流程數 | 百分比 |
|------|--------|--------|--------|
| features | 7 | 215 | 96.8% |
| scan | 3 | 7 | 3.2% |

---

## 攻擊類型分布

| 攻擊類型 | 流程數 | 百分比 | 相關模組 |
|---------|--------|--------|----------|
| injection | 141 | 63.5% | function_sqli, function_xss |
| ssrf | 35 | 15.8% | function_ssrf |
| access_control | 19 | 8.6% | function_idor |
| authentication | 8 | 3.6% | function_authn_go |
| business_logic | 8 | 3.6% | function_bizlogic |
| language_engine | 7 | 3.2% | typescript_engine, rust_engine |
| cryptographic | 4 | 1.8% | function_crypto |

---

## 語言分布

| 語言 | 流程數 | 百分比 | 模組數 |
|------|--------|--------|--------|
| Unknown | 207 | 93.2% | 5 |
| Rust | 8 | 3.6% | 2 |
| Go | 4 | 1.8% | 1 |
| Typescript | 3 | 1.4% | 1 |
