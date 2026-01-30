# 外部模組多語言分類報告

生成時間: 2026-01-23 15:18:12

---

## 總體統計

- **總模組數**: 11
- **總流程數**: 628

## 可操作性分析

> 基於原則.md 的5大判斷原則（邊界、序列化、拓撲學、命名慣例、框架約定）

- ✅ **可操作流程**: 363 (57.8%)
- ❌ **不可操作流程**: 265 (42.2%)

### 按語言分類

| 語言 | 可操作 | 不可操作 | 可操作率 |
|------|--------|----------|----------|
| Go | 13 | 0 | 100.0% |
| Python | 342 | 265 | 56.3% |
| Rust | 4 | 0 | 100.0% |
| TypeScript | 4 | 0 | 100.0% |

## 模組列表

| 模組名稱 | 語言 | 類型 | 流程數 |
|---------|------|------|--------|
| function_xss | Python | injection | 174 |
| function_sqli | Python | injection | 146 |
| function_ssrf | Python | ssrf | 76 |
| function_web_scanner | Python | unknown | 74 |
| function_bizlogic | Python | business_logic | 53 |
| function_postex | Python | unknown | 51 |
| function_idor | Python | access_control | 33 |
| function_authn_go | Go | authentication | 13 |
| function_crypto | Rust | cryptographic | 4 |
| typescript_engine | TypeScript | language_engine | 4 |
| function_info_leak | Python | unknown | 0 |

