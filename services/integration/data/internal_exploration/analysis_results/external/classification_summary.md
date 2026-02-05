# 外部模組多語言分類報告

生成時間: 2026-02-02 15:23:12

---

## 總體統計

- **總模組數**: 14
- **總流程數**: 525

## 可操作性分析

> 基於原則.md 的5大判斷原則（邊界、序列化、拓撲學、命名慣例、框架約定）

- ✅ **可操作流程**: 287 (54.7%)
- ❌ **不可操作流程**: 238 (45.3%)

### 按語言分類

| 語言 | 可操作 | 不可操作 | 可操作率 |
|------|--------|----------|----------|
| Go | 2 | 0 | 100.0% |
| Python | 283 | 238 | 54.3% |
| Rust | 2 | 0 | 100.0% |

## 模組列表

| 模組名稱 | 語言 | 類型 | 流程數 |
|---------|------|------|--------|
| function_sqli | Python | injection | 115 |
| function_xss | Python | injection | 97 |
| function_web_scanner | Python | unknown | 74 |
| function_ssrf | Python | ssrf | 64 |
| function_bizlogic | Python | business_logic | 53 |
| function_postex | Python | unknown | 49 |
| function_idor | Python | access_control | 25 |
| python_engine | Python | unknown | 24 |
| function_info_leak | Python | unknown | 20 |
| function_authn_go | Go | authentication | 1 |
| go_engine | Go | unknown | 1 |
| function_crypto | Rust | cryptographic | 1 |
| rust_engine | Rust | unknown | 1 |
| typescript_engine | TypeScript | language_engine | 0 |

