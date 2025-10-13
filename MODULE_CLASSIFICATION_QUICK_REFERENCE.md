# AIVA 十二個建議模組快速分類表

## 📊 模組歸屬四大分類

| # | 模組名稱 | 歸屬模組 | 優先級 | 可行性 | 語言 | 實施方式 |
|---|---------|---------|--------|--------|------|---------|
| 1 | **Function-SCA**<br>軟體組成分析 | **Function** | **P0** | ⭐⭐⭐⭐⭐ | Go | 新建 `function_sca_go/` |
| 2 | **Function-SAST**<br>靜態應用程式安全測試 | **Scan** | **P1** | ⭐⭐⭐⭐ | Rust | 新建 `static_analyzer_rust/` |
| 3 | **Function-CSPM**<br>雲端安全態勢管理 | **Function** | **P1** | ⭐⭐⭐⭐ | Go | 新建 `function_cspm_go/` |
| 4 | **Module-AttackPath**<br>攻擊路徑分析 | **Integration** | **P0** | ⭐⭐⭐⭐⭐ | Python | `integration/attack_path_analyzer/` |
| 5 | **Module-AuthN**<br>認證安全測試 | **Function** | **P1** | ⭐⭐⭐⭐ | Go | 新建 `function_authn_go/` |
| 6 | **Module-APISec**<br>API 安全攻擊 | **Function** | **P0** | ⭐⭐⭐⭐⭐ | Python | 擴展 `function_idor/` |
| 7 | **Module-Secrets**<br>憑證洩漏掃描 | **Scan** | **P0** | ⭐⭐⭐⭐⭐ | Rust | 擴展 `info_gatherer_rust/` |
| 8 | **Module-PostEx**<br>漏洞利用與後滲透 | **Function** | **P2** | ⭐⭐⭐ | Python | 新建 `function_postex/` ⚠️ |
| 9 | **Module-ThreatIntel**<br>威脅情資整合 | **Core** | **P1** | ⭐⭐⭐⭐⭐ | Python | `core/analysis/threat_intel/` |
| 10 | **Module-Remediation**<br>自動化修復 | **Integration** | **P2** | ⭐⭐⭐ | Python | `integration/remediation/` 🤖 |
| 11 | **Module-BizLogic**<br>業務邏輯濫用測試 | **Function** | **P3** | ⭐⭐ | Python | 新建 `function_bizlogic/` 🤖 |
| 12 | **Module-AuthZ**<br>授權模型繪製 | **Function** | **P1** | ⭐⭐⭐⭐ | Python | 擴展 `function_idor/` |

## 🎯 按四大模組分類

### Core 模組（1 個）
- ✅ **Module-ThreatIntel** (P1) - 威脅情資整合

### Scan 模組（2 個）
- ✅ **Function-SAST** (P1) - 靜態原始碼分析
- ✅ **Module-Secrets** (P0) - 憑證洩漏掃描

### Function 模組（7 個）
- ✅ **Function-SCA** (P0) - 軟體組成分析
- ✅ **Function-CSPM** (P1) - 雲端安全態勢管理
- ✅ **Module-AuthN** (P1) - 認證安全測試
- ✅ **Module-APISec** (P0) - API 安全攻擊
- ⚠️ **Module-PostEx** (P2) - 漏洞利用（需嚴格控制）
- ⚠️ **Module-BizLogic** (P3) - 業務邏輯（需 AI）
- ✅ **Module-AuthZ** (P1) - 授權模型繪製

### Integration 模組（2 個）
- ✅ **Module-AttackPath** (P0) - 攻擊路徑分析
- ⚠️ **Module-Remediation** (P2) - 自動化修復（需 LLM）

## 📈 語言分布

| 語言 | 模組數量 | 模組列表 |
|------|---------|---------|
| **Python** | 6 | APISec, AttackPath, PostEx, ThreatIntel, Remediation, BizLogic, AuthZ |
| **Go** | 3 | SCA, CSPM, AuthN |
| **Rust** | 2 | SAST, Secrets |

## 🚀 實施優先級時程

### P0 級（3 個月）- 4 個模組
1. ✅ **Module-APISec** (2週) - 擴展現有 IDOR
2. ✅ **Function-SCA** (4週) - Go + OSV-Scanner
3. ✅ **Module-Secrets** (3週) - Rust + Git 歷史掃描
4. ✅ **Module-AttackPath** (6週) - Python + Neo4j

**預期成果**：
- OWASP Top 10 覆蓋率：40% → 80%
- 漏洞類型：4 種 → 10+ 種
- 新增攻擊路徑視覺化能力

### P1 級（6 個月）- 5 個模組
5. ✅ **Function-SAST** (8週)
6. ✅ **Function-CSPM** (4週)
7. ✅ **Module-AuthN** (4週)
8. ✅ **Module-ThreatIntel** (3週)
9. ✅ **Module-AuthZ** (3週)

**預期成果**：
- 漏洞類型：10+ 種 → 15+ 種
- 新增原始碼分析能力
- 新增 IaC 掃描能力

### P2 級（1 年）- 2 個模組
10. ⚠️ **Module-PostEx** (8週) - 需嚴格權限控制
11. ⚠️ **Module-Remediation** (12週) - 需 LLM 整合

### P3 級（暫緩）- 1 個模組
12. ❌ **Module-BizLogic** - 技術不成熟

## ⚠️ 特別注意事項

### 需嚴格控制的模組
- **Module-PostEx**：可能造成實際損害，需要：
  - 預設關閉
  - 審計日誌
  - 僅限沙盒環境

### 需 AI/LLM 整合的模組
- **Module-Remediation**：需 OpenAI API 或 Claude API
- **Module-BizLogic**：極度依賴 AI 理解業務邏輯

## ✅ 可立即開始實施的模組（基於現有程式碼）

### 1. Module-APISec（最容易）
**現有基礎**：
- `services/function/function_idor/aiva_func_idor/cross_user_tester.py`
- `services/function/function_idor/aiva_func_idor/vertical_escalation_tester.py`

**新增內容**：
```python
# 新增檔案：bfla_tester.py（函式級授權測試）
# 新增檔案：mass_assignment_tester.py（巨量賦值測試）
```

### 2. Module-Secrets（現有 Rust 掃描器擴展）
**現有基礎**：
- `services/scan/info_gatherer_rust/src/scanner.rs`（已有 regex 掃描）

**新增內容**：
```rust
// 新增模組：git_history_scanner.rs
// 新增模組：entropy_detector.rs
```

### 3. Function-SCA（新建但技術成熟）
**整合現有工具**：
- Google OSV-Scanner（開源）
- Trivy（開源）

**實施難度**：低（主要是 API 整合）

### 4. Module-AttackPath（現有 Neo4j 基礎）
**現有基礎**：
- `docker-compose.yml` 已包含 Neo4j
- `services/core/aiva_core/analysis/strategy_generator.py` 有 VulnerabilityCorrelationAnalyzer

**新增內容**：
```python
# 新增檔案：attack_path_engine.py
# 新增檔案：graph_builder.py
# 新增檔案：path_ranker.py
```

## 📝 下一步行動

1. ✅ 創建詳細的 P0 模組實施計劃
2. ✅ 設計各模組的數據合約擴展（更新 `DATA_CONTRACT.md`）
3. ⏳ 準備開發環境（Go 模組初始化、Rust 依賴）
4. ⏳ 建立模組開發模板
5. ⏳ 開始實施 Module-APISec（最小成本、最快見效）
