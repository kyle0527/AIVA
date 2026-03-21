# AIVA 專案策略評估報告

> 建立日期：2026-03-21
> 來源：完整專案分析（整合模組 + 功能模組 + AI 核心模組）
> 健康度評分：6.5 / 10

---

## 一、架構核心原則（已確認）

```
AI Commander → subprocess + CLI + JSON → 功能模組
```

- **不需要** MQ Worker（設計決策，已文件確認）
- 執行入口：`aiva_external_executor.py`（多語言統一執行器）
- 非同步由 `asyncio.to_thread()` 處理
- 2026-01-12 已驗證移除中間層的決策正確

---

## 二、繼續發展（Keep & Invest）

### 高優先 — 核心 AI

| 模組 | 狀態 | 理由 |
|------|------|------|
| `core/aiva_core/cognitive_core/` | ✅ 完整 | 神經網路 + RAG + 決策引擎，系統大腦 |
| `internal_loop_connector.py`（2,068 行） | ✅ v11.1 成熟 | RAG 知識注入架構完整 |
| `external_loop_connector.py` | ✅ 邏輯完整 | 學習回饋迴路，長期競爭力 |
| `task_planning/attack_coordinator.py`（1,532 行） | ✅ 已整合 | 攻擊編排核心，主系統調用 |

### 高優先 — 掃描引擎

| 模組 | 狀態 | 理由 |
|------|------|------|
| `scan/go_scanners/`（CSPM/SCA/secrets/vulndb） | ✅ 完整 | Go 原生 4 個掃描器 |
| `scan/info_gatherer_rust/` | ✅ 完整 | Rust 偵察引擎 |
| `scan/aiva_scan_node/` | ✅ 完整 | TypeScript + Playwright 動態掃描 |
| `scan/coordinators/multi_engine_coordinator.py` | ✅ 驗證有效 | 多引擎協調器 |

### 高優先 — 已完成功能模組

| 模組 | 完成度 | 可用操作數 |
|------|:------:|:----------:|
| `function_sqli` | ✅ 100% | 76（6 種引擎） |
| `function_xss` | ✅ 100% | 45（3 種模式） |
| `function_ssrf` | ✅ 100% | 26（含 DNS rebinding） |
| `function_idor` | ✅ 100% | 18（水平 + 垂直提權） |
| `function_info_leak` | ✅ 100% | 10（50+ 種資訊類型） |

### 中優先 — 補完缺口

| 模組 | 需要做什麼 | 可用操作數（現有） |
|------|----------|:-----------------:|
| `function_postex` | 補 CLI 入口，接通 executor | 26 |
| `function_web_scanner` | 修復 4 個 NotImplementedError | 27 |
| `function_exploit` | 補 CLI 入口 | 16 |
| `function_bizlogic` | 補 CLI 入口 | 22 |
| `function_forensic` | 補 CLI 入口 | 9 |
| `function_crypto` | Python binding 接通 Rust binary | ~4 |
| `function_authn_go` | 完成 Go 主體邏輯 | ~3 |

---

## 三、建議廢棄（Deprecate / Remove）

| 項目 | 理由 | 處置方式 |
|------|------|----------|
| `integration/coordinators/`（BaseCoordinator, XSSCoordinator） | README 明確：「已實現但從未被 AI 調用」 | 直接刪除 |
| `function_*/legacy/` 5 個資料夾 | 孤立舊碼，無測試，增加混亂 | 直接刪除 |
| `function_steganography/` | 與核心滲透測試不相關 | 降低優先度 / 移除 |
| `function_wordlist_generator/` | 外部工具可取代（cewl/crunch） | 廢棄，改用 subprocess 呼叫外部工具 |
| `function_social_engineering/` | 邊界不清，框架未完成 | 重新評估必要性 |
| `function_reverse_engineering/` | 複雜度極高，短期不可達可用狀態 | 暫停開發 |
| `_archive/` 15 個歷史檔案 | 歷史記錄 | git tag 後刪除 |

---

## 四、建議新增（Add / Build Next）

### 1. 在 Linux 上重新執行分類器（最高優先）
```bash
python services/core/aiva_core/internal_exploration/aiva_external_classifier.py
```
重新產生 `external_classification.json`，讓路徑變成 Linux 格式。

### 2. 補全 CAPABILITY_CONFIGS 對應（高優先）
參見：`docs/capability_mapping_todo.md`

### 3. 統一 CLI 入口驗證
每個功能模組補充標準 CLI 參數介面，讓 `aiva_external_executor.py` 呼叫更可靠。

### 4. RAG 知識庫內容擴充
架構完整但知識庫內容稀薄：
- 更多 CVE / MITRE ATT&CK 向量化資料
- 歷史攻擊路徑的結構化學習資料

### 5. Dashboard 強化（`services/dashboard/`）
- 即時掃描進度可視化
- RAG 知識庫查詢介面
- 攻擊路徑圖（NetworkX → 視覺化）

---

## 五、優先順序總覽

```
立即（這週）──── 修復 2 個致命 Bug（見 critical_bugs.md）
             ──── 重新跑 external_classifier.py 產生 Linux 路徑 JSON
本月         ──── 補全 CAPABILITY_CONFIGS（capability_mapping_todo.md）
             ──── 接通 postex / web_scanner / exploit / bizlogic CLI 入口
             ──── 清除 legacy/ 資料夾 + 廢棄 coordinators/
下季         ──── RAG 知識庫擴充
             ──── Dashboard 強化
             ──── function_crypto / function_authn_go 完成
廢棄         ──── steganography / social_engineering / wordlist / reverse_eng
```

---

## 六、能力數量摘要

| 類別 | 枚舉總數 | 有實際對應模組 | 缺少對應 |
|------|:--------:|:--------------:|:--------:|
| Attack | 40 | 7 | 33 |
| Scan | 19 | 6 | 13 |
| Recon | 16 | 0 | 16 |
| Analysis | 14 | 2 | 12 |
| Forensic | 12 | 2 | 10 |
| Exploit | 11 | 4 | 7 |
| Report | 5 | 0 | 5 |
| **合計** | **117** | **~22** | **~95** |

功能模組實際可呼叫操作：**~180 個**（AST 靜態掃描確認）
`external_classification.json` 記錄的 flows：**525 個**（287 個 `is_operable=True`）
