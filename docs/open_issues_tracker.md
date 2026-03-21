# AIVA 未解決事項追蹤

> 建立日期：2026-03-21
> 涵蓋範圍：本次分析會話（整合模組分析 → 策略評估 → AI 模組操作問題）
> 狀態：⬜ 待處理 / 🔄 進行中 / ✅ 完成 / ❌ 廢棄

---

## 第一類：致命 Bug（必須修復才能正常運作）

| ID | 狀態 | 問題 | 位置 | 詳細 |
|----|:----:|------|------|------|
| BUG-001 | ⬜ | `operable` 應為 `is_operable`，525 flows 全部失效 | `aiva_external_executor.py` | `critical_bugs_verified.md#BUG-001` |
| BUG-002 | ⬜ | `external_classification.json` 含 521 個 Windows 路徑，Linux 無法執行 | `data/internal_exploration/external_classification.json` | `critical_bugs_verified.md#BUG-002` |
| BUG-003 | ⬜ | `CAPABILITY_CONFIGS` 全部 14 筆 `module/class/entry = None` | `aiva_common/enums/capabilities.py` | `capability_mapping_todo.md` |
| BUG-004 | ⬜ | core `capability_registry` 本地快取永遠空，sync 從未自動呼叫 | `core_capabilities/capability_registry.py` | `critical_bugs_verified.md#BUG-004` |

---

## 第二類：高嚴重度問題（影響主要功能）

| ID | 狀態 | 問題 | 位置 |
|----|:----:|------|------|
| BUG-005 | ⬜ | `attack_coordinator` unified_executor 為 None 時 fallback 缺少主要功能 | `attack_coordinator.py` |
| BUG-006 | ⬜ | RAG 為 None 時靜默回傳空結果，錯誤不傳播 | `internal_loop_connector.py` |
| BUG-007 | ⬜ | 訓練結果（新權重）不持久化，重啟後消失 | `external_loop_connector.py` |

---

## 第三類：設計問題（不影響立即運作，但長期有害）

| ID | 狀態 | 問題 | 影響範圍 |
|----|:----:|------|---------|
| D-001 | ⬜ | 能力登錄有兩套（core proxy vs integration），sync 需手動觸發 | 全域能力查詢 |
| D-002 | ⬜ | external / internal classification JSON 產生時機不同，可能不一致 | 能力發現 |
| D-003 | ⬜ | `web_tools.py` 有 4 個 `NotImplementedError` | `function_web_scanner` |
| D-004 | ⬜ | 全系統「靜默成功」模式，錯誤被 logger 吞掉後繼續執行 | 全域 |
| D-005 | ⬜ | `external_loop_connector` 訓練觸發條件為魔術數字，無調整機制 | 學習迴路 |

---

## 第四類：能力補全工作（CAPABILITY_CONFIGS 對應）

參見 `docs/capability_mapping_todo.md` 完整清單，以下為摘要：

### 高優先（有現成模組，直接填入）

| ID | 狀態 | 工作 |
|----|:----:|------|
| CAP-001 | ⬜ | `function_idor` → `idor` enum 補全 |
| CAP-002 | ⬜ | `function_postex` → `privilege_escalation_*`, `lateral_movement`, `persistence_install` |
| CAP-003 | ⬜ | `function_web_scanner` → port scan / tech detection / subdomain / dir / crawler |
| CAP-004 | ⬜ | `function_info_leak` → `secret_detection` |
| CAP-005 | ⬜ | `function_bizlogic` → 新增 `race_condition`, `price_manipulation`, `workflow_bypass` enum |
| CAP-006 | ⬜ | `function_exploit` → 新增 `exploit_execute`, `payload_generate` enum |
| CAP-007 | ⬜ | `function_forensic` → `memory_analysis`, `disk_image`, `timeline_analysis` |

### 中優先（模組需確認介面）

| ID | 狀態 | 工作 |
|----|:----:|------|
| CAP-008 | ⬜ | `function_authn_go` → 完成 Go 主體後接入 |
| CAP-009 | ⬜ | `function_crypto` → 接通 Rust `crypto-scanner` binary |

### 低優先（需新增工具模組）

| ID | 狀態 | 工作 |
|----|:----:|------|
| CAP-010 | ⬜ | `whois_lookup`, `dns_lookup` → 整合 `dnspython` / `python-whois` |
| CAP-011 | ⬜ | Recon 類 16 個 enum 全部無對應模組（需新建或整合外部工具） |
| CAP-012 | ⬜ | Report 類 5 個 enum 無對應模組 |

---

## 第五類：廢棄清理工作

| ID | 狀態 | 工作 | 理由 |
|----|:----:|------|------|
| CLN-001 | ⬜ | 刪除 `integration/coordinators/`（BaseCoordinator, XSSCoordinator） | 從未被 AI 調用 |
| CLN-002 | ⬜ | 刪除 `function_sqli/legacy/` | 孤立舊碼 |
| CLN-003 | ⬜ | 刪除 `function_xss/legacy/`（如存在） | 孤立舊碼 |
| CLN-004 | ⬜ | 刪除 `function_postex/legacy/` | 孤立舊碼 |
| CLN-005 | ⬜ | 刪除 `function_forensic/legacy/` | 孤立舊碼 |
| CLN-006 | ⬜ | 刪除 `function_steganography/legacy/` | 孤立舊碼 |
| CLN-007 | ⬜ | 刪除 `function_social_engineering/legacy/` | 孤立舊碼 |
| CLN-008 | ⬜ | 刪除 `function_wordlist_generator/legacy/` | 孤立舊碼 |
| CLN-009 | ⬜ | 刪除 `function_reverse_engineering/legacy/` | 孤立舊碼 |
| CLN-010 | ⬜ | git tag 後刪除 `_archive/` 15 個歷史檔案 | 無用歷史記錄 |

---

## 第六類：功能模組開發（補完現有半成品）

| ID | 狀態 | 模組 | 缺少什麼 |
|----|:----:|------|---------|
| DEV-001 | ⬜ | `function_postex` | CLI 入口接通 `aiva_external_executor.py` |
| DEV-002 | ⬜ | `function_web_scanner` | 修復 4 個 NotImplementedError + CLI 入口 |
| DEV-003 | ⬜ | `function_exploit` | CLI 入口接通執行器 |
| DEV-004 | ⬜ | `function_bizlogic` | CLI 入口接通執行器 |
| DEV-005 | ⬜ | `function_forensic` | CLI 入口接通執行器 |
| DEV-006 | ⬜ | `function_crypto` | Python binding 接通 Rust binary |
| DEV-007 | ⬜ | `function_authn_go` | 完成 Go 主體邏輯 |

---

## 第七類：長期架構改善

| ID | 狀態 | 工作 |
|----|:----:|------|
| ARCH-001 | ⬜ | RAG 知識庫擴充（CVE / MITRE ATT&CK 向量化資料） |
| ARCH-002 | ⬜ | Dashboard 強化（掃描進度 / RAG 查詢介面 / 攻擊路徑圖） |
| ARCH-003 | ⬜ | 建立端對端 CLI 測試（每個模組驗證 JSON 輸出格式） |
| ARCH-004 | ⬜ | 統一錯誤傳播機制（取代靜默成功模式） |
| ARCH-005 | ⬜ | 訓練權重持久化機制 |
| ARCH-006 | ⬜ | 能力版本管理（舊能力更新時不破壞相容性） |

---

## 本次分析產出的文件

| 文件 | 路徑 | 用途 |
|------|------|------|
| 策略評估報告 | `docs/project_strategic_assessment.md` | 什麼要繼續 / 廢棄 / 新增 |
| 已驗證 Bug 清單 | `docs/critical_bugs_verified.md` | 具體 Bug + 修復方法 |
| 能力補全待辦 | `docs/capability_mapping_todo.md` | CAPABILITY_CONFIGS 補全範本 |
| 未解決事項追蹤 | `docs/open_issues_tracker.md` | 本文件 |

---

## 建議下一步行動（本地執行）

```bash
# Step 1：修復 BUG-001（10 分鐘）
grep -n '"operable"' services/core/aiva_core/internal_exploration/aiva_external_executor.py
# 將所有 f.get("operable") 改成 f.get("is_operable", False)

# Step 2：重新產生分類 JSON（BUG-002）
cd /path/to/AIVA
python services/core/aiva_core/internal_exploration/aiva_external_classifier.py

# Step 3：補全 CAPABILITY_CONFIGS（BUG-003）
# 參見 docs/capability_mapping_todo.md 中的範本，貼入：
# services/aiva_common/enums/capabilities.py

# Step 4：在啟動時加入 sync（BUG-004）
# 在 services/core/app.py 初始化階段加入：
# await capability_registry.sync_from_integration_registry()
```
