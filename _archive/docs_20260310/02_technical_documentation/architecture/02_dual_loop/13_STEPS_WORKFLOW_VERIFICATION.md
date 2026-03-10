# AIVA 完整 13 步驟工作流程驗證文檔

**創建日期**: 2025年12月28日  
**版本**: v1.0  
**狀態**: 架構定義完成

---

## 📑 目錄

- [概述](#概述)
- [雙閉環架構](#雙閉環架構)
- [完整 13 步驟工作流程](#完整-13-步驟工作流程)
- [掃描模組的功用](#掃描模組的功用)
- [各步驟詳細說明](#各步驟詳細說明)
- [模組職責對照表](#模組職責對照表)
- [驗證檢查清單](#驗證檢查清單)

---

## 概述

AIVA 採用**雙閉環 (Dual Loop)** 架構實現 AI 自我優化。完整工作流程包含 **13 個步驟**，涵蓋從自我認知到實戰執行再到學習優化的完整循環。

### 核心設計原則

```
┌────────────────────────────────────────────────────────────────┐
│                    AIVA 雙閉環架構核心原則                        │
├────────────────────────────────────────────────────────────────┤
│  • aiva_core 下令操作，異步功能只在 AI 內部                       │
│  • 其他模組接到命令直接執行（同步/阻塞式）                         │
│  • 內閉環：AI 自我認知 → 知道有哪些能力、問題、解法               │
│  • 外閉環：使用內閉環發現的能力 → 實戰執行 → 學習優化             │
└────────────────────────────────────────────────────────────────┘
```

---

## 雙閉環架構

### 內閉環 (Internal Loop) - 自我認知

```
探索(對內) → 分析 → RAG 注入 → 了解自身能力
   │           │        │            │
   ▼           ▼        ▼            ▼
ModuleExplorer → CapabilityAnalyzer → InternalLoopConnector → RAG Knowledge Base
```

**目的**: 讓 AI 了解自己有哪些能力、可以做什麼、限制是什麼

**關鍵組件**:
- `services/core/aiva_core/internal_exploration/` - 內部探索模組
- `services/core/aiva_core/cognitive_core/internal_loop_connector.py` - 內閉環連接器
- `services/core/aiva_core/cognitive_core/rag/` - RAG 知識庫

### 外閉環 (External Loop) - 實戰執行

```
掃描(對外) → 攻擊 → 結果分析 → 經驗學習 → 收集優化方向
   │          │        │           │            │
   ▼          ▼        ▼           ▼            ▼
Scan模組  → Features → Integration → ExternalLoopConnector → 學習系統
```

**目的**: 使用已知能力進行實戰，收集經驗並優化策略

**關鍵組件**:
- `services/scan/` - 多語言掃描引擎
- `services/features/` - 功能檢測模組
- `services/integration/` - 結果整合
- `services/core/aiva_core/cognitive_core/external_loop_connector.py` - 外閉環連接器

---

## 完整 13 步驟工作流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           AIVA 13 步驟完整工作流程                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║              【Phase 0: 內閉環 - 自我認知】                       ║   │
│  ╠══════════════════════════════════════════════════════════════════╣   │
│  ║  步驟 1: 模組探索 (Module Exploration)                            ║   │
│  ║         - 掃描 services/ 下所有模組                               ║   │
│  ║         - 發現 Python/Go/Rust/TypeScript 能力                    ║   │
│  ║                                                                   ║   │
│  ║  步驟 2: 能力分析 (Capability Analysis)                           ║   │
│  ║         - 解析函數簽名、參數、返回值                              ║   │
│  ║         - 分類能力 (Scanning/Attacking/Analysis/Utility)          ║   │
│  ║                                                                   ║   │
│  ║  步驟 3: RAG 注入 (Knowledge Base Injection)                      ║   │
│  ║         - 將能力資訊注入向量數據庫                                 ║   │
│  ║         - 建立語義查詢索引                                        ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                    ↓                                     │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║              【Phase 1: 情報收集 - 目標分析】                     ║   │
│  ╠══════════════════════════════════════════════════════════════════╣   │
│  ║  步驟 4: 目標接收 (Target Reception)                              ║   │
│  ║         - 接收目標 URL/IP 列表                                   ║   │
│  ║         - 初步驗證目標可達性                                      ║   │
│  ║                                                                   ║   │
│  ║  步驟 5: 能力查詢 (Capability Query)                              ║   │
│  ║         - AI 查詢 RAG: "我有哪些掃描能力？"                       ║   │
│  ║         - 獲取可用掃描引擎列表                                    ║   │
│  ║                                                                   ║   │
│  ║  步驟 6: 策略規劃 (Strategy Planning)                             ║   │
│  ║         - 基於目標類型選擇掃描策略                                ║   │
│  ║         - 生成 AICommand 調度序列                                 ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                    ↓                                     │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║              【Phase 2: 外閉環 - 實戰執行】                       ║   │
│  ╠══════════════════════════════════════════════════════════════════╣   │
│  ║  步驟 7: 掃描執行 (Scan Execution)                                ║   │
│  ║         - 調度多語言掃描引擎 (Go/Rust/TS/Python)                  ║   │
│  ║         - 【掃描模組職責】: 發送請求到靶場，收集響應              ║   │
│  ║                                                                   ║   │
│  ║  步驟 8: 結果解析 (Result Parsing)                                ║   │
│  ║         - 解析各引擎返回的原始結果                                ║   │
│  ║         - 轉換為統一 UnifiedVulnerabilityFinding                  ║   │
│  ║                                                                   ║   │
│  ║  步驟 9: 漏洞確認 (Vulnerability Verification)                    ║   │
│  ║         - 對高置信度發現進行二次驗證                              ║   │
│  ║         - 調用 Features 模組進行深度檢測                          ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                    ↓                                     │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║              【Phase 3: 攻擊執行 - 漏洞利用】                     ║   │
│  ╠══════════════════════════════════════════════════════════════════╣   │
│  ║  步驟 10: 攻擊規劃 (Attack Planning)                              ║   │
│  ║          - AI 查詢: "這個漏洞怎麼利用？"                          ║   │
│  ║          - 選擇攻擊向量和 Payload                                 ║   │
│  ║                                                                   ║   │
│  ║  步驟 11: 攻擊執行 (Attack Execution)                             ║   │
│  ║          - 執行 SQLi/XSS/SSRF/IDOR 等攻擊                        ║   │
│  ║          - 記錄攻擊請求和響應                                     ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                    ↓                                     │
│  ╔══════════════════════════════════════════════════════════════════╗   │
│  ║              【Phase 4: 學習優化 - 經驗積累】                     ║   │
│  ╠══════════════════════════════════════════════════════════════════╣   │
│  ║  步驟 12: 偏差分析 (Deviation Analysis)                           ║   │
│  ║          - 比較預期結果與實際結果                                 ║   │
│  ║          - 識別誤報/漏報原因                                      ║   │
│  ║                                                                   ║   │
│  ║  步驟 13: 經驗學習 (Experience Learning)                          ║   │
│  ║          - 更新 Payload 效率評分                                  ║   │
│  ║          - 調整策略權重                                           ║   │
│  ║          - 反饋到 RAG 知識庫                                      ║   │
│  ╚══════════════════════════════════════════════════════════════════╝   │
│                                    ↓                                     │
│                           返回步驟 4 (新目標)                            │
│                           或步驟 1 (重新探索)                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 掃描模組的功用

### 🎯 Scan 模組定位

**Scan 模組 (`services/scan/`) 是步驟 7 的執行者**

```
┌─────────────────────────────────────────────────────────────────┐
│                    掃描模組職責 (Scan Module)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ 應該做的事情:                                               │
│  ─────────────────                                              │
│  1. 接收 aiva_core 的 AICommand (目標 + 掃描參數)               │
│  2. 調度對應語言的掃描引擎                                      │
│  3. 實際發送 HTTP 請求到靶場                                    │
│  4. 收集靶場返回的響應                                          │
│  5. 解析響應，判斷是否存在漏洞特徵                              │
│  6. 返回結構化結果給 aiva_core                                  │
│                                                                 │
│  ❌ 不應該做的事情:                                              │
│  ─────────────────                                              │
│  1. 不決定掃描什麼目標 (由 aiva_core 決定)                      │
│  2. 不決定使用什麼策略 (由 AI 決策層決定)                       │
│  3. 不分析漏洞嚴重程度 (由 Integration 分析)                    │
│  4. 不學習經驗 (由 ExternalLoopConnector 負責)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 引擎分工

| 引擎 | 語言 | 職責 | 適合場景 |
|------|------|------|----------|
| **Go Engine** | Go | 參數模糊測試、SSRF、快速掃描 | 大量目標、廉價並發 |
| **Rust Engine** | Rust | HTTP Smuggling、認證爆破 | 需要極致性能 |
| **TypeScript Engine** | Node.js | DOM XSS、SPA 爬蟲 | 需要瀏覽器環境 |
| **Python Engine** | Python | XXE、反序列化、被動分析 | 複雜邏輯處理 |

### 🔄 掃描模組工作流程

```
aiva_core (步驟 6)              Scan 模組 (步驟 7)              靶場
       │                              │                          │
       │  AICommand                   │                          │
       │  {target, scan_type, params} │                          │
       │─────────────────────────────>│                          │
       │                              │                          │
       │                              │  HTTP Request            │
       │                              │  (Payload 注入)          │
       │                              │────────────────────────>│
       │                              │                          │
       │                              │  HTTP Response           │
       │                              │  (響應內容)              │
       │                              │<────────────────────────│
       │                              │                          │
       │                              │ 解析響應                  │
       │                              │ 判斷漏洞特徵              │
       │                              │                          │
       │  ScanResult                  │                          │
       │  {findings, stats, errors}   │                          │
       │<─────────────────────────────│                          │
       │                              │                          │
```

---

## 各步驟詳細說明

### 步驟 1: 模組探索 (Module Exploration)

**執行模組**: `services/core/aiva_core/internal_exploration/module_explorer.py`

**輸入**: 無（自動掃描 services/ 目錄）

**輸出**: 模組列表及其能力概要

**驗證方法**:
```powershell
python -c "from services.core.aiva_core.internal_exploration import ModuleExplorer; print('OK')"
```

---

### 步驟 2: 能力分析 (Capability Analysis)

**執行模組**: `services/core/aiva_core/internal_exploration/capability_analyzer.py`

**輸入**: 模組列表

**輸出**: 結構化能力定義 (ModuleCapability Schema)

**能力分類**:
- `SCANNING`: 端口掃描、漏洞掃描、服務識別
- `ATTACKING`: SQL注入、XSS、業務邏輯漏洞
- `ANALYSIS`: 數據分析、結果解析、偏差分析
- `UTILITY`: 編碼/解碼、加密/解密、數據轉換

---

### 步驟 3: RAG 注入 (Knowledge Base Injection)

**執行模組**: `services/core/aiva_core/cognitive_core/internal_loop_connector.py`

**輸入**: 能力定義列表

**輸出**: RAG 知識庫更新

**執行命令**:
```powershell
python scripts\core\update_self_awareness.py
```

---

### 步驟 4-6: 情報收集與策略規劃

**執行模組**: `services/core/aiva_core/task_planning/`

**關鍵功能**:
- 目標驗證
- RAG 查詢
- AICommand 生成

---

### 步驟 7: 掃描執行 (Scan Execution)

**執行模組**: `services/scan/`

**⚠️ 重要**: 這是掃描模組的核心職責

**真正的 SSRF 檢測應該**:
1. 向靶場發送包含回調 URL 的請求
2. 等待靶場後端訪問回調 URL
3. 若回調被觸發，確認 SSRF 漏洞存在

**錯誤示範** (僅 URL 參數注入):
```
# 這不是真正的 SSRF 測試
GET /api?url=http://169.254.169.254/metadata HTTP/1.1
```

**正確示範** (真正的 SSRF 測試):
```
# 靶場應該去請求攻擊者的 URL，返回內容
POST /profile/image/url HTTP/1.1
Content-Type: application/json

{"imageUrl": "http://169.254.169.254/latest/meta-data/"}
```

---

### 步驟 8-9: 結果解析與漏洞確認

**執行模組**: 
- `services/integration/coordinators/base_coordinator.py`
- `services/features/features_ready/`

---

### 步驟 10-11: 攻擊規劃與執行

**執行模組**: `services/features/features_ready/`

**功能模組**:
- `function_sqli/` - SQL 注入攻擊
- `function_xss/` - XSS 攻擊
- `function_ssrf/` - SSRF 攻擊
- `function_idor/` - IDOR 攻擊

---

### 步驟 12-13: 偏差分析與經驗學習

**執行模組**: 
- `services/integration/aiva_integration/attack_path_analyzer/`
- `services/core/aiva_core/cognitive_core/external_loop_connector.py`

---

## 模組職責對照表

| 步驟 | 模組 | 職責 | 調用方式 |
|------|------|------|----------|
| 1-3 | aiva_core/internal_exploration | 內閉環 - 自我認知 | 定時/手動觸發 |
| 4 | aiva_core/task_planning | 目標接收 | API/CLI |
| 5-6 | aiva_core/cognitive_core | AI 決策 | 內部調用 |
| 7 | scan/* | **掃描執行** | AICommand |
| 8 | integration/coordinators | 結果解析 | 回調 |
| 9 | features/features_ready | 漏洞確認 | AICommand |
| 10-11 | features/features_ready | 攻擊執行 | AICommand |
| 12-13 | integration + cognitive_core | 學習優化 | 自動觸發 |

---

## 驗證檢查清單

### ✅ 內閉環驗證

- [ ] 步驟 1: ModuleExplorer 能掃描所有模組
- [ ] 步驟 2: CapabilityAnalyzer 能解析能力
- [ ] 步驟 3: InternalLoopConnector 能注入 RAG
- [ ] RAG 查詢返回正確能力列表

### ✅ 外閉環驗證

- [ ] 步驟 7: 掃描引擎**實際發送請求**到靶場
- [ ] 步驟 7: 靶場**有響應記錄**
- [ ] 步驟 8: 結果能轉換為統一 Schema
- [ ] 步驟 9: Features 模組能進行深度檢測

### ✅ 學習閉環驗證

- [ ] 步驟 12: 偏差分析能識別誤報
- [ ] 步驟 13: 經驗能反饋到 RAG

---

## 參考文檔

- [INTERNAL_LOOP_EXECUTION_GUIDE.md](../../../guides/INTERNAL_LOOP_EXECUTION_GUIDE.md) - 內閉環執行手冊
- [Scan README](../../scan/README.md) - 掃描模組說明
- [Features README](../../features/README.md) - 功能模組說明
- [dual_loop.py](../../aiva_common/schemas/dual_loop.py) - 雙閉環 Schema 定義

---

*文檔版本: v1.0 | 最後更新: 2025-12-28*
