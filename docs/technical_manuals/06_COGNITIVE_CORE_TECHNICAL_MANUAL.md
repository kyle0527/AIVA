# AIVA Cognitive Core 技術手冊

**版本**: v5.0.0 | **狀態**: ✅ Production Ready | **路徑**: `services/core/aiva_core/cognitive_core/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [七大子系統](#2-七大子系統)
3. [根目錄核心整合檔案](#3-根目錄核心整合檔案)
4. [decision/ — Bug Bounty 決策子系統](#4-decision--bug-bounty-決策子系統-v440)
5. [embedded_knowledge/ — 嵌入知識庫](#5-embedded_knowledge--嵌入知識庫-v100)
6. [neural/ — 5M 決策引擎](#6-neural--5m-決策引擎)
7. [learning_system/ — 統一經驗學習](#7-learning_system--統一經驗學習)
8. [anti_hallucination/ — 反幻覺驗證](#8-anti_hallucination--反幻覺驗證)
9. [完成狀態](#9-完成狀態)
   - 9.1 [已完成功能](#91-已完成功能-)
   - 9.2 [待完成 / 目標功能](#92-待完成--目標功能-)
10. [與其他模組的整合](#10-與其他模組的整合)
11. [搭配閱讀](#11-搭配閱讀)

---

## 1. 模組概述

Cognitive Core 是 AIVA 的「大腦」——負責 AI 決策、持續學習、知識檢索與反幻覺驗證。包含 48 個 Python 檔案，7 個核心子系統。

**品質指標**：
- 嵌入知識庫回應延遲 < 1ms
- 記憶體用量 ~8MB（指紋 + CVE 資料）
- 反幻覺模組準確率 >95%
- 去語意化向量驗證：12/12 通過

---

## 2. 七大子系統

```
cognitive_core/
├── decision/           決策支援（7 個檔案）
├── embedded_knowledge/ 嵌入知識庫（8 個檔案）⭐ v1.0.0 新增
├── neural/             神經網路（5 個檔案）
├── rag/                檢索增強生成（7 個檔案）
├── learning_system/    學習系統（18 個檔案）
├── anti_hallucination/ 反幻覺（3 個檔案）
└── [根目錄]            7 個核心整合檔案
```

---

## 3. 根目錄核心整合檔案

| 檔案 | 功能 |
|---|---|
| `capability_orchestrator.py` | AI 決策引擎核心（總協調）|
| `capability_encoder.py` | 512 維去語意化向量編碼（v2.1）|
| `ai_capability_query.py` | 使用者友善能力分析介面 |
| `internal_loop_connector.py` | 能力分析注入 RAG |
| `external_loop_connector.py` | 執行結果傳送至 Integration |

---

## 4. decision/ — Bug Bounty 決策子系統（v4.4.0）

針對 HackerOne/Bugcrowd 優化的決策引擎：

### 4.1 四個核心決策方法

```python
# 1. 掃描策略選擇
decide_scan_strategy(target_info) -> ScanStrategy
# 決定使用 nmap 或 masscan，依據目標規模與時間預算

# 2. Phase 1 策略
decide_phase1_strategy(scan_results) -> Phase1Plan
# ROI 閾值：$75/hr
# 預期回報低於閾值 → 跳過深度掃描

# 3. Phase 2 目標優先順序
decide_phase2_targets(phase1_results) -> PriorityList
# Tier 1: Critical $10,000+
# Tier 2: High    $5,000+
# Tier 3: Medium  $2,000+

# 4. Phase 2 結果評估
evaluate_phase2_results(results) -> NextActions
# CVSS ≥9.0 → 立即報告
# CVSS 7.0-8.9 → 優先處理
# CVSS <7.0 → 排入佇列
```

### 4.2 增強決策代理

`enhanced_decision_agent.py`（v4.4.0）提供 4 種決策方法：
1. **Neural 方法**：5M 神經網路評分
2. **RAG 方法**：歷史經驗檢索
3. **Embedded 方法**：本地知識庫查詢
4. **Hybrid 方法**：三者融合決策

---

## 5. embedded_knowledge/ — 嵌入知識庫（v1.0.0）

無需外部 API 的本地化漏洞知識庫，~3,200 行，stateless classmethods：

### 5.1 核心元件

```python
VulnerabilityDetector   # 漏洞偵測
  ├── SQLi 指紋庫（400+）
  ├── XSS payload 庫
  ├── SSRF 探測端點
  └── IDOR 模式識別

CVEIdentifier           # CVE 識別
  └── 8 個 CVSS ≥9.0 高危 CVE

WAFBypassEngine         # WAF 繞過
  ├── 20+ 繞過技術
  └── 6 家 WAF 廠商
      （Cloudflare, Akamai, AWS WAF, F5, Imperva, ModSecurity）

WebArchitectureAnalyzer # Web 架構分析
  ├── GraphQL 端點偵測
  ├── JWT 實作安全分析
  ├── REST API 模式識別
  └── WebSocket 安全分析
```

### 5.2 WAF 繞過技術分類（20+）

| 類別 | 數量 |
|---|---|
| 編碼繞過（URL/雙重/Unicode/HTML/Base64/Hex）| 6 |
| 大小寫混淆 | 3 |
| 注釋注入 | 4 |
| HTTP 參數污染 | 3 |
| 分塊傳輸 | 2 |
| 其他技術 | 2+ |

---

## 6. neural/ — 5M 決策引擎

### 6.1 架構

```
輸入: 512 維特徵向量（由 capability_encoder.py 生成）
  │
  ▼
隱藏層（多層感知機）
  │
  ▼
輸出: 100 維攻擊策略評分矩陣
```

### 6.2 關鍵檔案

| 檔案 | 功能 |
|---|---|
| `real_neural_core.py` | 5M 決策引擎主體 |
| `ai_model_manager.py` | AI 模型生命週期管理 |
| `weight_manager.py` | 權重持久化與版本管理 |

### 6.3 去語意化向量（v2.1）

**目的**：不使用語言模型，改用 Feature Hashing 確定性產生 512 維向量：

```
優點：
  ✓ 可重現（相同輸入 → 永遠相同向量）
  ✓ 不依賴外部 NLU 服務
  ✓ 環境特徵直接檢索
  ✓ 支援 PostgreSQL backend

驗證：12/12 測試通過
```

---

## 7. learning_system/ — 統一經驗學習

**18 個檔案**，負責持續學習與知識積累：

```
攻擊執行完成
  │
  ▼
KnowledgeExtractor（提取學習要點）
  │
  ▼
ExperienceReplayMemory（存入記憶）
  │
  ▼
向量化 → RAG vector_store（可被未來查詢使用）
  │
  ▼
ModelTrainer（定期觸發模型更新）
```

---

## 8. anti_hallucination/ — 反幻覺驗證

**三層防護**：

1. 決策前：查詢 RAG 確認知識基礎
2. 決策中：交叉驗證多個來源（`CrossSourceVerify`）
3. 決策後：信心評分標記，低於閾值送人工複審（`UncertaintyMarker`）

---

## 9. 完成狀態

### 9.1 已完成功能 ✅

| 功能 | 版本 | 說明 |
|---|---|---|
| 5M 神經網路決策引擎 | v4.1.1 | 生產就緒，權重檔案存在 |
| Bug Bounty 決策引擎 | v4.4.0 | 4 大決策方法 |
| 去語意化反射引擎 | v2.1 | 12/12 驗證測試通過 |
| 嵌入式安全知識庫 | v1.0.0 | 8 檔案，~3,200 行，< 1ms 回應 |
| VulnerabilityDetector | ✅ | 400+ SQLi 指紋 |
| CVEIdentifier | ✅ | 8 個 CVSS ≥9.0 CVE |
| WAFBypassEngine | ✅ | 20+ 技術，6 廠商 |
| WebArchitectureAnalyzer | ✅ | GraphQL, JWT, WebSocket |
| 反幻覺模組 | ✅ | >95% 精度 |
| 學習系統（基礎）| ✅ | 經驗重放記憶體 |
| DecisionContext 修復 | ✅ | 補齊 environment_features |
| UTC 相容性修復 | ✅ | 5 個檔案 |

### 9.2 待完成 / 目標功能 🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| RAG P1 實際執行驗證 | P1 | 對真實目標實測，收集錯誤資料 |
| 決策算法強化 | P2 | 基於目標指紋優化 Tier 分類 |
| XXE 嵌入知識支援 | P2 | 新增 XXE 到 VulnerabilityDetector |
| File Upload 知識支援 | P2 | 惡意檔案上傳偵測規則 |
| WAF 廠商擴展 | P2 | 新增 Sucuri、StackPath 等廠商 |
| CVE 庫動態更新 | P2 | 連接 NVD API 自動更新 8 個固定 CVE |
| 攻擊鏈知識編碼 | P2 | 將 Phase1→Phase2→PostEx 鏈式攻擊編入知識庫 |
| 模型線上微調 | P3 | 基於執行結果的即時權重更新 |
| 知識庫版本管理 | P3 | 追蹤知識庫變更歷史 |
| 多目標並行決策 | P3 | 同時處理多個目標的決策請求 |

---

## 10. 與其他模組的整合

```
capability_orchestrator.py
  │
  ├── → embedded_knowledge/（本地知識查詢）
  ├── → rag/rag_engine.py（向量知識查詢）
  ├── → neural/real_neural_core.py（神經決策）
  ├── → anti_hallucination/（結果驗證）
  │
  ├── ← internal_loop_connector.py（能力分析注入）
  └── → external_loop_connector.py（結果送 Integration）
```

---

## 11. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第2冊_AI決策流程.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md`
- **技術手冊**：`docs/technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md`
