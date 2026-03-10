# AIVA Cognitive Core 技術手冊

**版本**: v5.0.0
**狀態**: Production Ready
**路徑**: `services/core/aiva_core/cognitive_core/`

---

## 1. 模組概述

Cognitive Core 是 AIVA 的「大腦」——負責 AI 決策、持續學習、知識檢索與反幻覺驗證。包含 48 個 Python 檔案，7 個核心子系統。

---

## 2. 七大子系統

```
cognitive_core/
├── decision/           決策支援（7 個檔案）
├── embedded_knowledge/ 嵌入知識庫（8 個檔案）⭐ v1.0.0
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
| `capability_orchestrator.py` | AI 決策引擎核心（總協調） |
| `capability_encoder.py` | 512 維去語意化向量編碼（v2.1） |
| `ai_capability_query.py` | 使用者友善能力分析介面 |
| `internal_loop_connector.py` | 能力分析注入 RAG |
| `external_loop_connector.py` | 執行結果傳送至 Integration |

---

## 4. decision/ — Bug Bounty 決策子系統

**版本**: v4.4.0，針對 HackerOne/Bugcrowd 優化

### 4.1 四個核心決策方法

```python
# 1. 掃描策略選擇
decide_scan_strategy(target_info) -> ScanStrategy
# 決定使用 nmap 或 masscan，依據：
# - 目標規模（IP 數量）
# - 掃描深度需求
# - 時間預算

# 2. Phase 1 策略
decide_phase1_strategy(scan_results) -> Phase1Plan
# ROI 閾值：$75/hr
# 若預期回報低於閾值，跳過深度掃描

# 3. Phase 2 目標優先順序
decide_phase2_targets(phase1_results) -> PriorityList
# Tier 1: Critical 漏洞 - $10,000+
# Tier 2: High 漏洞   - $5,000+
# Tier 3: Medium 漏洞 - $2,000+

# 4. Phase 2 結果評估
evaluate_phase2_results(results) -> NextActions
# 依 CVSS 評分決定後續行動
```

### 4.2 決策輔助元件

- **Tier-based 優先系統**：自動計算 Bug Bounty 投資回報
- **CVSS 引導**：9.0+ 立即報告，7.0+ 優先處理
- **HackerOne/Bugcrowd 特化**：理解各平台獎勵結構

---

## 5. embedded_knowledge/ — 嵌入知識庫（v1.0.0）

無需外部 API 的本地化漏洞知識庫：

### 5.1 核心元件

```python
VulnerabilityDetector   # 漏洞偵測
  ├── SQL Injection 指紋（400+）
  ├── XSS payload 庫
  ├── SSRF 探測點
  └── IDOR 模式識別

CVEIdentifier           # CVE 識別
  └── 8 個 CVSS ≥9.0 高危 CVE

WAFBypassEngine         # WAF 繞過
  ├── 20+ 繞過技術
  └── 6 家 WAF 廠商（Cloudflare, Akamai, AWS WAF, F5, Imperva, ModSecurity）

WebArchitectureAnalyzer # Web 架構分析
  ├── GraphQL 端點偵測
  ├── JWT 實作分析
  ├── REST API 模式識別
  └── WebSocket 安全分析
```

### 5.2 WAF 繞過技術（20+）

| 類別 | 技術數量 |
|---|---|
| 編碼繞過 | 6（URL/雙重/Unicode/HTML/Base64/Hex） |
| 大小寫混淆 | 3 |
| 注釋注入 | 4 |
| HTTP 參數污染 | 3 |
| 分塊傳輸 | 2 |
| 其他 | 2+ |

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

**目的**：不使用語言模型，改用特徵哈希（Feature Hashing）確定性產生 512 維向量：

```
優點：
  ✓ 可重現（相同輸入→相同向量）
  ✓ 不依賴外部 NLU 服務
  ✓ 環境特徵直接檢索
  ✓ 支援 PostgreSQL backend

驗證：12/12 測試通過
```

---

## 7. learning_system/ — 統一經驗學習

**18 個檔案**，負責持續學習與知識積累：

```python
# 核心功能
ExperienceReplayMemory   # 經驗重放記憶
KnowledgeExtractor       # 知識提取
ModelTrainer             # 模型訓練觸發
ContinuousLearning       # 持續學習管線
```

### 學習流程

```
攻擊執行完成
  │
  ▼
結果收集（Integration 回傳）
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

確保 AI 決策基於實際知識，而非生成幻覺：

```python
FactValidator      # 事實驗證（交叉比對多個知識來源）
CrossSourceVerify  # 跨來源一致性驗證
UncertaintyMarker  # 低信心決策標記
```

**三層防護**：
1. 決策前：查詢 RAG 確認知識基礎
2. 決策中：交叉驗證多個來源
3. 決策後：信心評分標記，低於閾值送人工複審

---

## 9. 與其他模組的整合

```
capability_orchestrator.py
  │
  ├── → embedded_knowledge/ (本地知識查詢)
  ├── → rag/rag_engine.py (向量知識查詢)
  ├── → neural/real_neural_core.py (神經決策)
  ├── → anti_hallucination/ (結果驗證)
  │
  ├── ← internal_loop_connector.py (能力分析注入)
  └── → external_loop_connector.py (結果送 Integration)
```

---

## 10. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第2冊_AI決策流程.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md`
- **技術手冊**：`docs/technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md`
