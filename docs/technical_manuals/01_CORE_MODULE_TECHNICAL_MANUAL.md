# AIVA Core 模組技術手冊

**版本**: v4.1.1 | **狀態**: ✅ Production Ready | **路徑**: `services/core/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [架構設計](#2-架構設計)
   - 2.1 [四種操作模式](#21-四種操作模式)
   - 2.2 [13 步驟工作流](#22-13-步驟工作流)
3. [核心元件](#3-核心元件)
   - 3.1 [關鍵檔案](#31-關鍵檔案)
   - 3.2 [Bug Bounty 決策引擎](#32-bug-bounty-決策引擎-v440)
   - 3.3 [神經網路決策引擎](#33-神經網路決策引擎)
4. [資料流](#4-資料流)
5. [完成狀態](#5-完成狀態)
   - 5.1 [已完成功能](#51-已完成功能-)
   - 5.2 [待完成 / 目標功能](#52-待完成--目標功能-)
6. [反幻覺機制](#6-反幻覺機制)
7. [依賴配置](#7-依賴配置)
8. [與其他模組的整合](#8-與其他模組的整合)
9. [搭配閱讀](#9-搭配閱讀)

---

## 1. 模組概述

Core 模組是 AIVA 系統的主要 AI 決策中樞，負責接收任務指令、協調各子系統執行，並輸出最終決策結果。包含三個主要子系統整合：

| 子系統 | 說明 |
|---|---|
| AIVA Core | 500M 參數 BioNeuron + RAG 系統 |
| AI Core | 事件驅動增強系統 |
| AIVA Core v1 | 輕量工作流引擎 |

**規模**：60+ Python 模組，約 25,000 行程式碼

---

## 2. 架構設計

### 2.1 四種操作模式

```
┌─────────────────────────────────────────────┐
│              AIVA Core 操作模式              │
├───────────┬───────────┬────────┬────────────┤
│ UI 引導模式 │ AI 自主模式 │ 對話模式 │  混合模式  │
└───────────┴───────────┴────────┴────────────┘
```

- **UI 引導模式**：使用者透過介面指定目標與參數
- **AI 自主模式**：AI 完全自主決策攻擊路徑
- **對話模式**：即時對話式漏洞分析
- **混合模式**：人機協作，AI 提供建議由人確認

### 2.2 13 步驟工作流

```
Phase 0 接收與偵察
  Step 0: CLI 執行模式（276 flows 已分類）
  Step 1: 目標接收與初始化

Phase 1 深度掃描 + AI 決策 1
  Step 2: 端口掃描（nmap/masscan AI 選擇）
  Step 3: 服務識別與技術棧分析
  Step 4: AI 決策點 1 → decide_scan_strategy()
  Step 5: 深度漏洞掃描

Phase 2 攻擊測試 + AI 決策 2
  Step 6: AI 決策點 2 → decide_phase2_targets()
  Step 7: 多模組並行攻擊測試
  Step 8: AI 決策點 3 → decide_phase2_strategy()
  Step 9: 精確攻擊執行

Phase 3 評估 + 學習
  Step 10: 結果收集與驗證
  Step 11: evaluate_phase2_results()
  Step 12: 報告生成
  Step 13: 經驗學習寫回 RAG
```

---

## 3. 核心元件

### 3.1 關鍵檔案

| 檔案 | 功能 |
|---|---|
| `aiva_core/cognitive_core/capability_orchestrator.py` | AI 決策引擎核心 |
| `aiva_core/cognitive_core/neural/real_neural_core.py` | 5M 決策引擎（512→100 輸出） |
| `aiva_core/__init__.py` | 模組匯出與初始化（24KB） |
| `aiva_core/ai_executor_interface.py` | AI 執行器介面（11.8KB） |
| `aiva_core/planner/orchestrator.py` | 攻擊編排器 |
| `aiva_core/execution/plan_executor.py` | 計畫執行引擎 |

### 3.2 Bug Bounty 決策引擎（v4.4.0）

```python
# 4 個核心決策方法
decide_scan_strategy()      # 掃描工具智能選擇（nmap/masscan）
decide_phase1_strategy()    # Phase1 深度掃描決策（ROI 閾值 $75/hr）
decide_phase2_targets()     # 攻擊目標優先順序
                            #   Tier 1: Critical $10k+
                            #   Tier 2: High $5k+
                            #   Tier 3: Medium $2k+
evaluate_phase2_results()   # 結果評估與後續行動（CVSS 指引）
```

### 3.3 神經網路決策引擎

- **架構**：512 維輸入 → 隱藏層 → 100 維輸出
- **參數量**：5M
- **輸入特徵**：從 `capability_encoder.py` 產生的 512 維去語意化向量
- **輸出**：攻擊策略優先度分數矩陣
- **權重檔案**：`aiva_real_weights.pth`（已存在）

---

## 4. 資料流

```
使用者輸入 URL/目標
      │
      ▼
cognitive_core/capability_orchestrator.py
      │  ← 查詢 RAG 知識庫
      │  ← 執行 5M Neural 決策
      │  ← 呼叫 embedded_knowledge
      ▼
planner/orchestrator.py（攻擊計畫生成）
      │
      ▼
execution/plan_executor.py
      │  → 呼叫 features/ 模組
      │  → 呼叫 scan/ 模組
      ▼
結果 → integration/ → 報告
      │
      ▼
learning_system/（經驗寫回）
```

---

## 5. 完成狀態

### 5.1 已完成功能 ✅

| 功能 | 版本 | 說明 |
|---|---|---|
| 5M 神經網路決策引擎 | v4.1.1 | 512→100 輸出，生產就緒 |
| Bug Bounty 決策引擎 | v4.4.0 | 4 大決策方法，HackerOne/Bugcrowd 優化 |
| 去語意化反射引擎 | v2.1 | 12/12 驗證測試通過 |
| 嵌入式安全知識庫 | v1.0.0 | 400+ SQLi 指紋，20+ WAF 繞過技術 |
| RAG 整合 | ✅ | 512 維向量檢索，P0 階段完成 |
| 反幻覺模組 | ✅ | >95% 事實驗證精度 |
| 學習系統 | ✅ | 經驗重放記憶體，持續學習管線 |
| P0-P2 架構修復 | 2025-11-15 | 所有跨模組 AI 協同問題已解決 |
| UTC 相容性修復 | ✅ | 5 個檔案已修復 |
| 13 步驟工作流 | ✅ | 276 flows 已分類可呼叫 |

### 5.2 待完成 / 目標功能 🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| AI Recorder 延遲優化 | P1 | 目標 <100ms（目前 ~200ms） |
| RAG P1 實際執行驗證 | P1 | 對真實目標實測，收集錯誤並優化 |
| 多目標並行編排 | P2 | 同時處理多個 Bug Bounty 目標 |
| 自動 PoC 生成 | P2 | 漏洞確認後自動產生概念驗證程式碼 |
| 攻擊鏈組合決策 | P2 | Phase1 → Phase2 → PostEx 自動串接 |
| 目標指紋識別強化 | P2 | 基於目標技術棧自動調整決策參數 |
| 模型線上微調 | P3 | 基於執行回饋的即時權重更新 |
| 多平台 Bug Bounty 支援 | P3 | 擴展至 Intigriti、YesWeHack 等平台 |
| Tier 4 漏洞分類 | P3 | 支援 Info/N/A 類漏洞的 ROI 計算 |
| Web UI 決策視覺化 | P3 | 即時顯示 AI 決策理由與信心分數 |

---

## 6. 反幻覺機制

為確保 AI 決策可靠性，系統包含三層驗證：

1. **事實驗證**（`anti_hallucination/`）：交叉比對多個知識來源
2. **信心評分**：每個決策附帶置信度（HIGH ≥0.8 / MEDIUM 0.5-0.8 / LOW <0.5）
3. **不確定性標記**：低信心決策會標記供人工複審

---

## 7. 依賴配置

| 層級 | 大小 | 用途 |
|---|---|---|
| minimal | 65 MB | 基本運算 |
| standard | ~500 MB | 一般使用 |
| full | 4.5 GB | 完整 AI 功能 |
| dev | full + 開發工具 | 開發環境 |

---

## 8. 與其他模組的整合

| 模組 | 關係 | 介面 |
|---|---|---|
| `features/` | 呼叫方 | `feature_step_executor.py` |
| `scan/` | 呼叫方 | 各引擎 CLI 介面 |
| `integration/` | 結果傳送 | `ai_executor_interface.py` |
| `aiva_common/` | 依賴 | schemas, enums, config |
| `cognitive_core/rag/` | 內部依賴 | `rag_engine.py` |

---

## 9. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第2冊_AI決策流程.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第3冊_執行與適應.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第2-2冊_13步驟黑盒測試架構詳解.md`
- **技術手冊**：`docs/technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/07_RAG_SYSTEM_TECHNICAL_MANUAL.md`
