# AIVA AI 內部完整運作流程

## 📑 目錄

- [🔍 關鍵問題澄清](#-關鍵問題澄清)
  - [Q: 內部閉環/外部學習是「被禁用」還是「還沒完成」？](#q-內部閉環外部學習是被禁用還是還沒完成)
- [📊 完整運作流程圖](#-完整運作流程圖)
- [📋 各環節腳本清單](#-各環節腳本清單)
  - [1. 入口層](#1-入口層)
  - [2. 狀態管理層](#2-狀態管理層)
  - [3. 命令路由層](#3-命令路由層)
  - [4. 執行規劃層](#4-執行規劃層)
  - [5. AI 決策引擎 (核心)](#5-ai-決策引擎-核心)
  - [6. RAG 知識庫層](#6-rag-知識庫層)
  - [7. 統一執行層](#7-統一執行層)
  - [8. 學習系統](#8-學習系統)
- [🚨 待解決問題清單](#-待解決問題清單)
  - [優先級 P0 (阻塞)](#優先級-p0-阻塞)
  - [優先級 P1 (嚴重)](#優先級-p1-嚴重)
  - [優先級 P2 (中等)](#優先級-p2-中等)
  - [優先級 P3 (低)](#優先級-p3-低)
- [🔄 完整調用鏈](#-完整調用鏈)
- [📊 模組統計](#-模組統計)
- [🎯 下一步行動](#-下一步行動)
  - [階段一: 啟用內部閉環 (P0)](#階段一-啟用內部閉環-p0)
  - [階段二: 補全 RAG 組件 (P1)](#階段二-補全-rag-組件-p1)
  - [階段三: 優化學習系統 (P2)](#階段三-優化學習系統-p2)
- [🎯 模組啟動機制分析](#-模組啟動機制分析)
  - [每個模組「誰啟動它」？](#每個模組誰啟動它)
  - [啟動方式對照表](#啟動方式對照表)

---


> **版本**: v1.1  
> **日期**: 2026-01-12  
> **目的**: 繪製 AI 從接收指令到執行完成的完整內部流程，標註各環節腳本和待解問題

---

## 🔍 關鍵問題澄清

### Q: 內部閉環/外部學習是「被禁用」還是「還沒完成」？

**答案：✅ 已完成，但 ⚠️ 連接層未啟用**

| 組件 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| ExternalLoopConnector | `external_loop_connector.py` (447行) | ✅ **已實現** | 完整實現偏差分析、訓練觸發、權重更新 |
| ExternalLearningListener | `event_listener.py` (266行) | ✅ **已實現** | 監聽 TASK_COMPLETED 事件 |
| periodic_update | ❌ 不存在 | ❌ **檔案不存在** | `update_self_awareness.py` 從未建立 |

**問題根源**：
```python
# app.py:54-59 (import 被註釋)
# ⚠️ 暫時註釋：這些模組尚未實現
# from services.core.aiva_core.internal_exploration.connectors.update_self_awareness import (
#     periodic_update,                    # ← 這個檔案不存在！
# )
# from services.core.aiva_core.external_learning.connectors.external_loop_connector import (
#     ExternalLoopConnector,              # ← 路徑錯誤！正確位置在 cognitive_core/
# )
```

**結論**：
1. `ExternalLoopConnector` 已完成，但 import 路徑寫錯了
2. `periodic_update` 對應的檔案從未創建
3. 這是「**連接層配置問題**」，不是「功能未實現」

---

## 📊 完整運作流程圖

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AIVA AI 內部完整運作流程                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ [入口層] app.py (FastAPI)                                                 │  │
│  │ 位置: services/core/aiva_core/service_backbone/api/app.py (433行)         │  │
│  │ 功能: 系統唯一入口，HTTP API 端點                                          │  │
│  │ 狀態: ✅ 已實現                                                           │  │
│  │ ⚠️ 問題: Internal/External loops disabled (模組未實現)                    │  │
│  └────────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                             │
│                                   ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ [狀態管理層] CoreServiceCoordinator                                       │  │
│  │ 位置: .../service_backbone/coordination/core_service_coordinator.py (655行)│  │
│  │ 功能: 被動的狀態管理器和服務工廠                                           │  │
│  │ 核心方法: process_command() - 主命令處理入口                               │  │
│  │ 狀態: ✅ 已實現                                                           │  │
│  └────────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                             │
│                    ┌──────────────┼──────────────┐                              │
│                    ▼              ▼              ▼                              │
│  ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────┐│
│  │ [命令路由]           │ │ [上下文管理]     │ │ [執行規劃]                      ││
│  │ CommandRouter       │ │ ContextManager  │ │ ExecutionPlanner               ││
│  │ command_router.py   │ │ context_manager │ │ execution_planner.py           ││
│  │ (534行)             │ │ .py             │ │ (573行)                        ││
│  │ ✅ 已實現           │ │ ✅ 已實現       │ │ ✅ 已實現                       ││
│  └──────────┬──────────┘ └────────────────┘ └────────────────────────────────┘ │
│             │                                                                   │
│             ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ [AI 決策引擎] CapabilityOrchestrator                                      │  │
│  │ 位置: .../cognitive_core/capability_orchestrator.py (1118行)              │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ plan() 方法 - 主決策流程                                            │  │  │
│  │  │                                                                     │  │  │
│  │  │  1. _query_relevant_capabilities() → 查詢 RAG 知識庫                │  │  │
│  │  │     ├─ 使用 InternalLoopConnector.query_capabilities()              │  │  │
│  │  │     ├─ 語義匹配 + 精確匹配                                          │  │  │
│  │  │     └─ ⚠️ InternalLoopConnector 可能不可用時用 fallback             │  │  │
│  │  │                                                                     │  │  │
│  │  │  2. _filter_available_capabilities() → 過濾可用能力                 │  │  │
│  │  │     ├─ 檢查 CLI 是否可用                                            │  │  │
│  │  │     ├─ 檢查健康狀態                                                 │  │  │
│  │  │     └─ 過濾禁用項目                                                 │  │  │
│  │  │                                                                     │  │  │
│  │  │  3. _select_best_capabilities() → 選擇最佳能力組合                  │  │  │
│  │  │     ├─ 計算權重分數                                                 │  │  │
│  │  │     ├─ 考慮歷史表現                                                 │  │  │
│  │  │     └─ 取 top-N                                                     │  │  │
│  │  │                                                                     │  │  │
│  │  │  4. _generate_execution_sequence() → 生成執行序列                   │  │  │
│  │  │     └─ 按依賴和優先級排序                                           │  │  │
│  │  │                                                                     │  │  │
│  │  │  5. _capabilities_to_cli_commands() → 轉換為 CLI 命令               │  │  │
│  │  │     └─ 生成可執行的命令列表                                          │  │  │
│  │  │                                                                     │  │  │
│  │  │  → 輸出: CapabilityPlan (計劃 ID, CLI 命令, 預估時間)               │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ execute() 方法 - 執行計劃                                           │  │  │
│  │  │                                                                     │  │  │
│  │  │  1. 使用 AsyncProcessManager 執行 CLI 命令                          │  │  │
│  │  │  2. 收集遙測數據 (HTTP 狀態碼、WAF 檢測等)                          │  │  │
│  │  │  3. 返回 ExecutionResult                                            │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │ 狀態: ✅ 已實現                                                           │  │
│  │ ⚠️ 問題: 無 - 核心決策引擎已完整實現                                      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                   │                                             │
│                                   ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ [RAG 知識庫] InternalLoopConnector                                        │  │
│  │ 位置: .../cognitive_core/internal_loop_connector.py (2036行)              │  │
│  │ 功能: 連接內部探索結果到 RAG 知識庫                                        │  │
│  │                                                                           │  │
│  │  數據來源:                                                                │  │
│  │  internal_exploration (三階段管道)                                        │  │
│  │   ├─ aiva_flow_analyzer.py → 分析 flow                                    │  │
│  │   ├─ aiva_flow_classifier.py → 分類 flow                                  │  │
│  │   └─ aiva_cli_implementation.py → 生成 CLI                                │  │
│  │                                                                           │  │
│  │  核心功能:                                                                │  │
│  │   ├─ query_capabilities() - 查詢能力 (RAG)                                │  │
│  │   ├─ sync_capabilities() - 同步能力到 RAG                                 │  │
│  │   └─ CapabilityScopeClassifier - 能力範圍分類                             │  │
│  │                                                                           │  │
│  │ 狀態: ⚠️ 部分實現                                                         │  │
│  │ ⚠️ 問題:                                                                  │  │
│  │   - ModuleExplorer 未實現 (用 aiva_flow_analyzer 替代)                    │  │
│  │   - CapabilityAnalyzer 未實現 (用 aiva_flow_classifier 替代)              │  │
│  │   - CapabilityRegistry 未實現 (dual-write disabled)                       │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                   │                                             │
│                                   ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ [學習系統] LearningSystem (learning_system/)                              │  │
│  │ 位置: .../cognitive_core/learning_system/                                 │  │
│  │                                                                           │  │
│  │  子模組:                                                                  │  │
│  │   ├─ learning/                                                            │  │
│  │   │   ├─ continuous_learning.py - 持續學習引擎                            │  │
│  │   │   └─ model_trainer.py - 模型訓練 (DQN/PPO)                            │  │
│  │   ├─ experience_manager.py - 經驗管理                                     │  │
│  │   ├─ training/ - 訓練數據                                                 │  │
│  │   └─ analysis/ - 分析模組                                                 │  │
│  │                                                                           │  │
│  │ 狀態: ⚠️ 部分實現                                                         │  │
│  │ ⚠️ 問題:                                                                  │  │
│  │   - PyTorch not available 時 DQN/PPO training disabled                    │  │
│  │   - 內部閉環更新被禁用 (app.py 中註釋)                                     │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 各環節腳本清單

### 1. 入口層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `app.py` | `services/core/aiva_core/service_backbone/api/app.py` | 433 | FastAPI 入口，HTTP API 端點 | ✅ 已實現 |

**主要功能**:
- `startup()` - 系統啟動，初始化協調器
- `process_phase0_results()` - 處理 Phase0 結果
- `process_scan_results()` - 處理掃描結果
- `process_function_results()` - 處理功能結果
- `monitor_execution_status()` - 監控執行狀態

---

### 2. 狀態管理層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `core_service_coordinator.py` | `.../service_backbone/coordination/` | 655 | 狀態管理器和服務工廠 | ✅ 已實現 |

**主要功能**:
- `process_command()` - 命令處理主入口
- `get_service_status()` - 獲取服務狀態
- `_initialize_core_components()` - 初始化核心組件
- `_initialize_shared_services()` - 初始化共享服務

---

### 3. 命令路由層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `command_router.py` | `.../task_planning/` | 534 | 智能命令路由 | ✅ 已實現 |

**主要功能**:
- `route_command()` - 路由命令到正確處理器
- `_initialize_intelligent_routes()` - 初始化路由映射
- `_initialize_ai_keywords()` - 初始化 AI 關鍵詞
- `get_command_stats()` - 獲取命令統計

---

### 4. 執行規劃層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `execution_planner.py` | `.../task_planning/planner/` | 573 | 執行計劃編排 | ✅ 已實現 |

**主要功能**:
- `create_execution_plan()` - 創建執行計劃
- `execute_plan()` - 執行計劃
- `_check_resources()` - 檢查資源可用性
- `get_execution_stats()` - 獲取執行統計

---

### 5. AI 決策引擎 (核心)

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `capability_orchestrator.py` | `.../cognitive_core/` | 1118 | **AI 主決策引擎** | ✅ 已實現 |

**核心 `plan()` 流程**:
```python
async def plan(requirement: TaskRequirement) -> CapabilityPlan:
    # 1️⃣ 查詢相關能力
    relevant = await self._query_relevant_capabilities(requirement)
    
    # 2️⃣ 過濾可用能力
    available = await self._filter_available_capabilities(relevant)
    
    # 3️⃣ 選擇最佳組合
    selected = await self._select_best_capabilities(requirement, available)
    
    # 4️⃣ 生成執行序列
    sequence = await self._generate_execution_sequence(requirement, selected)
    
    # 5️⃣ 轉換為 CLI 命令
    commands = self._capabilities_to_cli_commands(sequence)
    
    return CapabilityPlan(
        plan_id=...,
        cli_commands=commands,
        estimated_duration=...
    )
```

**核心 `execute()` 流程**:
```python
async def execute(plan: CapabilityPlan) -> ExecutionResult:
    # 使用 AsyncProcessManager 執行 CLI 命令
    for cli_cmd in plan.cli_commands:
        result = await process_manager.run_command_with_telemetry(
            cmd=cmd_list,
            timeout=plan.estimated_duration
        )
        # 收集遙測數據供 AI 學習
```

---

### 6. RAG 知識庫層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `internal_loop_connector.py` | `.../cognitive_core/` | 2036 | RAG 連接器 | ⚠️ 部分實現 |

**主要功能**:
- `query_capabilities()` - 查詢能力 (RAG 語義搜索)
- `sync_capabilities()` - 同步能力到 RAG
- `CapabilityScopeClassifier` - 能力範圍分類器

---

### 7. 統一執行層

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `unified_executor.py` | `.../task_planning/` | 824 | 統一執行接口 | ✅ 已實現 |

**主要功能**:
- `execute()` - 統一執行接口 (靶場和實戰)
- `execute_with_context()` - 帶上下文執行
- 整合 CapabilityOrchestrator + ContinuousLearningEngine

---

### 8. 學習系統

| 腳本 | 路徑 | 行數 | 功能 | 狀態 |
|------|------|------|------|------|
| `continuous_learning.py` | `.../learning_system/learning/` | - | 持續學習引擎 | ⚠️ 部分實現 |
| `model_trainer.py` | `.../learning_system/learning/` | - | DQN/PPO 訓練 | ⚠️ PyTorch 依賴 |
| `experience_manager.py` | `.../learning_system/` | - | 經驗管理 | ✅ 已實現 |

---

## 🚨 待解決問題清單

### 優先級 P0 (阻塞)

| 問題 | 位置 | 說明 | 影響 |
|------|------|------|------|
| 內部閉環禁用 | `app.py:131-144` | `periodic_update()` 被註釋 | 無法自動更新 RAG |
| 外部學習禁用 | `app.py:131-144` | `ExternalLoopConnector` 被註釋 | 無法從外部學習 |

### 優先級 P1 (嚴重)

| 問題 | 位置 | 說明 | 影響 |
|------|------|------|------|
| ModuleExplorer 未實現 | `internal_loop_connector.py:426` | 使用替代方案 | 能力發現受限 |
| CapabilityAnalyzer 未實現 | `internal_loop_connector.py:427` | 使用替代方案 | 能力分析受限 |
| CapabilityRegistry 未實現 | `internal_loop_connector.py:434` | dual-write disabled | 無法持久化能力 |

### 優先級 P2 (中等)

| 問題 | 位置 | 說明 | 影響 |
|------|------|------|------|
| PyTorch 依賴 | `model_trainer.py:40` | DQN/PPO 訓練 disabled | 強化學習不可用 |
| InternalLoopConnector fallback | `capability_orchestrator.py:321` | 可能使用 fallback | 查詢質量降低 |
| CapabilityEncoder 不可用 | `vector_store.py:165` | 使用 hash embedding | 向量品質降低 |

### 優先級 P3 (低)

| 問題 | 位置 | 說明 | 影響 |
|------|------|------|------|
| TODO: 文本解析 | `ast_parser.py:219` | 待實現 | 解析受限 |
| TODO: 額外動作檢測 | `plan_executor.py:689` | 待實現 | 監控受限 |
| TODO: 真實協同邏輯 | `ai_controller.py:318` | 待實現 | 多 AI 協同 |

---

## 🔄 完整調用鏈

```
HTTP Request
    │
    ▼
app.py (FastAPI)
    │
    ├─[POST /command]───────────────────────────────────────────────┐
    │                                                               │
    ▼                                                               ▼
CoreServiceCoordinator.process_command()                 Background Tasks
    │                                                   (phase0/scan/function)
    ├─ CommandRouter.route_command()
    │   └─ 判斷: AI vs 非AI, 複雜度, 類型
    │
    ├─ ContextManager.create_context()
    │   └─ 創建執行上下文
    │
    ├─ ExecutionPlanner.create_execution_plan()
    │   └─ 根據命令類型制定步驟
    │
    └─ ExecutionPlanner.execute_plan()
        │
        ├─[requires_ai=true]───────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
CapabilityOrchestrator.plan()                              Simple Executor
    │                                                     (直接執行)
    ├─ _query_relevant_capabilities()
    │   └─ InternalLoopConnector.query_capabilities()
    │       └─ RAG 語義搜索
    │
    ├─ _filter_available_capabilities()
    │   └─ 檢查 CLI/健康狀態
    │
    ├─ _select_best_capabilities()
    │   └─ 權重計算 + 歷史表現
    │
    ├─ _generate_execution_sequence()
    │   └─ 依賴排序
    │
    └─ _capabilities_to_cli_commands()
        └─ 生成 CLI 命令列表
            │
            ▼
CapabilityOrchestrator.execute()
    │
    ├─ AsyncProcessManager.run_command_with_telemetry()
    │   └─ 執行 CLI 命令 + 收集遙測
    │
    └─ ExecutionResult
        │
        ▼
LearningSystem (if enabled)
    │
    ├─ ContinuousLearningEngine.learn_from_execution()
    │   └─ 更新經驗庫
    │
    └─ ModelTrainer (if PyTorch available)
        └─ 更新 DQN/PPO 模型
```

---

## 📊 模組統計

| 層級 | 模組數 | 總行數 | 實現狀態 |
|------|--------|--------|----------|
| 入口層 | 1 | 433 | ✅ 100% |
| 狀態管理層 | 1 | 655 | ✅ 100% |
| 命令路由層 | 1 | 534 | ✅ 100% |
| 執行規劃層 | 1 | 573 | ✅ 100% |
| AI 決策引擎 | 1 | 1118 | ✅ 100% |
| RAG 知識庫層 | 1 | 2036 | ⚠️ 80% |
| 統一執行層 | 1 | 824 | ✅ 100% |
| 學習系統 | 3+ | ~1500 | ⚠️ 60% |

**總計**: ~7700+ 行核心代碼

---

## 🎯 下一步行動

### 階段一: 啟用內部閉環 (P0)
1. 實現 `periodic_update()` 或確認替代方案
2. 實現 `ExternalLoopConnector` 或確認不需要
3. 取消 `app.py` 中的相關註釋

### 階段二: 補全 RAG 組件 (P1)
1. 確認 `aiva_flow_analyzer` 完全替代 `ModuleExplorer`
2. 確認 `aiva_flow_classifier` 完全替代 `CapabilityAnalyzer`
3. 評估是否需要 `CapabilityRegistry`

### 階段三: 優化學習系統 (P2)
1. 添加 PyTorch 依賴或提供替代訓練器
2. 優化 `CapabilityEncoder` 向量化
3. 實現真實的多 AI 協同邏輯

---

## 🎯 模組啟動機制分析

### 每個模組「誰啟動它」？

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              模組啟動觸發關係圖                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 觸發源頭 (3種)                                                          │    │
│  │                                                                         │    │
│  │   1️⃣ HTTP Request (外部觸發)                                           │    │
│  │      └─ 用戶/系統發送 POST /command 請求                                │    │
│  │                                                                         │    │
│  │   2️⃣ Background Task (內部定時)                                        │    │
│  │      └─ asyncio.create_task() 創建的後台任務                            │    │
│  │                                                                         │    │
│  │   3️⃣ Message Queue (事件驅動)                                          │    │
│  │      └─ MessageBroker 訂閱的事件觸發                                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ [1] app.py (FastAPI) - 系統啟動時自動執行                               │    │
│  │     啟動者: uvicorn / python -m uvicorn                                 │    │
│  │                                                                         │    │
│  │     startup() 自動觸發:                                                 │    │
│  │     ├─ CoreServiceCoordinator.start() ← 直接調用                        │    │
│  │     ├─ process_phase0_results() ← asyncio.create_task() 後台           │    │
│  │     ├─ process_scan_results() ← asyncio.create_task() 後台             │    │
│  │     ├─ process_function_results() ← asyncio.create_task() 後台         │    │
│  │     └─ monitor_execution_status() ← asyncio.create_task() 後台         │    │
│  └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ [2] CoreServiceCoordinator - 被 app.py 啟動                             │    │
│  │     啟動者: await coordinator.start()                                   │    │
│  │                                                                         │    │
│  │     process_command() 被調用時觸發:                                      │    │
│  │     ├─ CommandRouter.route_command() ← 直接方法調用                     │    │
│  │     ├─ ContextManager.create_context() ← 直接方法調用                   │    │
│  │     ├─ ExecutionPlanner.create_execution_plan() ← 直接方法調用          │    │
│  │     └─ ExecutionPlanner.execute_plan() ← 直接方法調用                   │    │
│  └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ [3] CapabilityOrchestrator - 被 ExecutionPlanner 或 UnifiedExecutor 調用│    │
│  │     啟動者: 當命令需要 AI 決策時，由上層實例化並調用                      │    │
│  │                                                                         │    │
│  │     plan() / execute() 觸發:                                            │    │
│  │     ├─ InternalLoopConnector.query_capabilities() ← 延遲加載            │    │
│  │     ├─ AsyncProcessManager.run_command_with_telemetry() ← 直接調用      │    │
│  │     └─ ContinuousLearningEngine.learn_from_execution() ← 條件調用       │    │
│  └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ [4] LearningSystem - 被執行結果觸發                                     │    │
│  │     啟動者: 事件驅動 (TASK_COMPLETED) 或直接方法調用                     │    │
│  │                                                                         │    │
│  │     ExternalLearningListener.start_listening() (如果啟用):              │    │
│  │     └─ MessageBroker.subscribe("task.completed.*") ← 事件訂閱           │    │
│  │         └─ ExternalLoopConnector.process_execution_result() ← 回調      │    │
│  │             └─ ModelTrainer.train() ← 延遲加載                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 啟動方式對照表

| 模組 | 啟動方式 | 觸發者 | 觸發時機 |
|------|----------|--------|----------|
| `app.py` | 外部命令 | `uvicorn` 或 `python -m` | 系統啟動 |
| `CoreServiceCoordinator` | 直接調用 | `app.py startup()` | 系統啟動 |
| `CommandRouter` | 直接調用 | `CoreServiceCoordinator` | 每次命令 |
| `ExecutionPlanner` | 直接調用 | `CoreServiceCoordinator` | 每次命令 |
| `CapabilityOrchestrator` | 實例化調用 | `ExecutionPlanner` / `UnifiedExecutor` | AI 決策需求 |
| `InternalLoopConnector` | 延遲加載 | `CapabilityOrchestrator` | 首次查詢 |
| `ExternalLoopConnector` | 延遲加載 | `ExternalLearningListener` | 首次事件 |
| `ExternalLearningListener` | 後台任務 | ❌ **未啟用** (app.py 註釋) | - |
| `ContinuousLearningEngine` | 延遲加載 | `UnifiedExecutor` | 學習啟用時 |
| `ModelTrainer` | 延遲加載 | `ExternalLoopConnector` / `ContinuousLearningEngine` | 訓練觸發 |

---

*文檔生成時間: 2026-01-12*
