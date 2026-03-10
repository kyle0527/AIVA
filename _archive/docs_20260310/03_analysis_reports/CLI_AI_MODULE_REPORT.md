# AIVA CLI 指令分析報告

## 📑 目錄

- [依照 AI 模組分類](#依照-ai-模組分類)
- [📊 總覽](#-總覽)
- [🎯 模組分布總覽](#-模組分布總覽)
- [🤖 AI 能力類型分布](#-ai-能力類型分布)
- [🌐 AI對外能力 (10 個)](#-ai對外能力-10-個)
  - [Flow 29 - AI 能力查詢](#flow-29---ai-能力查詢)
  - [Flow 45 - 增強決策代理](#flow-45---增強決策代理)
  - [Flow 143 - 能力編排器](#flow-143---能力編排器)
  - [Flow 147 - 外部迴路連接器](#flow-147---外部迴路連接器)
  - [Flow 174 - 計畫執行器](#flow-174---計畫執行器)
  - [Flow 356 - 攻擊鏈](#flow-356---攻擊鏈)
  - [Flow 442, 458, 491, 495](#flow-442-458-491-495)
- [🧠 AI內部能力 (3 個)](#-ai內部能力-3-個)
  - [Flow 51 - 內部迴路連接器](#flow-51---內部迴路連接器)
  - [Flow 323 - 強化學習模型](#flow-323---強化學習模型)
  - [Flow 464 - 內部迴路連接器 (副本)](#flow-464---內部迴路連接器-副本)
- [⚙️ AI組件 (7 個)](#-ai組件-7-個)
  - [Flow 117 - AI 控制器](#flow-117---ai-控制器)
  - [Flow 134 - AI 模型管理器](#flow-134---ai-模型管理器)
  - [Flow 137 - 真實神經核心](#flow-137---真實神經核心)
  - [Flow 140 - 模型訓練器](#flow-140---模型訓練器)
  - [Flow 141 - RL 訓練器](#flow-141---rl-訓練器)
  - [Flow 322 - RAG 引擎](#flow-322---rag-引擎)
  - [Flow 430 - AI 模型管理器 (副本)](#flow-430---ai-模型管理器-副本)
- [🔄 混合組件 (2 個)](#-混合組件-2-個)
  - [Flow 33 - 能力註冊表](#flow-33---能力註冊表)
  - [Flow 446 - 能力註冊表 (副本)](#flow-446---能力註冊表-副本)
- [📈 模組 × AI 類型 交叉分析](#-模組-ai-類型-交叉分析)
- [🚀 推薦的 CLI 操作](#-推薦的-cli-操作)
  - [📋 查詢 AI 能力](#-查詢-ai-能力)
  - [🧠 認知核心操作](#-認知核心操作)
  - [📊 學習系統操作](#-學習系統操作)
  - [🔧 服務骨幹操作](#-服務骨幹操作)
- [📝 結論](#-結論)

---

## 依照 AI 模組分類

> 生成日期: 2026-01-09  
> 資料來源: `latest_classification.json` (Schema v3.3)

---

## 📊 總覽

| 統計項目 | 數值 |
|---------|------|
| **總計 Flow** | 676 個 |
| **AI 相關能力** | 22 個 (3.3%) |
| **程式組件** | 654 個 (96.7%) |

---

## 🎯 模組分布總覽

| 模組 | 總數 | AI 能力數 | AI 佔比 |
|------|------|----------|---------|
| **unknown** | 523 | 0 | 0.0% |
| **service_backbone** | 41 | 1 | 2.4% |
| **cognitive_core** | 32 | 14 | 43.8% |
| **core_capabilities** | 22 | 3 | 13.6% |
| **task_planning** | 21 | 1 | 4.8% |
| **internal_exploration** | 20 | 0 | 0.0% |
| **learning_system** | 17 | 3 | 17.6% |

---

## 🤖 AI 能力類型分布

| 類型 | 數量 | 說明 |
|------|------|------|
| 🌐 **AI對外能力** | 10 | 對外提供的 AI 服務接口 |
| 🧠 **AI內部能力** | 3 | 內部 AI 處理邏輯 |
| ⚙️ **AI組件** | 7 | AI 相關基礎組件 |
| 🔄 **混合組件** | 2 | 混合 AI 與程式功能 |

---

## 🌐 AI對外能力 (10 個)

這些是 AIVA 對外提供的 AI 服務接口。

### Flow 29 - AI 能力查詢
- **模組**: `cognitive_core`
- **端點**: `ai_capability_query`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.ai_capability_query query
  ```
- **參數**:
  - `question` (str) - 必填
  - `top_k` (int) = 5

### Flow 45 - 增強決策代理
- **模組**: `cognitive_core`
- **端點**: `enhanced_decision_agent`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent query
  ```
- **參數**:
  - `knowledge_base` (any)
  - `experience_manager` (any)

### Flow 143 - 能力編排器
- **模組**: `cognitive_core`
- **端點**: `capability_orchestrator`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.capability_orchestrator query
  ```
- **參數**:
  - `task_type` (str) - 必填
  - `target` (str) - 必填
  - `objectives` (List[str]) - 必填
  - `**kwargs` (dict)

### Flow 147 - 外部迴路連接器
- **模組**: `cognitive_core`
- **端點**: `external_loop_connector`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.external_loop_connector query
  ```

### Flow 174 - 計畫執行器
- **模組**: `task_planning`
- **端點**: `plan_executor`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.task_planning.executor.plan_executor execute
  ```
- **參數**:
  - `message_broker` (MessageBroker | None)
  - `unified_tracer` (UnifiedTracer | None)
  - `storage_backend` (Any | None)

### Flow 356 - 攻擊鏈
- **模組**: `core_capabilities`
- **端點**: `attack_chain`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.core_capabilities.attack.attack_chain
  ```
- **參數**:
  - `chain_id` (str) - 必填

### Flow 442, 458, 491, 495
這些是重複的 Flow (不同入口但相同端點)：
- `ai_capability_query` (Flow 442)
- `enhanced_decision_agent` (Flow 458)
- `capability_orchestrator` (Flow 491)
- `external_loop_connector` (Flow 495)

---

## 🧠 AI內部能力 (3 個)

這些是 AIVA 的內部 AI 處理邏輯。

### Flow 51 - 內部迴路連接器
- **模組**: `cognitive_core`
- **端點**: `internal_loop_connector`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.internal_loop_connector query
  ```
- **參數**:
  - `file_path` (str) - 必填
- **路徑**: session_state_manager → rich_cli → postgresql_vector_store → internal_loop_connector

### Flow 323 - 強化學習模型
- **模組**: `learning_system`
- **端點**: `rl_models`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.learning_system.learning.rl_models
  ```
- **路徑**: session_state_manager → storage_manager → experience_manager → rl_models

### Flow 464 - 內部迴路連接器 (副本)
- 與 Flow 51 相同端點，不同入口路徑

---

## ⚙️ AI組件 (7 個)

這些是 AI 相關的基礎組件。

### Flow 117 - AI 控制器
- **模組**: `service_backbone`
- **端點**: `ai_controller`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.service_backbone.coordination.ai_controller
  ```

### Flow 134 - AI 模型管理器
- **模組**: `cognitive_core`
- **端點**: `ai_model_manager`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.neural.ai_model_manager query
  ```
- **參數**:
  - `experience_repository` (any) - 必填

### Flow 137 - 真實神經核心
- **模組**: `cognitive_core`
- **端點**: `real_neural_core`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.neural.real_neural_core query
  ```

### Flow 140 - 模型訓練器
- **模組**: `learning_system`
- **端點**: `model_trainer`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.learning_system.learning.model_trainer
  ```
- **參數**:
  - `model_dir` (Path | None)
  - `storage_backend` (Any | None)

### Flow 141 - RL 訓練器
- **模組**: `learning_system`
- **端點**: `rl_trainers`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.learning_system.learning.rl_trainers
  ```
- **參數** (12 個):
  - `state_dim` (int) - 必填
  - `action_dim` (int) - 必填
  - `learning_rate` (float) = 0.0003
  - `gamma` (float) = 0.99
  - `gae_lambda` (float) = 0.95
  - ... 還有 7 個參數

### Flow 322 - RAG 引擎
- **模組**: `cognitive_core`
- **端點**: `rag_engine`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.cognitive_core.rag.rag_engine query
  ```
- **參數**:
  - `knowledge_base` ('KnowledgeBase') - 必填

### Flow 430 - AI 模型管理器 (副本)
- 與 Flow 134 相同端點，不同入口路徑

---

## 🔄 混合組件 (2 個)

這些是混合 AI 與程式功能的組件。

### Flow 33 - 能力註冊表
- **模組**: `core_capabilities`
- **端點**: `capability_registry`
- **CLI**: 
  ```bash
  python -m services.core.aiva_core.core_capabilities.capability_registry
  ```
- **參數**:
  - `force_refresh` (bool) = False

### Flow 446 - 能力註冊表 (副本)
- 與 Flow 33 相同端點，不同入口路徑

---

## 📈 模組 × AI 類型 交叉分析

| 模組 | AI對外能力 | AI內部能力 | AI組件 | 混合組件 | **總計** |
|------|-----------|-----------|--------|---------|---------|
| **cognitive_core** | 8 | 2 | 4 | 0 | **14** |
| **core_capabilities** | 1 | 0 | 0 | 2 | **3** |
| **learning_system** | 0 | 1 | 2 | 0 | **3** |
| **service_backbone** | 0 | 0 | 1 | 0 | **1** |
| **task_planning** | 1 | 0 | 0 | 0 | **1** |
| **總計** | **10** | **3** | **7** | **2** | **22** |

---

## 🚀 推薦的 CLI 操作

### 📋 查詢 AI 能力
```bash
# 查詢特定能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query query --question "如何執行攻擊鏈"

# 列出所有能力
python -m services.core.aiva_core.core_capabilities.capability_registry
```

### 🧠 認知核心操作
```bash
# 決策查詢
python -m services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent query

# 內部迴路處理
python -m services.core.aiva_core.cognitive_core.internal_loop_connector query --file_path "path/to/file"

# RAG 查詢
python -m services.core.aiva_core.cognitive_core.rag.rag_engine query
```

### 📊 學習系統操作
```bash
# 訓練模型
python -m services.core.aiva_core.cognitive_core.learning_system.learning.model_trainer

# RL 訓練
python -m services.core.aiva_core.cognitive_core.learning_system.learning.rl_trainers --state_dim 10 --action_dim 4
```

### 🔧 服務骨幹操作
```bash
# AI 控制器
python -m services.core.aiva_core.service_backbone.coordination.ai_controller
```

---

## 📝 結論

1. **認知核心 (cognitive_core)** 是 AI 能力最集中的模組，包含 14 個 AI 相關能力 (佔總 AI 能力的 63.6%)

2. **AI 對外能力** 主要集中在查詢和決策功能

3. **學習系統 (learning_system)** 提供 RL 訓練和模型管理功能

4. **unknown 模組** 有 523 個 Flow 需要進一步分類

5. 部分 Flow 是重複的 (不同入口到相同端點)，建議後續去重或合併
