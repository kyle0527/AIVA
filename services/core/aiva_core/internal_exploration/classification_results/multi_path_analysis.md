# 多路徑終點分析報告

生成時間: 2026-01-31 01:04:39
找到 61 個有多條路徑到達的終點

---

## 終點: ai_models

**說明**: ai_models - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 認知核心模組(學習子系統), unknown

### 路徑詳細對比

#### 路徑 1 (Flow 3)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: model_trainer[AI組件] → ai_models[AI組件]

**完整腳本列表**:
1. model_trainer
2. ai_models

#### 路徑 2 (Flow 15)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: plan_executor[AI對外能力] → ai_models[AI組件]

**完整腳本列表**:
1. plan_executor
2. ai_models

#### 路徑 3 (Flow 50)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: scenario_manager[程式組件] → ai_models[AI組件]

**完整腳本列表**:
1. scenario_manager
2. ai_models

#### 路徑 4 (Flow 111)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: plan_comparator[程式組件] → ai_models[AI組件]

**完整腳本列表**:
1. plan_comparator
2. ai_models

#### 路徑 5 (Flow 137)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: unified_tracer[程式組件] → ai_models[AI組件]

**完整腳本列表**:
1. unified_tracer
2. ai_models

#### 路徑 6 (Flow 232)

- **長度**: 2 步
- **主要模組**: unknown
- **執行順序**: backends[程式組件] → ai_models[AI組件]

**完整腳本列表**:
1. backends
2. ai_models

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - model_trainer
- 路徑 2 獨有: 1 個腳本
  - plan_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統), unknown
- 路徑 2 主要涉及: unknown, 任務規劃模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: enhanced_decision_agent

**說明**: enhanced_decision_agent - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 認知核心模組, 核心能力模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 8)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: scan_result_processor[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. scan_result_processor
2. enhanced_decision_agent

#### 路徑 2 (Flow 25)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. enhanced_decision_agent
2. enhanced_decision_agent

#### 路徑 3 (Flow 35)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. enhanced_decision_agent

#### 路徑 4 (Flow 197)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: app[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. app
2. enhanced_decision_agent

#### 路徑 5 (Flow 256)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: two_phase_scan_orchestrator[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. two_phase_scan_orchestrator
2. enhanced_decision_agent

#### 路徑 6 (Flow 398)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: attack_coordinator[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. attack_coordinator
2. enhanced_decision_agent

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - scan_result_processor
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_executor

**說明**: unified_executor - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 認知核心模組, 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 9)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: unified_executor[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. unified_executor
2. unified_executor

#### 路徑 2 (Flow 14)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: backends[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. backends
2. unified_executor

#### 路徑 3 (Flow 61)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 4 (Flow 63)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: learning_adapter[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. learning_adapter
2. unified_executor

#### 路徑 5 (Flow 266)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: external_loop_connector[AI對外能力] → unified_executor[程式組件]

**完整腳本列表**:
1. external_loop_connector
2. unified_executor

#### 路徑 6 (Flow 391)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: plan_builder[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. plan_builder
2. unified_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - backends

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組
- 路徑 2 主要涉及: 服務骨幹模組, 任務規劃模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: message_broker

**說明**: message_broker - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 認知核心模組, 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 22)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: message_broker[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. message_broker
2. message_broker

#### 路徑 2 (Flow 117)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: plan_executor[AI對外能力] → message_broker[程式組件]

**完整腳本列表**:
1. plan_executor
2. message_broker

#### 路徑 3 (Flow 146)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: event_listener[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. event_listener
2. message_broker

#### 路徑 4 (Flow 287)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: dispatcher[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. dispatcher
2. message_broker

#### 路徑 5 (Flow 341)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: dispatcher_base[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. dispatcher_base
2. message_broker

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - plan_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組
- 路徑 2 主要涉及: 任務規劃模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: internal_loop_connector

**說明**: internal_loop_connector - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 47)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_orchestrator[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. capability_orchestrator
2. internal_loop_connector

#### 路徑 2 (Flow 103)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_registry[混合組件] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. capability_registry
2. internal_loop_connector

#### 路徑 3 (Flow 108)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: internal_loop_connector[AI內部能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. internal_loop_connector
2. internal_loop_connector

#### 路徑 4 (Flow 238)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. enhanced_decision_agent
2. internal_loop_connector

#### 路徑 5 (Flow 354)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. ai_capability_query
2. internal_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 1 個腳本
  - capability_registry

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組
- 路徑 2 主要涉及: 認知核心模組, 核心能力模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: core_analyzer

**說明**: core_analyzer - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 4 步
- **平均路徑長度**: 2.40 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 73)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. run_analysis
2. core_analyzer

#### 路徑 2 (Flow 84)

- **長度**: 4 步
- **主要模組**: 內探模組
- **執行順序**: analyze_results[程式組件] → analyze_results[程式組件] → analyze_results[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. analyze_results
2. analyze_results
3. analyze_results
4. core_analyzer

#### 路徑 3 (Flow 177)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. sync_experiences
2. core_analyzer

#### 路徑 4 (Flow 284)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_missing_function_connections[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. analyze_missing_function_connections
2. core_analyzer

#### 路徑 5 (Flow 311)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. core_analyzer
2. core_analyzer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - run_analysis
- 路徑 2 獨有: 1 個腳本
  - analyze_results

**使用場景差異推測**:

---

## 終點: aiva_internal_executor

**說明**: aiva_internal_executor - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組, 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 106)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: internal_loop_connector[AI內部能力] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. internal_loop_connector
2. aiva_internal_executor

#### 路徑 2 (Flow 112)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: aiva_cli[程式組件] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. aiva_cli
2. aiva_internal_executor

#### 路徑 3 (Flow 153)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: aiva_internal_executor[程式組件] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. aiva_internal_executor
2. aiva_internal_executor

#### 路徑 4 (Flow 160)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. sync_experiences
2. aiva_internal_executor

#### 路徑 5 (Flow 362)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: unified_executor_controller[程式組件] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. unified_executor_controller
2. aiva_internal_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - internal_loop_connector
- 路徑 2 獨有: 1 個腳本
  - aiva_cli

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 內探模組
- 路徑 2 主要涉及: 核心能力模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: vector_store

**說明**: vector_store - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.20 步
- **涉及模組**: 認知核心模組, 核心能力模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 128)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. vector_store

#### 路徑 2 (Flow 194)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: sync_experiences[程式組件] → sync_experiences[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. sync_experiences
2. sync_experiences
3. vector_store

#### 路徑 3 (Flow 206)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_executor
2. vector_store

#### 路徑 4 (Flow 275)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: assistant[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. assistant
2. vector_store

#### 路徑 5 (Flow 393)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → vector_store[程式組件]

**完整腳本列表**:
1. ai_capability_query
2. vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_vector_store
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

---

## 終點: unified_function_caller

**說明**: unified_function_caller - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.25 步
- **涉及模組**: 核心能力模組, 服務骨幹模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 13)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: enhanced_unified_caller[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. enhanced_unified_caller
2. unified_function_caller

#### 路徑 2 (Flow 71)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: assistant[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. assistant
2. unified_function_caller

#### 路徑 3 (Flow 125)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: unified_function_caller[程式組件] → unified_function_caller[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. unified_function_caller
2. unified_function_caller
3. unified_function_caller

#### 路徑 4 (Flow 212)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_unified_caller
- 路徑 2 獨有: 1 個腳本
  - assistant

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組
- 路徑 2 主要涉及: 核心能力模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_tracer

**說明**: unified_tracer - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 19)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_tracer[程式組件] → unified_tracer[程式組件]

**完整腳本列表**:
1. unified_tracer
2. unified_tracer

#### 路徑 2 (Flow 118)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: plan_executor[AI對外能力] → unified_tracer[程式組件]

**完整腳本列表**:
1. plan_executor
2. unified_tracer

#### 路徑 3 (Flow 332)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: enhanced_decision_agent[AI對外能力] → unified_tracer[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. unified_tracer

#### 路徑 4 (Flow 345)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: trace_recorder[程式組件] → unified_tracer[程式組件]

**完整腳本列表**:
1. trace_recorder
2. unified_tracer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - plan_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組(學習子系統), 任務規劃模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: experience_manager

**說明**: experience_manager - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 27)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_executor[程式組件] → experience_manager[程式組件]

**完整腳本列表**:
1. unified_executor
2. experience_manager

#### 路徑 2 (Flow 107)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: experience_manager[程式組件] → experience_manager[程式組件]

**完整腳本列表**:
1. experience_manager
2. experience_manager

#### 路徑 3 (Flow 204)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → experience_manager[程式組件]

**完整腳本列表**:
1. ai_model_manager
2. experience_manager

#### 路徑 4 (Flow 234)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: continuous_learning[程式組件] → experience_manager[程式組件]

**完整腳本列表**:
1. continuous_learning
2. experience_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統), 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: analyze_dataflow_breakpoints

**說明**: analyze_dataflow_breakpoints - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 30)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_dataflow_breakpoints[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. analyze_dataflow_breakpoints
2. analyze_dataflow_breakpoints

#### 路徑 2 (Flow 44)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. run_analysis
2. analyze_dataflow_breakpoints

#### 路徑 3 (Flow 175)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. sync_experiences
2. analyze_dataflow_breakpoints

#### 路徑 4 (Flow 338)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. core_analyzer
2. analyze_dataflow_breakpoints

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - run_analysis

**使用場景差異推測**:

---

## 終點: real_neural_core

**說明**: real_neural_core - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 45)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: multilang_coordinator[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. multilang_coordinator
2. real_neural_core

#### 路徑 2 (Flow 62)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: real_bio_net_adapter[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. real_bio_net_adapter
2. real_neural_core

#### 路徑 3 (Flow 237)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → real_neural_core[AI組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. real_neural_core

#### 路徑 4 (Flow 268)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: real_neural_core[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. real_neural_core
2. real_neural_core

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - multilang_coordinator
- 路徑 2 獨有: 1 個腳本
  - real_bio_net_adapter

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_registry

**說明**: capability_registry - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.50 步
- **涉及模組**: 認知核心模組, 核心能力模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 48)

- **長度**: 3 步
- **主要模組**: 核心能力模組
- **執行順序**: capability_orchestrator[AI對外能力] → capability_registry[混合組件] → capability_registry[混合組件]

**完整腳本列表**:
1. capability_orchestrator
2. capability_registry
3. capability_registry

#### 路徑 2 (Flow 100)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: capability_registry[混合組件] → capability_registry[混合組件]

**完整腳本列表**:
1. capability_registry
2. capability_registry

#### 路徑 3 (Flow 211)

- **長度**: 3 步
- **主要模組**: 核心能力模組
- **執行順序**: task_executor[程式組件] → capability_registry[混合組件] → capability_registry[混合組件]

**完整腳本列表**:
1. task_executor
2. capability_registry
3. capability_registry

#### 路徑 4 (Flow 298)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: skill_graph[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. skill_graph
2. capability_registry

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 核心能力模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: base

**說明**: base - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 55)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → base[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. base

#### 路徑 2 (Flow 154)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vulnerability_detection[程式組件] → base[程式組件]

**完整腳本列表**:
1. vulnerability_detection
2. base

#### 路徑 3 (Flow 199)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → base[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. base

#### 路徑 4 (Flow 286)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: cve_identification[程式組件] → base[程式組件]

**完整腳本列表**:
1. cve_identification
2. base

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE
- 路徑 2 獨有: 1 個腳本
  - vulnerability_detection

**使用場景差異推測**:

---

## 終點: practical_analyzer

**說明**: practical_analyzer - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 68)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: practical_analyzer[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. practical_analyzer
2. practical_analyzer

#### 路徑 2 (Flow 123)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. run_analysis
2. practical_analyzer

#### 路徑 3 (Flow 178)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. sync_experiences
2. practical_analyzer

#### 路徑 4 (Flow 340)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. core_analyzer
2. practical_analyzer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - run_analysis

**使用場景差異推測**:

---

## 終點: knowledge_base

**說明**: knowledge_base - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 102)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_registry[混合組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. capability_registry
2. knowledge_base

#### 路徑 2 (Flow 141)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → knowledge_base[程式組件]

**完整腳本列表**:
1. ai_capability_query
2. knowledge_base

#### 路徑 3 (Flow 207)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. unified_executor
2. knowledge_base

#### 路徑 4 (Flow 276)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: assistant[程式組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. assistant
2. knowledge_base

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - capability_registry
- 路徑 2 獨有: 1 個腳本
  - ai_capability_query

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: analyze_missing_function_connections

**說明**: analyze_missing_function_connections - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 110)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_missing_function_connections[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. analyze_missing_function_connections
2. analyze_missing_function_connections

#### 路徑 2 (Flow 176)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. sync_experiences
2. analyze_missing_function_connections

#### 路徑 3 (Flow 214)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. run_analysis
2. analyze_missing_function_connections

#### 路徑 4 (Flow 339)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. core_analyzer
2. analyze_missing_function_connections

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: model_trainer

**說明**: model_trainer - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 198)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_executor[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. unified_executor
2. model_trainer

#### 路徑 2 (Flow 203)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. ai_model_manager
2. model_trainer

#### 路徑 3 (Flow 205)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: external_loop_connector[AI對外能力] → model_trainer[AI組件]

**完整腳本列表**:
1. external_loop_connector
2. model_trainer

#### 路徑 4 (Flow 235)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: continuous_learning[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. continuous_learning
2. model_trainer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 1 個腳本
  - ai_model_manager

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統), 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組, 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: rag_trigger

**說明**: rag_trigger - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 4)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: rag_trigger[AI組件] → rag_trigger[AI組件]

**完整腳本列表**:
1. rag_trigger
2. rag_trigger

#### 路徑 2 (Flow 135)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: experience_manager[程式組件] → rag_trigger[AI組件]

**完整腳本列表**:
1. experience_manager
2. rag_trigger

#### 路徑 3 (Flow 372)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_decision_core[AI組件] → rag_trigger[AI組件]

**完整腳本列表**:
1. ai_decision_core
2. rag_trigger

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - experience_manager

**使用場景差異推測**:

---

## 終點: execution_orchestrator

**說明**: execution_orchestrator - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 18)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → execution_orchestrator[程式組件]

**完整腳本列表**:
1. unified_executor
2. execution_orchestrator

#### 路徑 2 (Flow 120)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_orchestrator[AI對外能力] → execution_orchestrator[程式組件]

**完整腳本列表**:
1. capability_orchestrator
2. execution_orchestrator

#### 路徑 3 (Flow 389)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: execution_orchestrator[程式組件] → execution_orchestrator[程式組件]

**完整腳本列表**:
1. execution_orchestrator
2. execution_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 1 個腳本
  - capability_orchestrator

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: analyze_connection_recommendations

**說明**: analyze_connection_recommendations - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 70)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → analyze_connection_recommendations[程式組件]

**完整腳本列表**:
1. run_analysis
2. analyze_connection_recommendations

#### 路徑 2 (Flow 174)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → analyze_connection_recommendations[程式組件]

**完整腳本列表**:
1. sync_experiences
2. analyze_connection_recommendations

#### 路徑 3 (Flow 251)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_connection_recommendations[程式組件] → analyze_connection_recommendations[程式組件]

**完整腳本列表**:
1. analyze_connection_recommendations
2. analyze_connection_recommendations

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - run_analysis
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: permission_matrix

**說明**: permission_matrix - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.33 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 97)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: policy_manager[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. policy_manager
2. permission_matrix

#### 路徑 2 (Flow 165)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: sync_experiences[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. sync_experiences
2. permission_matrix

#### 路徑 3 (Flow 258)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: permission_matrix[程式組件] → permission_matrix[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. permission_matrix
2. permission_matrix
3. permission_matrix

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - policy_manager
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 服務骨幹模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_embedding

**說明**: aiva_embedding - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 98)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: real_neural_core[AI組件] → aiva_embedding[程式組件]

**完整腳本列表**:
1. real_neural_core
2. aiva_embedding

#### 路徑 2 (Flow 138)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → aiva_embedding[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. aiva_embedding

#### 路徑 3 (Flow 291)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → aiva_embedding[程式組件]

**完整腳本列表**:
1. vector_store
2. aiva_embedding

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - real_neural_core
- 路徑 2 獨有: 1 個腳本
  - unified_vector_store

**使用場景差異推測**:

---

## 終點: rag_engine

**說明**: rag_engine - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 208)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → rag_engine[AI組件]

**完整腳本列表**:
1. unified_executor
2. rag_engine

#### 路徑 2 (Flow 236)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → rag_engine[AI組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. rag_engine

#### 路徑 3 (Flow 364)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: rag_engine[AI組件] → rag_engine[AI組件]

**完整腳本列表**:
1. rag_engine
2. rag_engine

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: execution_planner

**說明**: execution_planner - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.33 步
- **涉及模組**: 認知核心模組, 服務骨幹模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 273)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: execution_planner[程式組件] → execution_planner[程式組件]

**完整腳本列表**:
1. execution_planner
2. execution_planner

#### 路徑 2 (Flow 330)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → execution_planner[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. execution_planner

#### 路徑 3 (Flow 376)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: core_service_coordinator[程式組件] → execution_planner[程式組件] → execution_planner[程式組件]

**完整腳本列表**:
1. core_service_coordinator
2. execution_planner
3. execution_planner

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

---

## 終點: notification_system

**說明**: notification_system - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.50 步
- **涉及模組**: 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 1)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: notification_system[程式組件] → notification_system[程式組件]

**完整腳本列表**:
1. notification_system
2. notification_system

#### 路徑 2 (Flow 134)

- **長度**: 3 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: experience_manager[程式組件] → notification_system[程式組件] → notification_system[程式組件]

**完整腳本列表**:
1. experience_manager
2. notification_system
3. notification_system

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - experience_manager

**使用場景差異推測**:

---

## 終點: models

**說明**: models - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 6)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → models[AI組件]

**完整腳本列表**:
1. backends
2. models

#### 路徑 2 (Flow 33)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: command_repository[程式組件] → models[AI組件]

**完整腳本列表**:
1. command_repository
2. models

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - backends
- 路徑 2 獨有: 1 個腳本
  - command_repository

**使用場景差異推測**:

---

## 終點: scalable_bio_trainer

**說明**: scalable_bio_trainer - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 7)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → scalable_bio_trainer[AI內部能力]

**完整腳本列表**:
1. ai_model_manager
2. scalable_bio_trainer

#### 路徑 2 (Flow 385)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: scalable_bio_trainer[AI內部能力] → scalable_bio_trainer[AI內部能力]

**完整腳本列表**:
1. scalable_bio_trainer
2. scalable_bio_trainer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - ai_model_manager
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: app

**說明**: app - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 核心能力模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 11)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: app[程式組件] → app[程式組件]

**完整腳本列表**:
1. app
2. app

#### 路徑 2 (Flow 302)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: multilang_coordinator[程式組件] → app[程式組件]

**完整腳本列表**:
1. multilang_coordinator
2. app

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - multilang_coordinator

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組
- 路徑 2 主要涉及: 核心能力模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_orchestrator

**說明**: capability_orchestrator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 16)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → capability_orchestrator[AI對外能力]

**完整腳本列表**:
1. unified_executor
2. capability_orchestrator

#### 路徑 2 (Flow 99)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_orchestrator[AI對外能力] → capability_orchestrator[AI對外能力]

**完整腳本列表**:
1. capability_orchestrator
2. capability_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_encoder

**說明**: capability_encoder - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 20)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. vector_store
2. capability_encoder

#### 路徑 2 (Flow 139)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_encoder[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. capability_encoder
2. capability_encoder

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - vector_store
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: web_architecture

**說明**: web_architecture - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 23)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → web_architecture[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. web_architecture

#### 路徑 2 (Flow 243)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → web_architecture[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. web_architecture

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

---

## 終點: cve_identification

**說明**: cve_identification - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 3 - 3 步
- **平均路徑長度**: 3.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 24)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → cve_identification[程式組件] → cve_identification[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. cve_identification
3. cve_identification

#### 路徑 2 (Flow 241)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → cve_identification[程式組件] → cve_identification[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. cve_identification
3. cve_identification

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

---

## 終點: enhanced_classifier_processor

**說明**: enhanced_classifier_processor - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 28)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: enhanced_classifier_processor[程式組件] → enhanced_classifier_processor[程式組件]

**完整腳本列表**:
1. enhanced_classifier_processor
2. enhanced_classifier_processor

#### 路徑 2 (Flow 162)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → enhanced_classifier_processor[程式組件]

**完整腳本列表**:
1. sync_experiences
2. enhanced_classifier_processor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: monitoring

**說明**: monitoring - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 29)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: monitoring[程式組件] → monitoring[程式組件]

**完整腳本列表**:
1. monitoring
2. monitoring

#### 路徑 2 (Flow 386)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: ai_manager[AI組件] → monitoring[程式組件]

**完整腳本列表**:
1. ai_manager
2. monitoring

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - ai_manager

**使用場景差異推測**:

---

## 終點: system_self_explorer

**說明**: system_self_explorer - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 32)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: system_self_explorer[程式組件] → system_self_explorer[程式組件]

**完整腳本列表**:
1. system_self_explorer
2. system_self_explorer

#### 路徑 2 (Flow 371)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: ai_decision_core[AI組件] → system_self_explorer[程式組件]

**完整腳本列表**:
1. ai_decision_core
2. system_self_explorer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - ai_decision_core

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: waf_bypass

**說明**: waf_bypass - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 4 步
- **平均路徑長度**: 3.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 36)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → INTEGRATION_EXAMPLE[程式組件] → waf_bypass[程式組件] → waf_bypass[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. INTEGRATION_EXAMPLE
3. waf_bypass
4. waf_bypass

#### 路徑 2 (Flow 200)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → waf_bypass[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. waf_bypass

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

---

## 終點: policy_manager

**說明**: policy_manager - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 39)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: strategy_engine[程式組件] → policy_manager[程式組件]

**完整腳本列表**:
1. strategy_engine
2. policy_manager

#### 路徑 2 (Flow 96)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: policy_manager[程式組件] → policy_manager[程式組件]

**完整腳本列表**:
1. policy_manager
2. policy_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - strategy_engine
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: unified_executor_controller

**說明**: unified_executor_controller - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 65)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: ai_executor_interface[AI組件] → unified_executor_controller[程式組件]

**完整腳本列表**:
1. ai_executor_interface
2. unified_executor_controller

#### 路徑 2 (Flow 163)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → unified_executor_controller[程式組件]

**完整腳本列表**:
1. sync_experiences
2. unified_executor_controller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - ai_executor_interface
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組, 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_internal_classifier

**說明**: aiva_internal_classifier - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 67)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: enhanced_classifier_processor[程式組件] → aiva_internal_classifier[程式組件]

**完整腳本列表**:
1. enhanced_classifier_processor
2. aiva_internal_classifier

#### 路徑 2 (Flow 159)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → aiva_internal_classifier[程式組件]

**完整腳本列表**:
1. sync_experiences
2. aiva_internal_classifier

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_classifier_processor
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: enhanced_capability_integrator

**說明**: enhanced_capability_integrator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 69)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: enhanced_capability_integrator[程式組件] → enhanced_capability_integrator[程式組件]

**完整腳本列表**:
1. enhanced_capability_integrator
2. enhanced_capability_integrator

#### 路徑 2 (Flow 161)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → enhanced_capability_integrator[程式組件]

**完整腳本列表**:
1. sync_experiences
2. enhanced_capability_integrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - sync_experiences

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組
- 路徑 2 主要涉及: 認知核心模組, 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: vulnerability_detection

**說明**: vulnerability_detection - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 90)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. vulnerability_detection

#### 路徑 2 (Flow 240)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. vulnerability_detection

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

---

## 終點: ast_trace_comparator

**說明**: ast_trace_comparator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 95)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ast_trace_comparator[程式組件] → ast_trace_comparator[程式組件]

**完整腳本列表**:
1. ast_trace_comparator
2. ast_trace_comparator

#### 路徑 2 (Flow 229)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: external_loop_connector[AI對外能力] → ast_trace_comparator[程式組件]

**完整腳本列表**:
1. external_loop_connector
2. ast_trace_comparator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - external_loop_connector

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組, 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_vector_store

**說明**: unified_vector_store - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 101)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_registry[混合組件] → unified_vector_store[程式組件]

**完整腳本列表**:
1. capability_registry
2. unified_vector_store

#### 路徑 2 (Flow 131)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → unified_vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. unified_vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - capability_registry
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: external_loop_connector

**說明**: external_loop_connector - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 113)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: event_listener[程式組件] → external_loop_connector[AI對外能力]

**完整腳本列表**:
1. event_listener
2. external_loop_connector

#### 路徑 2 (Flow 239)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → external_loop_connector[AI對外能力]

**完整腳本列表**:
1. enhanced_decision_agent
2. external_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - event_listener
- 路徑 2 獨有: 1 個腳本
  - enhanced_decision_agent

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: backends

**說明**: backends - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 115)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: storage_manager[程式組件] → backends[程式組件]

**完整腳本列表**:
1. storage_manager
2. backends

#### 路徑 2 (Flow 260)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → backends[程式組件]

**完整腳本列表**:
1. backends
2. backends

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - storage_manager
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: enhanced_unified_caller

**說明**: enhanced_unified_caller - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 3 - 3 步
- **平均路徑長度**: 3.00 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 124)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: unified_function_caller[程式組件] → unified_function_caller[程式組件] → enhanced_unified_caller[程式組件]

**完整腳本列表**:
1. unified_function_caller
2. unified_function_caller
3. enhanced_unified_caller

#### 路徑 2 (Flow 246)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件] → enhanced_unified_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller
3. enhanced_unified_caller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - task_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組
- 路徑 2 主要涉及: 任務規劃模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: execution_status_monitor

**說明**: execution_status_monitor - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 132)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: mode_manager[程式組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. mode_manager
2. execution_status_monitor

#### 路徑 2 (Flow 210)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: task_executor[程式組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. task_executor
2. execution_status_monitor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - mode_manager
- 路徑 2 獨有: 1 個腳本
  - task_executor

**使用場景差異推測**:

---

## 終點: weight_manager

**說明**: weight_manager - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 133)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: weight_manager[程式組件] → weight_manager[程式組件]

**完整腳本列表**:
1. weight_manager
2. weight_manager

#### 路徑 2 (Flow 202)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: external_loop_connector[AI對外能力] → weight_manager[程式組件]

**完整腳本列表**:
1. external_loop_connector
2. weight_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - external_loop_connector

**使用場景差異推測**:

---

## 終點: logging_formatter

**說明**: logging_formatter - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 核心能力模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 145)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → logging_formatter[程式組件]

**完整腳本列表**:
1. logging_formatter
2. logging_formatter

#### 路徑 2 (Flow 220)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: multilang_coordinator[程式組件] → logging_formatter[程式組件]

**完整腳本列表**:
1. multilang_coordinator
2. logging_formatter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - multilang_coordinator

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組
- 路徑 2 主要涉及: 核心能力模組, 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: postgresql_vector_store

**說明**: postgresql_vector_store - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 155)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: postgresql_vector_store[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. postgresql_vector_store
2. postgresql_vector_store

#### 路徑 2 (Flow 294)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. postgresql_vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - unified_vector_store

**使用場景差異推測**:

---

## 終點: aiva_external_executor

**說明**: aiva_external_executor - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 158)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → aiva_external_executor[程式組件]

**完整腳本列表**:
1. sync_experiences
2. aiva_external_executor

#### 路徑 2 (Flow 363)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: unified_executor_controller[程式組件] → aiva_external_executor[程式組件]

**完整腳本列表**:
1. unified_executor_controller
2. aiva_external_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - sync_experiences
- 路徑 2 獨有: 1 個腳本
  - unified_executor_controller

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 內探模組
- 路徑 2 主要涉及: 內探模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ai_service

**說明**: ai_service - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 164)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: sync_experiences[程式組件] → ai_service[AI組件]

**完整腳本列表**:
1. sync_experiences
2. ai_service

#### 路徑 2 (Flow 304)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: ai_service[AI組件] → ai_service[AI組件]

**完整腳本列表**:
1. ai_service
2. ai_service

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - sync_experiences
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 服務骨幹模組
- 路徑 2 主要涉及: 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ai_manager

**說明**: ai_manager - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 168)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: sync_experiences[程式組件] → ai_manager[AI組件]

**完整腳本列表**:
1. sync_experiences
2. ai_manager

#### 路徑 2 (Flow 387)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: ai_manager[AI組件] → ai_manager[AI組件]

**完整腳本列表**:
1. ai_manager
2. ai_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - sync_experiences
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 服務骨幹模組
- 路徑 2 主要涉及: 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: core_service_coordinator

**說明**: core_service_coordinator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.50 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 196)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: app[程式組件] → core_service_coordinator[程式組件]

**完整腳本列表**:
1. app
2. core_service_coordinator

#### 路徑 2 (Flow 282)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: core_service_coordinator[程式組件] → core_service_coordinator[程式組件] → core_service_coordinator[程式組件]

**完整腳本列表**:
1. core_service_coordinator
2. core_service_coordinator
3. core_service_coordinator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - app
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: ai_capability_query

**說明**: ai_capability_query - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 215)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → ai_capability_query[AI對外能力]

**完整腳本列表**:
1. ai_capability_query
2. ai_capability_query

#### 路徑 2 (Flow 322)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_menu[AI組件] → ai_capability_query[AI對外能力]

**完整腳本列表**:
1. ai_menu
2. ai_capability_query

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - ai_menu

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組
- 路徑 2 主要涉及: 認知核心模組, 核心能力模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_flow_analyzer

**說明**: aiva_flow_analyzer - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 252)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: aiva_flow_analyzer[程式組件] → aiva_flow_analyzer[程式組件]

**完整腳本列表**:
1. aiva_flow_analyzer
2. aiva_flow_analyzer

#### 路徑 2 (Flow 337)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → aiva_flow_analyzer[程式組件]

**完整腳本列表**:
1. core_analyzer
2. aiva_flow_analyzer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - core_analyzer

**使用場景差異推測**:

---

## 終點: module_knowledge_manager

**說明**: module_knowledge_manager - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統), 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 257)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: execution_status_monitor[程式組件] → module_knowledge_manager[程式組件]

**完整腳本列表**:
1. execution_status_monitor
2. module_knowledge_manager

#### 路徑 2 (Flow 335)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: event_listener[程式組件] → module_knowledge_manager[程式組件]

**完整腳本列表**:
1. event_listener
2. module_knowledge_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - execution_status_monitor
- 路徑 2 獨有: 1 個腳本
  - event_listener

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統), 任務規劃模組
- 路徑 2 主要涉及: 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: assistant

**說明**: assistant - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.50 步
- **涉及模組**: 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 279)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: assistant[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. assistant
2. assistant

#### 路徑 2 (Flow 321)

- **長度**: 3 步
- **主要模組**: 核心能力模組
- **執行順序**: ai_menu[AI組件] → assistant[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. ai_menu
2. assistant
3. assistant

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - ai_menu

**使用場景差異推測**:

---

## 終點: trace_recorder

**說明**: trace_recorder - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.50 步
- **涉及模組**: 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 315)

- **長度**: 3 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_tracer[程式組件] → execution_tracer[程式組件] → trace_recorder[程式組件]

**完整腳本列表**:
1. unified_tracer
2. execution_tracer
3. trace_recorder

#### 路徑 2 (Flow 379)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: trace_recorder[程式組件] → trace_recorder[程式組件]

**完整腳本列表**:
1. trace_recorder
2. trace_recorder

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 2 個腳本
  - execution_tracer, unified_tracer
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

