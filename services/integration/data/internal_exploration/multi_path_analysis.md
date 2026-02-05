# 多路徑終點分析報告

生成時間: 2026-02-01 19:43:17
找到 41 個有多條路徑到達的終點

---

## 終點: unified_executor

**說明**: unified_executor - 功能組件

- **路徑總數**: 12
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 認知核心模組(學習子系統), 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 3)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: learning_adapter[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. learning_adapter
2. unified_executor

#### 路徑 2 (Flow 35)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 3 (Flow 42)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 4 (Flow 47)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 5 (Flow 63)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 6 (Flow 102)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: backends[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. backends
2. unified_executor

#### 路徑 7 (Flow 103)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: backends[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. backends
2. unified_executor

#### 路徑 8 (Flow 108)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 9 (Flow 121)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: plan_builder[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. plan_builder
2. unified_executor

#### 路徑 10 (Flow 128)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: scenario_manager[程式組件] → unified_executor[程式組件]

**完整腳本列表**:
1. scenario_manager
2. unified_executor

#### 路徑 11 (Flow 142)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: external_loop_connector[AI對外能力] → unified_executor[程式組件]

**完整腳本列表**:
1. external_loop_connector
2. unified_executor

#### 路徑 12 (Flow 143)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: external_loop_connector[AI對外能力] → unified_executor[程式組件]

**完整腳本列表**:
1. external_loop_connector
2. unified_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - learning_adapter
- 路徑 2 獨有: 1 個腳本
  - scenario_manager

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: vector_store

**說明**: vector_store - 功能組件

- **路徑總數**: 8
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 26)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: assistant[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. assistant
2. vector_store

#### 路徑 2 (Flow 46)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → vector_store[程式組件]

**完整腳本列表**:
1. ai_capability_query
2. vector_store

#### 路徑 3 (Flow 49)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. vector_store

#### 路徑 4 (Flow 55)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: sync_experiences[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. sync_experiences
2. vector_store

#### 路徑 5 (Flow 70)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_executor
2. vector_store

#### 路徑 6 (Flow 132)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: sync_experiences[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. sync_experiences
2. vector_store

#### 路徑 7 (Flow 134)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. vector_store

#### 路徑 8 (Flow 151)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - assistant
- 路徑 2 獨有: 1 個腳本
  - ai_capability_query

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: enhanced_decision_agent

**說明**: enhanced_decision_agent - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 服務骨幹模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 2)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: app[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. app
2. enhanced_decision_agent

#### 路徑 2 (Flow 61)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: scan_result_processor[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. scan_result_processor
2. enhanced_decision_agent

#### 路徑 3 (Flow 76)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: app[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. app
2. enhanced_decision_agent

#### 路徑 4 (Flow 92)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: attack_coordinator[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. attack_coordinator
2. enhanced_decision_agent

#### 路徑 5 (Flow 99)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: scan_result_processor[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. scan_result_processor
2. enhanced_decision_agent

#### 路徑 6 (Flow 107)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: two_phase_scan_orchestrator[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. two_phase_scan_orchestrator
2. enhanced_decision_agent

#### 路徑 7 (Flow 112)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → enhanced_decision_agent[AI對外能力]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. enhanced_decision_agent

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - app
- 路徑 2 獨有: 1 個腳本
  - scan_result_processor

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 認知核心模組
- 路徑 2 主要涉及: 認知核心模組, 核心能力模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_function_caller

**說明**: unified_function_caller - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 服務骨幹模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 1)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller

#### 路徑 2 (Flow 14)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller

#### 路徑 3 (Flow 37)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: assistant[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. assistant
2. unified_function_caller

#### 路徑 4 (Flow 64)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: enhanced_unified_caller[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. enhanced_unified_caller
2. unified_function_caller

#### 路徑 5 (Flow 109)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller

#### 路徑 6 (Flow 160)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: task_executor[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. task_executor
2. unified_function_caller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: base

**說明**: base - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 4)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: cve_identification[程式組件] → base[程式組件]

**完整腳本列表**:
1. cve_identification
2. base

#### 路徑 2 (Flow 5)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vulnerability_detection[程式組件] → base[程式組件]

**完整腳本列表**:
1. vulnerability_detection
2. base

#### 路徑 3 (Flow 28)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → base[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. base

#### 路徑 4 (Flow 50)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → base[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. base

#### 路徑 5 (Flow 152)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → base[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. base

#### 路徑 6 (Flow 168)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vulnerability_detection[程式組件] → base[程式組件]

**完整腳本列表**:
1. vulnerability_detection
2. base

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - cve_identification
- 路徑 2 獨有: 1 個腳本
  - vulnerability_detection

**使用場景差異推測**:

---

## 終點: models

**說明**: models - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 6)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: command_repository[程式組件] → models[AI組件]

**完整腳本列表**:
1. command_repository
2. models

#### 路徑 2 (Flow 32)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → models[AI組件]

**完整腳本列表**:
1. backends
2. models

#### 路徑 3 (Flow 78)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → models[AI組件]

**完整腳本列表**:
1. backends
2. models

#### 路徑 4 (Flow 93)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → models[AI組件]

**完整腳本列表**:
1. backends
2. models

#### 路徑 5 (Flow 117)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: backends[程式組件] → models[AI組件]

**完整腳本列表**:
1. backends
2. models

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - command_repository
- 路徑 2 獨有: 1 個腳本
  - backends

**使用場景差異推測**:

---

## 終點: message_broker

**說明**: message_broker - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 認知核心模組(學習子系統), 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 8)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: plan_executor[AI對外能力] → message_broker[程式組件]

**完整腳本列表**:
1. plan_executor
2. message_broker

#### 路徑 2 (Flow 16)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: dispatcher[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. dispatcher
2. message_broker

#### 路徑 3 (Flow 100)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: event_listener[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. event_listener
2. message_broker

#### 路徑 4 (Flow 127)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: dispatcher_base[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. dispatcher_base
2. message_broker

#### 路徑 5 (Flow 133)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: dispatcher[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. dispatcher
2. message_broker

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - plan_executor
- 路徑 2 獨有: 1 個腳本
  - dispatcher

**使用場景差異推測**:

---

## 終點: core_analyzer

**說明**: core_analyzer - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 31)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_results[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. analyze_results
2. core_analyzer

#### 路徑 2 (Flow 62)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. run_analysis
2. core_analyzer

#### 路徑 3 (Flow 79)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: analyze_missing_function_connections[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. analyze_missing_function_connections
2. core_analyzer

#### 路徑 4 (Flow 90)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. run_analysis
2. core_analyzer

#### 路徑 5 (Flow 157)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: sync_experiences[程式組件] → core_analyzer[程式組件]

**完整腳本列表**:
1. sync_experiences
2. core_analyzer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - analyze_results
- 路徑 2 獨有: 1 個腳本
  - run_analysis

**使用場景差異推測**:

---

## 終點: internal_loop_connector

**說明**: internal_loop_connector - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 19)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. enhanced_decision_agent
2. internal_loop_connector

#### 路徑 2 (Flow 48)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. ai_capability_query
2. internal_loop_connector

#### 路徑 3 (Flow 84)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_registry[混合組件] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. capability_registry
2. internal_loop_connector

#### 路徑 4 (Flow 129)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_orchestrator[AI對外能力] → internal_loop_connector[AI內部能力]

**完整腳本列表**:
1. capability_orchestrator
2. internal_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - ai_capability_query

**使用場景差異推測**:

---

## 終點: vulnerability_detection

**說明**: vulnerability_detection - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 21)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. vulnerability_detection

#### 路徑 2 (Flow 29)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. vulnerability_detection

#### 路徑 3 (Flow 30)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. vulnerability_detection

#### 路徑 4 (Flow 145)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → vulnerability_detection[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. vulnerability_detection

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: cve_identification

**說明**: cve_identification - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 22)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → cve_identification[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. cve_identification

#### 路徑 2 (Flow 67)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → cve_identification[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. cve_identification

#### 路徑 3 (Flow 98)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → cve_identification[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. cve_identification

#### 路徑 4 (Flow 146)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → cve_identification[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. cve_identification

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE

**使用場景差異推測**:

---

## 終點: waf_bypass

**說明**: waf_bypass - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 23)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → waf_bypass[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. waf_bypass

#### 路徑 2 (Flow 80)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → waf_bypass[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. waf_bypass

#### 路徑 3 (Flow 147)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → waf_bypass[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. waf_bypass

#### 路徑 4 (Flow 153)

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
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE

**使用場景差異推測**:

---

## 終點: web_architecture

**說明**: web_architecture - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 24)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → web_architecture[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. web_architecture

#### 路徑 2 (Flow 148)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → web_architecture[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. web_architecture

#### 路徑 3 (Flow 162)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: INTEGRATION_EXAMPLE[程式組件] → web_architecture[程式組件]

**完整腳本列表**:
1. INTEGRATION_EXAMPLE
2. web_architecture

#### 路徑 4 (Flow 167)

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
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - INTEGRATION_EXAMPLE

**使用場景差異推測**:

---

## 終點: knowledge_base

**說明**: knowledge_base - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 27)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: assistant[程式組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. assistant
2. knowledge_base

#### 路徑 2 (Flow 71)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. unified_executor
2. knowledge_base

#### 路徑 3 (Flow 81)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: ai_capability_query[AI對外能力] → knowledge_base[程式組件]

**完整腳本列表**:
1. ai_capability_query
2. knowledge_base

#### 路徑 4 (Flow 83)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: capability_registry[混合組件] → knowledge_base[程式組件]

**完整腳本列表**:
1. capability_registry
2. knowledge_base

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - assistant
- 路徑 2 獨有: 1 個腳本
  - unified_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 核心能力模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: model_trainer

**說明**: model_trainer - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 41)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: continuous_learning[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. continuous_learning
2. model_trainer

#### 路徑 2 (Flow 52)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: external_loop_connector[AI對外能力] → model_trainer[AI組件]

**完整腳本列表**:
1. external_loop_connector
2. model_trainer

#### 路徑 3 (Flow 86)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. ai_model_manager
2. model_trainer

#### 路徑 4 (Flow 116)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_executor[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. unified_executor
2. model_trainer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - continuous_learning
- 路徑 2 獨有: 1 個腳本
  - external_loop_connector

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組(學習子系統), 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_encoder

**說明**: capability_encoder - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 69)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. vector_store
2. capability_encoder

#### 路徑 2 (Flow 88)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. vector_store
2. capability_encoder

#### 路徑 3 (Flow 89)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. vector_store
2. capability_encoder

#### 路徑 4 (Flow 113)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: vector_store[程式組件] → capability_encoder[程式組件]

**完整腳本列表**:
1. vector_store
2. capability_encoder

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: execution_status_monitor

**說明**: execution_status_monitor - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組

### 路徑詳細對比

#### 路徑 1 (Flow 7)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: mode_manager[程式組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. mode_manager
2. execution_status_monitor

#### 路徑 2 (Flow 12)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: task_executor[程式組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. task_executor
2. execution_status_monitor

#### 路徑 3 (Flow 15)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: mode_manager[程式組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. mode_manager
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

## 終點: unified_tracer

**說明**: unified_tracer - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 9)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: plan_executor[AI對外能力] → unified_tracer[程式組件]

**完整腳本列表**:
1. plan_executor
2. unified_tracer

#### 路徑 2 (Flow 118)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: trace_recorder[程式組件] → unified_tracer[程式組件]

**完整腳本列表**:
1. trace_recorder
2. unified_tracer

#### 路徑 3 (Flow 141)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: enhanced_decision_agent[AI對外能力] → unified_tracer[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. unified_tracer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - plan_executor
- 路徑 2 獨有: 1 個腳本
  - trace_recorder

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_registry

**說明**: capability_registry - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 任務規劃模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 13)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: task_executor[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. task_executor
2. capability_registry

#### 路徑 2 (Flow 59)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: skill_graph[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. skill_graph
2. capability_registry

#### 路徑 3 (Flow 130)

- **長度**: 2 步
- **主要模組**: 核心能力模組
- **執行順序**: capability_orchestrator[AI對外能力] → capability_registry[混合組件]

**完整腳本列表**:
1. capability_orchestrator
2. capability_registry

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - task_executor
- 路徑 2 獨有: 1 個腳本
  - skill_graph

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 核心能力模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: real_neural_core

**說明**: real_neural_core - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 18)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → real_neural_core[AI組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. real_neural_core

#### 路徑 2 (Flow 77)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: real_bio_net_adapter[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. real_bio_net_adapter
2. real_neural_core

#### 路徑 3 (Flow 159)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: multilang_coordinator[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. multilang_coordinator
2. real_neural_core

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - real_bio_net_adapter

**使用場景差異推測**:

---

## 終點: experience_manager

**說明**: experience_manager - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 40)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: continuous_learning[程式組件] → experience_manager[程式組件]

**完整腳本列表**:
1. continuous_learning
2. experience_manager

#### 路徑 2 (Flow 87)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → experience_manager[程式組件]

**完整腳本列表**:
1. ai_model_manager
2. experience_manager

#### 路徑 3 (Flow 101)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: unified_executor[程式組件] → experience_manager[程式組件]

**完整腳本列表**:
1. unified_executor
2. experience_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - continuous_learning
- 路徑 2 獨有: 1 個腳本
  - ai_model_manager

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 認知核心模組(學習子系統), 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: permission_matrix

**說明**: permission_matrix - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 服務骨幹模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 44)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: policy_manager[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. policy_manager
2. permission_matrix

#### 路徑 2 (Flow 95)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: policy_manager[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. policy_manager
2. permission_matrix

#### 路徑 3 (Flow 156)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: sync_experiences[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. sync_experiences
2. permission_matrix

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: module_knowledge_manager

**說明**: module_knowledge_manager - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 53)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: event_listener[程式組件] → module_knowledge_manager[程式組件]

**完整腳本列表**:
1. event_listener
2. module_knowledge_manager

#### 路徑 2 (Flow 66)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: execution_status_monitor[程式組件] → module_knowledge_manager[程式組件]

**完整腳本列表**:
1. execution_status_monitor
2. module_knowledge_manager

#### 路徑 3 (Flow 85)

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
  - event_listener
- 路徑 2 獨有: 1 個腳本
  - execution_status_monitor

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統)
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_embedding

**說明**: aiva_embedding - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 97)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_vector_store[程式組件] → aiva_embedding[程式組件]

**完整腳本列表**:
1. unified_vector_store
2. aiva_embedding

#### 路徑 2 (Flow 119)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: real_neural_core[AI組件] → aiva_embedding[程式組件]

**完整腳本列表**:
1. real_neural_core
2. aiva_embedding

#### 路徑 3 (Flow 155)

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
  - unified_vector_store
- 路徑 2 獨有: 1 個腳本
  - real_neural_core

**使用場景差異推測**:

---

## 終點: scalable_bio_trainer

**說明**: scalable_bio_trainer - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 110)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → scalable_bio_trainer[AI內部能力]

**完整腳本列表**:
1. ai_model_manager
2. scalable_bio_trainer

#### 路徑 2 (Flow 149)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → scalable_bio_trainer[AI內部能力]

**完整腳本列表**:
1. ai_model_manager
2. scalable_bio_trainer

#### 路徑 3 (Flow 158)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_model_manager[AI組件] → scalable_bio_trainer[AI內部能力]

**完整腳本列表**:
1. ai_model_manager
2. scalable_bio_trainer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: logging_formatter

**說明**: logging_formatter - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 123)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: multilang_coordinator[程式組件] → logging_formatter[程式組件]

**完整腳本列表**:
1. multilang_coordinator
2. logging_formatter

#### 路徑 2 (Flow 136)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: multilang_coordinator[程式組件] → logging_formatter[程式組件]

**完整腳本列表**:
1. multilang_coordinator
2. logging_formatter

#### 路徑 3 (Flow 154)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: multilang_coordinator[程式組件] → logging_formatter[程式組件]

**完整腳本列表**:
1. multilang_coordinator
2. logging_formatter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: scan_execution_planner

**說明**: scan_execution_planner - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 3 步
- **平均路徑長度**: 2.33 步
- **涉及模組**: 任務規劃模組, 服務骨幹模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 139)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → scan_execution_planner[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. scan_execution_planner

#### 路徑 2 (Flow 140)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → scan_execution_planner[程式組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. scan_execution_planner

#### 路徑 3 (Flow 171)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: core_service_coordinator[程式組件] → task_execution_planner[程式組件] → scan_execution_planner[程式組件]

**完整腳本列表**:
1. core_service_coordinator
2. task_execution_planner
3. scan_execution_planner

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: rag_engine

**說明**: rag_engine - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 17)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → rag_engine[AI組件]

**完整腳本列表**:
1. enhanced_decision_agent
2. rag_engine

#### 路徑 2 (Flow 72)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → rag_engine[AI組件]

**完整腳本列表**:
1. unified_executor
2. rag_engine

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - unified_executor

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: external_loop_connector

**說明**: external_loop_connector - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 20)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: enhanced_decision_agent[AI對外能力] → external_loop_connector[AI對外能力]

**完整腳本列表**:
1. enhanced_decision_agent
2. external_loop_connector

#### 路徑 2 (Flow 45)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: event_listener[程式組件] → external_loop_connector[AI對外能力]

**完整腳本列表**:
1. event_listener
2. external_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - enhanced_decision_agent
- 路徑 2 獨有: 1 個腳本
  - event_listener

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組
- 路徑 2 主要涉及: 認知核心模組(學習子系統), 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: rl_models

**說明**: rl_models - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 25)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: rl_trainers[AI組件] → rl_models[AI內部能力]

**完整腳本列表**:
1. rl_trainers
2. rl_models

#### 路徑 2 (Flow 54)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: rl_trainers[AI組件] → rl_models[AI內部能力]

**完整腳本列表**:
1. rl_trainers
2. rl_models

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: command_router

**說明**: command_router - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 34)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: core_service_coordinator[程式組件] → command_router[程式組件]

**完整腳本列表**:
1. core_service_coordinator
2. command_router

#### 路徑 2 (Flow 169)

- **長度**: 2 步
- **主要模組**: 任務規劃模組
- **執行順序**: core_service_coordinator[程式組件] → command_router[程式組件]

**完整腳本列表**:
1. core_service_coordinator
2. command_router

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: capability_orchestrator

**說明**: capability_orchestrator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 56)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → capability_orchestrator[AI對外能力]

**完整腳本列表**:
1. unified_executor
2. capability_orchestrator

#### 路徑 2 (Flow 57)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: unified_executor[程式組件] → capability_orchestrator[AI對外能力]

**完整腳本列表**:
1. unified_executor
2. capability_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: execution_orchestrator

**說明**: execution_orchestrator - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 任務規劃模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 58)

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

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - unified_executor
- 路徑 2 獨有: 1 個腳本
  - capability_orchestrator

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 認知核心模組
- 路徑 2 主要涉及: 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_internal_executor

**說明**: aiva_internal_executor - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 內探模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 73)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: aiva_cli[程式組件] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. aiva_cli
2. aiva_internal_executor

#### 路徑 2 (Flow 137)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: internal_loop_connector[AI內部能力] → aiva_internal_executor[程式組件]

**完整腳本列表**:
1. internal_loop_connector
2. aiva_internal_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - aiva_cli
- 路徑 2 獨有: 1 個腳本
  - internal_loop_connector

**使用場景差異推測**:

- 路徑 1 主要涉及: 內探模組, 核心能力模組
- 路徑 2 主要涉及: 內探模組, 認知核心模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: rag_trigger

**說明**: rag_trigger - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統), 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 74)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: ai_decision_core[AI組件] → rag_trigger[AI組件]

**完整腳本列表**:
1. ai_decision_core
2. rag_trigger

#### 路徑 2 (Flow 126)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: experience_manager[程式組件] → rag_trigger[AI組件]

**完整腳本列表**:
1. experience_manager
2. rag_trigger

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - ai_decision_core
- 路徑 2 獨有: 1 個腳本
  - experience_manager

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組(學習子系統), 認知核心模組
- 路徑 2 主要涉及: 認知核心模組(學習子系統)
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: analyze_missing_function_connections

**說明**: analyze_missing_function_connections - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 96)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. run_analysis
2. analyze_missing_function_connections

#### 路徑 2 (Flow 165)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → analyze_missing_function_connections[程式組件]

**完整腳本列表**:
1. core_analyzer
2. analyze_missing_function_connections

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - run_analysis
- 路徑 2 獨有: 1 個腳本
  - core_analyzer

**使用場景差異推測**:

---

## 終點: real_bio_net_adapter

**說明**: real_bio_net_adapter - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組, 核心能力模組

### 路徑詳細對比

#### 路徑 1 (Flow 104)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: analysis_engine[程式組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. analysis_engine
2. real_bio_net_adapter

#### 路徑 2 (Flow 105)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: analysis_engine[程式組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. analysis_engine
2. real_bio_net_adapter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: backends

**說明**: backends - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組

### 路徑詳細對比

#### 路徑 1 (Flow 114)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: storage_manager[程式組件] → backends[程式組件]

**完整腳本列表**:
1. storage_manager
2. backends

#### 路徑 2 (Flow 115)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: storage_manager[程式組件] → backends[程式組件]

**完整腳本列表**:
1. storage_manager
2. backends

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: rl_trainers

**說明**: rl_trainers - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 認知核心模組(學習子系統)

### 路徑詳細對比

#### 路徑 1 (Flow 122)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: model_trainer[AI組件] → rl_trainers[AI組件]

**完整腳本列表**:
1. model_trainer
2. rl_trainers

#### 路徑 2 (Flow 138)

- **長度**: 2 步
- **主要模組**: 認知核心模組(學習子系統)
- **執行順序**: model_trainer[AI組件] → rl_trainers[AI組件]

**完整腳本列表**:
1. model_trainer
2. rl_trainers

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: analyze_dataflow_breakpoints

**說明**: analyze_dataflow_breakpoints - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 124)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. run_analysis
2. analyze_dataflow_breakpoints

#### 路徑 2 (Flow 164)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → analyze_dataflow_breakpoints[程式組件]

**完整腳本列表**:
1. core_analyzer
2. analyze_dataflow_breakpoints

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - run_analysis
- 路徑 2 獨有: 1 個腳本
  - core_analyzer

**使用場景差異推測**:

---

## 終點: practical_analyzer

**說明**: practical_analyzer - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 內探模組

### 路徑詳細對比

#### 路徑 1 (Flow 161)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: run_analysis[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. run_analysis
2. practical_analyzer

#### 路徑 2 (Flow 166)

- **長度**: 2 步
- **主要模組**: 內探模組
- **執行順序**: core_analyzer[程式組件] → practical_analyzer[程式組件]

**完整腳本列表**:
1. core_analyzer
2. practical_analyzer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - run_analysis
- 路徑 2 獨有: 1 個腳本
  - core_analyzer

**使用場景差異推測**:

---

