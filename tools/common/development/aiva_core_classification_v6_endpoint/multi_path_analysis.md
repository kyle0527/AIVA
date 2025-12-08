# 多路徑終點分析報告

生成時間: 2025-12-08 08:55:22
找到 62 個有多條路徑到達的終點

---

## 終點: ai_model_manager

**說明**: ai_model_manager - 功能組件

- **路徑總數**: 13
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.23 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 7)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. model_trainer
3. ai_model_manager

#### 路徑 2 (Flow 13)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → enhanced_decision_agent[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. enhanced_decision_agent
5. ai_model_manager

#### 路徑 3 (Flow 57)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. train_classifier
4. model_trainer
5. ai_model_manager

#### 路徑 4 (Flow 60)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. model_trainer
4. ai_model_manager

#### 路徑 5 (Flow 63)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. ai_model_manager

#### 路徑 6 (Flow 92)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. ai_model_manager

#### 路徑 7 (Flow 199)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. model_trainer
5. ai_model_manager

#### 路徑 8 (Flow 214)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. ai_model_manager

#### 路徑 9 (Flow 246)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → enhanced_decision_agent[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. enhanced_decision_agent
5. ai_model_manager

#### 路徑 10 (Flow 248)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_model_manager

#### 路徑 11 (Flow 290)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. model_trainer
5. ai_model_manager

#### 路徑 12 (Flow 293)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. ai_model_manager

#### 路徑 13 (Flow 318)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: monitoring[程式組件] → optimized_core[程式組件] → train_classifier[程式組件] → model_trainer[AI組件] → ai_model_manager[AI組件]

**完整腳本列表**:
1. monitoring
2. optimized_core
3. train_classifier
4. model_trainer
5. ai_model_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - model_trainer
- 路徑 2 獨有: 3 個腳本
  - rl_trainers, capability_orchestrator, enhanced_decision_agent

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: model_trainer

**說明**: model_trainer - 功能組件

- **路徑總數**: 12
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.25 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 5)

- **長度**: 2 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. model_trainer

#### 路徑 2 (Flow 11)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → train_classifier[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. train_classifier
5. model_trainer

#### 路徑 3 (Flow 55)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. train_classifier
4. model_trainer

#### 路徑 4 (Flow 58)

- **長度**: 3 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. model_trainer

#### 路徑 5 (Flow 79)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. model_trainer

#### 路徑 6 (Flow 96)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → optimized_core[程式組件] → train_classifier[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. optimized_core
4. train_classifier
5. model_trainer

#### 路徑 7 (Flow 182)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. model_trainer

#### 路徑 8 (Flow 197)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. model_trainer

#### 路徑 9 (Flow 211)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. model_trainer

#### 路徑 10 (Flow 287)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. train_classifier
5. model_trainer

#### 路徑 11 (Flow 288)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. model_trainer

#### 路徑 12 (Flow 316)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: monitoring[程式組件] → optimized_core[程式組件] → train_classifier[程式組件] → model_trainer[AI組件]

**完整腳本列表**:
1. monitoring
2. optimized_core
3. train_classifier
4. model_trainer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 3 個腳本
  - rl_trainers, capability_orchestrator, train_classifier

**使用場景差異推測**:

- 路徑 1 主要涉及: 外學模組
- 路徑 2 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

- 路徑長度差異顯著 (3 步)
- **推測**: 路徑 1 可能是快速路徑或直接調用,路徑 2 可能包含更多處理邏輯

---

## 終點: app

**說明**: app - 功能組件

- **路徑總數**: 12
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 29)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → event_listener[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. event_listener
5. app

#### 路徑 2 (Flow 48)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_status_monitor
4. app

#### 路徑 3 (Flow 120)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → event_listener[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. event_listener
5. app

#### 路徑 4 (Flow 141)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → execution_status_monitor[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. execution_status_monitor
5. app

#### 路徑 5 (Flow 155)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. app

#### 路徑 6 (Flow 166)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → core_service_coordinator[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. core_service_coordinator
5. app

#### 路徑 7 (Flow 172)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → session_state_manager[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. session_state_manager
5. app

#### 路徑 8 (Flow 195)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → dynamic_strategy_adjustment[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. dynamic_strategy_adjustment
5. app

#### 路徑 9 (Flow 234)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scan_module_interface[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scan_module_interface
5. app

#### 路徑 10 (Flow 241)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scan_result_processor[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scan_result_processor
5. app

#### 路徑 11 (Flow 281)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件] → app[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. execution_status_monitor
5. app

#### 路徑 12 (Flow 321)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: initial_surface[程式組件] → scan_module_interface[程式組件] → app[程式組件]

**完整腳本列表**:
1. initial_surface
2. scan_module_interface
3. app

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, event_listener
- 路徑 2 獨有: 1 個腳本
  - execution_status_monitor

**使用場景差異推測**:

---

## 終點: train_classifier

**說明**: train_classifier - 功能組件

- **路徑總數**: 11
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.36 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 10)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. train_classifier

#### 路徑 2 (Flow 17)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. train_classifier

#### 路徑 3 (Flow 39)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. train_classifier

#### 路徑 4 (Flow 54)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. train_classifier

#### 路徑 5 (Flow 95)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → optimized_core[程式組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. optimized_core
4. train_classifier

#### 路徑 6 (Flow 102)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_orchestrator[混合組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_orchestrator
5. train_classifier

#### 路徑 7 (Flow 168)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → optimized_core[程式組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. optimized_core
5. train_classifier

#### 路徑 8 (Flow 210)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. train_classifier

#### 路徑 9 (Flow 264)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. capability_orchestrator
5. train_classifier

#### 路徑 10 (Flow 286)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. train_classifier

#### 路徑 11 (Flow 315)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: monitoring[程式組件] → optimized_core[程式組件] → train_classifier[程式組件]

**完整腳本列表**:
1. monitoring
2. optimized_core
3. train_classifier

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - postgresql_vector_store

**使用場景差異推測**:

---

## 終點: command_repository

**說明**: command_repository - 功能組件

- **路徑總數**: 9
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.56 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 4)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. command_repository

#### 路徑 2 (Flow 16)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. command_repository

#### 路徑 3 (Flow 24)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_capability_query[混合組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. ai_capability_query
5. command_repository

#### 路徑 4 (Flow 38)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. command_repository

#### 路徑 5 (Flow 88)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. command_repository

#### 路徑 6 (Flow 100)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_capability_query[混合組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_capability_query
5. command_repository

#### 路徑 7 (Flow 126)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_analyzer[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_analyzer
5. command_repository

#### 路徑 8 (Flow 259)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store
5. command_repository

#### 路徑 9 (Flow 300)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → command_repository[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store
5. command_repository

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 3 個腳本
  - postgresql_vector_store, rl_trainers, capability_orchestrator

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

- 路徑長度差異顯著 (3 步)
- **推測**: 路徑 1 可能是快速路徑或直接調用,路徑 2 可能包含更多處理邏輯

---

## 終點: real_neural_core

**說明**: real_neural_core - 功能組件

- **路徑總數**: 9
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 3.89 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 65)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. real_neural_core

#### 路徑 2 (Flow 90)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. real_neural_core

#### 路徑 3 (Flow 216)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. real_neural_core

#### 路徑 4 (Flow 249)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. real_neural_core

#### 路徑 5 (Flow 261)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store
5. real_neural_core

#### 路徑 6 (Flow 295)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. real_neural_core

#### 路徑 7 (Flow 302)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store
5. real_neural_core

#### 路徑 8 (Flow 303)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. real_neural_core

#### 路徑 9 (Flow 305)

- **長度**: 2 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → real_neural_core[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. real_neural_core

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - vector_store

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: assistant

**說明**: assistant - 功能組件

- **路徑總數**: 8
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.88 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 19)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. assistant

#### 路徑 2 (Flow 31)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → unified_function_caller[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. unified_function_caller
5. assistant

#### 路徑 3 (Flow 41)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. assistant

#### 路徑 4 (Flow 82)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. assistant

#### 路徑 5 (Flow 161)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → unified_function_caller[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. unified_function_caller
5. assistant

#### 路徑 6 (Flow 185)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. assistant

#### 路徑 7 (Flow 232)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. assistant

#### 路徑 8 (Flow 252)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → knowledge_base[程式組件] → assistant[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. knowledge_base
5. assistant

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - postgresql_vector_store, capability_orchestrator
- 路徑 2 獨有: 2 個腳本
  - aiva_flow_classifier_final, unified_function_caller

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: analysis_engine

**說明**: analysis_engine - 功能組件

- **路徑總數**: 8
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.12 步
- **涉及模組**: 服務骨幹模組, 外學模組, 任務規劃模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 62)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. analysis_engine

#### 路徑 2 (Flow 91)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. analysis_engine

#### 路徑 3 (Flow 116)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. analysis_engine

#### 路徑 4 (Flow 127)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_analyzer[程式組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_analyzer
5. analysis_engine

#### 路徑 5 (Flow 146)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → plan_executor[混合組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. plan_executor
5. analysis_engine

#### 路徑 6 (Flow 213)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. analysis_engine

#### 路徑 7 (Flow 221)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. analysis_engine

#### 路徑 8 (Flow 292)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → analysis_engine[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. analysis_engine

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: training_orchestrator

**說明**: training_orchestrator - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.43 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 6)

- **長度**: 3 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. model_trainer
3. training_orchestrator

#### 路徑 2 (Flow 56)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → train_classifier[程式組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. train_classifier
4. model_trainer
5. training_orchestrator

#### 路徑 3 (Flow 59)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. model_trainer
4. training_orchestrator

#### 路徑 4 (Flow 198)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. model_trainer
5. training_orchestrator

#### 路徑 5 (Flow 220)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. training_orchestrator

#### 路徑 6 (Flow 289)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. model_trainer
5. training_orchestrator

#### 路徑 7 (Flow 317)

- **長度**: 5 步
- **主要模組**: 外學模組
- **執行順序**: monitoring[程式組件] → optimized_core[程式組件] → train_classifier[程式組件] → model_trainer[AI組件] → training_orchestrator[程式組件]

**完整腳本列表**:
1. monitoring
2. optimized_core
3. train_classifier
4. model_trainer
5. training_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - rl_trainers, train_classifier

**使用場景差異推測**:

- 路徑 1 主要涉及: 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: postgresql_vector_store

**說明**: postgresql_vector_store - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.71 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 14)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store

#### 路徑 2 (Flow 36)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store

#### 路徑 3 (Flow 104)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_orchestrator
5. postgresql_vector_store

#### 路徑 4 (Flow 118)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. postgresql_vector_store

#### 路徑 5 (Flow 136)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. postgresql_vector_store

#### 路徑 6 (Flow 266)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. capability_orchestrator
5. postgresql_vector_store

#### 路徑 7 (Flow 275)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. postgresql_vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 1 個腳本
  - aiva_flow_classifier_final

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: backends

**說明**: 後端存儲 - 數據持久化

- **路徑總數**: 7
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.71 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 23)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_capability_query[混合組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. ai_capability_query
5. backends

#### 路徑 2 (Flow 70)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends

#### 路徑 3 (Flow 99)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_capability_query[混合組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_capability_query
5. backends

#### 路徑 4 (Flow 125)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_analyzer[程式組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_analyzer
5. backends

#### 路徑 5 (Flow 173)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends

#### 路徑 6 (Flow 258)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store
5. backends

#### 路徑 7 (Flow 299)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store
5. backends

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, ai_capability_query
- 路徑 2 獨有: 1 個腳本
  - vector_store

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ai_controller

**說明**: ai_controller - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.57 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 32)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. ai_controller

#### 路徑 2 (Flow 68)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. ai_controller

#### 路徑 3 (Flow 133)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. ai_controller

#### 路徑 4 (Flow 163)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_controller

#### 路徑 5 (Flow 257)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store
5. ai_controller

#### 路徑 6 (Flow 272)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. ai_controller

#### 路徑 7 (Flow 298)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → ai_controller[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store
5. ai_controller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - aiva_flow_classifier_final
- 路徑 2 獨有: 1 個腳本
  - vector_store

**使用場景差異推測**:

---

## 終點: bizlogic_attack_executor

**說明**: bizlogic_attack_executor - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.71 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 35)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. bizlogic_attack_executor

#### 路徑 2 (Flow 81)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. bizlogic_attack_executor

#### 路徑 3 (Flow 135)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. bizlogic_attack_executor

#### 路徑 4 (Flow 159)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → enhanced_unified_caller[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. enhanced_unified_caller
5. bizlogic_attack_executor

#### 路徑 5 (Flow 184)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. bizlogic_attack_executor

#### 路徑 6 (Flow 226)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. bizlogic_attack_executor

#### 路徑 7 (Flow 274)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → bizlogic_attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. bizlogic_attack_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - aiva_flow_classifier_final
- 路徑 2 獨有: 2 個腳本
  - backends, vector_store

**使用場景差異推測**:

---

## 終點: ai_commander

**說明**: ai_commander - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.43 步
- **涉及模組**: 服務骨幹模組, 外學模組, 任務規劃模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 46)

- **長度**: 3 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. ai_commander

#### 路徑 2 (Flow 113)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. ai_commander

#### 路徑 3 (Flow 138)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_commander

#### 路徑 4 (Flow 206)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. ai_commander

#### 路徑 5 (Flow 224)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → attack_executor[程式組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. attack_executor
5. ai_commander

#### 路徑 6 (Flow 236)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → two_phase_scan_orchestrator[程式組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. two_phase_scan_orchestrator
5. ai_commander

#### 路徑 7 (Flow 279)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → ai_commander[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. ai_commander

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 3 個腳本
  - neural_network, multilang_coordinator, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: scenario_manager

**說明**: scenario_manager - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 61)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. scenario_manager

#### 路徑 2 (Flow 80)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. scenario_manager

#### 路徑 3 (Flow 183)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. scenario_manager

#### 路徑 4 (Flow 212)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. scenario_manager

#### 路徑 5 (Flow 219)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scenario_manager

#### 路徑 6 (Flow 291)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. scenario_manager

#### 路徑 7 (Flow 313)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: strategy_generator[程式組件] → scenario_manager[程式組件]

**完整腳本列表**:
1. strategy_generator
2. scenario_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - backends, vector_store

**使用場景差異推測**:

---

## 終點: real_bio_net_adapter

**說明**: real_bio_net_adapter - 功能組件

- **路徑總數**: 7
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 64)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. real_bio_net_adapter

#### 路徑 2 (Flow 89)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. real_bio_net_adapter

#### 路徑 3 (Flow 215)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. real_bio_net_adapter

#### 路徑 4 (Flow 260)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store
5. real_bio_net_adapter

#### 路徑 5 (Flow 294)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. real_bio_net_adapter

#### 路徑 6 (Flow 301)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store
5. real_bio_net_adapter

#### 路徑 7 (Flow 304)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → real_bio_net_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. real_bio_net_adapter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 1 個腳本
  - vector_store

**使用場景差異推測**:

---

## 終點: authz_mapper

**說明**: authz_mapper - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 3.33 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 2)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → permission_matrix[程式組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. permission_matrix
3. authz_mapper

#### 路徑 2 (Flow 53)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. authz_mapper

#### 路徑 3 (Flow 209)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. authz_mapper

#### 路徑 4 (Flow 285)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. authz_mapper

#### 路徑 5 (Flow 307)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. logging_formatter
2. authz_mapper

#### 路徑 6 (Flow 310)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → permission_matrix[程式組件] → authz_mapper[程式組件]

**完整腳本列表**:
1. logging_formatter
2. permission_matrix
3. authz_mapper

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - permission_matrix
- 路徑 2 獨有: 1 個腳本
  - rl_trainers

**使用場景差異推測**:

---

## 終點: optimized_core

**說明**: optimized_core - 功能組件

- **路徑總數**: 6
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 50)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件] → unified_memory_manager[程式組件] → optimized_core[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_status_monitor
4. unified_memory_manager
5. optimized_core

#### 路徑 2 (Flow 75)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → optimized_core[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. optimized_core

#### 路徑 3 (Flow 94)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → optimized_core[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. optimized_core

#### 路徑 4 (Flow 167)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → optimized_core[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. optimized_core

#### 路徑 5 (Flow 178)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → optimized_core[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. optimized_core

#### 路徑 6 (Flow 314)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: monitoring[程式組件] → optimized_core[程式組件]

**完整腳本列表**:
1. monitoring
2. optimized_core

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - unified_memory_manager, execution_status_monitor
- 路徑 2 獨有: 2 個腳本
  - backends, vector_store

**使用場景差異推測**:

---

## 終點: enhanced_decision_agent

**說明**: 增強決策代理 - 程式邏輯決策

- **路徑總數**: 5
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.60 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 12)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → enhanced_decision_agent[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. enhanced_decision_agent

#### 路徑 2 (Flow 103)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_orchestrator[混合組件] → enhanced_decision_agent[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_orchestrator
5. enhanced_decision_agent

#### 路徑 3 (Flow 244)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → anti_hallucination_module[程式組件] → enhanced_decision_agent[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. anti_hallucination_module
5. enhanced_decision_agent

#### 路徑 4 (Flow 245)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → enhanced_decision_agent[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. enhanced_decision_agent

#### 路徑 5 (Flow 265)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → enhanced_decision_agent[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. capability_orchestrator
5. enhanced_decision_agent

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

---

## 終點: capability_registry

**說明**: 能力註冊表 - 能力登記管理

- **路徑總數**: 5
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.60 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 25)

- **長度**: 4 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. capability_registry

#### 路徑 2 (Flow 107)

- **長度**: 5 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → internal_loop_connector[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. internal_loop_connector
5. capability_registry

#### 路徑 3 (Flow 109)

- **長度**: 4 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_registry[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_registry

#### 路徑 4 (Flow 130)

- **長度**: 5 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. capability_registry

#### 路徑 5 (Flow 269)

- **長度**: 5 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → capability_registry[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. capability_registry

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, rl_trainers
- 路徑 2 獨有: 3 個腳本
  - neural_network, internal_loop_connector, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_function_caller

**說明**: unified_function_caller - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.60 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 30)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. unified_function_caller

#### 路徑 2 (Flow 132)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. unified_function_caller

#### 路徑 3 (Flow 158)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → enhanced_unified_caller[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. enhanced_unified_caller
5. unified_function_caller

#### 路徑 4 (Flow 160)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. unified_function_caller

#### 路徑 5 (Flow 271)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → unified_function_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. unified_function_caller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: cli_integration_example

**說明**: cli_integration_example - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.60 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 34)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → cli_integration_example[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. cli_integration_example

#### 路徑 2 (Flow 134)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → cli_integration_example[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. cli_integration_example

#### 路徑 3 (Flow 192)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → storage_manager[程式組件] → cli_integration_example[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. storage_manager
5. cli_integration_example

#### 路徑 4 (Flow 193)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → cli_integration_example[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. cli_integration_example

#### 路徑 5 (Flow 273)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → cli_integration_example[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. cli_integration_example

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: execution_planner

**說明**: execution_planner - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.20 步
- **涉及模組**: 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 51)

- **長度**: 3 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_planner[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_planner

#### 路徑 2 (Flow 144)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → plan_executor[混合組件] → execution_planner[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. plan_executor
5. execution_planner

#### 路徑 3 (Flow 149)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → execution_planner[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. execution_planner

#### 路徑 4 (Flow 208)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → execution_planner[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. execution_planner

#### 路徑 5 (Flow 283)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → execution_planner[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. execution_planner

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 3 個腳本
  - neural_network, plan_executor, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 外學模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: payload_generator

**說明**: payload_generator - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 52)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_planner[程式組件] → payload_generator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_planner
4. payload_generator

#### 路徑 2 (Flow 150)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → execution_planner[程式組件] → payload_generator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. execution_planner
5. payload_generator

#### 路徑 3 (Flow 231)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → payload_generator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. payload_generator

#### 路徑 4 (Flow 284)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → execution_planner[程式組件] → payload_generator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. execution_planner
5. payload_generator

#### 路徑 5 (Flow 312)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → payload_generator[程式組件]

**完整腳本列表**:
1. logging_formatter
2. payload_generator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: vector_store

**說明**: vector_store - 功能組件

- **路徑總數**: 5
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.20 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 67)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store

#### 路徑 2 (Flow 218)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. vector_store

#### 路徑 3 (Flow 254)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → knowledge_base[程式組件] → vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. knowledge_base
5. vector_store

#### 路徑 4 (Flow 256)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. vector_store

#### 路徑 5 (Flow 297)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: matrix_visualizer

**說明**: matrix_visualizer - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 4 步
- **平均路徑長度**: 3.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 3)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → permission_matrix[程式組件] → matrix_visualizer[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. permission_matrix
3. matrix_visualizer

#### 路徑 2 (Flow 162)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → matrix_visualizer[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. matrix_visualizer

#### 路徑 3 (Flow 308)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → matrix_visualizer[程式組件]

**完整腳本列表**:
1. logging_formatter
2. matrix_visualizer

#### 路徑 4 (Flow 311)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → permission_matrix[程式組件] → matrix_visualizer[程式組件]

**完整腳本列表**:
1. logging_formatter
2. permission_matrix
3. matrix_visualizer

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - permission_matrix
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: capability_orchestrator

**說明**: 能力編排器 - 功能協調管理

- **路徑總數**: 4
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 9)

- **長度**: 3 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator

#### 路徑 2 (Flow 101)

- **長度**: 4 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_orchestrator[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_orchestrator

#### 路徑 3 (Flow 201)

- **長度**: 5 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. capability_orchestrator

#### 路徑 4 (Flow 263)

- **長度**: 4 步
- **主要模組**: 核心能力模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. capability_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: nlg_system

**說明**: nlg_system - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 15)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → nlg_system[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. nlg_system

#### 路徑 2 (Flow 37)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → nlg_system[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. nlg_system

#### 路徑 3 (Flow 108)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → nlg_system[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. nlg_system

#### 路徑 4 (Flow 306)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → nlg_system[程式組件]

**完整腳本列表**:
1. logging_formatter
2. nlg_system

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 1 個腳本
  - aiva_flow_classifier_final

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: aiva_flow_classifier_final

**說明**: aiva_flow_classifier_final - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 21)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final

#### 路徑 2 (Flow 128)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final

#### 路徑 3 (Flow 202)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. aiva_flow_classifier_final

#### 路徑 4 (Flow 267)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ai_capability_query

**說明**: AI能力查詢器 - 預設指令查詢

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 22)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_capability_query[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. ai_capability_query

#### 路徑 2 (Flow 98)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_capability_query[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_capability_query

#### 路徑 3 (Flow 129)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → ai_capability_query[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. ai_capability_query

#### 路徑 4 (Flow 268)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_capability_query[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. ai_capability_query

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: event_listener

**說明**: event_listener - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 28)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → event_listener[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. event_listener

#### 路徑 2 (Flow 119)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → event_listener[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. event_listener

#### 路徑 3 (Flow 131)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_classifier_final[程式組件] → event_listener[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_classifier_final
5. event_listener

#### 路徑 4 (Flow 270)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → event_listener[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. aiva_flow_classifier_final
5. event_listener

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ai_summary_plugin

**說明**: ai_summary_plugin - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.75 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 33)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → ai_controller[AI組件] → ai_summary_plugin[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. ai_controller
5. ai_summary_plugin

#### 路徑 2 (Flow 69)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → ai_controller[AI組件] → ai_summary_plugin[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. ai_controller
5. ai_summary_plugin

#### 路徑 3 (Flow 164)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_controller[AI組件] → ai_summary_plugin[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_controller
5. ai_summary_plugin

#### 路徑 4 (Flow 238)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_summary_plugin[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_summary_plugin

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - aiva_flow_classifier_final
- 路徑 2 獨有: 1 個腳本
  - vector_store

**使用場景差異推測**:

---

## 終點: execution_status_monitor

**說明**: execution_status_monitor - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 47)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_status_monitor

#### 路徑 2 (Flow 140)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. execution_status_monitor

#### 路徑 3 (Flow 207)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. execution_status_monitor

#### 路徑 4 (Flow 280)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. execution_status_monitor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_memory_manager

**說明**: unified_memory_manager - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.75 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 49)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件] → unified_memory_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. execution_status_monitor
4. unified_memory_manager

#### 路徑 2 (Flow 142)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → execution_status_monitor[程式組件] → unified_memory_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. execution_status_monitor
5. unified_memory_manager

#### 路徑 3 (Flow 239)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ai_summary_plugin[AI組件] → unified_memory_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ai_summary_plugin
5. unified_memory_manager

#### 路徑 4 (Flow 282)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → execution_status_monitor[程式組件] → unified_memory_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. execution_status_monitor
5. unified_memory_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 1 個腳本
  - rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: weight_manager

**說明**: weight_manager - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 66)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → weight_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. weight_manager

#### 路徑 2 (Flow 217)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → weight_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. weight_manager

#### 路徑 3 (Flow 250)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → weight_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. weight_manager

#### 路徑 4 (Flow 296)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → weight_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. weight_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: result_collector

**說明**: result_collector - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.75 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 77)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → result_collector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. result_collector

#### 路徑 2 (Flow 169)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → result_collector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. result_collector

#### 路徑 3 (Flow 180)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → result_collector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. result_collector

#### 路徑 4 (Flow 242)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scan_result_processor[程式組件] → result_collector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scan_result_processor
5. result_collector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: scan_module_interface

**說明**: scan_module_interface - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 83)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → scan_module_interface[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. scan_module_interface

#### 路徑 2 (Flow 186)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → scan_module_interface[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. scan_module_interface

#### 路徑 3 (Flow 233)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scan_module_interface[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scan_module_interface

#### 路徑 4 (Flow 320)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: initial_surface[程式組件] → scan_module_interface[程式組件]

**完整腳本列表**:
1. initial_surface
2. scan_module_interface

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - rl_trainers, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: scan_result_processor

**說明**: scan_result_processor - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.75 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 86)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → scan_result_processor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. scan_result_processor

#### 路徑 2 (Flow 189)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → scan_result_processor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. scan_result_processor

#### 路徑 3 (Flow 237)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → two_phase_scan_orchestrator[程式組件] → scan_result_processor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. two_phase_scan_orchestrator
5. scan_result_processor

#### 路徑 4 (Flow 240)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → scan_result_processor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. scan_result_processor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - rl_trainers, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: skill_graph

**說明**: skill_graph - 功能組件

- **路徑總數**: 4
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.75 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 87)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → skill_graph[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. skill_graph

#### 路徑 2 (Flow 117)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → skill_graph[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. skill_graph

#### 路徑 3 (Flow 190)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → skill_graph[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. skill_graph

#### 路徑 4 (Flow 247)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → skill_graph[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. skill_graph

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 3 個腳本
  - neural_network, multilang_coordinator, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: rl_trainers

**說明**: rl_trainers - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 4 步
- **平均路徑長度**: 3.00 步
- **涉及模組**: 認知核心模組, 外學模組

### 路徑詳細對比

#### 路徑 1 (Flow 8)

- **長度**: 2 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers

#### 路徑 2 (Flow 200)

- **長度**: 4 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers

#### 路徑 3 (Flow 262)

- **長度**: 3 步
- **主要模組**: 外學模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 外學模組
- 路徑 2 主要涉及: 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: attack_validator

**說明**: attack_validator - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 18)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → attack_validator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. attack_validator

#### 路徑 2 (Flow 40)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → attack_validator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. attack_validator

#### 路徑 3 (Flow 225)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → attack_validator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. attack_validator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 1 個腳本
  - aiva_flow_classifier_final

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: unified_vector_store

**說明**: unified_vector_store - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 5 - 5 步
- **平均路徑長度**: 5.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 20)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → capability_orchestrator[混合組件] → postgresql_vector_store[程式組件] → unified_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. capability_orchestrator
4. postgresql_vector_store
5. unified_vector_store

#### 路徑 2 (Flow 42)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → postgresql_vector_store[程式組件] → unified_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. postgresql_vector_store
5. unified_vector_store

#### 路徑 3 (Flow 253)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → knowledge_base[程式組件] → unified_vector_store[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. knowledge_base
5. unified_vector_store

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - capability_orchestrator
- 路徑 2 獨有: 1 個腳本
  - aiva_flow_classifier_final

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: experience_manager

**說明**: experience_manager - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 26)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → capability_registry[混合組件] → experience_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. capability_registry
5. experience_manager

#### 路徑 2 (Flow 110)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_registry[混合組件] → experience_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_registry
5. experience_manager

#### 路徑 3 (Flow 121)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → experience_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. experience_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: check_flow_details

**說明**: check_flow_details - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 43)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → check_flow_details[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. check_flow_details

#### 路徑 2 (Flow 203)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → check_flow_details[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. check_flow_details

#### 路徑 3 (Flow 276)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → check_flow_details[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. check_flow_details

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: find_testable_flows

**說明**: find_testable_flows - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 44)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → find_testable_flows[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. find_testable_flows

#### 路徑 2 (Flow 204)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → find_testable_flows[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. find_testable_flows

#### 路徑 3 (Flow 277)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → find_testable_flows[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. find_testable_flows

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: verify_classification

**說明**: verify_classification - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 3 - 5 步
- **平均路徑長度**: 4.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 45)

- **長度**: 3 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → verify_classification[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. verify_classification

#### 路徑 2 (Flow 205)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → rl_trainers[AI組件] → verify_classification[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. rl_trainers
5. verify_classification

#### 路徑 3 (Flow 278)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_trainers[AI組件] → verify_classification[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_trainers
4. verify_classification

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 0 個腳本
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: external_loop_connector

**說明**: 外部循環連接器 - 系統接口整合

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 71)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → external_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. external_loop_connector

#### 路徑 2 (Flow 105)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → external_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. external_loop_connector

#### 路徑 3 (Flow 174)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → external_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. external_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: internal_loop_connector

**說明**: 內部循環連接器 - 內部API協調

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 72)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → internal_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. internal_loop_connector

#### 路徑 2 (Flow 106)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → internal_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. internal_loop_connector

#### 路徑 3 (Flow 175)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → internal_loop_connector[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. internal_loop_connector

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: plan_executor

**說明**: 計劃執行器 - 任務執行管理

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 73)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → plan_executor[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. plan_executor

#### 路徑 2 (Flow 143)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → plan_executor[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. plan_executor

#### 路徑 3 (Flow 176)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → plan_executor[混合組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. plan_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: protocol_adapter

**說明**: protocol_adapter - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 74)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → protocol_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. protocol_adapter

#### 路徑 2 (Flow 154)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → protocol_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. protocol_adapter

#### 路徑 3 (Flow 177)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → protocol_adapter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. protocol_adapter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: task_dispatcher

**說明**: task_dispatcher - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 78)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → task_dispatcher[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. task_dispatcher

#### 路徑 2 (Flow 170)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → task_dispatcher[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. task_dispatcher

#### 路徑 3 (Flow 181)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → task_dispatcher[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. task_dispatcher

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 2
- 路徑 1 獨有: 3 個腳本
  - rl_trainers, backends, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: two_phase_scan_orchestrator

**說明**: two_phase_scan_orchestrator - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 84)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → two_phase_scan_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. two_phase_scan_orchestrator

#### 路徑 2 (Flow 187)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → two_phase_scan_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. two_phase_scan_orchestrator

#### 路徑 3 (Flow 235)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → two_phase_scan_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. two_phase_scan_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - rl_trainers, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: attack_executor

**說明**: attack_executor - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.67 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 147)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → plan_executor[混合組件] → attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. plan_executor
5. attack_executor

#### 路徑 2 (Flow 223)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. attack_executor

#### 路徑 3 (Flow 228)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → exploit_manager_legacy[程式組件] → attack_executor[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. exploit_manager_legacy
5. attack_executor

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - plan_executor
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: exploit_orchestrator

**說明**: exploit_orchestrator - 功能組件

- **路徑總數**: 3
- **路徑長度範圍**: 2 - 5 步
- **平均路徑長度**: 3.67 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 229)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → exploit_manager_legacy[程式組件] → exploit_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. exploit_manager_legacy
5. exploit_orchestrator

#### 路徑 2 (Flow 230)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → exploit_orchestrator[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. exploit_orchestrator

#### 路徑 3 (Flow 319)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: initial_surface[程式組件] → exploit_orchestrator[程式組件]

**完整腳本列表**:
1. initial_surface
2. exploit_orchestrator

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - exploit_manager_legacy
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: permission_matrix

**說明**: permission_matrix - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 2 - 2 步
- **平均路徑長度**: 2.00 步
- **涉及模組**: 服務骨幹模組, 外學模組

### 路徑詳細對比

#### 路徑 1 (Flow 1)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. permission_matrix

#### 路徑 2 (Flow 309)

- **長度**: 2 步
- **主要模組**: 服務骨幹模組
- **執行順序**: logging_formatter[程式組件] → permission_matrix[程式組件]

**完整腳本列表**:
1. logging_formatter
2. permission_matrix

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 1
- 路徑 1 獨有: 1 個腳本
  - scalable_bio_trainer
- 路徑 2 獨有: 1 個腳本
  - logging_formatter

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 服務骨幹模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: trace_recorder

**說明**: trace_recorder - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 5 - 5 步
- **平均路徑長度**: 5.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 核心能力模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 27)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → aiva_flow_classifier_final[程式組件] → capability_registry[混合組件] → trace_recorder[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. aiva_flow_classifier_final
4. capability_registry
5. trace_recorder

#### 路徑 2 (Flow 111)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → capability_registry[混合組件] → trace_recorder[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. capability_registry
5. trace_recorder

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - aiva_flow_classifier_final, rl_trainers
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 核心能力模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 核心能力模組, 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: message_broker

**說明**: message_broker - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 5 - 5 步
- **平均路徑長度**: 5.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 76)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. message_broker

#### 路徑 2 (Flow 179)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → message_broker[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. message_broker

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - rl_trainers, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: to_functions

**說明**: to_functions - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 5 - 5 步
- **平均路徑長度**: 5.00 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 85)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → rl_trainers[AI組件] → vector_store[程式組件] → backends[程式組件] → to_functions[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. rl_trainers
3. vector_store
4. backends
5. to_functions

#### 路徑 2 (Flow 188)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → backends[程式組件] → to_functions[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. backends
5. to_functions

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 3
- 路徑 1 獨有: 2 個腳本
  - rl_trainers, vector_store
- 路徑 2 獨有: 2 個腳本
  - neural_network, rl_models

**使用場景差異推測**:

- 路徑 1 主要涉及: 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: enhanced_unified_caller

**說明**: enhanced_unified_caller - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 114)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → enhanced_unified_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. enhanced_unified_caller

#### 路徑 2 (Flow 156)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → enhanced_unified_caller[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. enhanced_unified_caller

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - multilang_coordinator
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: storage_manager

**說明**: 存儲管理器 - 存儲資源管理

- **路徑總數**: 2
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 115)

- **長度**: 5 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → multilang_coordinator[程式組件] → storage_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. multilang_coordinator
5. storage_manager

#### 路徑 2 (Flow 191)

- **長度**: 4 步
- **主要模組**: 認知核心模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → storage_manager[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. storage_manager

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - multilang_coordinator
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 認知核心模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

## 終點: ast_parser

**說明**: ast_parser - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 123)

- **長度**: 5 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_analyzer[程式組件] → ast_parser[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_analyzer
5. ast_parser

#### 路徑 2 (Flow 148)

- **長度**: 4 步
- **主要模組**: 服務骨幹模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → ast_parser[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. ast_parser

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - aiva_flow_analyzer
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

---

## 終點: task_converter

**說明**: task_converter - 功能組件

- **路徑總數**: 2
- **路徑長度範圍**: 4 - 5 步
- **平均路徑長度**: 4.50 步
- **涉及模組**: 服務骨幹模組, 任務規劃模組, 外學模組, 認知核心模組

### 路徑詳細對比

#### 路徑 1 (Flow 124)

- **長度**: 5 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → aiva_flow_analyzer[程式組件] → task_converter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. aiva_flow_analyzer
5. task_converter

#### 路徑 2 (Flow 151)

- **長度**: 4 步
- **主要模組**: 任務規劃模組
- **執行順序**: scalable_bio_trainer[AI組件] → neural_network[AI組件] → rl_models[AI組件] → task_converter[程式組件]

**完整腳本列表**:
1. scalable_bio_trainer
2. neural_network
3. rl_models
4. task_converter

### 路徑差異分析

**路徑 1 vs 路徑 2 對比**:

- 共同腳本數: 4
- 路徑 1 獨有: 1 個腳本
  - aiva_flow_analyzer
- 路徑 2 獨有: 0 個腳本

**使用場景差異推測**:

- 路徑 1 主要涉及: 任務規劃模組, 認知核心模組, 服務骨幹模組, 外學模組
- 路徑 2 主要涉及: 任務規劃模組, 認知核心模組, 外學模組
- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景

---

