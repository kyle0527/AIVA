# 各模組能力分佈詳細分析報告

## 📑 目錄

- [📊 整體統計](#-整體統計)
- [🔹 對內能力詳細](#-對內能力詳細)
  - [各模組詳細](#各模組詳細)
    - [cognitive_core](#cognitive_core)
    - [internal_exploration](#internal_exploration)
    - [service_backbone](#service_backbone)
- [🔸 對外能力詳細](#-對外能力詳細)
  - [各模組詳細](#各模組詳細)
    - [core_capabilities](#core_capabilities)
    - [external_learning](#external_learning)
- [🔶 混合能力詳細](#-混合能力詳細)
    - [task_planning](#task_planning)
- [⚪ 通用組件詳細](#-通用組件詳細)
    - [unknown](#unknown)
- [💡 關鍵洞察](#-關鍵洞察)
  - [1. 路徑冗餘度](#1-路徑冗餘度)
  - [2. 能力豐富度](#2-能力豐富度)
  - [3. 內外能力對比](#3-內外能力對比)
  - [4. 設計理念](#4-設計理念)

---


> **生成時間**: 2026-01-01  
> **分析維度**: 模組 × 能力類型 × 路徑密度

## 📊 整體統計

| 模組 | 類型 | Flows | 能力數 | 路徑密度 |
|------|------|-------|--------|----------|
| internal_exploration | 對內能力 | 201 | 10 | 20.10 |
| service_backbone | 對內能力 | 163 | 22 | 7.41 |
| core_capabilities | 對外能力 | 131 | 15 | 8.73 |
| cognitive_core | 對內能力 | 124 | 18 | 6.89 |
| external_learning | 對外能力 | 99 | 12 | 8.25 |
| unknown | 通用組件 | 74 | 9 | 8.22 |
| task_planning | 混合能力 | 48 | 11 | 4.36 |


## 🔹 對內能力詳細

**包含模組**: cognitive_core, internal_exploration, service_backbone

| 指標 | 數值 |
|------|------|
| 總 Flows | 488 |
| 總能力數 | 50 |
| 平均路徑密度 | 9.76 |

### 各模組詳細


#### cognitive_core

- **Flows**: 124
- **能力數**: 18
- **路徑密度**: 6.89

**Top 5 能力**:

1. `ai_model_manager.py`: 22 flows (17.7%)
2. `real_neural_core.py`: 13 flows (10.5%)
3. `internal_loop_connector.py`: 13 flows (10.5%)
4. `skill_graph.py`: 13 flows (10.5%)
5. `real_bio_net_adapter.py`: 10 flows (8.1%)

#### internal_exploration

- **Flows**: 201
- **能力數**: 10
- **路徑密度**: 20.10

**Top 5 能力**:

1. `run_analysis.py`: 49 flows (24.4%)
2. `core_analyzer.py`: 39 flows (19.4%)
3. `practical_analyzer.py`: 35 flows (17.4%)
4. `aiva_exploration_pipeline.py`: 21 flows (10.4%)
5. `aiva_cli_implementation.py`: 16 flows (8.0%)

#### service_backbone

- **Flows**: 163
- **能力數**: 22
- **路徑密度**: 7.41

**Top 5 能力**:

1. `app.py`: 19 flows (11.7%)
2. `command_repository.py`: 16 flows (9.8%)
3. `result_collector.py`: 14 flows (8.6%)
4. `optimized_core.py`: 13 flows (8.0%)
5. `ai_controller.py`: 12 flows (7.4%)


## 🔸 對外能力詳細

**包含模組**: core_capabilities, external_learning

| 指標 | 數值 |
|------|------|
| 總 Flows | 230 |
| 總能力數 | 27 |
| 平均路徑密度 | 8.52 |

### 各模組詳細


#### core_capabilities

- **Flows**: 131
- **能力數**: 15
- **路徑密度**: 8.73

**Top 5 能力**:

1. `assistant.py`: 21 flows (16.0%)
2. `bizlogic_attack_executor.py`: 18 flows (13.7%)
3. `analysis_engine.py`: 14 flows (10.7%)
4. `scan_result_processor.py`: 12 flows (9.2%)
5. `capability_registry.py`: 10 flows (7.6%)

#### external_learning

- **Flows**: 99
- **能力數**: 12
- **路徑密度**: 8.25

**Top 5 能力**:

1. `model_trainer.py`: 23 flows (23.2%)
2. `train_classifier.py`: 17 flows (17.2%)
3. `training_orchestrator.py`: 15 flows (15.2%)
4. `scenario_manager.py`: 11 flows (11.1%)
5. `experience_manager.py`: 7 flows (7.1%)


## 🔶 混合能力詳細

**包含模組**: task_planning


#### task_planning

- **Flows**: 48
- **能力數**: 11
- **路徑密度**: 4.36

**Top 5 能力**:

1. `ai_commander.py`: 20 flows (41.7%)
2. `plan_executor.py`: 8 flows (16.7%)
3. `task_executor.py`: 5 flows (10.4%)
4. `execution_planner.py`: 5 flows (10.4%)
5. `plan_comparator.py`: 4 flows (8.3%)


## ⚪ 通用組件詳細

**包含模組**: unknown


#### unknown

- **Flows**: 74
- **能力數**: 9
- **路徑密度**: 8.22

**Top 5 能力**:

1. `system_connectivity_checker.py`: 26 flows (35.1%)
2. `server.py`: 11 flows (14.9%)
3. `dashboard.py`: 10 flows (13.5%)
4. `improved_ui.py`: 6 flows (8.1%)
5. `rich_cli.py`: 6 flows (8.1%)


## 💡 關鍵洞察

### 1. 路徑冗餘度

- **最高路徑密度**: internal_exploration (20.10)
  - 每個能力平均有 20.1 條不同路徑
  - 提供高度靈活的路徑選擇

- **最低路徑密度**: task_planning (4.36)
  - 每個能力平均有 4.4 條路徑
  - 路徑較為直接

### 2. 能力豐富度

- **能力最豐富**: service_backbone (22 種能力)
  - 類型: 對內能力
  - 體現模組的多樣性

- **能力最專注**: unknown (9 種能力)
  - 類型: 通用組件
  - 體現模組的專業性

### 3. 內外能力對比

**對內能力**:
- 能力數: 50
- Flows: 488
- 平均密度: 9.76
- 特點: 路徑選擇豐富，靈活性高

**對外能力**:
- 能力數: 27
- Flows: 230
- 平均密度: 8.52
- 特點: 路徑選擇豐富

### 4. 設計理念

1. **內部模組**: 通常有更高的路徑密度，提供靈活性和容錯性
2. **外部模組**: 更專注，路徑更直接，確保執行效率
3. **混合能力**: 在靈活性和效率之間取得平衡
4. **通用組件**: 提供跨模組的共享功能支持

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
