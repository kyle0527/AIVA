# 任務規劃 模組詳細說明

## 📊 整體統計

- **總能力數**: 13 個
- **總數據流**: 48 條
- **多路徑能力**: 6 個（有多條不同路徑）
- **單路徑能力**: 7 個（只有一條路徑）
- **平均每能力流數**: 3.7 條

## 🔍 核心概念說明

### 能力 vs 數據流
- **能力** = 從某個起點到某個終點的功能（例如：從 session_state_manager 到 app）
- **數據流** = 完整的路徑，包含所有中間經過的腳本
- **多路徑能力** = 同一個能力有多條不同的執行路徑

### 為什麼會有多路徑？
因為從起點到終點可能有不同的方式：
- 經過不同的中間模組
- 使用不同的處理邏輯
- 適應不同的場景需求

---

## 📋 完整能力清單（13個）


### 1. ai_commander
- **完整路徑**: `session_state_manager → ai_commander`
- **數據流數量**: 17 條 🔀 多路徑
- **路徑示例** (共17條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`
  - ... 還有 14 條其他路徑

### 2. plan_executor
- **完整路徑**: `session_state_manager → plan_executor`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py`
  - ... 還有 4 條其他路徑

### 3. task_executor
- **完整路徑**: `session_state_manager → task_executor`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\task_executor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\task_executor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\task_executor.py`
  - ... 還有 2 條其他路徑

### 4. execution_planner
- **完整路徑**: `session_state_manager → execution_planner`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py`
  - ... 還有 2 條其他路徑

### 5. plan_comparator
- **完整路徑**: `session_state_manager → plan_comparator`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\plan_comparator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\plan_comparator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\plan_comparator.py`
  - ... 還有 1 條其他路徑

### 6. ai_commander
- **完整路徑**: `scalable_bio_trainer → ai_commander`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander.py`

### 7. plan_executor
- **完整路徑**: `scalable_bio_trainer → plan_executor`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py`

### 8. attack_plan_mapper
- **完整路徑**: `session_state_manager → attack_plan_mapper`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\attack_plan_mapper.py`

### 9. execution_status_monitor
- **完整路徑**: `session_state_manager → execution_status_monitor`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\execution_status_monitor.py`

### 10. ast_parser
- **完整路徑**: `session_state_manager → ast_parser`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\ast_parser.py`

### 11. task_converter
- **完整路徑**: `session_state_manager → task_converter`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\task_converter.py`

### 12. task_generator
- **完整路徑**: `session_state_manager → task_generator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\task_generator.py`

### 13. tool_selector
- **完整路徑**: `session_state_manager → tool_selector`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\tool_selector.py`

---

## ✅ 數據驗證

- 多路徑能力: 6 個
- 單路徑能力: 7 個
- **總計**: 6 + 7 = 13 個能力 ✓

- 多路徑能力的流: 41 條
- 單路徑能力的流: 7 條
- **總計**: 41 + 7 = 48 條數據流 ✓
