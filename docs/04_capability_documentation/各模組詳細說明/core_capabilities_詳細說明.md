# 核心能力 模組詳細說明

## 📊 整體統計

- **總能力數**: 25 個
- **總數據流**: 131 條
- **多路徑能力**: 15 個（有多條不同路徑）
- **單路徑能力**: 10 個（只有一條路徑）
- **平均每能力流數**: 5.2 條

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

## 📋 完整能力清單（25個）


### 1. assistant
- **完整路徑**: `session_state_manager → assistant`
- **數據流數量**: 20 條 🔀 多路徑
- **路徑示例** (共20條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\dialog\assistant.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\dialog\assistant.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\dialog\assistant.py`
  - ... 還有 17 條其他路徑

### 2. bizlogic_attack_executor
- **完整路徑**: `session_state_manager → bizlogic_attack_executor`
- **數據流數量**: 17 條 🔀 多路徑
- **路徑示例** (共17條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor.py`
  - ... 還有 14 條其他路徑

### 3. scan_result_processor
- **完整路徑**: `session_state_manager → scan_result_processor`
- **數據流數量**: 11 條 🔀 多路徑
- **路徑示例** (共11條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py`
  - ... 還有 8 條其他路徑

### 4. analysis_engine
- **完整路徑**: `session_state_manager → analysis_engine`
- **數據流數量**: 11 條 🔀 多路徑
- **路徑示例** (共11條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`
  - ... 還有 8 條其他路徑

### 5. capability_registry
- **完整路徑**: `session_state_manager → capability_registry`
- **數據流數量**: 10 條 🔀 多路徑
- **路徑示例** (共10條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py`
  - ... 還有 7 條其他路徑

### 6. scan_module_interface
- **完整路徑**: `session_state_manager → scan_module_interface`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`
  - ... 還有 4 條其他路徑

### 7. two_phase_scan_orchestrator
- **完整路徑**: `session_state_manager → two_phase_scan_orchestrator`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py`
  - ... 還有 4 條其他路徑

### 8. to_functions
- **完整路徑**: `session_state_manager → to_functions`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\output\to_functions.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\output\to_functions.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\output\to_functions.py`
  - ... 還有 3 條其他路徑

### 9. multilang_coordinator
- **完整路徑**: `session_state_manager → multilang_coordinator`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator.py`
  - ... 還有 3 條其他路徑

### 10. attack_validator
- **完整路徑**: `session_state_manager → attack_validator`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_validator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_validator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_validator.py`
  - ... 還有 3 條其他路徑

### 11. attack_executor
- **完整路徑**: `session_state_manager → attack_executor`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_executor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_executor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_executor.py`
  - ... 還有 3 條其他路徑

### 12. exploit_orchestrator
- **完整路徑**: `session_state_manager → exploit_orchestrator`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py`
  - ... 還有 2 條其他路徑

### 13. payload_generator
- **完整路徑**: `session_state_manager → payload_generator`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator.py`
  - ... 還有 1 條其他路徑

### 14. analysis_engine
- **完整路徑**: `scalable_bio_trainer → analysis_engine`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine.py`

### 15. attack_chain
- **完整路徑**: `session_state_manager → attack_chain`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\command_callback.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_chain.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_chain.py`

### 16. bizlogic_attack_executor
- **完整路徑**: `scalable_bio_trainer → bizlogic_attack_executor`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor.py`

### 17. assistant
- **完整路徑**: `scalable_bio_trainer → assistant`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\dialog\assistant.py`

### 18. scan_module_interface
- **完整路徑**: `scalable_bio_trainer → scan_module_interface`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`

### 19. two_phase_scan_orchestrator
- **完整路徑**: `scalable_bio_trainer → two_phase_scan_orchestrator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator.py`

### 20. to_functions
- **完整路徑**: `scalable_bio_trainer → to_functions`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\output\to_functions.py`

### 21. scan_result_processor
- **完整路徑**: `scalable_bio_trainer → scan_result_processor`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py`

### 22. exploit_orchestrator
- **完整路徑**: `initial_surface → exploit_orchestrator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\initial_surface.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator.py`

### 23. scan_module_interface
- **完整路徑**: `initial_surface → scan_module_interface`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\initial_surface.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py`

### 24. payload_generator
- **完整路徑**: `logging_formatter → payload_generator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator.py`

### 25. exploit_manager_legacy
- **完整路徑**: `session_state_manager → exploit_manager_legacy`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_manager_legacy.py`

---

## ✅ 數據驗證

- 多路徑能力: 15 個
- 單路徑能力: 10 個
- **總計**: 15 + 10 = 25 個能力 ✓

- 多路徑能力的流: 121 條
- 單路徑能力的流: 10 條
- **總計**: 121 + 10 = 131 條數據流 ✓
