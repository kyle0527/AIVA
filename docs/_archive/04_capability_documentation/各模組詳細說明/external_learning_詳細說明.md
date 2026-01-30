# 外部學習 模組詳細說明

## 📊 整體統計

- **總能力數**: 24 個
- **總數據流**: 99 條
- **多路徑能力**: 15 個（有多條不同路徑）
- **單路徑能力**: 9 個（只有一條路徑）
- **平均每能力流數**: 4.1 條

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

## 📋 完整能力清單（24個）


### 1. model_trainer
- **完整路徑**: `session_state_manager → model_trainer`
- **數據流數量**: 14 條 🔀 多路徑
- **路徑示例** (共14條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  - ... 還有 11 條其他路徑

### 2. train_classifier
- **完整路徑**: `session_state_manager → train_classifier`
- **數據流數量**: 12 條 🔀 多路徑
- **路徑示例** (共12條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`
  - ... 還有 9 條其他路徑

### 3. training_orchestrator
- **完整路徑**: `session_state_manager → training_orchestrator`
- **數據流數量**: 9 條 🔀 多路徑
- **路徑示例** (共9條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  - ... 還有 6 條其他路徑

### 4. scenario_manager
- **完整路徑**: `session_state_manager → scenario_manager`
- **數據流數量**: 8 條 🔀 多路徑
- **路徑示例** (共8條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`
  - ... 還有 5 條其他路徑

### 5. model_trainer
- **完整路徑**: `scalable_bio_trainer → model_trainer`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`
  - ... 還有 4 條其他路徑

### 6. experience_manager
- **完整路徑**: `session_state_manager → experience_manager`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py`
  - ... 還有 4 條其他路徑

### 7. trace_recorder
- **完整路徑**: `session_state_manager → trace_recorder`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py`
  - ... 還有 3 條其他路徑

### 8. event_listener
- **完整路徑**: `session_state_manager → event_listener`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\event_listener.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\event_listener.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\event_listener.py`
  - ... 還有 3 條其他路徑

### 9. training_orchestrator
- **完整路徑**: `scalable_bio_trainer → training_orchestrator`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`
  - ... 還有 1 條其他路徑

### 10. rl_models
- **完整路徑**: `session_state_manager → rl_models`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py`
  - ... 還有 1 條其他路徑

### 11. train_classifier
- **完整路徑**: `scalable_bio_trainer → train_classifier`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`

### 12. scenario_manager
- **完整路徑**: `scalable_bio_trainer → scenario_manager`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager.py`

### 13. ast_trace_comparator
- **完整路徑**: `session_state_manager → ast_trace_comparator`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\ast_trace_comparator.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\ast_trace_comparator.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\ast_trace_comparator.py`

### 14. rl_trainers
- **完整路徑**: `scalable_bio_trainer → rl_trainers`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py`

### 15. rl_trainers
- **完整路徑**: `session_state_manager → rl_trainers`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py`

### 16. train_classifier
- **完整路徑**: `monitoring → train_classifier`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\monitoring.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`

### 17. model_trainer
- **完整路徑**: `monitoring → model_trainer`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\monitoring.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`

### 18. training_orchestrator
- **完整路徑**: `monitoring → training_orchestrator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\monitoring.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`

### 19. rl_models
- **完整路徑**: `scalable_bio_trainer → rl_models`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py`

### 20. dynamic_strategy_adjustment
- **完整路徑**: `session_state_manager → dynamic_strategy_adjustment`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\dynamic_strategy_adjustment.py`

### 21. risk_assessment_engine
- **完整路徑**: `session_state_manager → risk_assessment_engine`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\risk_assessment_engine.py`

### 22. train_classifier
- **完整路徑**: `websocket_manager → train_classifier`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\ui\websocket_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py`

### 23. model_trainer
- **完整路徑**: `websocket_manager → model_trainer`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\ui\websocket_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py`

### 24. training_orchestrator
- **完整路徑**: `websocket_manager → training_orchestrator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\ui\websocket_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator.py`

---

## ✅ 數據驗證

- 多路徑能力: 15 個
- 單路徑能力: 9 個
- **總計**: 15 + 9 = 24 個能力 ✓

- 多路徑能力的流: 90 條
- 單路徑能力的流: 9 條
- **總計**: 90 + 9 = 99 條數據流 ✓
