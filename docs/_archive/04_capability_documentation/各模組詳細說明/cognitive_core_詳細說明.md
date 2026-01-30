# 認知核心 模組詳細說明

## 📊 整體統計

- **總能力數**: 29 個
- **總數據流**: 124 條
- **多路徑能力**: 19 個（有多條不同路徑）
- **單路徑能力**: 10 個（只有一條路徑）
- **平均每能力流數**: 4.3 條

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

## 📋 完整能力清單（29個）


### 1. ai_model_manager
- **完整路徑**: `session_state_manager → ai_model_manager`
- **數據流數量**: 13 條 🔀 多路徑
- **路徑示例** (共13條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\enhanced_decision_agent.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  - ... 還有 10 條其他路徑

### 2. internal_loop_connector
- **完整路徑**: `session_state_manager → internal_loop_connector`
- **數據流數量**: 12 條 🔀 多路徑
- **路徑示例** (共12條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py`
  - ... 還有 9 條其他路徑

### 3. skill_graph
- **完整路徑**: `session_state_manager → skill_graph`
- **數據流數量**: 12 條 🔀 多路徑
- **路徑示例** (共12條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\skill_graph.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\skill_graph.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\skill_graph.py`
  - ... 還有 9 條其他路徑

### 4. ai_model_manager
- **完整路徑**: `scalable_bio_trainer → ai_model_manager`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`
  - ... 還有 4 條其他路徑

### 5. external_loop_connector
- **完整路徑**: `session_state_manager → external_loop_connector`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\external_loop_connector.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\external_loop_connector.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\external_loop_connector.py`
  - ... 還有 4 條其他路徑

### 6. real_neural_core
- **完整路徑**: `session_state_manager → real_neural_core`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  - ... 還有 4 條其他路徑

### 7. real_neural_core
- **完整路徑**: `scalable_bio_trainer → real_neural_core`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core.py`
  - ... 還有 3 條其他路徑

### 8. postgresql_vector_store
- **完整路徑**: `session_state_manager → postgresql_vector_store`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py`
  - ... 還有 3 條其他路徑

### 9. test_scope_management
- **完整路徑**: `session_state_manager → test_scope_management`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\test_scope_management.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\test_scope_management.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\test_scope_management.py`
  - ... 還有 3 條其他路徑

### 10. unified_vector_store
- **完整路徑**: `session_state_manager → unified_vector_store`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py`
  - ... 還有 3 條其他路徑

### 11. vector_store
- **完整路徑**: `session_state_manager → vector_store`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py`
  - ... 還有 3 條其他路徑

### 12. real_bio_net_adapter
- **完整路徑**: `scalable_bio_trainer → real_bio_net_adapter`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  - ... 還有 2 條其他路徑

### 13. ai_capability_query
- **完整路徑**: `session_state_manager → ai_capability_query`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py`
  - ... 還有 2 條其他路徑

### 14. real_bio_net_adapter
- **完整路徑**: `session_state_manager → real_bio_net_adapter`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\knowledge_base.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter.py`
  - ... 還有 2 條其他路徑

### 15. enhanced_decision_agent
- **完整路徑**: `session_state_manager → enhanced_decision_agent`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\enhanced_decision_agent.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\anti_hallucination\anti_hallucination_module.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\enhanced_decision_agent.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\enhanced_decision_agent.py`

### 16. weight_manager
- **完整路徑**: `scalable_bio_trainer → weight_manager`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\weight_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\weight_manager.py`

### 17. vector_store
- **完整路徑**: `scalable_bio_trainer → vector_store`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py`

### 18. neural_network
- **完整路徑**: `session_state_manager → neural_network`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py`

### 19. weight_manager
- **完整路徑**: `session_state_manager → weight_manager`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\weight_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\weight_manager.py`

### 20. ai_model_manager
- **完整路徑**: `monitoring → ai_model_manager`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\monitoring.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`

### 21. external_loop_connector
- **完整路徑**: `scalable_bio_trainer → external_loop_connector`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\external_loop_connector.py`

### 22. internal_loop_connector
- **完整路徑**: `scalable_bio_trainer → internal_loop_connector`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py`

### 23. skill_graph
- **完整路徑**: `scalable_bio_trainer → skill_graph`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\skill_graph.py`

### 24. neural_network
- **完整路徑**: `scalable_bio_trainer → neural_network`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py`

### 25. capability_orchestrator
- **完整路徑**: `session_state_manager → capability_orchestrator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py`

### 26. anti_hallucination_module
- **完整路徑**: `session_state_manager → anti_hallucination_module`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\anti_hallucination\anti_hallucination_module.py`

### 27. knowledge_base
- **完整路徑**: `session_state_manager → knowledge_base`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\knowledge_base.py`

### 28. rag_engine
- **完整路徑**: `session_state_manager → rag_engine`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\rag_engine.py`

### 29. ai_model_manager
- **完整路徑**: `websocket_manager → ai_model_manager`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\ui\websocket_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager.py`

---

## ✅ 數據驗證

- 多路徑能力: 19 個
- 單路徑能力: 10 個
- **總計**: 19 + 10 = 29 個能力 ✓

- 多路徑能力的流: 114 條
- 單路徑能力的流: 10 條
- **總計**: 114 + 10 = 124 條數據流 ✓
