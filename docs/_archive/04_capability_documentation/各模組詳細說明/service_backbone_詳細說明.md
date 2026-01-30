# 服務骨幹 模組詳細說明

## 📊 整體統計

- **總能力數**: 37 個
- **總數據流**: 163 條
- **多路徑能力**: 20 個（有多條不同路徑）
- **單路徑能力**: 17 個（只有一條路徑）
- **平均每能力流數**: 4.4 條

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

## 📋 完整能力清單（37個）


### 1. app
- **完整路徑**: `session_state_manager → app`
- **數據流數量**: 18 條 🔀 多路徑
- **路徑示例** (共18條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\app.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\app.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\app.py`
  - ... 還有 15 條其他路徑

### 2. result_collector
- **完整路徑**: `session_state_manager → result_collector`
- **數據流數量**: 13 條 🔀 多路徑
- **路徑示例** (共13條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\result_collector.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\result_collector.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\result_collector.py`
  - ... 還有 10 條其他路徑

### 3. command_repository
- **完整路徑**: `session_state_manager → command_repository`
- **數據流數量**: 13 條 🔀 多路徑
- **路徑示例** (共13條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`
  - ... 還有 10 條其他路徑

### 4. unified_function_caller
- **完整路徑**: `session_state_manager → unified_function_caller`
- **數據流數量**: 11 條 🔀 多路徑
- **路徑示例** (共11條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller.py`
  - ... 還有 8 條其他路徑

### 5. cli_integration_example
- **完整路徑**: `session_state_manager → cli_integration_example`
- **數據流數量**: 11 條 🔀 多路徑
- **路徑示例** (共11條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\examples\cli_integration_example.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\examples\cli_integration_example.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\examples\cli_integration_example.py`
  - ... 還有 8 條其他路徑

### 6. optimized_core
- **完整路徑**: `session_state_manager → optimized_core`
- **數據流數量**: 10 條 🔀 多路徑
- **路徑示例** (共10條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`
  - ... 還有 7 條其他路徑

### 7. task_dispatcher
- **完整路徑**: `session_state_manager → task_dispatcher`
- **數據流數量**: 10 條 🔀 多路徑
- **路徑示例** (共10條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\task_dispatcher.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\task_dispatcher.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\task_dispatcher.py`
  - ... 還有 7 條其他路徑

### 8. ai_controller
- **完整路徑**: `session_state_manager → ai_controller`
- **數據流數量**: 10 條 🔀 多路徑
- **路徑示例** (共10條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller.py`
  - ... 還有 7 條其他路徑

### 9. backends
- **完整路徑**: `session_state_manager → backends`
- **數據流數量**: 9 條 🔀 多路徑
- **路徑示例** (共9條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py`
  - ... 還有 6 條其他路徑

### 10. protocol_adapter
- **完整路徑**: `session_state_manager → protocol_adapter`
- **數據流數量**: 7 條 🔀 多路徑
- **路徑示例** (共7條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\adapters\protocol_adapter.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\adapters\protocol_adapter.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\adapters\protocol_adapter.py`
  - ... 還有 4 條其他路徑

### 11. message_broker
- **完整路徑**: `session_state_manager → message_broker`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py`
  - ... 還有 3 條其他路徑

### 12. enhanced_unified_caller
- **完整路徑**: `session_state_manager → enhanced_unified_caller`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller.py`
  - ... 還有 3 條其他路徑

### 13. storage_manager
- **完整路徑**: `session_state_manager → storage_manager`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager.py`
  - ... 還有 3 條其他路徑

### 14. command_repository
- **完整路徑**: `scalable_bio_trainer → command_repository`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository.py`

### 15. parallel_processor
- **完整路徑**: `session_state_manager → parallel_processor`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\parallel_processor.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\parallel_processor.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\parallel_processor.py`

### 16. ai_controller
- **完整路徑**: `scalable_bio_trainer → ai_controller`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller.py`

### 17. backends
- **完整路徑**: `scalable_bio_trainer → backends`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py`

### 18. optimized_core
- **完整路徑**: `scalable_bio_trainer → optimized_core`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`

### 19. authz_mapper
- **完整路徑**: `logging_formatter → authz_mapper`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper.py`

### 20. matrix_visualizer
- **完整路徑**: `logging_formatter → matrix_visualizer`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer.py`

### 21. optimized_core
- **完整路徑**: `monitoring → optimized_core`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\monitoring.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core.py`

### 22. permission_matrix
- **完整路徑**: `scalable_bio_trainer → permission_matrix`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py`

### 23. authz_mapper
- **完整路徑**: `scalable_bio_trainer → authz_mapper`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper.py`

### 24. matrix_visualizer
- **完整路徑**: `scalable_bio_trainer → matrix_visualizer`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer.py`

### 25. protocol_adapter
- **完整路徑**: `scalable_bio_trainer → protocol_adapter`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\adapters\protocol_adapter.py`

### 26. message_broker
- **完整路徑**: `scalable_bio_trainer → message_broker`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker.py`

### 27. result_collector
- **完整路徑**: `scalable_bio_trainer → result_collector`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\result_collector.py`

### 28. task_dispatcher
- **完整路徑**: `scalable_bio_trainer → task_dispatcher`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\task_dispatcher.py`

### 29. app
- **完整路徑**: `initial_surface → app`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\initial_surface.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\app.py`

### 30. permission_matrix
- **完整路徑**: `logging_formatter → permission_matrix`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\utils\logging_formatter.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix.py`

### 31. context_manager
- **完整路徑**: `session_state_manager → context_manager`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\context_manager.py`

### 32. unified_memory_manager
- **完整路徑**: `session_state_manager → unified_memory_manager`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\execution_status_monitor.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\unified_memory_manager.py`

### 33. authz_mapper
- **完整路徑**: `session_state_manager → authz_mapper`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper.py`

### 34. matrix_visualizer
- **完整路徑**: `session_state_manager → matrix_visualizer`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer.py`

### 35. core_service_coordinator
- **完整路徑**: `session_state_manager → core_service_coordinator`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\core_service_coordinator.py`

### 36. session_state_manager
- **完整路徑**: `session_state_manager → session_state_manager`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\state\session_state_manager.py`

### 37. db_helper
- **完整路徑**: `websocket_manager → db_helper`
- **數據流數量**: 1 條 ➡️ 單路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\ui\websocket_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\db_helper.py`

---

## ✅ 數據驗證

- 多路徑能力: 20 個
- 單路徑能力: 17 個
- **總計**: 20 + 17 = 37 個能力 ✓

- 多路徑能力的流: 146 條
- 單路徑能力的流: 17 條
- **總計**: 146 + 17 = 163 條數據流 ✓
