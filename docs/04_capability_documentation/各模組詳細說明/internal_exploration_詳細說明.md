# 內部探索 模組詳細說明

## 📊 整體統計

- **總能力數**: 19 個
- **總數據流**: 201 條
- **多路徑能力**: 19 個（有多條不同路徑）
- **單路徑能力**: 0 個（只有一條路徑）
- **平均每能力流數**: 10.6 條

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

## 📋 完整能力清單（19個）


### 1. run_analysis
- **完整路徑**: `session_state_manager → run_analysis`
- **數據流數量**: 44 條 🔀 多路徑
- **路徑示例** (共44條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\dashboard.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  - ... 還有 41 條其他路徑

### 2. core_analyzer
- **完整路徑**: `session_state_manager → core_analyzer`
- **數據流數量**: 33 條 🔀 多路徑
- **路徑示例** (共33條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\dashboard.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  - ... 還有 30 條其他路徑

### 3. practical_analyzer
- **完整路徑**: `session_state_manager → practical_analyzer`
- **數據流數量**: 30 條 🔀 多路徑
- **路徑示例** (共30條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  - ... 還有 27 條其他路徑

### 4. aiva_cli_implementation
- **完整路徑**: `session_state_manager → aiva_cli_implementation`
- **數據流數量**: 14 條 🔀 多路徑
- **路徑示例** (共14條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py`
  - ... 還有 11 條其他路徑

### 5. aiva_exploration_pipeline
- **完整路徑**: `session_state_manager → aiva_exploration_pipeline`
- **數據流數量**: 13 條 🔀 多路徑
- **路徑示例** (共13條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py`
  - ... 還有 10 條其他路徑

### 6. analyze_missing_function_connections
- **完整路徑**: `session_state_manager → analyze_missing_function_connections`
- **數據流數量**: 10 條 🔀 多路徑
- **路徑示例** (共10條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\dashboard.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  - ... 還有 7 條其他路徑

### 7. aiva_exploration_pipeline
- **完整路徑**: `scalable_bio_trainer → aiva_exploration_pipeline`
- **數據流數量**: 8 條 🔀 多路徑
- **路徑示例** (共8條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\aiva_exploration_pipeline.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\aiva_exploration_pipeline.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py`
  - ... 還有 5 條其他路徑

### 8. analyze_dataflow_breakpoints
- **完整路徑**: `session_state_manager → analyze_dataflow_breakpoints`
- **數據流數量**: 8 條 🔀 多路徑
- **路徑示例** (共8條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\dashboard.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\improved_ui.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py`
  - ... 還有 5 條其他路徑

### 9. core_analyzer
- **完整路徑**: `scalable_bio_trainer → core_analyzer`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py`
  - ... 還有 3 條其他路徑

### 10. analyze_connection_recommendations
- **完整路徑**: `session_state_manager → analyze_connection_recommendations`
- **數據流數量**: 6 條 🔀 多路徑
- **路徑示例** (共6條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query.py → C:\D\fold7\AIVA-git\services\core\ui\rich_cli.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations.py`
  - ... 還有 3 條其他路徑

### 11. practical_analyzer
- **完整路徑**: `scalable_bio_trainer → practical_analyzer`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer.py`
  - ... 還有 2 條其他路徑

### 12. run_analysis
- **完整路徑**: `scalable_bio_trainer → run_analysis`
- **數據流數量**: 5 條 🔀 多路徑
- **路徑示例** (共5條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis.py`
  - ... 還有 2 條其他路徑

### 13. analyze_missing_function_connections
- **完整路徑**: `scalable_bio_trainer → analyze_missing_function_connections`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections.py`
  - ... 還有 1 條其他路徑

### 14. verify_rl_models
- **完整路徑**: `session_state_manager → verify_rl_models`
- **數據流數量**: 4 條 🔀 多路徑
- **路徑示例** (共4條，顯示前3條):
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`
  3. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`
  - ... 還有 1 條其他路徑

### 15. verify_rl_models
- **完整路徑**: `scalable_bio_trainer → verify_rl_models`
- **數據流數量**: 3 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`
  3. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models.py`

### 16. aiva_cli_implementation
- **完整路徑**: `scalable_bio_trainer → aiva_cli_implementation`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation.py`

### 17. analyze_dataflow_breakpoints
- **完整路徑**: `scalable_bio_trainer → analyze_dataflow_breakpoints`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints.py`

### 18. analyze_results
- **完整路徑**: `scalable_bio_trainer → analyze_results`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_results.py`
  2. `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\scalable_bio_trainer.py → C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_results.py`

### 19. analyze_results
- **完整路徑**: `session_state_manager → analyze_results`
- **數據流數量**: 2 條 🔀 多路徑
- **路徑詳情**:
  1. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_results.py`
  2. `C:\D\fold7\AIVA-git\services\core\session_state_manager.py → C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers.py → C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_results.py`

---

## ✅ 數據驗證

- 多路徑能力: 19 個
- 單路徑能力: 0 個
- **總計**: 19 + 0 = 19 個能力 ✓

- 多路徑能力的流: 201 條
- 單路徑能力的流: 0 條
- **總計**: 201 + 0 = 201 條數據流 ✓
