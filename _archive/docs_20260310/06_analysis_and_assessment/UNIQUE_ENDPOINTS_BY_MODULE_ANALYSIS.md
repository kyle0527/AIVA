# 各模組獨特終點能力分析

## 📑 目錄

- [分析目的](#分析目的)
- [核心概念](#核心概念)
- [各模組統計](#各模組統計)
  - [cognitive_core](#cognitive_core)
  - [task_planning](#task_planning)
  - [external_learning](#external_learning)
  - [core_capabilities](#core_capabilities)
  - [service_backbone](#service_backbone)
- [整體統計](#整體統計)
- [路徑最多的終點 (Top 10)](#路徑最多的終點-top-10)
- [關鍵發現](#關鍵發現)
- [結論](#結論)

---


**分析日期**: 2026-01-01

## 分析目的

扣除多路徑到達同一終點的重複計算後，統計各模組真正的獨特終點能力數量，
並分析這些獨特能力的對內/對外特徵。

## 核心概念

- **獨特終點**: 不同的終點腳本（無論有多少條路徑到達）
- **多路徑終點**: 有多條不同起點路徑到達的終點
- **單路徑終點**: 只有一條固定路徑的終點
- **對內能力**: 起點和終點在同一模組內
- **對外能力**: 起點和終點在不同模組

## 各模組統計

### cognitive_core

- **總 Flows 數**: 85
- **獨特終點總數**: 9
  - 多路徑終點: 5 (55.6%)
  - 單路徑終點: 4 (44.4%)

- **對內能力**: 9 (100.0%)
- **對外能力**: 0 (0.0%)

**對內能力列表** (9 個):

- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\ai_capability_query` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\enhanced_decision_agent` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager` (4 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\neural_network` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_neural_core` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\rag_engine` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\ai_controller` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\storage_manager` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\ai_commander` (2 條路徑)

### task_planning

- **總 Flows 數**: 60
- **獨特終點總數**: 10
  - 多路徑終點: 3 (30.0%)
  - 單路徑終點: 7 (70.0%)

- **對內能力**: 10 (100.0%)
- **對外能力**: 0 (0.0%)

**對內能力列表** (10 個):

- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_executor` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\bizlogic_attack_executor` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\task_dispatcher` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\attack_plan_mapper` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\plan_executor` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\task_executor` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\execution_planner` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\plan_comparator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\task_converter` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\task_generator` (1 條路徑)

### external_learning

- **總 Flows 數**: 54
- **獨特終點總數**: 5
  - 多路徑終點: 5 (100.0%)
  - 單路徑終點: 0 (0.0%)

- **對內能力**: 5 (100.0%)
- **對外能力**: 0 (0.0%)

**對內能力列表** (5 個):

- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer` (4 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_models` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\rl_trainers` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator` (4 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\verify_rl_models` (2 條路徑)

### core_capabilities

- **總 Flows 數**: 13
- **獨特終點總數**: 3
  - 多路徑終點: 0 (0.0%)
  - 單路徑終點: 3 (100.0%)

- **對內能力**: 3 (100.0%)
- **對外能力**: 0 (0.0%)

**對內能力列表** (3 個):

- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\capability_orchestrator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_chain` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry` (1 條路徑)

### service_backbone

- **總 Flows 數**: 628
- **獨特終點總數**: 71
  - 多路徑終點: 36 (50.7%)
  - 單路徑終點: 35 (49.3%)

- **對內能力**: 71 (100.0%)
- **對外能力**: 0 (0.0%)

**對內能力列表** (71 個):

- `C:\D\fold7\AIVA-git\services\core\aiva_core\__init__` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\anti_hallucination\anti_hallucination_module` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\decision\skill_graph` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\external_loop_connector` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\internal_loop_connector` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\real_bio_net_adapter` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\weight_manager` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\knowledge_base` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\postgresql_vector_store` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\unified_vector_store` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\rag\vector_store` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\test_scope_management` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\analysis\analysis_engine` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\attack_validator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_manager_legacy` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\exploit_orchestrator` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\attack\payload_generator` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\dialog\assistant` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface` (3 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\multilang_coordinator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\orchestration\two_phase_scan_orchestrator` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\output\to_functions` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\processing\scan_result_processor` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier` (4 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\ast_trace_comparator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\dynamic_strategy_adjustment` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\analysis\risk_assessment_engine` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\event_listener` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\experience_manager` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\tracing\trace_recorder` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\scenario_manager` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\aiva_exploration_pipeline` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_cli_implementation` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_connection_recommendations` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_dataflow_breakpoints` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_missing_function_connections` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\analyze_results` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\core_analyzer` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\practical_analyzer` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\self_healing\run_analysis` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\adapters\protocol_adapter` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\app` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\enhanced_unified_caller` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\api\unified_function_caller` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper` (3 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer` (3 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\context_manager` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\core_service_coordinator` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core` (3 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\message_broker` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\messaging\result_collector` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\parallel_processor` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\performance\unified_memory_manager` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\state\session_state_manager` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\backends` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\command_repository` (2 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\db_helper` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\storage\examples\cli_integration_example` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\executor\execution_status_monitor` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\ast_parser` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\aiva_core\task_planning\planner\tool_selector` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\tools\system_connectivity_checker` (3 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\auto_server` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\command_callback` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\dashboard` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\improved_ui` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\rich_cli` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\server` (1 條路徑)
- `C:\D\fold7\AIVA-git\services\core\ui\server_v3` (1 條路徑)

## 整體統計

- **總 Flows 數**: 840
- **獨特終點總數**: 98
  - 多路徑終點: 49 (50.0%)
  - 單路徑終點: 49 (50.0%)

- **對內能力總數**: 98 (100.0%)
- **對外能力總數**: 0 (0.0%)

- **平均每終點路徑數**: 1.63

## 路徑最多的終點 (Top 10)

| 排名 | 終點名稱 | 路徑數 | 所屬模組 |
|------|----------|--------|----------|
| 1 | `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\ai_model\train_classifier` | 4 | service_backbone |
| 2 | `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\learning\model_trainer` | 4 | external_learning |
| 3 | `C:\D\fold7\AIVA-git\services\core\aiva_core\external_learning\training\training_orchestrator` | 4 | external_learning |
| 4 | `C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core\neural\ai_model_manager` | 4 | cognitive_core |
| 5 | `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\coordination\optimized_core` | 3 | service_backbone |
| 6 | `C:\D\fold7\AIVA-git\services\core\tools\system_connectivity_checker` | 3 | service_backbone |
| 7 | `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\authz_mapper` | 3 | service_backbone |
| 8 | `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\matrix_visualizer` | 3 | service_backbone |
| 9 | `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\ingestion\scan_module_interface` | 3 | service_backbone |
| 10 | `C:\D\fold7\AIVA-git\services\core\aiva_core\service_backbone\authz\permission_matrix` | 2 | service_backbone |

## 關鍵發現

1. **終點複用率**: 49/98 = 50.0%
   - 意義: 超過一半的能力終點被多條路徑使用，顯示系統高度模組化

2. **獨特終點分布**:
   - 對內能力占 100.0%
   - 對外能力占 0.0%
   - 說明: 多數獨特終點服務於模組內部流程

3. **單路徑終點** (49 個):
   - 這些是真正「獨特」的能力，只有一條固定路徑
   - 可能是專用功能或入口點

4. **多路徑終點** (49 個):
   - 這些能力提供多種訪問方式
   - 提高系統彈性和容錯能力
   - 平均每個多路徑終點有 2.3 條路徑

## 結論

1. AIVA 系統共有 **98 個獨特終點能力**
2. 這些能力通過 **840 條 flows** 被調用
3. **50.0%** 的終點提供多種訪問路徑
4. 對內能力 (100.0%) 多於對外能力 (0.0%)
5. 系統設計強調**模組內聚**和**路徑冗餘**
