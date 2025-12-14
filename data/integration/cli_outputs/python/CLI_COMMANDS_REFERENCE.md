# AIVA Core CLI 指令參考手冊

> 生成時間: 2025-12-12 21:11:24
> 來源設定檔: classification_data.json
> 總流程數: 368

## 快速指令索引

此表格列出所有可用流程及其對應的 CLI 執行指令。AI 代理可根據需求檢索此表。

| ID | 任務路徑 (Path) | 主要模組 | CLI 指令 |
|:---:|---|---|---|
| 1 | monitoring -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1` |
| 2 | monitoring -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 2` |
| 3 | monitoring -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 3` |
| 4 | monitoring -> optimized_core -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 4` |
| 5 | monitoring -> optimized_core -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 5` |
| 6 | logging_formatter -> nlg_system | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 6` |
| 7 | logging_formatter -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 7` |
| 8 | logging_formatter -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 8` |
| 9 | logging_formatter -> permission_matrix | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 9` |
| 10 | logging_formatter -> permission_matrix -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 10` |
| 11 | logging_formatter -> permission_matrix -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11` |
| 12 | logging_formatter -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 12` |
| 13 | scalable_bio_trainer -> permission_matrix | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 13` |
| 14 | scalable_bio_trainer -> permission_matrix -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 14` |
| 15 | scalable_bio_trainer -> permission_matrix -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 15` |
| 16 | scalable_bio_trainer -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 16` |
| 17 | scalable_bio_trainer -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 17` |
| 18 | scalable_bio_trainer -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 18` |
| 19 | scalable_bio_trainer -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 19` |
| 20 | scalable_bio_trainer -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 20` |
| 21 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 21` |
| 22 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 22` |
| 23 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 23` |
| 24 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 24` |
| 25 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> enhanced_decision_agent -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 25` |
| 26 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 26` |
| 27 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> nlg_system | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 27` |
| 28 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 28` |
| 29 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 29` |
| 30 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 30` |
| 31 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 31` |
| 32 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 32` |
| 33 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 33` |
| 34 | scalable_bio_trainer -> rl_trainers -> capability_orchestrator -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 34` |
| 35 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 35` |
| 36 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 36` |
| 37 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 37` |
| 38 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 38` |
| 39 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 39` |
| 40 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 40` |
| 41 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 41` |
| 42 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 42` |
| 43 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 43` |
| 44 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 44` |
| 45 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 45` |
| 46 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> event_listener -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 46` |
| 47 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 47` |
| 48 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 48` |
| 49 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 49` |
| 50 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> ai_controller -> ai_summary_plugin | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 50` |
| 51 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 51` |
| 52 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 52` |
| 53 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 53` |
| 54 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 54` |
| 55 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> nlg_system | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 55` |
| 56 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 56` |
| 57 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 57` |
| 58 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 58` |
| 59 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 59` |
| 60 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 60` |
| 61 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 61` |
| 62 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 62` |
| 63 | scalable_bio_trainer -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 63` |
| 64 | scalable_bio_trainer -> rl_trainers -> execution_status_monitor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 64` |
| 65 | scalable_bio_trainer -> rl_trainers -> execution_status_monitor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 65` |
| 66 | scalable_bio_trainer -> rl_trainers -> execution_status_monitor -> unified_memory_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 66` |
| 67 | scalable_bio_trainer -> rl_trainers -> execution_status_monitor -> unified_memory_manager -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 67` |
| 68 | scalable_bio_trainer -> rl_trainers -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 68` |
| 69 | scalable_bio_trainer -> rl_trainers -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 69` |
| 70 | scalable_bio_trainer -> rl_trainers -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 70` |
| 71 | scalable_bio_trainer -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 71` |
| 72 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 72` |
| 73 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 73` |
| 74 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 74` |
| 75 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 75` |
| 76 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 76` |
| 77 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 77` |
| 78 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 78` |
| 79 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 79` |
| 80 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 80` |
| 81 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 81` |
| 82 | scalable_bio_trainer -> rl_trainers -> generate_capability_index | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 82` |
| 83 | scalable_bio_trainer -> rl_trainers -> generate_capability_list | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 83` |
| 84 | scalable_bio_trainer -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 84` |
| 85 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 85` |
| 86 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 86` |
| 87 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 87` |
| 88 | scalable_bio_trainer -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 88` |
| 89 | scalable_bio_trainer -> rl_trainers -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 89` |
| 90 | scalable_bio_trainer -> rl_trainers -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 90` |
| 91 | scalable_bio_trainer -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 91` |
| 92 | scalable_bio_trainer -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 92` |
| 93 | scalable_bio_trainer -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 93` |
| 94 | scalable_bio_trainer -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 94` |
| 95 | scalable_bio_trainer -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 95` |
| 96 | scalable_bio_trainer -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 96` |
| 97 | scalable_bio_trainer -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 97` |
| 98 | scalable_bio_trainer -> rl_trainers -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 98` |
| 99 | scalable_bio_trainer -> rl_trainers -> vector_store -> ai_controller -> ai_summary_plugin | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 99` |
| 100 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 100` |
| 101 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 101` |
| 102 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 102` |
| 103 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 103` |
| 104 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 104` |
| 105 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 105` |
| 106 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 106` |
| 107 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 107` |
| 108 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 108` |
| 109 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 109` |
| 110 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 110` |
| 111 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 111` |
| 112 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 112` |
| 113 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 113` |
| 114 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 114` |
| 115 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 115` |
| 116 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 116` |
| 117 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 117` |
| 118 | scalable_bio_trainer -> rl_trainers -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 118` |
| 119 | scalable_bio_trainer -> rl_trainers -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 119` |
| 120 | scalable_bio_trainer -> rl_trainers -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 120` |
| 121 | scalable_bio_trainer -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 121` |
| 122 | scalable_bio_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 122` |
| 123 | scalable_bio_trainer -> neural_network | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 123` |
| 124 | scalable_bio_trainer -> neural_network -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 124` |
| 125 | scalable_bio_trainer -> neural_network -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 125` |
| 126 | scalable_bio_trainer -> neural_network -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 126` |
| 127 | scalable_bio_trainer -> neural_network -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 127` |
| 128 | scalable_bio_trainer -> neural_network -> rl_models -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 128` |
| 129 | scalable_bio_trainer -> neural_network -> rl_models -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 129` |
| 130 | scalable_bio_trainer -> neural_network -> rl_models -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 130` |
| 131 | scalable_bio_trainer -> neural_network -> rl_models -> capability_orchestrator | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 131` |
| 132 | scalable_bio_trainer -> neural_network -> rl_models -> capability_orchestrator -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 132` |
| 133 | scalable_bio_trainer -> neural_network -> rl_models -> capability_orchestrator -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 133` |
| 134 | scalable_bio_trainer -> neural_network -> rl_models -> capability_orchestrator -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 134` |
| 135 | scalable_bio_trainer -> neural_network -> rl_models -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 135` |
| 136 | scalable_bio_trainer -> neural_network -> rl_models -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 136` |
| 137 | scalable_bio_trainer -> neural_network -> rl_models -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 137` |
| 138 | scalable_bio_trainer -> neural_network -> rl_models -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 138` |
| 139 | scalable_bio_trainer -> neural_network -> rl_models -> nlg_system | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 139` |
| 140 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 140` |
| 141 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 141` |
| 142 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 142` |
| 143 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 143` |
| 144 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 144` |
| 145 | scalable_bio_trainer -> neural_network -> rl_models -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 145` |
| 146 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 146` |
| 147 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 147` |
| 148 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 148` |
| 149 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 149` |
| 150 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 150` |
| 151 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 151` |
| 152 | scalable_bio_trainer -> neural_network -> rl_models -> multilang_coordinator -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 152` |
| 153 | scalable_bio_trainer -> neural_network -> rl_models -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 153` |
| 154 | scalable_bio_trainer -> neural_network -> rl_models -> event_listener -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 154` |
| 155 | scalable_bio_trainer -> neural_network -> rl_models -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 155` |
| 156 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 156` |
| 157 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 157` |
| 158 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 158` |
| 159 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 159` |
| 160 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 160` |
| 161 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 161` |
| 162 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 162` |
| 163 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 163` |
| 164 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 164` |
| 165 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 165` |
| 166 | scalable_bio_trainer -> neural_network -> rl_models -> context_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 166` |
| 167 | scalable_bio_trainer -> neural_network -> rl_models -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 167` |
| 168 | scalable_bio_trainer -> neural_network -> rl_models -> attack_plan_mapper | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 168` |
| 169 | scalable_bio_trainer -> neural_network -> rl_models -> execution_status_monitor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 169` |
| 170 | scalable_bio_trainer -> neural_network -> rl_models -> execution_status_monitor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 170` |
| 171 | scalable_bio_trainer -> neural_network -> rl_models -> execution_status_monitor -> unified_memory_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 171` |
| 172 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 172` |
| 173 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 173` |
| 174 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor -> plan_comparator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 174` |
| 175 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 175` |
| 176 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 176` |
| 177 | scalable_bio_trainer -> neural_network -> rl_models -> plan_executor -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 177` |
| 178 | scalable_bio_trainer -> neural_network -> rl_models -> ast_parser | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 178` |
| 179 | scalable_bio_trainer -> neural_network -> rl_models -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 179` |
| 180 | scalable_bio_trainer -> neural_network -> rl_models -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 180` |
| 181 | scalable_bio_trainer -> neural_network -> rl_models -> task_converter | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 181` |
| 182 | scalable_bio_trainer -> neural_network -> rl_models -> task_generator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 182` |
| 183 | scalable_bio_trainer -> neural_network -> rl_models -> tool_selector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 183` |
| 184 | scalable_bio_trainer -> neural_network -> rl_models -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 184` |
| 185 | scalable_bio_trainer -> neural_network -> rl_models -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 185` |
| 186 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 186` |
| 187 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 187` |
| 188 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 188` |
| 189 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 189` |
| 190 | scalable_bio_trainer -> neural_network -> rl_models -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 190` |
| 191 | scalable_bio_trainer -> neural_network -> rl_models -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 191` |
| 192 | scalable_bio_trainer -> neural_network -> rl_models -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 192` |
| 193 | scalable_bio_trainer -> neural_network -> rl_models -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 193` |
| 194 | scalable_bio_trainer -> neural_network -> rl_models -> ai_controller -> ai_summary_plugin | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 194` |
| 195 | scalable_bio_trainer -> neural_network -> rl_models -> core_service_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 195` |
| 196 | scalable_bio_trainer -> neural_network -> rl_models -> core_service_coordinator -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 196` |
| 197 | scalable_bio_trainer -> neural_network -> rl_models -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 197` |
| 198 | scalable_bio_trainer -> neural_network -> rl_models -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 198` |
| 199 | scalable_bio_trainer -> neural_network -> rl_models -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 199` |
| 200 | scalable_bio_trainer -> neural_network -> rl_models -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 200` |
| 201 | scalable_bio_trainer -> neural_network -> rl_models -> session_state_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 201` |
| 202 | scalable_bio_trainer -> neural_network -> rl_models -> session_state_manager -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 202` |
| 203 | scalable_bio_trainer -> neural_network -> rl_models -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 203` |
| 204 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 204` |
| 205 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 205` |
| 206 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 206` |
| 207 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 207` |
| 208 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 208` |
| 209 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 209` |
| 210 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 210` |
| 211 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 211` |
| 212 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 212` |
| 213 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 213` |
| 214 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 214` |
| 215 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 215` |
| 216 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 216` |
| 217 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 217` |
| 218 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 218` |
| 219 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 219` |
| 220 | scalable_bio_trainer -> neural_network -> rl_models -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 220` |
| 221 | scalable_bio_trainer -> neural_network -> rl_models -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 221` |
| 222 | scalable_bio_trainer -> neural_network -> rl_models -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 222` |
| 223 | scalable_bio_trainer -> neural_network -> rl_models -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 223` |
| 224 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 224` |
| 225 | scalable_bio_trainer -> neural_network -> rl_models -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 225` |
| 226 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 226` |
| 227 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 227` |
| 228 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 228` |
| 229 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 229` |
| 230 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 230` |
| 231 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 231` |
| 232 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_connection_recommendations -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 232` |
| 233 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 233` |
| 234 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 234` |
| 235 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 235` |
| 236 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 236` |
| 237 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 237` |
| 238 | scalable_bio_trainer -> neural_network -> rl_models -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 238` |
| 239 | scalable_bio_trainer -> neural_network -> rl_models -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 239` |
| 240 | scalable_bio_trainer -> neural_network -> rl_models -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 240` |
| 241 | scalable_bio_trainer -> neural_network -> rl_models -> generate_capability_index | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 241` |
| 242 | scalable_bio_trainer -> neural_network -> rl_models -> generate_capability_list | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 242` |
| 243 | scalable_bio_trainer -> neural_network -> rl_models -> dynamic_strategy_adjustment | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 243` |
| 244 | scalable_bio_trainer -> neural_network -> rl_models -> dynamic_strategy_adjustment -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 244` |
| 245 | scalable_bio_trainer -> neural_network -> rl_models -> risk_assessment_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 245` |
| 246 | scalable_bio_trainer -> neural_network -> rl_models -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 246` |
| 247 | scalable_bio_trainer -> neural_network -> rl_models -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 247` |
| 248 | scalable_bio_trainer -> neural_network -> rl_models -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 248` |
| 249 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 249` |
| 250 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> capability_orchestrator | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 250` |
| 251 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 251` |
| 252 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 252` |
| 253 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> execution_status_monitor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 253` |
| 254 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 254` |
| 255 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 255` |
| 256 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 256` |
| 257 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 257` |
| 258 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 258` |
| 259 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 259` |
| 260 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> generate_capability_index | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 260` |
| 261 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> generate_capability_list | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 261` |
| 262 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 262` |
| 263 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 263` |
| 264 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 264` |
| 265 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 265` |
| 266 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 266` |
| 267 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 267` |
| 268 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 268` |
| 269 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 269` |
| 270 | scalable_bio_trainer -> neural_network -> rl_models -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 270` |
| 271 | scalable_bio_trainer -> neural_network -> rl_models -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 271` |
| 272 | scalable_bio_trainer -> neural_network -> rl_models -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 272` |
| 273 | scalable_bio_trainer -> neural_network -> rl_models -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 273` |
| 274 | scalable_bio_trainer -> neural_network -> rl_models -> attack_chain | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 274` |
| 275 | scalable_bio_trainer -> neural_network -> rl_models -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 275` |
| 276 | scalable_bio_trainer -> neural_network -> rl_models -> attack_executor -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 276` |
| 277 | scalable_bio_trainer -> neural_network -> rl_models -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 277` |
| 278 | scalable_bio_trainer -> neural_network -> rl_models -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 278` |
| 279 | scalable_bio_trainer -> neural_network -> rl_models -> exploit_manager_legacy | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 279` |
| 280 | scalable_bio_trainer -> neural_network -> rl_models -> exploit_manager_legacy -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 280` |
| 281 | scalable_bio_trainer -> neural_network -> rl_models -> exploit_manager_legacy -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 281` |
| 282 | scalable_bio_trainer -> neural_network -> rl_models -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 282` |
| 283 | scalable_bio_trainer -> neural_network -> rl_models -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 283` |
| 284 | scalable_bio_trainer -> neural_network -> rl_models -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 284` |
| 285 | scalable_bio_trainer -> neural_network -> rl_models -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 285` |
| 286 | scalable_bio_trainer -> neural_network -> rl_models -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 286` |
| 287 | scalable_bio_trainer -> neural_network -> rl_models -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 287` |
| 288 | scalable_bio_trainer -> neural_network -> rl_models -> two_phase_scan_orchestrator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 288` |
| 289 | scalable_bio_trainer -> neural_network -> rl_models -> two_phase_scan_orchestrator -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 289` |
| 290 | scalable_bio_trainer -> neural_network -> rl_models -> ai_summary_plugin | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 290` |
| 291 | scalable_bio_trainer -> neural_network -> rl_models -> ai_summary_plugin -> unified_memory_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 291` |
| 292 | scalable_bio_trainer -> neural_network -> rl_models -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 292` |
| 293 | scalable_bio_trainer -> neural_network -> rl_models -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 293` |
| 294 | scalable_bio_trainer -> neural_network -> rl_models -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 294` |
| 295 | scalable_bio_trainer -> neural_network -> rl_models -> anti_hallucination_module | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 295` |
| 296 | scalable_bio_trainer -> neural_network -> rl_models -> anti_hallucination_module -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 296` |
| 297 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 297` |
| 298 | scalable_bio_trainer -> neural_network -> rl_models -> enhanced_decision_agent -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 298` |
| 299 | scalable_bio_trainer -> neural_network -> rl_models -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 299` |
| 300 | scalable_bio_trainer -> neural_network -> rl_models -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 300` |
| 301 | scalable_bio_trainer -> neural_network -> rl_models -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 301` |
| 302 | scalable_bio_trainer -> neural_network -> rl_models -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 302` |
| 303 | scalable_bio_trainer -> neural_network -> rl_models -> knowledge_base | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 303` |
| 304 | scalable_bio_trainer -> neural_network -> rl_models -> knowledge_base -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 304` |
| 305 | scalable_bio_trainer -> neural_network -> rl_models -> knowledge_base -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 305` |
| 306 | scalable_bio_trainer -> neural_network -> rl_models -> knowledge_base -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 306` |
| 307 | scalable_bio_trainer -> neural_network -> rl_models -> rag_engine | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 307` |
| 308 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 308` |
| 309 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 309` |
| 310 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 310` |
| 311 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 311` |
| 312 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 312` |
| 313 | scalable_bio_trainer -> neural_network -> rl_models -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 313` |
| 314 | scalable_bio_trainer -> neural_network -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 314` |
| 315 | scalable_bio_trainer -> neural_network -> rl_trainers -> capability_orchestrator | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 315` |
| 316 | scalable_bio_trainer -> neural_network -> rl_trainers -> capability_orchestrator -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 316` |
| 317 | scalable_bio_trainer -> neural_network -> rl_trainers -> capability_orchestrator -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 317` |
| 318 | scalable_bio_trainer -> neural_network -> rl_trainers -> capability_orchestrator -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 318` |
| 319 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 319` |
| 320 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 320` |
| 321 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 321` |
| 322 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 322` |
| 323 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 323` |
| 324 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 324` |
| 325 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 325` |
| 326 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 326` |
| 327 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 327` |
| 328 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 328` |
| 329 | scalable_bio_trainer -> neural_network -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 329` |
| 330 | scalable_bio_trainer -> neural_network -> rl_trainers -> execution_status_monitor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 330` |
| 331 | scalable_bio_trainer -> neural_network -> rl_trainers -> execution_status_monitor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 331` |
| 332 | scalable_bio_trainer -> neural_network -> rl_trainers -> execution_status_monitor -> unified_memory_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 332` |
| 333 | scalable_bio_trainer -> neural_network -> rl_trainers -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 333` |
| 334 | scalable_bio_trainer -> neural_network -> rl_trainers -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 334` |
| 335 | scalable_bio_trainer -> neural_network -> rl_trainers -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 335` |
| 336 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 336` |
| 337 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 337` |
| 338 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 338` |
| 339 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 339` |
| 340 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 340` |
| 341 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 341` |
| 342 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 342` |
| 343 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 343` |
| 344 | scalable_bio_trainer -> neural_network -> rl_trainers -> generate_capability_index | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 344` |
| 345 | scalable_bio_trainer -> neural_network -> rl_trainers -> generate_capability_list | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 345` |
| 346 | scalable_bio_trainer -> neural_network -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 346` |
| 347 | scalable_bio_trainer -> neural_network -> rl_trainers -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 347` |
| 348 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 348` |
| 349 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 349` |
| 350 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 350` |
| 351 | scalable_bio_trainer -> neural_network -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 351` |
| 352 | scalable_bio_trainer -> neural_network -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 352` |
| 353 | scalable_bio_trainer -> neural_network -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 353` |
| 354 | scalable_bio_trainer -> neural_network -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 354` |
| 355 | scalable_bio_trainer -> neural_network -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 355` |
| 356 | scalable_bio_trainer -> neural_network -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 356` |
| 357 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 357` |
| 358 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 358` |
| 359 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 359` |
| 360 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 360` |
| 361 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 361` |
| 362 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 362` |
| 363 | scalable_bio_trainer -> neural_network -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 363` |
| 364 | scalable_bio_trainer -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 364` |
| 365 | scalable_bio_trainer -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 365` |
| 366 | initial_surface -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 366` |
| 367 | initial_surface -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 367` |
| 368 | initial_surface -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 368` |
