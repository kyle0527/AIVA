# AIVA Core CLI 指令參考手冊

> 生成時間: 2025-12-25 09:50:12
> 來源設定檔: latest_classification.json
> 總流程數: 840

## 快速指令索引

此表格列出所有可用流程及其對應的 CLI 執行指令。AI 代理可根據需求檢索此表。

| ID | 任務路徑 (Path) | 主要模組 | CLI 指令 |
|:---:|---|---|---|
| 1 | monitoring -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1` |
| 2 | monitoring -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 2` |
| 3 | monitoring -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 3` |
| 4 | monitoring -> optimized_core -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 4` |
| 5 | monitoring -> optimized_core -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 5` |
| 6 | scalable_bio_trainer -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 6` |
| 7 | scalable_bio_trainer -> permission_matrix | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 7` |
| 8 | scalable_bio_trainer -> permission_matrix -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 8` |
| 9 | scalable_bio_trainer -> permission_matrix -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 9` |
| 10 | scalable_bio_trainer -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 10` |
| 11 | scalable_bio_trainer -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11` |
| 12 | scalable_bio_trainer -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 12` |
| 13 | scalable_bio_trainer -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 13` |
| 14 | scalable_bio_trainer -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 14` |
| 15 | scalable_bio_trainer -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 15` |
| 16 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 16` |
| 17 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 17` |
| 18 | scalable_bio_trainer -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 18` |
| 19 | scalable_bio_trainer -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 19` |
| 20 | scalable_bio_trainer -> rl_trainers -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 20` |
| 21 | scalable_bio_trainer -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 21` |
| 22 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 22` |
| 23 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 23` |
| 24 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 24` |
| 25 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 25` |
| 26 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 26` |
| 27 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 27` |
| 28 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 28` |
| 29 | scalable_bio_trainer -> rl_trainers -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 29` |
| 30 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 30` |
| 31 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 31` |
| 32 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 32` |
| 33 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 33` |
| 34 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 34` |
| 35 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 35` |
| 36 | scalable_bio_trainer -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 36` |
| 37 | scalable_bio_trainer -> rl_trainers -> analyze_results | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 37` |
| 38 | scalable_bio_trainer -> rl_trainers -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 38` |
| 39 | scalable_bio_trainer -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 39` |
| 40 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 40` |
| 41 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 41` |
| 42 | scalable_bio_trainer -> rl_trainers -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 42` |
| 43 | scalable_bio_trainer -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 43` |
| 44 | scalable_bio_trainer -> rl_trainers -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 44` |
| 45 | scalable_bio_trainer -> rl_trainers -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 45` |
| 46 | scalable_bio_trainer -> rl_trainers -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 46` |
| 47 | scalable_bio_trainer -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 47` |
| 48 | scalable_bio_trainer -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 48` |
| 49 | scalable_bio_trainer -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 49` |
| 50 | scalable_bio_trainer -> rl_trainers -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 50` |
| 51 | scalable_bio_trainer -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 51` |
| 52 | scalable_bio_trainer -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 52` |
| 53 | scalable_bio_trainer -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 53` |
| 54 | scalable_bio_trainer -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 54` |
| 55 | scalable_bio_trainer -> rl_trainers -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 55` |
| 56 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 56` |
| 57 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 57` |
| 58 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 58` |
| 59 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 59` |
| 60 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 60` |
| 61 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 61` |
| 62 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 62` |
| 63 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 63` |
| 64 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 64` |
| 65 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 65` |
| 66 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 66` |
| 67 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 67` |
| 68 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 68` |
| 69 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 69` |
| 70 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 70` |
| 71 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 71` |
| 72 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 72` |
| 73 | scalable_bio_trainer -> rl_trainers -> vector_store -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 73` |
| 74 | scalable_bio_trainer -> rl_trainers -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 74` |
| 75 | scalable_bio_trainer -> rl_trainers -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 75` |
| 76 | scalable_bio_trainer -> rl_trainers -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 76` |
| 77 | scalable_bio_trainer -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 77` |
| 78 | scalable_bio_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 78` |
| 79 | scalable_bio_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 79` |
| 80 | scalable_bio_trainer -> neural_network | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 80` |
| 81 | scalable_bio_trainer -> neural_network -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 81` |
| 82 | scalable_bio_trainer -> neural_network -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 82` |
| 83 | scalable_bio_trainer -> neural_network -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 83` |
| 84 | scalable_bio_trainer -> neural_network -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 84` |
| 85 | scalable_bio_trainer -> neural_network -> rl_models -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 85` |
| 86 | scalable_bio_trainer -> neural_network -> rl_models -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 86` |
| 87 | scalable_bio_trainer -> neural_network -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 87` |
| 88 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 88` |
| 89 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 89` |
| 90 | scalable_bio_trainer -> neural_network -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 90` |
| 91 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 91` |
| 92 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 92` |
| 93 | scalable_bio_trainer -> neural_network -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 93` |
| 94 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 94` |
| 95 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 95` |
| 96 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 96` |
| 97 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 97` |
| 98 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 98` |
| 99 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 99` |
| 100 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 100` |
| 101 | scalable_bio_trainer -> neural_network -> rl_trainers -> analyze_results | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 101` |
| 102 | scalable_bio_trainer -> neural_network -> rl_trainers -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 102` |
| 103 | scalable_bio_trainer -> neural_network -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 103` |
| 104 | scalable_bio_trainer -> neural_network -> rl_trainers -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 104` |
| 105 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 105` |
| 106 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 106` |
| 107 | scalable_bio_trainer -> neural_network -> rl_trainers -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 107` |
| 108 | scalable_bio_trainer -> neural_network -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 108` |
| 109 | scalable_bio_trainer -> neural_network -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 109` |
| 110 | scalable_bio_trainer -> neural_network -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 110` |
| 111 | scalable_bio_trainer -> neural_network -> rl_trainers -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 111` |
| 112 | scalable_bio_trainer -> neural_network -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 112` |
| 113 | scalable_bio_trainer -> neural_network -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 113` |
| 114 | scalable_bio_trainer -> neural_network -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 114` |
| 115 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 115` |
| 116 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 116` |
| 117 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 117` |
| 118 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 118` |
| 119 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 119` |
| 120 | scalable_bio_trainer -> neural_network -> rl_trainers -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 120` |
| 121 | scalable_bio_trainer -> neural_network -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 121` |
| 122 | scalable_bio_trainer -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 122` |
| 123 | scalable_bio_trainer -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 123` |
| 124 | initial_surface -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 124` |
| 125 | initial_surface -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 125` |
| 126 | initial_surface -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 126` |
| 127 | logging_formatter -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 127` |
| 128 | logging_formatter -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 128` |
| 129 | logging_formatter -> permission_matrix | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 129` |
| 130 | logging_formatter -> permission_matrix -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 130` |
| 131 | logging_formatter -> permission_matrix -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 131` |
| 132 | logging_formatter -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 132` |
| 133 | session_state_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 133` |
| 134 | session_state_manager -> command_callback | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 134` |
| 135 | session_state_manager -> command_callback -> attack_chain | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 135` |
| 136 | session_state_manager -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 136` |
| 137 | session_state_manager -> dashboard -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 137` |
| 138 | session_state_manager -> dashboard -> improved_ui -> __init__ | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 138` |
| 139 | session_state_manager -> dashboard -> improved_ui -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 139` |
| 140 | session_state_manager -> dashboard -> improved_ui -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 140` |
| 141 | session_state_manager -> dashboard -> improved_ui -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 141` |
| 142 | session_state_manager -> dashboard -> improved_ui -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 142` |
| 143 | session_state_manager -> dashboard -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 143` |
| 144 | session_state_manager -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 144` |
| 145 | session_state_manager -> improved_ui -> __init__ | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 145` |
| 146 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 146` |
| 147 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 147` |
| 148 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 148` |
| 149 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 149` |
| 150 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 150` |
| 151 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 151` |
| 152 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 152` |
| 153 | session_state_manager -> improved_ui -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 153` |
| 154 | session_state_manager -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 154` |
| 155 | session_state_manager -> rich_cli -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 155` |
| 156 | session_state_manager -> rich_cli -> auto_server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 156` |
| 157 | session_state_manager -> rich_cli -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 157` |
| 158 | session_state_manager -> rich_cli -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 158` |
| 159 | session_state_manager -> rich_cli -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 159` |
| 160 | session_state_manager -> rich_cli -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 160` |
| 161 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 161` |
| 162 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 162` |
| 163 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 163` |
| 164 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 164` |
| 165 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 165` |
| 166 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 166` |
| 167 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 167` |
| 168 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 168` |
| 169 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 169` |
| 170 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 170` |
| 171 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 171` |
| 172 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 172` |
| 173 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 173` |
| 174 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 174` |
| 175 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 175` |
| 176 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 176` |
| 177 | session_state_manager -> rich_cli -> ai_capability_query -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 177` |
| 178 | session_state_manager -> rich_cli -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 178` |
| 179 | session_state_manager -> rich_cli -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 179` |
| 180 | session_state_manager -> rich_cli -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 180` |
| 181 | session_state_manager -> rich_cli -> capability_registry -> experience_manager -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 181` |
| 182 | session_state_manager -> rich_cli -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 182` |
| 183 | session_state_manager -> rich_cli -> capability_registry -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 183` |
| 184 | session_state_manager -> rich_cli -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 184` |
| 185 | session_state_manager -> rich_cli -> capability_registry -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 185` |
| 186 | session_state_manager -> rich_cli -> capability_registry -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 186` |
| 187 | session_state_manager -> rich_cli -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 187` |
| 188 | session_state_manager -> rich_cli -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 188` |
| 189 | session_state_manager -> rich_cli -> capability_registry -> trace_recorder -> ast_trace_comparator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 189` |
| 190 | session_state_manager -> rich_cli -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 190` |
| 191 | session_state_manager -> rich_cli -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 191` |
| 192 | session_state_manager -> rich_cli -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 192` |
| 193 | session_state_manager -> rich_cli -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 193` |
| 194 | session_state_manager -> rich_cli -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 194` |
| 195 | session_state_manager -> rich_cli -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 195` |
| 196 | session_state_manager -> rich_cli -> enhanced_unified_caller -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 196` |
| 197 | session_state_manager -> rich_cli -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 197` |
| 198 | session_state_manager -> rich_cli -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 198` |
| 199 | session_state_manager -> rich_cli -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 199` |
| 200 | session_state_manager -> rich_cli -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 200` |
| 201 | session_state_manager -> rich_cli -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 201` |
| 202 | session_state_manager -> rich_cli -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 202` |
| 203 | session_state_manager -> rich_cli -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 203` |
| 204 | session_state_manager -> rich_cli -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 204` |
| 205 | session_state_manager -> rich_cli -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 205` |
| 206 | session_state_manager -> rich_cli -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 206` |
| 207 | session_state_manager -> rich_cli -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 207` |
| 208 | session_state_manager -> rich_cli -> postgresql_vector_store -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 208` |
| 209 | session_state_manager -> rich_cli -> postgresql_vector_store -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 209` |
| 210 | session_state_manager -> rich_cli -> postgresql_vector_store -> dashboard -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 210` |
| 211 | session_state_manager -> rich_cli -> postgresql_vector_store -> dashboard -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 211` |
| 212 | session_state_manager -> rich_cli -> postgresql_vector_store -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 212` |
| 213 | session_state_manager -> rich_cli -> postgresql_vector_store -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 213` |
| 214 | session_state_manager -> rich_cli -> postgresql_vector_store -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 214` |
| 215 | session_state_manager -> rich_cli -> postgresql_vector_store -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 215` |
| 216 | session_state_manager -> rich_cli -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 216` |
| 217 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 217` |
| 218 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 218` |
| 219 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 219` |
| 220 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 220` |
| 221 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 221` |
| 222 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 222` |
| 223 | session_state_manager -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 223` |
| 224 | session_state_manager -> rich_cli -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 224` |
| 225 | session_state_manager -> rich_cli -> postgresql_vector_store -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 225` |
| 226 | session_state_manager -> rich_cli -> postgresql_vector_store -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 226` |
| 227 | session_state_manager -> rich_cli -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 227` |
| 228 | session_state_manager -> rich_cli -> postgresql_vector_store -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 228` |
| 229 | session_state_manager -> rich_cli -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 229` |
| 230 | session_state_manager -> rich_cli -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 230` |
| 231 | session_state_manager -> rich_cli -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 231` |
| 232 | session_state_manager -> rich_cli -> postgresql_vector_store -> unified_vector_store -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 232` |
| 233 | session_state_manager -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 233` |
| 234 | session_state_manager -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 234` |
| 235 | session_state_manager -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 235` |
| 236 | session_state_manager -> ai_capability_query -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 236` |
| 237 | session_state_manager -> ai_capability_query -> rich_cli -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 237` |
| 238 | session_state_manager -> ai_capability_query -> rich_cli -> auto_server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 238` |
| 239 | session_state_manager -> ai_capability_query -> rich_cli -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 239` |
| 240 | session_state_manager -> ai_capability_query -> rich_cli -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 240` |
| 241 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 241` |
| 242 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 242` |
| 243 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 243` |
| 244 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 244` |
| 245 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 245` |
| 246 | session_state_manager -> ai_capability_query -> rich_cli -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 246` |
| 247 | session_state_manager -> ai_capability_query -> rich_cli -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 247` |
| 248 | session_state_manager -> ai_capability_query -> rich_cli -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 248` |
| 249 | session_state_manager -> ai_capability_query -> rich_cli -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 249` |
| 250 | session_state_manager -> ai_capability_query -> rich_cli -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 250` |
| 251 | session_state_manager -> ai_capability_query -> rich_cli -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 251` |
| 252 | session_state_manager -> ai_capability_query -> rich_cli -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 252` |
| 253 | session_state_manager -> ai_capability_query -> rich_cli -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 253` |
| 254 | session_state_manager -> ai_capability_query -> rich_cli -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 254` |
| 255 | session_state_manager -> ai_capability_query -> rich_cli -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 255` |
| 256 | session_state_manager -> ai_capability_query -> rich_cli -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 256` |
| 257 | session_state_manager -> ai_capability_query -> rich_cli -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 257` |
| 258 | session_state_manager -> ai_capability_query -> rich_cli -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 258` |
| 259 | session_state_manager -> ai_capability_query -> rich_cli -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 259` |
| 260 | session_state_manager -> ai_capability_query -> rich_cli -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 260` |
| 261 | session_state_manager -> ai_capability_query -> rich_cli -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 261` |
| 262 | session_state_manager -> ai_capability_query -> rich_cli -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 262` |
| 263 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 263` |
| 264 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 264` |
| 265 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 265` |
| 266 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 266` |
| 267 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 267` |
| 268 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 268` |
| 269 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 269` |
| 270 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 270` |
| 271 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 271` |
| 272 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 272` |
| 273 | session_state_manager -> ai_capability_query -> rich_cli -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 273` |
| 274 | session_state_manager -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 274` |
| 275 | session_state_manager -> ai_capability_query -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 275` |
| 276 | session_state_manager -> ai_capability_query -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 276` |
| 277 | session_state_manager -> ai_capability_query -> backends -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 277` |
| 278 | session_state_manager -> ai_capability_query -> backends -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 278` |
| 279 | session_state_manager -> ai_capability_query -> backends -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 279` |
| 280 | session_state_manager -> ai_capability_query -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 280` |
| 281 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 281` |
| 282 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 282` |
| 283 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> plan_comparator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 283` |
| 284 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 284` |
| 285 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 285` |
| 286 | session_state_manager -> ai_capability_query -> backends -> plan_executor -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 286` |
| 287 | session_state_manager -> ai_capability_query -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 287` |
| 288 | session_state_manager -> ai_capability_query -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 288` |
| 289 | session_state_manager -> ai_capability_query -> backends -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 289` |
| 290 | session_state_manager -> ai_capability_query -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 290` |
| 291 | session_state_manager -> ai_capability_query -> backends -> message_broker -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 291` |
| 292 | session_state_manager -> ai_capability_query -> backends -> message_broker -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 292` |
| 293 | session_state_manager -> ai_capability_query -> backends -> message_broker -> parallel_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 293` |
| 294 | session_state_manager -> ai_capability_query -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 294` |
| 295 | session_state_manager -> ai_capability_query -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 295` |
| 296 | session_state_manager -> ai_capability_query -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 296` |
| 297 | session_state_manager -> ai_capability_query -> backends -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 297` |
| 298 | session_state_manager -> ai_capability_query -> backends -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 298` |
| 299 | session_state_manager -> ai_capability_query -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 299` |
| 300 | session_state_manager -> ai_capability_query -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 300` |
| 301 | session_state_manager -> ai_capability_query -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 301` |
| 302 | session_state_manager -> ai_capability_query -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 302` |
| 303 | session_state_manager -> ai_capability_query -> backends -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 303` |
| 304 | session_state_manager -> ai_capability_query -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 304` |
| 305 | session_state_manager -> ai_capability_query -> backends -> two_phase_scan_orchestrator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 305` |
| 306 | session_state_manager -> ai_capability_query -> backends -> two_phase_scan_orchestrator -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 306` |
| 307 | session_state_manager -> ai_capability_query -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 307` |
| 308 | session_state_manager -> ai_capability_query -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 308` |
| 309 | session_state_manager -> ai_capability_query -> backends -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 309` |
| 310 | session_state_manager -> ai_capability_query -> backends -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 310` |
| 311 | session_state_manager -> ai_capability_query -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 311` |
| 312 | session_state_manager -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 312` |
| 313 | session_state_manager -> capability_orchestrator | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 313` |
| 314 | session_state_manager -> capability_orchestrator -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 314` |
| 315 | session_state_manager -> capability_orchestrator -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 315` |
| 316 | session_state_manager -> capability_orchestrator -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 316` |
| 317 | session_state_manager -> capability_orchestrator -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 317` |
| 318 | session_state_manager -> capability_orchestrator -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 318` |
| 319 | session_state_manager -> capability_orchestrator -> enhanced_decision_agent -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 319` |
| 320 | session_state_manager -> capability_orchestrator -> enhanced_decision_agent -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 320` |
| 321 | session_state_manager -> capability_orchestrator -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 321` |
| 322 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 322` |
| 323 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 323` |
| 324 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> dashboard -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 324` |
| 325 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> dashboard -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 325` |
| 326 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 326` |
| 327 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 327` |
| 328 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 328` |
| 329 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 329` |
| 330 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 330` |
| 331 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 331` |
| 332 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 332` |
| 333 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 333` |
| 334 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 334` |
| 335 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 335` |
| 336 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 336` |
| 337 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> analyze_connection_recommendations -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 337` |
| 338 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 338` |
| 339 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 339` |
| 340 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 340` |
| 341 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 341` |
| 342 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 342` |
| 343 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 343` |
| 344 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 344` |
| 345 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 345` |
| 346 | session_state_manager -> capability_orchestrator -> postgresql_vector_store -> unified_vector_store -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 346` |
| 347 | session_state_manager -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 347` |
| 348 | session_state_manager -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 348` |
| 349 | session_state_manager -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 349` |
| 350 | session_state_manager -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 350` |
| 351 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 351` |
| 352 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 352` |
| 353 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> auto_server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 353` |
| 354 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 354` |
| 355 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 355` |
| 356 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 356` |
| 357 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 357` |
| 358 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 358` |
| 359 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 359` |
| 360 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 360` |
| 361 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 361` |
| 362 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 362` |
| 363 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 363` |
| 364 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 364` |
| 365 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 365` |
| 366 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 366` |
| 367 | session_state_manager -> internal_loop_connector -> capability_registry -> rich_cli -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 367` |
| 368 | session_state_manager -> internal_loop_connector -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 368` |
| 369 | session_state_manager -> internal_loop_connector -> capability_registry -> experience_manager -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 369` |
| 370 | session_state_manager -> internal_loop_connector -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 370` |
| 371 | session_state_manager -> internal_loop_connector -> capability_registry -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 371` |
| 372 | session_state_manager -> internal_loop_connector -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 372` |
| 373 | session_state_manager -> internal_loop_connector -> capability_registry -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 373` |
| 374 | session_state_manager -> internal_loop_connector -> capability_registry -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 374` |
| 375 | session_state_manager -> internal_loop_connector -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 375` |
| 376 | session_state_manager -> internal_loop_connector -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 376` |
| 377 | session_state_manager -> internal_loop_connector -> capability_registry -> trace_recorder -> ast_trace_comparator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 377` |
| 378 | session_state_manager -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 378` |
| 379 | session_state_manager -> internal_loop_connector -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 379` |
| 380 | session_state_manager -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 380` |
| 381 | session_state_manager -> capability_registry -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 381` |
| 382 | session_state_manager -> capability_registry -> rich_cli -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 382` |
| 383 | session_state_manager -> capability_registry -> rich_cli -> auto_server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 383` |
| 384 | session_state_manager -> capability_registry -> rich_cli -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 384` |
| 385 | session_state_manager -> capability_registry -> rich_cli -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 385` |
| 386 | session_state_manager -> capability_registry -> rich_cli -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 386` |
| 387 | session_state_manager -> capability_registry -> rich_cli -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 387` |
| 388 | session_state_manager -> capability_registry -> rich_cli -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 388` |
| 389 | session_state_manager -> capability_registry -> rich_cli -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 389` |
| 390 | session_state_manager -> capability_registry -> rich_cli -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 390` |
| 391 | session_state_manager -> capability_registry -> rich_cli -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 391` |
| 392 | session_state_manager -> capability_registry -> rich_cli -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 392` |
| 393 | session_state_manager -> capability_registry -> rich_cli -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 393` |
| 394 | session_state_manager -> capability_registry -> rich_cli -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 394` |
| 395 | session_state_manager -> capability_registry -> rich_cli -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 395` |
| 396 | session_state_manager -> capability_registry -> rich_cli -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 396` |
| 397 | session_state_manager -> capability_registry -> rich_cli -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 397` |
| 398 | session_state_manager -> capability_registry -> rich_cli -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 398` |
| 399 | session_state_manager -> capability_registry -> rich_cli -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 399` |
| 400 | session_state_manager -> capability_registry -> rich_cli -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 400` |
| 401 | session_state_manager -> capability_registry -> rich_cli -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 401` |
| 402 | session_state_manager -> capability_registry -> rich_cli -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 402` |
| 403 | session_state_manager -> capability_registry -> rich_cli -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 403` |
| 404 | session_state_manager -> capability_registry -> rich_cli -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 404` |
| 405 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 405` |
| 406 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 406` |
| 407 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 407` |
| 408 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 408` |
| 409 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 409` |
| 410 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 410` |
| 411 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 411` |
| 412 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 412` |
| 413 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 413` |
| 414 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 414` |
| 415 | session_state_manager -> capability_registry -> rich_cli -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 415` |
| 416 | session_state_manager -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 416` |
| 417 | session_state_manager -> capability_registry -> experience_manager -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 417` |
| 418 | session_state_manager -> capability_registry -> experience_manager -> rl_models -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 418` |
| 419 | session_state_manager -> capability_registry -> experience_manager -> rl_models -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 419` |
| 420 | session_state_manager -> capability_registry -> experience_manager -> rl_models -> neural_network | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 420` |
| 421 | session_state_manager -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 421` |
| 422 | session_state_manager -> capability_registry -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 422` |
| 423 | session_state_manager -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 423` |
| 424 | session_state_manager -> capability_registry -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 424` |
| 425 | session_state_manager -> capability_registry -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 425` |
| 426 | session_state_manager -> capability_registry -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 426` |
| 427 | session_state_manager -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 427` |
| 428 | session_state_manager -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 428` |
| 429 | session_state_manager -> capability_registry -> trace_recorder -> ast_trace_comparator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 429` |
| 430 | session_state_manager -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 430` |
| 431 | session_state_manager -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 431` |
| 432 | session_state_manager -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 432` |
| 433 | session_state_manager -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 433` |
| 434 | session_state_manager -> experience_manager -> rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 434` |
| 435 | session_state_manager -> experience_manager -> rl_models -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 435` |
| 436 | session_state_manager -> experience_manager -> rl_models -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 436` |
| 437 | session_state_manager -> experience_manager -> rl_models -> neural_network | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 437` |
| 438 | session_state_manager -> experience_manager -> rl_models -> neural_network -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 438` |
| 439 | session_state_manager -> experience_manager -> rl_models -> neural_network -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 439` |
| 440 | session_state_manager -> experience_manager -> rl_models -> neural_network -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 440` |
| 441 | session_state_manager -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 441` |
| 442 | session_state_manager -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 442` |
| 443 | session_state_manager -> context_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 443` |
| 444 | session_state_manager -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 444` |
| 445 | session_state_manager -> attack_plan_mapper | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 445` |
| 446 | session_state_manager -> execution_status_monitor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 446` |
| 447 | session_state_manager -> execution_status_monitor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 447` |
| 448 | session_state_manager -> execution_status_monitor -> unified_memory_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 448` |
| 449 | session_state_manager -> execution_status_monitor -> unified_memory_manager -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 449` |
| 450 | session_state_manager -> execution_status_monitor -> unified_memory_manager -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 450` |
| 451 | session_state_manager -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 451` |
| 452 | session_state_manager -> plan_executor -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 452` |
| 453 | session_state_manager -> plan_executor -> dashboard -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 453` |
| 454 | session_state_manager -> plan_executor -> dashboard -> improved_ui -> __init__ | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 454` |
| 455 | session_state_manager -> plan_executor -> dashboard -> improved_ui -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 455` |
| 456 | session_state_manager -> plan_executor -> dashboard -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 456` |
| 457 | session_state_manager -> plan_executor -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 457` |
| 458 | session_state_manager -> plan_executor -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 458` |
| 459 | session_state_manager -> plan_executor -> plan_comparator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 459` |
| 460 | session_state_manager -> plan_executor -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 460` |
| 461 | session_state_manager -> plan_executor -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 461` |
| 462 | session_state_manager -> plan_executor -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 462` |
| 463 | session_state_manager -> plan_executor -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 463` |
| 464 | session_state_manager -> plan_executor -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 464` |
| 465 | session_state_manager -> plan_executor -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 465` |
| 466 | session_state_manager -> plan_executor -> attack_executor -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 466` |
| 467 | session_state_manager -> ast_parser | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 467` |
| 468 | session_state_manager -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 468` |
| 469 | session_state_manager -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 469` |
| 470 | session_state_manager -> task_converter | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 470` |
| 471 | session_state_manager -> task_converter -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 471` |
| 472 | session_state_manager -> task_converter -> rich_cli -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 472` |
| 473 | session_state_manager -> task_converter -> rich_cli -> auto_server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 473` |
| 474 | session_state_manager -> task_converter -> rich_cli -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 474` |
| 475 | session_state_manager -> task_converter -> rich_cli -> server_v3 | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 475` |
| 476 | session_state_manager -> task_converter -> rich_cli -> ai_capability_query | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 476` |
| 477 | session_state_manager -> task_converter -> rich_cli -> ai_capability_query -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 477` |
| 478 | session_state_manager -> task_converter -> rich_cli -> ai_capability_query -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 478` |
| 479 | session_state_manager -> task_converter -> rich_cli -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 479` |
| 480 | session_state_manager -> task_converter -> rich_cli -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 480` |
| 481 | session_state_manager -> task_converter -> rich_cli -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 481` |
| 482 | session_state_manager -> task_converter -> rich_cli -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 482` |
| 483 | session_state_manager -> task_converter -> rich_cli -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 483` |
| 484 | session_state_manager -> task_converter -> rich_cli -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 484` |
| 485 | session_state_manager -> task_converter -> rich_cli -> multilang_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 485` |
| 486 | session_state_manager -> task_converter -> rich_cli -> multilang_coordinator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 486` |
| 487 | session_state_manager -> task_converter -> rich_cli -> event_listener | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 487` |
| 488 | session_state_manager -> task_converter -> rich_cli -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 488` |
| 489 | session_state_manager -> task_converter -> rich_cli -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 489` |
| 490 | session_state_manager -> task_converter -> rich_cli -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 490` |
| 491 | session_state_manager -> task_converter -> rich_cli -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 491` |
| 492 | session_state_manager -> task_converter -> rich_cli -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 492` |
| 493 | session_state_manager -> task_converter -> rich_cli -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 493` |
| 494 | session_state_manager -> task_converter -> rich_cli -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 494` |
| 495 | session_state_manager -> task_converter -> rich_cli -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 495` |
| 496 | session_state_manager -> task_converter -> rich_cli -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 496` |
| 497 | session_state_manager -> task_converter -> rich_cli -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 497` |
| 498 | session_state_manager -> task_converter -> rich_cli -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 498` |
| 499 | session_state_manager -> task_converter -> rich_cli -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 499` |
| 500 | session_state_manager -> task_converter -> rich_cli -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 500` |
| 501 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 501` |
| 502 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 502` |
| 503 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 503` |
| 504 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 504` |
| 505 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 505` |
| 506 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 506` |
| 507 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 507` |
| 508 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 508` |
| 509 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 509` |
| 510 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 510` |
| 511 | session_state_manager -> task_converter -> rich_cli -> postgresql_vector_store -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 511` |
| 512 | session_state_manager -> task_generator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 512` |
| 513 | session_state_manager -> tool_selector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 513` |
| 514 | session_state_manager -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 514` |
| 515 | session_state_manager -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 515` |
| 516 | session_state_manager -> enhanced_unified_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 516` |
| 517 | session_state_manager -> enhanced_unified_caller -> task_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 517` |
| 518 | session_state_manager -> enhanced_unified_caller -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 518` |
| 519 | session_state_manager -> enhanced_unified_caller -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 519` |
| 520 | session_state_manager -> enhanced_unified_caller -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 520` |
| 521 | session_state_manager -> unified_function_caller | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 521` |
| 522 | session_state_manager -> unified_function_caller -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 522` |
| 523 | session_state_manager -> authz_mapper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 523` |
| 524 | session_state_manager -> matrix_visualizer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 524` |
| 525 | session_state_manager -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 525` |
| 526 | session_state_manager -> core_service_coordinator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 526` |
| 527 | session_state_manager -> core_service_coordinator -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 527` |
| 528 | session_state_manager -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 528` |
| 529 | session_state_manager -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 529` |
| 530 | session_state_manager -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 530` |
| 531 | session_state_manager -> optimized_core -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 531` |
| 532 | session_state_manager -> optimized_core -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 532` |
| 533 | session_state_manager -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 533` |
| 534 | session_state_manager -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 534` |
| 535 | session_state_manager -> session_state_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 535` |
| 536 | session_state_manager -> session_state_manager -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 536` |
| 537 | session_state_manager -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 537` |
| 538 | session_state_manager -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 538` |
| 539 | session_state_manager -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 539` |
| 540 | session_state_manager -> backends -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 540` |
| 541 | session_state_manager -> backends -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 541` |
| 542 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> rich_cli | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 542` |
| 543 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> experience_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 543` |
| 544 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 544` |
| 545 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 545` |
| 546 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 546` |
| 547 | session_state_manager -> backends -> internal_loop_connector -> capability_registry -> trace_recorder | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 547` |
| 548 | session_state_manager -> backends -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 548` |
| 549 | session_state_manager -> backends -> internal_loop_connector -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 549` |
| 550 | session_state_manager -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 550` |
| 551 | session_state_manager -> backends -> plan_executor -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 551` |
| 552 | session_state_manager -> backends -> plan_executor -> dashboard -> improved_ui | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 552` |
| 553 | session_state_manager -> backends -> plan_executor -> dashboard -> server | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 553` |
| 554 | session_state_manager -> backends -> plan_executor -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 554` |
| 555 | session_state_manager -> backends -> plan_executor -> execution_planner -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 555` |
| 556 | session_state_manager -> backends -> plan_executor -> plan_comparator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 556` |
| 557 | session_state_manager -> backends -> plan_executor -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 557` |
| 558 | session_state_manager -> backends -> plan_executor -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 558` |
| 559 | session_state_manager -> backends -> plan_executor -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 559` |
| 560 | session_state_manager -> backends -> plan_executor -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 560` |
| 561 | session_state_manager -> backends -> plan_executor -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 561` |
| 562 | session_state_manager -> backends -> plan_executor -> attack_executor -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 562` |
| 563 | session_state_manager -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 563` |
| 564 | session_state_manager -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 564` |
| 565 | session_state_manager -> backends -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 565` |
| 566 | session_state_manager -> backends -> optimized_core -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 566` |
| 567 | session_state_manager -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 567` |
| 568 | session_state_manager -> backends -> message_broker -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 568` |
| 569 | session_state_manager -> backends -> message_broker -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 569` |
| 570 | session_state_manager -> backends -> message_broker -> parallel_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 570` |
| 571 | session_state_manager -> backends -> message_broker -> parallel_processor -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 571` |
| 572 | session_state_manager -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 572` |
| 573 | session_state_manager -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 573` |
| 574 | session_state_manager -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 574` |
| 575 | session_state_manager -> backends -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 575` |
| 576 | session_state_manager -> backends -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 576` |
| 577 | session_state_manager -> backends -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 577` |
| 578 | session_state_manager -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 578` |
| 579 | session_state_manager -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 579` |
| 580 | session_state_manager -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 580` |
| 581 | session_state_manager -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 581` |
| 582 | session_state_manager -> backends -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 582` |
| 583 | session_state_manager -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 583` |
| 584 | session_state_manager -> backends -> two_phase_scan_orchestrator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 584` |
| 585 | session_state_manager -> backends -> two_phase_scan_orchestrator -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 585` |
| 586 | session_state_manager -> backends -> two_phase_scan_orchestrator -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 586` |
| 587 | session_state_manager -> backends -> two_phase_scan_orchestrator -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 587` |
| 588 | session_state_manager -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 588` |
| 589 | session_state_manager -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 589` |
| 590 | session_state_manager -> backends -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 590` |
| 591 | session_state_manager -> backends -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 591` |
| 592 | session_state_manager -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 592` |
| 593 | session_state_manager -> storage_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 593` |
| 594 | session_state_manager -> storage_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 594` |
| 595 | session_state_manager -> cli_integration_example | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 595` |
| 596 | session_state_manager -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 596` |
| 597 | session_state_manager -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 597` |
| 598 | session_state_manager -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 598` |
| 599 | session_state_manager -> analyze_connection_recommendations | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 599` |
| 600 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 600` |
| 601 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 601` |
| 602 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 602` |
| 603 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 603` |
| 604 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 604` |
| 605 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 605` |
| 606 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 606` |
| 607 | session_state_manager -> analyze_connection_recommendations -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 607` |
| 608 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 608` |
| 609 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 609` |
| 610 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 610` |
| 611 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 611` |
| 612 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 612` |
| 613 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 613` |
| 614 | session_state_manager -> analyze_connection_recommendations -> analyze_missing_function_connections -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 614` |
| 615 | session_state_manager -> analyze_connection_recommendations -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 615` |
| 616 | session_state_manager -> analyze_connection_recommendations -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 616` |
| 617 | session_state_manager -> analyze_connection_recommendations -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 617` |
| 618 | session_state_manager -> analyze_connection_recommendations -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 618` |
| 619 | session_state_manager -> analyze_connection_recommendations -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 619` |
| 620 | session_state_manager -> analyze_connection_recommendations -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 620` |
| 621 | session_state_manager -> analyze_connection_recommendations -> practical_analyzer -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 621` |
| 622 | session_state_manager -> analyze_connection_recommendations -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 622` |
| 623 | session_state_manager -> analyze_connection_recommendations -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 623` |
| 624 | session_state_manager -> analyze_connection_recommendations -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 624` |
| 625 | session_state_manager -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 625` |
| 626 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 626` |
| 627 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 627` |
| 628 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 628` |
| 629 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 629` |
| 630 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 630` |
| 631 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 631` |
| 632 | session_state_manager -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 632` |
| 633 | session_state_manager -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 633` |
| 634 | session_state_manager -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 634` |
| 635 | session_state_manager -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 635` |
| 636 | session_state_manager -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 636` |
| 637 | session_state_manager -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 637` |
| 638 | session_state_manager -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 638` |
| 639 | session_state_manager -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 639` |
| 640 | session_state_manager -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 640` |
| 641 | session_state_manager -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 641` |
| 642 | session_state_manager -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 642` |
| 643 | session_state_manager -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 643` |
| 644 | session_state_manager -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 644` |
| 645 | session_state_manager -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 645` |
| 646 | session_state_manager -> analyze_missing_function_connections -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 646` |
| 647 | session_state_manager -> analyze_results | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 647` |
| 648 | session_state_manager -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 648` |
| 649 | session_state_manager -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 649` |
| 650 | session_state_manager -> core_analyzer -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 650` |
| 651 | session_state_manager -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 651` |
| 652 | session_state_manager -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 652` |
| 653 | session_state_manager -> dynamic_strategy_adjustment | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 653` |
| 654 | session_state_manager -> dynamic_strategy_adjustment -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 654` |
| 655 | session_state_manager -> risk_assessment_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 655` |
| 656 | session_state_manager -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 656` |
| 657 | session_state_manager -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 657` |
| 658 | session_state_manager -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 658` |
| 659 | session_state_manager -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 659` |
| 660 | session_state_manager -> rl_trainers | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 660` |
| 661 | session_state_manager -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 661` |
| 662 | session_state_manager -> rl_trainers -> aiva_exploration_pipeline -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 662` |
| 663 | session_state_manager -> rl_trainers -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 663` |
| 664 | session_state_manager -> rl_trainers -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 664` |
| 665 | session_state_manager -> rl_trainers -> aiva_cli_implementation -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 665` |
| 666 | session_state_manager -> rl_trainers -> aiva_exploration_pipeline | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 666` |
| 667 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 667` |
| 668 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 668` |
| 669 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 669` |
| 670 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 670` |
| 671 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 671` |
| 672 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 672` |
| 673 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 673` |
| 674 | session_state_manager -> rl_trainers -> analyze_dataflow_breakpoints -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 674` |
| 675 | session_state_manager -> rl_trainers -> analyze_missing_function_connections | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 675` |
| 676 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 676` |
| 677 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> core_analyzer -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 677` |
| 678 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> core_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 678` |
| 679 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 679` |
| 680 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 680` |
| 681 | session_state_manager -> rl_trainers -> analyze_missing_function_connections -> practical_analyzer -> run_analysis | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 681` |
| 682 | session_state_manager -> rl_trainers -> analyze_results | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 682` |
| 683 | session_state_manager -> rl_trainers -> verify_rl_models | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 683` |
| 684 | session_state_manager -> rl_trainers -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 684` |
| 685 | session_state_manager -> rl_trainers -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 685` |
| 686 | session_state_manager -> rl_trainers -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 686` |
| 687 | session_state_manager -> rl_trainers -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 687` |
| 688 | session_state_manager -> rl_trainers -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 688` |
| 689 | session_state_manager -> rl_trainers -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 689` |
| 690 | session_state_manager -> rl_trainers -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 690` |
| 691 | session_state_manager -> rl_trainers -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 691` |
| 692 | session_state_manager -> rl_trainers -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 692` |
| 693 | session_state_manager -> rl_trainers -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 693` |
| 694 | session_state_manager -> rl_trainers -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 694` |
| 695 | session_state_manager -> rl_trainers -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 695` |
| 696 | session_state_manager -> rl_trainers -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 696` |
| 697 | session_state_manager -> rl_trainers -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 697` |
| 698 | session_state_manager -> rl_trainers -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 698` |
| 699 | session_state_manager -> rl_trainers -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 699` |
| 700 | session_state_manager -> rl_trainers -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 700` |
| 701 | session_state_manager -> rl_trainers -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 701` |
| 702 | session_state_manager -> rl_trainers -> vector_store -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 702` |
| 703 | session_state_manager -> rl_trainers -> vector_store -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 703` |
| 704 | session_state_manager -> rl_trainers -> vector_store -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 704` |
| 705 | session_state_manager -> rl_trainers -> vector_store -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 705` |
| 706 | session_state_manager -> rl_trainers -> vector_store -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 706` |
| 707 | session_state_manager -> rl_trainers -> vector_store -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 707` |
| 708 | session_state_manager -> rl_trainers -> vector_store -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 708` |
| 709 | session_state_manager -> rl_trainers -> vector_store -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 709` |
| 710 | session_state_manager -> rl_trainers -> vector_store -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 710` |
| 711 | session_state_manager -> rl_trainers -> vector_store -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 711` |
| 712 | session_state_manager -> rl_trainers -> vector_store -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 712` |
| 713 | session_state_manager -> rl_trainers -> vector_store -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 713` |
| 714 | session_state_manager -> rl_trainers -> vector_store -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 714` |
| 715 | session_state_manager -> rl_trainers -> vector_store -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 715` |
| 716 | session_state_manager -> rl_trainers -> vector_store -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 716` |
| 717 | session_state_manager -> rl_trainers -> vector_store -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 717` |
| 718 | session_state_manager -> rl_trainers -> vector_store -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 718` |
| 719 | session_state_manager -> rl_trainers -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 719` |
| 720 | session_state_manager -> rl_trainers -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 720` |
| 721 | session_state_manager -> rl_trainers -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 721` |
| 722 | session_state_manager -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 722` |
| 723 | session_state_manager -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 723` |
| 724 | session_state_manager -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 724` |
| 725 | session_state_manager -> attack_chain | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 725` |
| 726 | session_state_manager -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 726` |
| 727 | session_state_manager -> attack_executor -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 727` |
| 728 | session_state_manager -> attack_validator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 728` |
| 729 | session_state_manager -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 729` |
| 730 | session_state_manager -> exploit_manager_legacy | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 730` |
| 731 | session_state_manager -> exploit_manager_legacy -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 731` |
| 732 | session_state_manager -> exploit_manager_legacy -> attack_executor -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 732` |
| 733 | session_state_manager -> exploit_manager_legacy -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 733` |
| 734 | session_state_manager -> exploit_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 734` |
| 735 | session_state_manager -> payload_generator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 735` |
| 736 | session_state_manager -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 736` |
| 737 | session_state_manager -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 737` |
| 738 | session_state_manager -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 738` |
| 739 | session_state_manager -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 739` |
| 740 | session_state_manager -> two_phase_scan_orchestrator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 740` |
| 741 | session_state_manager -> two_phase_scan_orchestrator -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 741` |
| 742 | session_state_manager -> two_phase_scan_orchestrator -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 742` |
| 743 | session_state_manager -> two_phase_scan_orchestrator -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 743` |
| 744 | session_state_manager -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 744` |
| 745 | session_state_manager -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 745` |
| 746 | session_state_manager -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 746` |
| 747 | session_state_manager -> anti_hallucination_module | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 747` |
| 748 | session_state_manager -> anti_hallucination_module -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 748` |
| 749 | session_state_manager -> anti_hallucination_module -> enhanced_decision_agent -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 749` |
| 750 | session_state_manager -> anti_hallucination_module -> enhanced_decision_agent -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 750` |
| 751 | session_state_manager -> enhanced_decision_agent | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 751` |
| 752 | session_state_manager -> enhanced_decision_agent -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 752` |
| 753 | session_state_manager -> enhanced_decision_agent -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 753` |
| 754 | session_state_manager -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 754` |
| 755 | session_state_manager -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 755` |
| 756 | session_state_manager -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 756` |
| 757 | session_state_manager -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 757` |
| 758 | session_state_manager -> weight_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 758` |
| 759 | session_state_manager -> knowledge_base | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 759` |
| 760 | session_state_manager -> knowledge_base -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 760` |
| 761 | session_state_manager -> knowledge_base -> unified_vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 761` |
| 762 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 762` |
| 763 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 763` |
| 764 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 764` |
| 765 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 765` |
| 766 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 766` |
| 767 | session_state_manager -> knowledge_base -> unified_vector_store -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 767` |
| 768 | session_state_manager -> knowledge_base -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 768` |
| 769 | session_state_manager -> knowledge_base -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 769` |
| 770 | session_state_manager -> knowledge_base -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 770` |
| 771 | session_state_manager -> knowledge_base -> vector_store -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 771` |
| 772 | session_state_manager -> knowledge_base -> vector_store -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 772` |
| 773 | session_state_manager -> knowledge_base -> vector_store -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 773` |
| 774 | session_state_manager -> knowledge_base -> vector_store -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 774` |
| 775 | session_state_manager -> knowledge_base -> vector_store -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 775` |
| 776 | session_state_manager -> knowledge_base -> vector_store -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 776` |
| 777 | session_state_manager -> knowledge_base -> vector_store -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 777` |
| 778 | session_state_manager -> knowledge_base -> vector_store -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 778` |
| 779 | session_state_manager -> knowledge_base -> vector_store -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 779` |
| 780 | session_state_manager -> knowledge_base -> vector_store -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 780` |
| 781 | session_state_manager -> knowledge_base -> vector_store -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 781` |
| 782 | session_state_manager -> knowledge_base -> vector_store -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 782` |
| 783 | session_state_manager -> knowledge_base -> vector_store -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 783` |
| 784 | session_state_manager -> knowledge_base -> vector_store -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 784` |
| 785 | session_state_manager -> knowledge_base -> vector_store -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 785` |
| 786 | session_state_manager -> knowledge_base -> vector_store -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 786` |
| 787 | session_state_manager -> knowledge_base -> vector_store -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 787` |
| 788 | session_state_manager -> knowledge_base -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 788` |
| 789 | session_state_manager -> knowledge_base -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 789` |
| 790 | session_state_manager -> knowledge_base -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 790` |
| 791 | session_state_manager -> rag_engine | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 791` |
| 792 | session_state_manager -> vector_store | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 792` |
| 793 | session_state_manager -> vector_store -> ai_controller | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 793` |
| 794 | session_state_manager -> vector_store -> backends | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 794` |
| 795 | session_state_manager -> vector_store -> backends -> external_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 795` |
| 796 | session_state_manager -> vector_store -> backends -> internal_loop_connector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 796` |
| 797 | session_state_manager -> vector_store -> backends -> internal_loop_connector -> test_scope_management | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 797` |
| 798 | session_state_manager -> vector_store -> backends -> internal_loop_connector -> capability_registry | core_capabilities | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 798` |
| 799 | session_state_manager -> vector_store -> backends -> internal_loop_connector -> aiva_cli_implementation | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 799` |
| 800 | session_state_manager -> vector_store -> backends -> plan_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 800` |
| 801 | session_state_manager -> vector_store -> backends -> plan_executor -> dashboard | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 801` |
| 802 | session_state_manager -> vector_store -> backends -> plan_executor -> execution_planner | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 802` |
| 803 | session_state_manager -> vector_store -> backends -> plan_executor -> plan_comparator | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 803` |
| 804 | session_state_manager -> vector_store -> backends -> plan_executor -> core_analyzer | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 804` |
| 805 | session_state_manager -> vector_store -> backends -> plan_executor -> analysis_engine | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 805` |
| 806 | session_state_manager -> vector_store -> backends -> plan_executor -> attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 806` |
| 807 | session_state_manager -> vector_store -> backends -> protocol_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 807` |
| 808 | session_state_manager -> vector_store -> backends -> optimized_core | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 808` |
| 809 | session_state_manager -> vector_store -> backends -> optimized_core -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 809` |
| 810 | session_state_manager -> vector_store -> backends -> message_broker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 810` |
| 811 | session_state_manager -> vector_store -> backends -> message_broker -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 811` |
| 812 | session_state_manager -> vector_store -> backends -> message_broker -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 812` |
| 813 | session_state_manager -> vector_store -> backends -> message_broker -> parallel_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 813` |
| 814 | session_state_manager -> vector_store -> backends -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 814` |
| 815 | session_state_manager -> vector_store -> backends -> task_dispatcher | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 815` |
| 816 | session_state_manager -> vector_store -> backends -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 816` |
| 817 | session_state_manager -> vector_store -> backends -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 817` |
| 818 | session_state_manager -> vector_store -> backends -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 818` |
| 819 | session_state_manager -> vector_store -> backends -> scenario_manager | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 819` |
| 820 | session_state_manager -> vector_store -> backends -> bizlogic_attack_executor | task_planning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 820` |
| 821 | session_state_manager -> vector_store -> backends -> assistant | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 821` |
| 822 | session_state_manager -> vector_store -> backends -> scan_module_interface | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 822` |
| 823 | session_state_manager -> vector_store -> backends -> scan_module_interface -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 823` |
| 824 | session_state_manager -> vector_store -> backends -> two_phase_scan_orchestrator | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 824` |
| 825 | session_state_manager -> vector_store -> backends -> two_phase_scan_orchestrator -> ai_commander | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 825` |
| 826 | session_state_manager -> vector_store -> backends -> two_phase_scan_orchestrator -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 826` |
| 827 | session_state_manager -> vector_store -> backends -> to_functions | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 827` |
| 828 | session_state_manager -> vector_store -> backends -> scan_result_processor | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 828` |
| 829 | session_state_manager -> vector_store -> backends -> scan_result_processor -> app | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 829` |
| 830 | session_state_manager -> vector_store -> backends -> scan_result_processor -> result_collector | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 830` |
| 831 | session_state_manager -> vector_store -> backends -> skill_graph | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 831` |
| 832 | session_state_manager -> vector_store -> command_repository | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 832` |
| 833 | session_state_manager -> vector_store -> real_bio_net_adapter | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 833` |
| 834 | session_state_manager -> vector_store -> real_neural_core | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 834` |
| 835 | websocket_manager -> db_helper | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 835` |
| 836 | websocket_manager -> train_classifier | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 836` |
| 837 | websocket_manager -> train_classifier -> model_trainer | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 837` |
| 838 | websocket_manager -> train_classifier -> model_trainer -> training_orchestrator | external_learning | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 838` |
| 839 | websocket_manager -> train_classifier -> model_trainer -> ai_model_manager | cognitive_core | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 839` |
| 840 | websocket_manager -> train_classifier -> model_trainer -> ai_model_manager -> system_connectivity_checker | service_backbone | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 840` |
