# 完整數據流詳細列表

生成時間: 2025-12-08 08:55:22
總數據流數量: 321

---

## 認知核心模組 (cognitive_core)

包含 53 條數據流

### Flow 7

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

3. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 12

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 13

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 22

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 32

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 33

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_summary_plugin
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

5. **AI組件**
   __init__
   ai_summary_plugin
   
   register_capability
   ai_summary_plugin
   
   discover_and_register
   ai_summary_plugin
   
   _process_module_path
   ai_summary_plugin
   
   _try_register_function
   ai_summary_plugin
   
   execute_capability
   ai_summary_plugin
   
   _update_avg_execution_time
   ai_summary_plugin
   
   list_capabilities
   ai_summary_plugin
   
   get_registry_stats
   ai_summary_plugin
   
   is_enabled
   ai_summary_plugin
   
   enable
   ai_summary_plugin
   
   disable
   ai_summary_plugin
   
   get_status
   ai_summary_plugin
   
   generate_summary
   ai_summary_plugin
   
   _build_summary_prompt
   ai_summary_plugin
   
   _classify_request_type
   ai_summary_plugin
   
   _get_complexity_level
   ai_summary_plugin
   
   _calculate_efficiency_score
   ai_summary_plugin
   
   _extract_recommendations
   ai_summary_plugin
   
   _identify_learning_points
   ai_summary_plugin
   
   _create_brief_summary
   ai_summary_plugin
   
   _enhance_detailed_summary
   ai_summary_plugin
   
   _extract_processing_steps
   ai_summary_plugin
   
   _estimate_resource_usage
   ai_summary_plugin
   
   _assess_improvement_potential
   ai_summary_plugin
   
   _record_summary_history
   ai_summary_plugin
   
   get_statistics
   ai_summary_plugin
   
   configure
   ai_summary_plugin
   
   reset
   ai_summary_plugin
   
   unload
   ai_summary_plugin
   - 模組: 認知核心模組

---

### Flow 46

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 57

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 60

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 63

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 65

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 68

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 69

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_summary_plugin
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

5. **AI組件**
   __init__
   ai_summary_plugin
   
   register_capability
   ai_summary_plugin
   
   discover_and_register
   ai_summary_plugin
   
   _process_module_path
   ai_summary_plugin
   
   _try_register_function
   ai_summary_plugin
   
   execute_capability
   ai_summary_plugin
   
   _update_avg_execution_time
   ai_summary_plugin
   
   list_capabilities
   ai_summary_plugin
   
   get_registry_stats
   ai_summary_plugin
   
   is_enabled
   ai_summary_plugin
   
   enable
   ai_summary_plugin
   
   disable
   ai_summary_plugin
   
   get_status
   ai_summary_plugin
   
   generate_summary
   ai_summary_plugin
   
   _build_summary_prompt
   ai_summary_plugin
   
   _classify_request_type
   ai_summary_plugin
   
   _get_complexity_level
   ai_summary_plugin
   
   _calculate_efficiency_score
   ai_summary_plugin
   
   _extract_recommendations
   ai_summary_plugin
   
   _identify_learning_points
   ai_summary_plugin
   
   _create_brief_summary
   ai_summary_plugin
   
   _enhance_detailed_summary
   ai_summary_plugin
   
   _extract_processing_steps
   ai_summary_plugin
   
   _estimate_resource_usage
   ai_summary_plugin
   
   _assess_improvement_potential
   ai_summary_plugin
   
   _record_summary_history
   ai_summary_plugin
   
   get_statistics
   ai_summary_plugin
   
   configure
   ai_summary_plugin
   
   reset
   ai_summary_plugin
   
   unload
   ai_summary_plugin
   - 模組: 認知核心模組

---

### Flow 90

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 92

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 93

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: neural_network
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

---

### Flow 98

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 103

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 113

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 115

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: storage_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   storage_manager
   
   initialize
   storage_manager
   
   _get_database_config
   storage_manager
   
   _create_backend
   storage_manager
   
   get_path
   storage_manager
   
   get_statistics
   storage_manager
   
   save_experience_sample
   storage_manager
   
   save_unified_experience_sample
   storage_manager
   
   get_experience_samples
   storage_manager
   
   save_trace
   storage_manager
   
   get_traces_by_session
   storage_manager
   
   save_training_session
   storage_manager
   
   save_command_execution
   storage_manager
   
   get_command_history
   storage_manager
   
   get_command_statistics
   storage_manager
   
   get_popular_capabilities
   storage_manager
   
   get_slow_executions
   storage_manager
   
   get_dir_size
   storage_manager
   - 模組: 認知核心模組

---

### Flow 129

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 133

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 138

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 163

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 164

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_summary_plugin
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

5. **AI組件**
   __init__
   ai_summary_plugin
   
   register_capability
   ai_summary_plugin
   
   discover_and_register
   ai_summary_plugin
   
   _process_module_path
   ai_summary_plugin
   
   _try_register_function
   ai_summary_plugin
   
   execute_capability
   ai_summary_plugin
   
   _update_avg_execution_time
   ai_summary_plugin
   
   list_capabilities
   ai_summary_plugin
   
   get_registry_stats
   ai_summary_plugin
   
   is_enabled
   ai_summary_plugin
   
   enable
   ai_summary_plugin
   
   disable
   ai_summary_plugin
   
   get_status
   ai_summary_plugin
   
   generate_summary
   ai_summary_plugin
   
   _build_summary_prompt
   ai_summary_plugin
   
   _classify_request_type
   ai_summary_plugin
   
   _get_complexity_level
   ai_summary_plugin
   
   _calculate_efficiency_score
   ai_summary_plugin
   
   _extract_recommendations
   ai_summary_plugin
   
   _identify_learning_points
   ai_summary_plugin
   
   _create_brief_summary
   ai_summary_plugin
   
   _enhance_detailed_summary
   ai_summary_plugin
   
   _extract_processing_steps
   ai_summary_plugin
   
   _estimate_resource_usage
   ai_summary_plugin
   
   _assess_improvement_potential
   ai_summary_plugin
   
   _record_summary_history
   ai_summary_plugin
   
   get_statistics
   ai_summary_plugin
   
   configure
   ai_summary_plugin
   
   reset
   ai_summary_plugin
   
   unload
   ai_summary_plugin
   - 模組: 認知核心模組

---

### Flow 191

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: storage_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   storage_manager
   
   initialize
   storage_manager
   
   _get_database_config
   storage_manager
   
   _create_backend
   storage_manager
   
   get_path
   storage_manager
   
   get_statistics
   storage_manager
   
   save_experience_sample
   storage_manager
   
   save_unified_experience_sample
   storage_manager
   
   get_experience_samples
   storage_manager
   
   save_trace
   storage_manager
   
   get_traces_by_session
   storage_manager
   
   save_training_session
   storage_manager
   
   save_command_execution
   storage_manager
   
   get_command_history
   storage_manager
   
   get_command_statistics
   storage_manager
   
   get_popular_capabilities
   storage_manager
   
   get_slow_executions
   storage_manager
   
   get_dir_size
   storage_manager
   - 模組: 認知核心模組

---

### Flow 199

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 206

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 214

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 216

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 224

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   attack_executor
   
   execute_plan_with_ai_analysis
   attack_executor
   
   _generate_feedback_data
   attack_executor
   
   execute_plan
   attack_executor
   
   _execute_step
   attack_executor
   
   _simulate_step
   attack_executor
   
   _real_execute_step
   attack_executor
   
   _map_step_type_to_exploit_type
   attack_executor
   
   _enhanced_safety_check
   attack_executor
   
   _safety_check
   attack_executor
   
   _create_aborted_result
   attack_executor
   
   _generate_recommendations
   attack_executor
   
   get_statistics
   attack_executor
   - 模組: 任務規劃模組

5. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 236

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   two_phase_scan_orchestrator
   
   execute_two_phase_scan
   two_phase_scan_orchestrator
   
   _execute_phase0
   two_phase_scan_orchestrator
   
   _execute_phase1
   two_phase_scan_orchestrator
   
   _analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   _select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 238

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_summary_plugin
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_summary_plugin
   
   register_capability
   ai_summary_plugin
   
   discover_and_register
   ai_summary_plugin
   
   _process_module_path
   ai_summary_plugin
   
   _try_register_function
   ai_summary_plugin
   
   execute_capability
   ai_summary_plugin
   
   _update_avg_execution_time
   ai_summary_plugin
   
   list_capabilities
   ai_summary_plugin
   
   get_registry_stats
   ai_summary_plugin
   
   is_enabled
   ai_summary_plugin
   
   enable
   ai_summary_plugin
   
   disable
   ai_summary_plugin
   
   get_status
   ai_summary_plugin
   
   generate_summary
   ai_summary_plugin
   
   _build_summary_prompt
   ai_summary_plugin
   
   _classify_request_type
   ai_summary_plugin
   
   _get_complexity_level
   ai_summary_plugin
   
   _calculate_efficiency_score
   ai_summary_plugin
   
   _extract_recommendations
   ai_summary_plugin
   
   _identify_learning_points
   ai_summary_plugin
   
   _create_brief_summary
   ai_summary_plugin
   
   _enhance_detailed_summary
   ai_summary_plugin
   
   _extract_processing_steps
   ai_summary_plugin
   
   _estimate_resource_usage
   ai_summary_plugin
   
   _assess_improvement_potential
   ai_summary_plugin
   
   _record_summary_history
   ai_summary_plugin
   
   get_statistics
   ai_summary_plugin
   
   configure
   ai_summary_plugin
   
   reset
   ai_summary_plugin
   
   unload
   ai_summary_plugin
   - 模組: 認知核心模組

---

### Flow 244

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   demo_anti_hallucination
   anti_hallucination_module
   
   __init__
   anti_hallucination_module
   
   _check_knowledge_base_health
   anti_hallucination_module
   
   _fallback_knowledge_validation
   anti_hallucination_module
   
   _get_technique_category
   anti_hallucination_module
   
   _validate_technique_consistency
   anti_hallucination_module
   
   _setup_logger
   anti_hallucination_module
   
   validate_attack_plan
   anti_hallucination_module
   
   _validate_single_step
   anti_hallucination_module
   
   _validate_with_knowledge_base_fallback
   anti_hallucination_module
   
   _validate_step_sequence
   anti_hallucination_module
   
   _is_known_technique
   anti_hallucination_module
   
   _extract_relevance_score
   anti_hallucination_module
   
   _validate_with_knowledge_base
   anti_hallucination_module
   
   _validate_step_logic
   anti_hallucination_module
   
   get_validation_stats
   anti_hallucination_module
   
   export_validation_report
   anti_hallucination_module
   
   reset_knowledge_base
   anti_hallucination_module
   - 模組: 服務骨幹模組

5. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 245

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 246

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 248

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 249

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 255

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: rag_engine
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rag_engine
   
   enhance_attack_plan
   rag_engine
   
   suggest_next_step
   rag_engine
   
   analyze_failure
   rag_engine
   
   get_relevant_payloads
   rag_engine
   
   learn_from_experience
   rag_engine
   
   _extract_successful_pattern
   rag_engine
   
   save_knowledge
   rag_engine
   
   get_statistics
   rag_engine
   - 模組: 認知核心模組

---

### Flow 257

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 261

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 265

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **AI組件**
   demo_enhanced_decision_agent
   enhanced_decision_agent
   
   __init__
   enhanced_decision_agent
   
   _setup_logger
   enhanced_decision_agent
   
   _initialize_decision_rules
   enhanced_decision_agent
   
   decide
   enhanced_decision_agent
   
   _convert_decision_to_intent
   enhanced_decision_agent
   
   make_decision
   enhanced_decision_agent
   
   _assess_risk_decision
   enhanced_decision_agent
   
   _make_experience_driven_decision
   enhanced_decision_agent
   
   _find_similar_experiences
   enhanced_decision_agent
   
   _calculate_similarity
   enhanced_decision_agent
   
   _apply_decision_rules
   enhanced_decision_agent
   
   _execute_rule_action
   enhanced_decision_agent
   
   _select_best_tool
   enhanced_decision_agent
   
   _suggest_alternative_strategy
   enhanced_decision_agent
   
   execute_decision
   enhanced_decision_agent
   
   _execute_tool_decision
   enhanced_decision_agent
   
   _execute_vulnerability_test
   enhanced_decision_agent
   
   _execute_mode_switch
   enhanced_decision_agent
   
   _execute_strategy_change
   enhanced_decision_agent
   
   _execute_stop
   enhanced_decision_agent
   
   _make_default_decision
   enhanced_decision_agent
   
   _record_decision
   enhanced_decision_agent
   
   get_decision_stats
   enhanced_decision_agent
   
   export_decision_analysis
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 268

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 272

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 279

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_commander
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_commander
   
   execute_command
   ai_commander
   
   _plan_attack
   ai_commander
   
   _build_plan_generation_prompt
   ai_commander
   
   _calculate_plan_confidence
   ai_commander
   
   _make_strategy_decision
   ai_commander
   
   _assess_risk_factors
   ai_commander
   
   _build_strategy_decision_prompt
   ai_commander
   
   _adjust_confidence_by_risk
   ai_commander
   
   _get_similar_decisions
   ai_commander
   
   _calculate_historical_confidence
   ai_commander
   
   _detect_vulnerabilities
   ai_commander
   
   _learn_from_experience
   ai_commander
   
   _train_model
   ai_commander
   
   _retrieve_knowledge
   ai_commander
   
   _coordinate_multilang
   ai_commander
   
   _execute_attack
   ai_commander
   
   _execute_two_phase_scan
   ai_commander
   
   run_training_session
   ai_commander
   
   get_status
   ai_commander
   
   save_state
   ai_commander
   
   add_experience
   ai_commander
   
   get_experiences
   ai_commander
   
   add_sample
   ai_commander
   
   get_statistics
   ai_commander
   
   export_to_jsonl
   ai_commander
   - 模組: 認知核心模組

---

### Flow 290

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 293

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

### Flow 295

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 298

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ai_controller
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **AI組件**
   demonstrate_unified_control
   ai_controller
   
   __init__
   ai_controller
   
   master_ai
   ai_controller
   
   process_specialized_request
   ai_controller
   
   _analyze_task_complexity
   ai_controller
   
   _direct_processing
   ai_controller
   
   _coordinated_code_fixing
   ai_controller
   
   _coordinated_detection
   ai_controller
   
   _multi_ai_coordination
   ai_controller
   
   get_summary_plugin_status
   ai_controller
   
   enable_summary_plugin
   ai_controller
   
   disable_summary_plugin
   ai_controller
   
   configure_summary_plugin
   ai_controller
   
   get_summary_statistics
   ai_controller
   
   reset_summary_plugin
   ai_controller
   
   unload_summary_plugin
   ai_controller
   
   _record_unified_decision
   ai_controller
   
   get_control_statistics
   ai_controller
   
   _classify_request_type
   ai_controller
   
   _get_complexity_level
   ai_controller
   
   _calculate_efficiency_score
   ai_controller
   
   _extract_recommendations
   ai_controller
   
   _identify_learning_points
   ai_controller
   
   _create_brief_summary
   ai_controller
   
   _enhance_detailed_summary
   ai_controller
   
   _extract_processing_steps
   ai_controller
   
   _estimate_resource_usage
   ai_controller
   
   _assess_improvement_potential
   ai_controller
   
   _record_summary_history
   ai_controller
   
   get_ai_summary_statistics
   ai_controller
   
   _generate_summary_recommendations
   ai_controller
   
   configure_summary_settings
   ai_controller
   
   get_latest_summaries
   ai_controller
   
   export_summary_report
   ai_controller
   
   generate_comprehensive_summary
   ai_controller
   
   _perform_quantitative_analysis
   ai_controller
   
   _analyze_trends
   ai_controller
   
   _generate_comprehensive_recommendations
   ai_controller
   - 模組: 認知核心模組

---

### Flow 302

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 303

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 305

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   create_real_ai_replacement
   real_neural_core
   
   __init__
   real_neural_core
   
   _build_5m_network
   real_neural_core
   
   _initialize_weights
   real_neural_core
   
   _build_legacy_network
   real_neural_core
   
   forward
   real_neural_core
   
   forward_with_aux
   real_neural_core
   
   save_weights
   real_neural_core
   
   load_weights
   real_neural_core
   
   encode_input
   real_neural_core
   
   _enhance_bug_bounty_context
   real_neural_core
   
   _extract_bug_bounty_features
   real_neural_core
   
   _extract_attack_intent_features
   real_neural_core
   
   _extract_target_features
   real_neural_core
   
   _extract_tool_features
   real_neural_core
   
   _extract_context_features
   real_neural_core
   
   generate_decision
   real_neural_core
   
   _prepare_decision_input
   real_neural_core
   
   _calculate_enhanced_confidence
   real_neural_core
   
   _analyze_decision_output
   real_neural_core
   
   _analyze_bug_bounty_decision
   real_neural_core
   
   _fallback_bug_bounty_decision
   real_neural_core
   
   decide
   real_neural_core
   
   train_step
   real_neural_core
   
   _compute_training_loss
   real_neural_core
   
   _compute_dual_output_loss
   real_neural_core
   
   _compute_single_output_loss
   real_neural_core
   
   _perform_backward_pass
   real_neural_core
   
   _update_training_statistics
   real_neural_core
   
   _calculate_gradient_norm
   real_neural_core
   
   save_model
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 318

- **長度**: 5 步
- **起點**: monitoring
- **終點**: ai_model_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   monitor_performance
   monitoring
   
   __init__
   monitoring
   
   record_duration
   monitoring
   
   increment_counter
   monitoring
   
   set_gauge
   monitoring
   
   _make_key
   monitoring
   
   get_metrics_summary
   monitoring
   
   update_component_health
   monitoring
   
   get_system_health_status
   monitoring
   
   check_component_freshness
   monitoring
   
   decorator
   monitoring
   
   wrapper
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **AI組件**
   __init__
   ai_model_manager
   
   initialize_models
   ai_model_manager
   
   train_models
   ai_model_manager
   
   _prepare_training_data
   ai_model_manager
   
   _create_no_data_result
   ai_model_manager
   
   _setup_training_config
   ai_model_manager
   
   _execute_training
   ai_model_manager
   
   _prepare_training_arrays
   ai_model_manager
   
   _has_real_sample_data
   ai_model_manager
   
   _extract_real_data_arrays
   ai_model_manager
   
   _generate_synthetic_data_arrays
   ai_model_manager
   
   _update_model_state
   ai_model_manager
   
   _create_success_result
   ai_model_manager
   
   _create_failure_result
   ai_model_manager
   
   make_decision
   ai_model_manager
   
   _validate_decision_with_scalable_net
   ai_model_manager
   
   _merge_dual_outputs
   ai_model_manager
   
   get_model_status
   ai_model_manager
   
   update_from_experience
   ai_model_manager
   
   _save_model
   ai_model_manager
   
   load_model
   ai_model_manager
   
   predict_batch
   ai_model_manager
   
   _create_experience_adapter
   ai_model_manager
   
   query_experiences
   ai_model_manager
   
   add_experience
   ai_model_manager
   
   get_training_samples
   ai_model_manager
   
   _sync_get_samples
   ai_model_manager
   
   _sync_add_experience
   ai_model_manager
   - 模組: 認知核心模組

---

## 任務規劃模組 (task_planning)

包含 27 條數據流

### Flow 35

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 51

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: execution_planner
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

---

### Flow 73

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: plan_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

---

### Flow 78

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: task_dispatcher
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   task_dispatcher
   
   _get_topic_for_tool
   task_dispatcher
   
   dispatch_attack_plan
   task_dispatcher
   
   dispatch_step
   task_dispatcher
   
   _build_task_payload
   task_dispatcher
   
   _build_message
   task_dispatcher
   
   dispatch_scan_task
   task_dispatcher
   
   dispatch_batch_tasks
   task_dispatcher
   
   send_control_command
   task_dispatcher
   
   send_feedback
   task_dispatcher
   
   request_status
   task_dispatcher
   
   register_callback
   task_dispatcher
   
   unregister_callback
   task_dispatcher
   
   trigger_callback
   task_dispatcher
   - 模組: 任務規劃模組

---

### Flow 81

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 124

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: task_converter
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

5. **程式組件**
   __repr__
   task_converter
   
   add_task
   task_converter
   
   get_task
   task_converter
   
   get_pending_tasks
   task_converter
   
   get_runnable_tasks
   task_converter
   
   __init__
   task_converter
   
   convert
   task_converter
   
   _topological_sort
   task_converter
   
   _get_node_priority
   task_converter
   
   _create_task_from_node
   task_converter
   
   _interpolate_variables
   task_converter
   
   _resolve_nested_variable
   task_converter
   - 模組: 任務規劃模組

---

### Flow 135

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 139

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: attack_plan_mapper
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   attack_plan_mapper
   
   map_decision_to_tasks
   attack_plan_mapper
   
   _map_vulnerability_to_module
   attack_plan_mapper
   
   _create_info_gathering_tasks
   attack_plan_mapper
   
   _create_exploitation_tasks
   attack_plan_mapper
   
   map_entire_plan
   attack_plan_mapper
   
   _generate_task_id
   attack_plan_mapper
   
   _extract_domain
   attack_plan_mapper
   - 模組: 任務規劃模組

---

### Flow 143

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: plan_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

---

### Flow 144

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: execution_planner
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

5. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

---

### Flow 145

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: plan_comparator
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

5. **程式組件**
   __init__
   plan_comparator
   
   compare
   plan_comparator
   
   _match_steps
   plan_comparator
   
   _calculate_step_similarity
   plan_comparator
   
   _count_extra_actions
   plan_comparator
   
   _calculate_sequence_accuracy
   plan_comparator
   
   _lcs_length
   plan_comparator
   
   _evaluate_goal_achievement
   plan_comparator
   
   _check_critical_steps
   plan_comparator
   
   _calculate_reward_score
   plan_comparator
   
   _calculate_quality_bonus
   plan_comparator
   
   generate_comparison_report
   plan_comparator
   
   _generate_recommendations
   plan_comparator
   - 模組: 任務規劃模組

---

### Flow 147

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

5. **程式組件**
   __init__
   attack_executor
   
   execute_plan_with_ai_analysis
   attack_executor
   
   _generate_feedback_data
   attack_executor
   
   execute_plan
   attack_executor
   
   _execute_step
   attack_executor
   
   _simulate_step
   attack_executor
   
   _real_execute_step
   attack_executor
   
   _map_step_type_to_exploit_type
   attack_executor
   
   _enhanced_safety_check
   attack_executor
   
   _safety_check
   attack_executor
   
   _create_aborted_result
   attack_executor
   
   _generate_recommendations
   attack_executor
   
   get_statistics
   attack_executor
   - 模組: 任務規劃模組

---

### Flow 149

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: execution_planner
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

---

### Flow 151

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: task_converter
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __repr__
   task_converter
   
   add_task
   task_converter
   
   get_task
   task_converter
   
   get_pending_tasks
   task_converter
   
   get_runnable_tasks
   task_converter
   
   __init__
   task_converter
   
   convert
   task_converter
   
   _topological_sort
   task_converter
   
   _get_node_priority
   task_converter
   
   _create_task_from_node
   task_converter
   
   _interpolate_variables
   task_converter
   
   _resolve_nested_variable
   task_converter
   - 模組: 任務規劃模組

---

### Flow 152

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: task_generator
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   from_strategy
   task_generator
   - 模組: 任務規劃模組

---

### Flow 157

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: task_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   enhanced_unified_caller
   
   __init__
   enhanced_unified_caller
   
   initialize
   enhanced_unified_caller
   
   _setup_protocol_adapters
   enhanced_unified_caller
   
   _init_endpoints
   enhanced_unified_caller
   
   call_function
   enhanced_unified_caller
   
   call_multiple_functions
   enhanced_unified_caller
   
   health_check
   enhanced_unified_caller
   
   cleanup
   enhanced_unified_caller
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   task_executor
   
   execute_task
   task_executor
   
   _execute_by_service_type
   task_executor
   
   _execute_scan_service
   task_executor
   
   _call_capability_dynamically
   task_executor
   
   _execute_function_service
   task_executor
   
   _execute_integration_service
   task_executor
   
   _execute_core_service
   task_executor
   
   _infer_capability_name
   task_executor
   - 模組: 任務規劃模組

---

### Flow 159

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   enhanced_unified_caller
   
   __init__
   enhanced_unified_caller
   
   initialize
   enhanced_unified_caller
   
   _setup_protocol_adapters
   enhanced_unified_caller
   
   _init_endpoints
   enhanced_unified_caller
   
   call_function
   enhanced_unified_caller
   
   call_multiple_functions
   enhanced_unified_caller
   
   health_check
   enhanced_unified_caller
   
   cleanup
   enhanced_unified_caller
   - 模組: 服務骨幹模組

5. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 170

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: task_dispatcher
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   task_dispatcher
   
   _get_topic_for_tool
   task_dispatcher
   
   dispatch_attack_plan
   task_dispatcher
   
   dispatch_step
   task_dispatcher
   
   _build_task_payload
   task_dispatcher
   
   _build_message
   task_dispatcher
   
   dispatch_scan_task
   task_dispatcher
   
   dispatch_batch_tasks
   task_dispatcher
   
   send_control_command
   task_dispatcher
   
   send_feedback
   task_dispatcher
   
   request_status
   task_dispatcher
   
   register_callback
   task_dispatcher
   
   unregister_callback
   task_dispatcher
   
   trigger_callback
   task_dispatcher
   - 模組: 任務規劃模組

---

### Flow 176

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: plan_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

---

### Flow 181

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: task_dispatcher
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   task_dispatcher
   
   _get_topic_for_tool
   task_dispatcher
   
   dispatch_attack_plan
   task_dispatcher
   
   dispatch_step
   task_dispatcher
   
   _build_task_payload
   task_dispatcher
   
   _build_message
   task_dispatcher
   
   dispatch_scan_task
   task_dispatcher
   
   dispatch_batch_tasks
   task_dispatcher
   
   send_control_command
   task_dispatcher
   
   send_feedback
   task_dispatcher
   
   request_status
   task_dispatcher
   
   register_callback
   task_dispatcher
   
   unregister_callback
   task_dispatcher
   
   trigger_callback
   task_dispatcher
   - 模組: 任務規劃模組

---

### Flow 184

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 208

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: execution_planner
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

---

### Flow 223

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   attack_executor
   
   execute_plan_with_ai_analysis
   attack_executor
   
   _generate_feedback_data
   attack_executor
   
   execute_plan
   attack_executor
   
   _execute_step
   attack_executor
   
   _simulate_step
   attack_executor
   
   _real_execute_step
   attack_executor
   
   _map_step_type_to_exploit_type
   attack_executor
   
   _enhanced_safety_check
   attack_executor
   
   _safety_check
   attack_executor
   
   _create_aborted_result
   attack_executor
   
   _generate_recommendations
   attack_executor
   
   get_statistics
   attack_executor
   - 模組: 任務規劃模組

---

### Flow 226

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 228

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   exploit_manager_legacy
   
   _initialize_exploits
   exploit_manager_legacy
   
   _register_exploit
   exploit_manager_legacy
   
   register_exploit
   exploit_manager_legacy
   
   get_exploits_by_type
   exploit_manager_legacy
   
   get_exploit
   exploit_manager_legacy
   
   execute_exploit
   exploit_manager_legacy
   
   _execute_exploit_by_type
   exploit_manager_legacy
   
   _test_idor_vulnerability
   exploit_manager_legacy
   
   _test_sql_injection
   exploit_manager_legacy
   
   _test_xss_vulnerability
   exploit_manager_legacy
   
   _test_auth_bypass
   exploit_manager_legacy
   
   _test_jwt_attack
   exploit_manager_legacy
   
   _test_graphql_injection
   exploit_manager_legacy
   
   get_statistics
   exploit_manager_legacy
   
   _count_by_type
   exploit_manager_legacy
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   attack_executor
   
   execute_plan_with_ai_analysis
   attack_executor
   
   _generate_feedback_data
   attack_executor
   
   execute_plan
   attack_executor
   
   _execute_step
   attack_executor
   
   _simulate_step
   attack_executor
   
   _real_execute_step
   attack_executor
   
   _map_step_type_to_exploit_type
   attack_executor
   
   _enhanced_safety_check
   attack_executor
   
   _safety_check
   attack_executor
   
   _create_aborted_result
   attack_executor
   
   _generate_recommendations
   attack_executor
   
   get_statistics
   attack_executor
   - 模組: 任務規劃模組

---

### Flow 274

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: bizlogic_attack_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   integration_example
   bizlogic_attack_executor
   
   __init__
   bizlogic_attack_executor
   
   execute_attack
   bizlogic_attack_executor
   
   _test_price_manipulation
   bizlogic_attack_executor
   
   _test_idor
   bizlogic_attack_executor
   
   _test_authorization_bypass
   bizlogic_attack_executor
   
   _test_race_condition
   bizlogic_attack_executor
   
   _test_coupon_abuse
   bizlogic_attack_executor
   
   _send_price_manipulation_request
   bizlogic_attack_executor
   
   _send_idor_request
   bizlogic_attack_executor
   
   _send_workflow_bypass_request
   bizlogic_attack_executor
   
   _send_race_condition_request
   bizlogic_attack_executor
   
   _send_coupon_request
   bizlogic_attack_executor
   
   test
   bizlogic_attack_executor
   - 模組: 任務規劃模組

---

### Flow 283

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: execution_planner
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

---

## 外學模組 (external_learning)

包含 23 條數據流

### Flow 5

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 6

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

3. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 8

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: rl_trainers
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

---

### Flow 11

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 55

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 56

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 58

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 59

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

4. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 79

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 96

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 97

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: rl_models
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

---

### Flow 182

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 197

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 198

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 200

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: rl_trainers
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

---

### Flow 211

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 220

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 262

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: rl_trainers
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

---

### Flow 287

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

5. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 288

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 289

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

### Flow 316

- **長度**: 4 步
- **起點**: monitoring
- **終點**: model_trainer
- **主要模組**: 外學模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   monitor_performance
   monitoring
   
   __init__
   monitoring
   
   record_duration
   monitoring
   
   increment_counter
   monitoring
   
   set_gauge
   monitoring
   
   _make_key
   monitoring
   
   get_metrics_summary
   monitoring
   
   update_component_health
   monitoring
   
   get_system_health_status
   monitoring
   
   check_component_freshness
   monitoring
   
   decorator
   monitoring
   
   wrapper
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

---

### Flow 317

- **長度**: 5 步
- **起點**: monitoring
- **終點**: training_orchestrator
- **主要模組**: 外學模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   monitor_performance
   monitoring
   
   __init__
   monitoring
   
   record_duration
   monitoring
   
   increment_counter
   monitoring
   
   set_gauge
   monitoring
   
   _make_key
   monitoring
   
   get_metrics_summary
   monitoring
   
   update_component_health
   monitoring
   
   get_system_health_status
   monitoring
   
   check_component_freshness
   monitoring
   
   decorator
   monitoring
   
   wrapper
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

4. **AI組件**
   __init__
   model_trainer
   
   train_supervised
   model_trainer
   
   train_reinforcement
   model_trainer
   
   train_dqn
   model_trainer
   
   train_ppo
   model_trainer
   
   _prepare_supervised_data
   model_trainer
   
   _extract_features
   model_trainer
   
   _prepare_rl_data
   model_trainer
   
   _build_state_vector
   model_trainer
   
   _encode_attack_type
   model_trainer
   
   _encode_action
   model_trainer
   
   _calculate_step_reward
   model_trainer
   
   _train_model_supervised
   model_trainer
   
   _train_model_rl
   model_trainer
   
   _evaluate_model
   model_trainer
   
   _save_model
   model_trainer
   
   load_model
   model_trainer
   
   test_on_scenario
   model_trainer
   
   _increment_version
   model_trainer
   
   _persist_training_result
   model_trainer
   - 模組: 外學模組

5. **程式組件**
   __init__
   training_orchestrator
   
   _create_default_scenario_manager
   training_orchestrator
   
   _create_default_rag_engine
   training_orchestrator
   
   _create_default_plan_executor
   training_orchestrator
   
   _create_default_experience_manager
   training_orchestrator
   
   _create_default_model_trainer
   training_orchestrator
   
   run_training_episode
   training_orchestrator
   
   _extract_experience_samples
   training_orchestrator
   
   _calculate_step_reward
   training_orchestrator
   
   _assess_finding_quality
   training_orchestrator
   
   _calculate_quality_score
   training_orchestrator
   
   _generate_learning_tags
   training_orchestrator
   
   run_training_batch
   training_orchestrator
   
   train_model
   training_orchestrator
   
   get_training_statistics
   training_orchestrator
   
   save_session
   training_orchestrator
   
   _save_single_session
   training_orchestrator
   
   _generate_ai_attack_plan
   training_orchestrator
   
   _analyze_target_context
   training_orchestrator
   
   _select_attack_tactics
   training_orchestrator
   
   _build_attack_plan
   training_orchestrator
   
   _technique_to_steps
   training_orchestrator
   
   _map_method_to_type
   training_orchestrator
   
   _generate_payload_for_method
   training_orchestrator
   
   _get_expected_outcome
   training_orchestrator
   
   _get_success_criteria
   training_orchestrator
   - 模組: 外學模組

---

## 核心能力模組 (core_capabilities)

包含 10 條數據流

### Flow 9

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: capability_orchestrator
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

---

### Flow 25

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 101

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: capability_orchestrator
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

---

### Flow 107

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   internal_loop_connector
   
   module_explorer
   internal_loop_connector
   
   capability_analyzer
   internal_loop_connector
   
   sync_capabilities_to_rag
   internal_loop_connector
   
   _enhance_capabilities
   internal_loop_connector
   
   _match_sub_category
   internal_loop_connector
   
   _categorize_capability
   internal_loop_connector
   
   _assess_complexity
   internal_loop_connector
   
   _generate_tags
   internal_loop_connector
   
   _build_invocation_metadata
   internal_loop_connector
   
   _get_go_module_port
   internal_loop_connector
   
   _get_rust_module_port
   internal_loop_connector
   
   _build_parameter_definitions
   internal_loop_connector
   
   _generate_param_example
   internal_loop_connector
   
   _build_return_definition
   internal_loop_connector
   
   _generate_usage_examples
   internal_loop_connector
   
   _convert_to_capability_model
   internal_loop_connector
   
   _build_basic_info_section
   internal_loop_connector
   
   _build_parameters_section
   internal_loop_connector
   
   _build_examples_section
   internal_loop_connector
   
   _build_health_section
   internal_loop_connector
   
   _build_dependencies_section
   internal_loop_connector
   
   _convert_to_documents
   internal_loop_connector
   
   _inject_to_rag
   internal_loop_connector
   
   query_self_awareness
   internal_loop_connector
   
   report_issue
   internal_loop_connector
   
   search_solution
   internal_loop_connector
   
   get_sync_status
   internal_loop_connector
   
   export_capabilities_json
   internal_loop_connector
   - 模組: 服務骨幹模組

5. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 109

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 130

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 201

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: capability_orchestrator
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

---

### Flow 222

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: attack_chain
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   attack_chain
   
   add_step
   attack_chain
   
   can_execute_step
   attack_chain
   
   get_next_steps
   attack_chain
   
   mark_step_completed
   attack_chain
   
   is_completed
   attack_chain
   
   get_progress
   attack_chain
   
   get_execution_path
   attack_chain
   
   get_summary
   attack_chain
   - 模組: 核心能力模組

---

### Flow 263

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: capability_orchestrator
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

---

### Flow 269

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

---

## 服務骨幹模組 (service_backbone)

包含 208 條數據流

### Flow 1

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: permission_matrix
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

---

### Flow 2

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

3. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 3

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: matrix_visualizer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

3. **程式組件**
   main
   matrix_visualizer
   
   make_subplots
   matrix_visualizer
   
   __init__
   matrix_visualizer
   
   generate_heatmap
   matrix_visualizer
   
   generate_coverage_chart
   matrix_visualizer
   
   generate_role_comparison_chart
   matrix_visualizer
   
   generate_html_report
   matrix_visualizer
   
   _generate_all_charts
   matrix_visualizer
   
   _get_analysis_data
   matrix_visualizer
   
   _get_html_template
   matrix_visualizer
   
   _render_html_template
   matrix_visualizer
   
   export_to_csv
   matrix_visualizer
   
   add_trace
   matrix_visualizer
   
   update_layout
   matrix_visualizer
   
   to_html
   matrix_visualizer
   
   write_html
   matrix_visualizer
   - 模組: 服務骨幹模組

---

### Flow 4

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 10

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 14

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 15

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: nlg_system
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   nlg_system
   
   _init_response_templates
   nlg_system
   
   _init_context_analyzers
   nlg_system
   
   generate_response
   nlg_system
   
   _analyze_context
   nlg_system
   
   _detect_intent
   nlg_system
   
   _extract_entities
   nlg_system
   
   _analyze_sentiment
   nlg_system
   
   _extract_technical_details
   nlg_system
   
   _determine_response_type
   nlg_system
   
   _select_template
   nlg_system
   
   _fill_template
   nlg_system
   
   _generate_result_detail
   nlg_system
   
   _extract_filename
   nlg_system
   
   _post_process_response
   nlg_system
   - 模組: 服務骨幹模組

---

### Flow 16

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 17

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 18

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: attack_validator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   attack_validator
   
   _load_false_positive_patterns
   attack_validator
   
   validate_result
   attack_validator
   
   _basic_validation
   attack_validator
   
   _default_validation
   attack_validator
   
   _validate_sql_injection
   attack_validator
   
   _validate_xss
   attack_validator
   
   _validate_command_injection
   attack_validator
   
   _check_false_positive
   attack_validator
   
   batch_validate
   attack_validator
   
   get_statistics
   attack_validator
   - 模組: 服務骨幹模組

---

### Flow 19

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 20

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_unified_vector_store
   unified_vector_store
   
   __init__
   unified_vector_store
   
   initialize
   unified_vector_store
   
   _migrate_from_legacy
   unified_vector_store
   
   _get_embedding_model
   unified_vector_store
   
   _simple_embedding
   unified_vector_store
   
   add_document
   unified_vector_store
   
   add_batch
   unified_vector_store
   
   search
   unified_vector_store
   
   delete_document
   unified_vector_store
   
   get_document
   unified_vector_store
   
   get_statistics
   unified_vector_store
   
   close
   unified_vector_store
   - 模組: 服務骨幹模組

---

### Flow 21

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: aiva_flow_classifier_final
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

---

### Flow 23

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

5. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 24

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 26

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: experience_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

5. **程式組件**
   integrate_with_repository_example
   experience_manager
   
   __init__
   experience_manager
   
   to_dict
   experience_manager
   
   push
   experience_manager
   
   _persist_to_integration
   experience_manager
   
   load_from_integration
   experience_manager
   
   sample
   experience_manager
   
   prioritized_sample
   experience_manager
   
   create_dataset
   experience_manager
   
   get_statistics
   experience_manager
   
   clear
   experience_manager
   
   __len__
   experience_manager
   
   __repr__
   experience_manager
   
   add_sample
   experience_manager
   
   get_high_quality_samples
   experience_manager
   - 模組: 服務骨幹模組

---

### Flow 27

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: trace_recorder
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

5. **程式組件**
   to_dict
   trace_recorder
   
   to_json
   trace_recorder
   
   add_entry
   trace_recorder
   
   get_entries_by_task
   trace_recorder
   
   get_entries_by_type
   trace_recorder
   
   finalize
   trace_recorder
   
   __init__
   trace_recorder
   
   start_trace
   trace_recorder
   
   record
   trace_recorder
   
   record_task_start
   trace_recorder
   
   record_task_end
   trace_recorder
   
   record_http_request
   trace_recorder
   
   record_http_response
   trace_recorder
   
   record_log
   trace_recorder
   
   record_error
   trace_recorder
   
   finalize_trace
   trace_recorder
   
   get_trace
   trace_recorder
   - 模組: 服務骨幹模組

---

### Flow 28

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: event_listener
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

---

### Flow 29

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 30

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

---

### Flow 31

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 34

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: cli_integration_example
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   example_usage
   cli_integration_example
   
   __init__
   cli_integration_example
   
   _load_flows_data
   cli_integration_example
   
   execute_capability
   cli_integration_example
   
   _select_flow_path
   cli_integration_example
   
   _execute_flow
   cli_integration_example
   - 模組: 服務骨幹模組

---

### Flow 36

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 37

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: nlg_system
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   nlg_system
   
   _init_response_templates
   nlg_system
   
   _init_context_analyzers
   nlg_system
   
   generate_response
   nlg_system
   
   _analyze_context
   nlg_system
   
   _detect_intent
   nlg_system
   
   _extract_entities
   nlg_system
   
   _analyze_sentiment
   nlg_system
   
   _extract_technical_details
   nlg_system
   
   _determine_response_type
   nlg_system
   
   _select_template
   nlg_system
   
   _fill_template
   nlg_system
   
   _generate_result_detail
   nlg_system
   
   _extract_filename
   nlg_system
   
   _post_process_response
   nlg_system
   - 模組: 服務骨幹模組

---

### Flow 38

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 39

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 40

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: attack_validator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   attack_validator
   
   _load_false_positive_patterns
   attack_validator
   
   validate_result
   attack_validator
   
   _basic_validation
   attack_validator
   
   _default_validation
   attack_validator
   
   _validate_sql_injection
   attack_validator
   
   _validate_xss
   attack_validator
   
   _validate_command_injection
   attack_validator
   
   _check_false_positive
   attack_validator
   
   batch_validate
   attack_validator
   
   get_statistics
   attack_validator
   - 模組: 服務骨幹模組

---

### Flow 41

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 42

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

4. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_unified_vector_store
   unified_vector_store
   
   __init__
   unified_vector_store
   
   initialize
   unified_vector_store
   
   _migrate_from_legacy
   unified_vector_store
   
   _get_embedding_model
   unified_vector_store
   
   _simple_embedding
   unified_vector_store
   
   add_document
   unified_vector_store
   
   add_batch
   unified_vector_store
   
   search
   unified_vector_store
   
   delete_document
   unified_vector_store
   
   get_document
   unified_vector_store
   
   get_statistics
   unified_vector_store
   
   close
   unified_vector_store
   - 模組: 服務骨幹模組

---

### Flow 43

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: check_flow_details
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   check_flow_details
   check_flow_details
   - 模組: 服務骨幹模組

---

### Flow 44

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: find_testable_flows
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   find_testable_flows
   find_testable_flows
   - 模組: 服務骨幹模組

---

### Flow 45

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: verify_classification
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   verify_classification
   verify_classification
   - 模組: 服務骨幹模組

---

### Flow 47

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: execution_status_monitor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

---

### Flow 48

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

4. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 49

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: unified_memory_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

4. **程式組件**
   __init__
   unified_memory_manager
   
   _generate_cache_key
   unified_memory_manager
   
   get_cached_prediction
   unified_memory_manager
   
   cache_prediction
   unified_memory_manager
   
   _evict_oldest_cache_entry
   unified_memory_manager
   
   clear_cache
   unified_memory_manager
   
   create_component_pool
   unified_memory_manager
   
   get_component_pool
   unified_memory_manager
   
   register_weak_ref
   unified_memory_manager
   
   start_monitoring
   unified_memory_manager
   
   stop_monitoring
   unified_memory_manager
   
   _monitor_memory
   unified_memory_manager
   
   _force_cleanup
   unified_memory_manager
   
   _cleanup_expired_cache
   unified_memory_manager
   
   process_batch
   unified_memory_manager
   
   process_large_dataset
   unified_memory_manager
   
   _get_memory_usage_mb
   unified_memory_manager
   
   _record_memory_usage
   unified_memory_manager
   
   optimize_memory
   unified_memory_manager
   
   get_comprehensive_stats
   unified_memory_manager
   
   _get_cache_stats
   unified_memory_manager
   
   _get_memory_stats
   unified_memory_manager
   
   _get_pool_stats
   unified_memory_manager
   
   get_component
   unified_memory_manager
   
   get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

---

### Flow 50

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

4. **程式組件**
   __init__
   unified_memory_manager
   
   _generate_cache_key
   unified_memory_manager
   
   get_cached_prediction
   unified_memory_manager
   
   cache_prediction
   unified_memory_manager
   
   _evict_oldest_cache_entry
   unified_memory_manager
   
   clear_cache
   unified_memory_manager
   
   create_component_pool
   unified_memory_manager
   
   get_component_pool
   unified_memory_manager
   
   register_weak_ref
   unified_memory_manager
   
   start_monitoring
   unified_memory_manager
   
   stop_monitoring
   unified_memory_manager
   
   _monitor_memory
   unified_memory_manager
   
   _force_cleanup
   unified_memory_manager
   
   _cleanup_expired_cache
   unified_memory_manager
   
   process_batch
   unified_memory_manager
   
   process_large_dataset
   unified_memory_manager
   
   _get_memory_usage_mb
   unified_memory_manager
   
   _record_memory_usage
   unified_memory_manager
   
   optimize_memory
   unified_memory_manager
   
   get_comprehensive_stats
   unified_memory_manager
   
   _get_cache_stats
   unified_memory_manager
   
   _get_memory_stats
   unified_memory_manager
   
   _get_pool_stats
   unified_memory_manager
   
   get_component
   unified_memory_manager
   
   get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

5. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 52

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: payload_generator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

4. **程式組件**
   __init__
   payload_generator
   
   _load_templates
   payload_generator
   
   generate_with_target_analysis
   payload_generator
   
   _analyze_target_environment
   payload_generator
   
   _select_payload_templates
   payload_generator
   
   _is_template_suitable
   payload_generator
   
   _customize_payloads
   payload_generator
   
   _validate_payloads
   payload_generator
   
   _validate_single_payload
   payload_generator
   
   _format_output
   payload_generator
   
   _generate_usage_recommendations
   payload_generator
   
   generate
   payload_generator
   
   _encode_payload
   payload_generator
   
   generate_fuzzing_payloads
   payload_generator
   
   get_statistics
   payload_generator
   - 模組: 服務骨幹模組

---

### Flow 53

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 54

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 61

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 62

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 64

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 66

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: weight_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   
   __init__
   weight_manager
   
   save_model_weights
   weight_manager
   
   load_model_weights
   weight_manager
   
   list_available_weights
   weight_manager
   
   _list_model_versions
   weight_manager
   
   _extract_version_info
   weight_manager
   
   _list_all_models
   weight_manager
   
   delete_weights
   weight_manager
   
   _find_weight_file
   weight_manager
   
   _calculate_file_hash
   weight_manager
   
   _save_metadata
   weight_manager
   
   _load_and_verify_metadata
   weight_manager
   
   _verify_model_compatibility
   weight_manager
   
   _create_backup
   weight_manager
   
   _cleanup_old_backups
   weight_manager
   - 模組: 服務骨幹模組

---

### Flow 67

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

---

### Flow 70

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 71

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: external_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   external_loop_connector
   
   comparator
   external_loop_connector
   
   trainer
   external_loop_connector
   
   weight_manager
   external_loop_connector
   
   process_execution_result
   external_loop_connector
   
   _analyze_deviations
   external_loop_connector
   
   _is_significant_deviation
   external_loop_connector
   
   _train_from_experience
   external_loop_connector
   
   _register_new_weights
   external_loop_connector
   
   get_loop_status
   external_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 72

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: internal_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   internal_loop_connector
   
   module_explorer
   internal_loop_connector
   
   capability_analyzer
   internal_loop_connector
   
   sync_capabilities_to_rag
   internal_loop_connector
   
   _enhance_capabilities
   internal_loop_connector
   
   _match_sub_category
   internal_loop_connector
   
   _categorize_capability
   internal_loop_connector
   
   _assess_complexity
   internal_loop_connector
   
   _generate_tags
   internal_loop_connector
   
   _build_invocation_metadata
   internal_loop_connector
   
   _get_go_module_port
   internal_loop_connector
   
   _get_rust_module_port
   internal_loop_connector
   
   _build_parameter_definitions
   internal_loop_connector
   
   _generate_param_example
   internal_loop_connector
   
   _build_return_definition
   internal_loop_connector
   
   _generate_usage_examples
   internal_loop_connector
   
   _convert_to_capability_model
   internal_loop_connector
   
   _build_basic_info_section
   internal_loop_connector
   
   _build_parameters_section
   internal_loop_connector
   
   _build_examples_section
   internal_loop_connector
   
   _build_health_section
   internal_loop_connector
   
   _build_dependencies_section
   internal_loop_connector
   
   _convert_to_documents
   internal_loop_connector
   
   _inject_to_rag
   internal_loop_connector
   
   query_self_awareness
   internal_loop_connector
   
   report_issue
   internal_loop_connector
   
   search_solution
   internal_loop_connector
   
   get_sync_status
   internal_loop_connector
   
   export_capabilities_json
   internal_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 74

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: protocol_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   create_http_adapter
   protocol_adapter
   
   send_request
   protocol_adapter
   
   handle_response
   protocol_adapter
   
   __init__
   protocol_adapter
   
   _adapt_request_data
   protocol_adapter
   
   _adapt_response_data
   protocol_adapter
   - 模組: 服務骨幹模組

---

### Flow 75

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 76

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   
   __init__
   message_broker
   
   connect
   message_broker
   
   _declare_exchanges
   message_broker
   
   publish_message
   message_broker
   
   subscribe
   message_broker
   
   create_rpc_client
   message_broker
   
   get_rpc_client
   message_broker
   
   disconnect
   message_broker
   
   setup
   message_broker
   
   _on_response
   message_broker
   
   call
   message_broker
   
   is_expired
   message_broker
   
   can_retry
   message_broker
   
   matches
   message_broker
   
   _match_pattern
   message_broker
   
   start_event_system
   message_broker
   
   stop_event_system
   message_broker
   
   publish_event
   message_broker
   
   subscribe_event
   message_broker
   
   unsubscribe_event
   message_broker
   
   _process_events
   message_broker
   
   _handle_event
   message_broker
   
   get_event_statistics
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 77

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: result_collector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   result_collector
   
   start
   result_collector
   
   _subscribe_scan_results
   result_collector
   
   _subscribe_function_results
   result_collector
   
   _subscribe_task_updates
   result_collector
   
   _subscribe_findings
   result_collector
   
   _handle_scan_result
   result_collector
   
   _handle_function_result
   result_collector
   
   _handle_task_update
   result_collector
   
   _handle_finding
   result_collector
   
   _store_result
   result_collector
   
   _trigger_handlers
   result_collector
   
   register_handler
   result_collector
   
   unregister_handler
   result_collector
   
   _set_pending_result
   result_collector
   
   wait_for_result
   result_collector
   
   get_recent_results
   result_collector
   
   get_statistics
   result_collector
   - 模組: 服務骨幹模組

---

### Flow 80

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 82

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 83

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scan_module_interface
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

---

### Flow 84

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: two_phase_scan_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   two_phase_scan_orchestrator
   
   execute_two_phase_scan
   two_phase_scan_orchestrator
   
   _execute_phase0
   two_phase_scan_orchestrator
   
   _execute_phase1
   two_phase_scan_orchestrator
   
   _analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   _select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 85

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: to_functions
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   to_function_message
   to_functions
   - 模組: 服務骨幹模組

---

### Flow 86

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scan_result_processor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

---

### Flow 87

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: skill_graph
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   skill_graph
   
   build_graph
   skill_graph
   
   _extract_success_rate
   skill_graph
   
   _extract_usage_count
   skill_graph
   
   _build_node_metadata
   skill_graph
   
   _create_skill_nodes
   skill_graph
   
   _analyze_relationships
   skill_graph
   
   _analyze_prerequisite_relationships
   skill_graph
   
   _analyze_tag_similarity_relationships
   skill_graph
   
   _analyze_language_ecosystem_relationships
   skill_graph
   
   _analyze_topic_relationships
   skill_graph
   
   _check_io_compatibility
   skill_graph
   
   _analyze_io_relationships
   skill_graph
   
   _is_compatible_io
   skill_graph
   
   _build_networkx_graph
   skill_graph
   
   find_optimal_path
   skill_graph
   
   _find_goal_capabilities
   skill_graph
   
   _create_skill_path
   skill_graph
   
   get_capability_recommendations
   skill_graph
   
   analyze_capability_centrality
   skill_graph
   
   initialize
   skill_graph
   
   rebuild_if_needed
   skill_graph
   
   find_execution_path
   skill_graph
   
   get_recommendations
   skill_graph
   
   analyze_centrality
   skill_graph
   
   get_graph_statistics
   skill_graph
   - 模組: 服務骨幹模組

---

### Flow 88

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 89

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

3. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

4. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 91

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 94

- **長度**: 3 步
- **起點**: scalable_bio_trainer
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 95

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 99

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

5. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 100

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   
   __init__
   ai_capability_query
   
   vector_store
   ai_capability_query
   
   kb
   ai_capability_query
   
   connector
   ai_capability_query
   
   query
   ai_capability_query
   
   display_results
   ai_capability_query
   
   _display_results_rich
   ai_capability_query
   
   _display_results_plain
   ai_capability_query
   
   show_statistics
   ai_capability_query
   
   _display_statistics_rich
   ai_capability_query
   
   _display_statistics_plain
   ai_capability_query
   
   get_workflow_recommendation
   ai_capability_query
   
   query_by_module
   ai_capability_query
   
   query_by_language
   ai_capability_query
   
   _handle_command_line
   ai_capability_query
   
   _handle_interactive_mode
   ai_capability_query
   
   main
   ai_capability_query
   - 模組: 認知核心模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 102

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 104

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 105

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: external_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   external_loop_connector
   
   comparator
   external_loop_connector
   
   trainer
   external_loop_connector
   
   weight_manager
   external_loop_connector
   
   process_execution_result
   external_loop_connector
   
   _analyze_deviations
   external_loop_connector
   
   _is_significant_deviation
   external_loop_connector
   
   _train_from_experience
   external_loop_connector
   
   _register_new_weights
   external_loop_connector
   
   get_loop_status
   external_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 106

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: internal_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   internal_loop_connector
   
   module_explorer
   internal_loop_connector
   
   capability_analyzer
   internal_loop_connector
   
   sync_capabilities_to_rag
   internal_loop_connector
   
   _enhance_capabilities
   internal_loop_connector
   
   _match_sub_category
   internal_loop_connector
   
   _categorize_capability
   internal_loop_connector
   
   _assess_complexity
   internal_loop_connector
   
   _generate_tags
   internal_loop_connector
   
   _build_invocation_metadata
   internal_loop_connector
   
   _get_go_module_port
   internal_loop_connector
   
   _get_rust_module_port
   internal_loop_connector
   
   _build_parameter_definitions
   internal_loop_connector
   
   _generate_param_example
   internal_loop_connector
   
   _build_return_definition
   internal_loop_connector
   
   _generate_usage_examples
   internal_loop_connector
   
   _convert_to_capability_model
   internal_loop_connector
   
   _build_basic_info_section
   internal_loop_connector
   
   _build_parameters_section
   internal_loop_connector
   
   _build_examples_section
   internal_loop_connector
   
   _build_health_section
   internal_loop_connector
   
   _build_dependencies_section
   internal_loop_connector
   
   _convert_to_documents
   internal_loop_connector
   
   _inject_to_rag
   internal_loop_connector
   
   query_self_awareness
   internal_loop_connector
   
   report_issue
   internal_loop_connector
   
   search_solution
   internal_loop_connector
   
   get_sync_status
   internal_loop_connector
   
   export_capabilities_json
   internal_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 108

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: nlg_system
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   nlg_system
   
   _init_response_templates
   nlg_system
   
   _init_context_analyzers
   nlg_system
   
   generate_response
   nlg_system
   
   _analyze_context
   nlg_system
   
   _detect_intent
   nlg_system
   
   _extract_entities
   nlg_system
   
   _analyze_sentiment
   nlg_system
   
   _extract_technical_details
   nlg_system
   
   _determine_response_type
   nlg_system
   
   _select_template
   nlg_system
   
   _fill_template
   nlg_system
   
   _generate_result_detail
   nlg_system
   
   _extract_filename
   nlg_system
   
   _post_process_response
   nlg_system
   - 模組: 服務骨幹模組

---

### Flow 110

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: experience_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

5. **程式組件**
   integrate_with_repository_example
   experience_manager
   
   __init__
   experience_manager
   
   to_dict
   experience_manager
   
   push
   experience_manager
   
   _persist_to_integration
   experience_manager
   
   load_from_integration
   experience_manager
   
   sample
   experience_manager
   
   prioritized_sample
   experience_manager
   
   create_dataset
   experience_manager
   
   get_statistics
   experience_manager
   
   clear
   experience_manager
   
   __len__
   experience_manager
   
   __repr__
   experience_manager
   
   add_sample
   experience_manager
   
   get_high_quality_samples
   experience_manager
   - 模組: 服務骨幹模組

---

### Flow 111

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: trace_recorder
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   
   __init__
   capability_registry
   
   to_dict
   capability_registry
   
   __new__
   capability_registry
   
   load_from_exploration
   capability_registry
   
   register_capability
   capability_registry
   
   get_capability
   capability_registry
   
   list_capabilities
   capability_registry
   
   list_modules
   capability_registry
   
   search_capabilities
   capability_registry
   
   get_statistics
   capability_registry
   
   clear
   capability_registry
   
   test_registry
   capability_registry
   - 模組: 核心能力模組

5. **程式組件**
   to_dict
   trace_recorder
   
   to_json
   trace_recorder
   
   add_entry
   trace_recorder
   
   get_entries_by_task
   trace_recorder
   
   get_entries_by_type
   trace_recorder
   
   finalize
   trace_recorder
   
   __init__
   trace_recorder
   
   start_trace
   trace_recorder
   
   record
   trace_recorder
   
   record_task_start
   trace_recorder
   
   record_task_end
   trace_recorder
   
   record_http_request
   trace_recorder
   
   record_http_response
   trace_recorder
   
   record_log
   trace_recorder
   
   record_error
   trace_recorder
   
   finalize_trace
   trace_recorder
   
   get_trace
   trace_recorder
   - 模組: 服務骨幹模組

---

### Flow 112

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: multilang_coordinator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

---

### Flow 114

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_unified_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   get_unified_caller
   enhanced_unified_caller
   
   __init__
   enhanced_unified_caller
   
   initialize
   enhanced_unified_caller
   
   _setup_protocol_adapters
   enhanced_unified_caller
   
   _init_endpoints
   enhanced_unified_caller
   
   call_function
   enhanced_unified_caller
   
   call_multiple_functions
   enhanced_unified_caller
   
   health_check
   enhanced_unified_caller
   
   cleanup
   enhanced_unified_caller
   - 模組: 服務骨幹模組

---

### Flow 116

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 117

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: skill_graph
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   skill_graph
   
   build_graph
   skill_graph
   
   _extract_success_rate
   skill_graph
   
   _extract_usage_count
   skill_graph
   
   _build_node_metadata
   skill_graph
   
   _create_skill_nodes
   skill_graph
   
   _analyze_relationships
   skill_graph
   
   _analyze_prerequisite_relationships
   skill_graph
   
   _analyze_tag_similarity_relationships
   skill_graph
   
   _analyze_language_ecosystem_relationships
   skill_graph
   
   _analyze_topic_relationships
   skill_graph
   
   _check_io_compatibility
   skill_graph
   
   _analyze_io_relationships
   skill_graph
   
   _is_compatible_io
   skill_graph
   
   _build_networkx_graph
   skill_graph
   
   find_optimal_path
   skill_graph
   
   _find_goal_capabilities
   skill_graph
   
   _create_skill_path
   skill_graph
   
   get_capability_recommendations
   skill_graph
   
   analyze_capability_centrality
   skill_graph
   
   initialize
   skill_graph
   
   rebuild_if_needed
   skill_graph
   
   find_execution_path
   skill_graph
   
   get_recommendations
   skill_graph
   
   analyze_centrality
   skill_graph
   
   get_graph_statistics
   skill_graph
   - 模組: 服務骨幹模組

---

### Flow 118

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   multilang_coordinator
   
   initialize
   multilang_coordinator
   
   check_module_availability
   multilang_coordinator
   
   execute_task
   multilang_coordinator
   
   _execute_python_task
   multilang_coordinator
   
   _select_best_language
   multilang_coordinator
   
   get_status
   multilang_coordinator
   
   enable_module
   multilang_coordinator
   
   disable_module
   multilang_coordinator
   
   _check_rust_service
   multilang_coordinator
   
   _check_go_service
   multilang_coordinator
   
   _check_typescript_service
   multilang_coordinator
   
   call_rust_ai
   multilang_coordinator
   
   call_go_ai
   multilang_coordinator
   
   call_typescript_ai
   multilang_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 119

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: event_listener
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

---

### Flow 120

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 121

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: experience_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   integrate_with_repository_example
   experience_manager
   
   __init__
   experience_manager
   
   to_dict
   experience_manager
   
   push
   experience_manager
   
   _persist_to_integration
   experience_manager
   
   load_from_integration
   experience_manager
   
   sample
   experience_manager
   
   prioritized_sample
   experience_manager
   
   create_dataset
   experience_manager
   
   get_statistics
   experience_manager
   
   clear
   experience_manager
   
   __len__
   experience_manager
   
   __repr__
   experience_manager
   
   add_sample
   experience_manager
   
   get_high_quality_samples
   experience_manager
   - 模組: 服務骨幹模組

---

### Flow 122

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: aiva_flow_analyzer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

---

### Flow 123

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: ast_parser
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

5. **程式組件**
   __repr__
   ast_parser
   
   add_node
   ast_parser
   
   add_edge
   ast_parser
   
   get_start_node
   ast_parser
   
   get_next_nodes
   ast_parser
   
   validate
   ast_parser
   
   __init__
   ast_parser
   
   parse_dict
   ast_parser
   
   parse_text
   ast_parser
   
   create_example_sqli_flow
   ast_parser
   - 模組: 服務骨幹模組

---

### Flow 125

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

5. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 126

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 127

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_python_file
   aiva_flow_analyzer
   
   main
   aiva_flow_analyzer
   
   __init__
   aiva_flow_analyzer
   
   _sanitize_id
   aiva_flow_analyzer
   
   _validate_direction
   aiva_flow_analyzer
   
   add
   aiva_flow_analyzer
   
   link
   aiva_flow_analyzer
   
   render_mermaid
   aiva_flow_analyzer
   
   _add_node_definitions
   aiva_flow_analyzer
   
   _get_node_definition
   aiva_flow_analyzer
   
   _add_node_connections
   aiva_flow_analyzer
   
   _get_connection_definition
   aiva_flow_analyzer
   
   _debug_print
   aiva_flow_analyzer
   
   visit_FunctionDef
   aiva_flow_analyzer
   
   visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   visit_If
   aiva_flow_analyzer
   
   visit_For
   aiva_flow_analyzer
   
   visit_While
   aiva_flow_analyzer
   
   visit_Break
   aiva_flow_analyzer
   
   visit_Continue
   aiva_flow_analyzer
   
   visit_Try
   aiva_flow_analyzer
   
   _process_try_body
   aiva_flow_analyzer
   
   _process_exception_handlers
   aiva_flow_analyzer
   
   _get_exception_handler_name
   aiva_flow_analyzer
   
   _process_try_else_finally
   aiva_flow_analyzer
   
   _process_else_clause
   aiva_flow_analyzer
   
   _process_finally_clause
   aiva_flow_analyzer
   
   visit_With
   aiva_flow_analyzer
   
   visit_Call
   aiva_flow_analyzer
   
   visit_Assign
   aiva_flow_analyzer
   
   visit_Expr
   aiva_flow_analyzer
   
   generic_visit
   aiva_flow_analyzer
   
   add_script
   aiva_flow_analyzer
   
   _analyze_script_head_tail
   aiva_flow_analyzer
   
   find_real_connections
   aiva_flow_analyzer
   
   _find_function_provider
   aiva_flow_analyzer
   
   build_data_flow_chains
   aiva_flow_analyzer
   
   _find_head_scripts
   aiva_flow_analyzer
   
   _build_all_paths_from_head
   aiva_flow_analyzer
   
   analyze_branch_patterns
   aiva_flow_analyzer
   
   add_graph
   aiva_flow_analyzer
   
   find_stitchable_graphs
   aiva_flow_analyzer
   
   _dfs_search_stitchable
   aiva_flow_analyzer
   
   generate_stitched_mermaid
   aiva_flow_analyzer
   
   _find_connection_function
   aiva_flow_analyzer
   
   _get_target_directory
   aiva_flow_analyzer
   
   analyze_directory
   aiva_flow_analyzer
   
   _generate_chain_mermaid
   aiva_flow_analyzer
   
   _extract_functions_from_graph
   aiva_flow_analyzer
   
   _find_meaningful_entry_functions
   aiva_flow_analyzer
   
   save_results
   aiva_flow_analyzer
   
   run_analysis
   aiva_flow_analyzer
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 128

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: aiva_flow_classifier_final
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

---

### Flow 131

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: event_listener
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

---

### Flow 132

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

---

### Flow 134

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: cli_integration_example
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   example_usage
   cli_integration_example
   
   __init__
   cli_integration_example
   
   _load_flows_data
   cli_integration_example
   
   execute_capability
   cli_integration_example
   
   _select_flow_path
   cli_integration_example
   
   _execute_flow
   cli_integration_example
   - 模組: 服務骨幹模組

---

### Flow 136

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 137

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: context_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_context_manager
   context_manager
   
   __init__
   context_manager
   
   create_context
   context_manager
   
   _update_session
   context_manager
   
   get_context
   context_manager
   
   update_context
   context_manager
   
   set_variable
   context_manager
   
   get_variable
   context_manager
   
   add_history
   context_manager
   
   get_context_history
   context_manager
   
   get_session_contexts
   context_manager
   
   get_session_info
   context_manager
   
   cleanup_context
   context_manager
   
   cleanup_session
   context_manager
   
   cleanup_expired_contexts
   context_manager
   
   cleanup_expired_sessions
   context_manager
   
   get_context_stats
   context_manager
   - 模組: 服務骨幹模組

---

### Flow 140

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: execution_status_monitor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

---

### Flow 141

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 142

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_memory_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   unified_memory_manager
   
   _generate_cache_key
   unified_memory_manager
   
   get_cached_prediction
   unified_memory_manager
   
   cache_prediction
   unified_memory_manager
   
   _evict_oldest_cache_entry
   unified_memory_manager
   
   clear_cache
   unified_memory_manager
   
   create_component_pool
   unified_memory_manager
   
   get_component_pool
   unified_memory_manager
   
   register_weak_ref
   unified_memory_manager
   
   start_monitoring
   unified_memory_manager
   
   stop_monitoring
   unified_memory_manager
   
   _monitor_memory
   unified_memory_manager
   
   _force_cleanup
   unified_memory_manager
   
   _cleanup_expired_cache
   unified_memory_manager
   
   process_batch
   unified_memory_manager
   
   process_large_dataset
   unified_memory_manager
   
   _get_memory_usage_mb
   unified_memory_manager
   
   _record_memory_usage
   unified_memory_manager
   
   optimize_memory
   unified_memory_manager
   
   get_comprehensive_stats
   unified_memory_manager
   
   _get_cache_stats
   unified_memory_manager
   
   _get_memory_stats
   unified_memory_manager
   
   _get_pool_stats
   unified_memory_manager
   
   get_component
   unified_memory_manager
   
   get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

---

### Flow 146

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **混合組件**
   __init__
   plan_executor
   
   execute_plan
   plan_executor
   
   _publish_completion_event
   plan_executor
   
   _execute_step
   plan_executor
   
   _prepare_task_payload
   plan_executor
   
   _send_task
   plan_executor
   
   _wait_for_result
   plan_executor
   
   _check_dependencies
   plan_executor
   
   _should_continue
   plan_executor
   
   _record_skipped_step
   plan_executor
   
   _calculate_metrics
   plan_executor
   
   _calculate_sequence_accuracy
   plan_executor
   
   _generate_recommendations
   plan_executor
   
   _persist_result
   plan_executor
   
   get_session
   plan_executor
   
   abort_session
   plan_executor
   - 模組: 任務規劃模組

5. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 148

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: ast_parser
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __repr__
   ast_parser
   
   add_node
   ast_parser
   
   add_edge
   ast_parser
   
   get_start_node
   ast_parser
   
   get_next_nodes
   ast_parser
   
   validate
   ast_parser
   
   __init__
   ast_parser
   
   parse_dict
   ast_parser
   
   parse_text
   ast_parser
   
   create_example_sqli_flow
   ast_parser
   - 模組: 服務骨幹模組

---

### Flow 150

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: payload_generator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

5. **程式組件**
   __init__
   payload_generator
   
   _load_templates
   payload_generator
   
   generate_with_target_analysis
   payload_generator
   
   _analyze_target_environment
   payload_generator
   
   _select_payload_templates
   payload_generator
   
   _is_template_suitable
   payload_generator
   
   _customize_payloads
   payload_generator
   
   _validate_payloads
   payload_generator
   
   _validate_single_payload
   payload_generator
   
   _format_output
   payload_generator
   
   _generate_usage_recommendations
   payload_generator
   
   generate
   payload_generator
   
   _encode_payload
   payload_generator
   
   generate_fuzzing_payloads
   payload_generator
   
   get_statistics
   payload_generator
   - 模組: 服務骨幹模組

---

### Flow 153

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: tool_selector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __repr__
   tool_selector
   
   __init__
   tool_selector
   
   select_tool
   tool_selector
   
   _select_service_type
   tool_selector
   
   _determine_endpoint_and_function
   tool_selector
   
   _prepare_parameters
   tool_selector
   
   _determine_routing_key
   tool_selector
   - 模組: 服務骨幹模組

---

### Flow 154

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: protocol_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   create_http_adapter
   protocol_adapter
   
   send_request
   protocol_adapter
   
   handle_response
   protocol_adapter
   
   __init__
   protocol_adapter
   
   _adapt_request_data
   protocol_adapter
   
   _adapt_response_data
   protocol_adapter
   - 模組: 服務骨幹模組

---

### Flow 155

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 156

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: enhanced_unified_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   enhanced_unified_caller
   
   __init__
   enhanced_unified_caller
   
   initialize
   enhanced_unified_caller
   
   _setup_protocol_adapters
   enhanced_unified_caller
   
   _init_endpoints
   enhanced_unified_caller
   
   call_function
   enhanced_unified_caller
   
   call_multiple_functions
   enhanced_unified_caller
   
   health_check
   enhanced_unified_caller
   
   cleanup
   enhanced_unified_caller
   - 模組: 服務骨幹模組

---

### Flow 158

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   enhanced_unified_caller
   
   __init__
   enhanced_unified_caller
   
   initialize
   enhanced_unified_caller
   
   _setup_protocol_adapters
   enhanced_unified_caller
   
   _init_endpoints
   enhanced_unified_caller
   
   call_function
   enhanced_unified_caller
   
   call_multiple_functions
   enhanced_unified_caller
   
   health_check
   enhanced_unified_caller
   
   cleanup
   enhanced_unified_caller
   - 模組: 服務骨幹模組

5. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

---

### Flow 160

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

---

### Flow 161

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 162

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: matrix_visualizer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   main
   matrix_visualizer
   
   make_subplots
   matrix_visualizer
   
   __init__
   matrix_visualizer
   
   generate_heatmap
   matrix_visualizer
   
   generate_coverage_chart
   matrix_visualizer
   
   generate_role_comparison_chart
   matrix_visualizer
   
   generate_html_report
   matrix_visualizer
   
   _generate_all_charts
   matrix_visualizer
   
   _get_analysis_data
   matrix_visualizer
   
   _get_html_template
   matrix_visualizer
   
   _render_html_template
   matrix_visualizer
   
   export_to_csv
   matrix_visualizer
   
   add_trace
   matrix_visualizer
   
   update_layout
   matrix_visualizer
   
   to_html
   matrix_visualizer
   
   write_html
   matrix_visualizer
   - 模組: 服務骨幹模組

---

### Flow 165

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: core_service_coordinator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   
   __init__
   core_service_coordinator
   
   _initialize_core_components
   core_service_coordinator
   
   _initialize_shared_services
   core_service_coordinator
   
   _setup_monitoring_and_config
   core_service_coordinator
   
   _apply_initial_config
   core_service_coordinator
   
   _configure_security_middleware
   core_service_coordinator
   
   _on_config_changed
   core_service_coordinator
   
   start
   core_service_coordinator
   
   _start_shared_services
   core_service_coordinator
   
   _start_core_components
   core_service_coordinator
   
   stop
   core_service_coordinator
   
   _stop_core_components
   core_service_coordinator
   
   _stop_shared_services
   core_service_coordinator
   
   _cleanup_on_failure
   core_service_coordinator
   
   get_service_status
   core_service_coordinator
   
   health_check
   core_service_coordinator
   - 模組: 服務骨幹模組

---

### Flow 166

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   
   __init__
   core_service_coordinator
   
   _initialize_core_components
   core_service_coordinator
   
   _initialize_shared_services
   core_service_coordinator
   
   _setup_monitoring_and_config
   core_service_coordinator
   
   _apply_initial_config
   core_service_coordinator
   
   _configure_security_middleware
   core_service_coordinator
   
   _on_config_changed
   core_service_coordinator
   
   start
   core_service_coordinator
   
   _start_shared_services
   core_service_coordinator
   
   _start_core_components
   core_service_coordinator
   
   stop
   core_service_coordinator
   
   _stop_core_components
   core_service_coordinator
   
   _stop_shared_services
   core_service_coordinator
   
   _cleanup_on_failure
   core_service_coordinator
   
   get_service_status
   core_service_coordinator
   
   health_check
   core_service_coordinator
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 167

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 168

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 169

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: result_collector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   result_collector
   
   start
   result_collector
   
   _subscribe_scan_results
   result_collector
   
   _subscribe_function_results
   result_collector
   
   _subscribe_task_updates
   result_collector
   
   _subscribe_findings
   result_collector
   
   _handle_scan_result
   result_collector
   
   _handle_function_result
   result_collector
   
   _handle_task_update
   result_collector
   
   _handle_finding
   result_collector
   
   _store_result
   result_collector
   
   _trigger_handlers
   result_collector
   
   register_handler
   result_collector
   
   unregister_handler
   result_collector
   
   _set_pending_result
   result_collector
   
   wait_for_result
   result_collector
   
   get_recent_results
   result_collector
   
   get_statistics
   result_collector
   - 模組: 服務骨幹模組

---

### Flow 171

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: session_state_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   session_state_manager
   
   record_scan_result
   session_state_manager
   
   record_task_update
   session_state_manager
   
   get_session_status
   session_state_manager
   
   get_session_context
   session_state_manager
   
   update_context
   session_state_manager
   
   update_session_status
   session_state_manager
   - 模組: 服務骨幹模組

---

### Flow 172

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   session_state_manager
   
   record_scan_result
   session_state_manager
   
   record_task_update
   session_state_manager
   
   get_session_status
   session_state_manager
   
   get_session_context
   session_state_manager
   
   update_context
   session_state_manager
   
   update_session_status
   session_state_manager
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 173

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 174

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: external_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   external_loop_connector
   
   comparator
   external_loop_connector
   
   trainer
   external_loop_connector
   
   weight_manager
   external_loop_connector
   
   process_execution_result
   external_loop_connector
   
   _analyze_deviations
   external_loop_connector
   
   _is_significant_deviation
   external_loop_connector
   
   _train_from_experience
   external_loop_connector
   
   _register_new_weights
   external_loop_connector
   
   get_loop_status
   external_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 175

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: internal_loop_connector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   internal_loop_connector
   
   module_explorer
   internal_loop_connector
   
   capability_analyzer
   internal_loop_connector
   
   sync_capabilities_to_rag
   internal_loop_connector
   
   _enhance_capabilities
   internal_loop_connector
   
   _match_sub_category
   internal_loop_connector
   
   _categorize_capability
   internal_loop_connector
   
   _assess_complexity
   internal_loop_connector
   
   _generate_tags
   internal_loop_connector
   
   _build_invocation_metadata
   internal_loop_connector
   
   _get_go_module_port
   internal_loop_connector
   
   _get_rust_module_port
   internal_loop_connector
   
   _build_parameter_definitions
   internal_loop_connector
   
   _generate_param_example
   internal_loop_connector
   
   _build_return_definition
   internal_loop_connector
   
   _generate_usage_examples
   internal_loop_connector
   
   _convert_to_capability_model
   internal_loop_connector
   
   _build_basic_info_section
   internal_loop_connector
   
   _build_parameters_section
   internal_loop_connector
   
   _build_examples_section
   internal_loop_connector
   
   _build_health_section
   internal_loop_connector
   
   _build_dependencies_section
   internal_loop_connector
   
   _convert_to_documents
   internal_loop_connector
   
   _inject_to_rag
   internal_loop_connector
   
   query_self_awareness
   internal_loop_connector
   
   report_issue
   internal_loop_connector
   
   search_solution
   internal_loop_connector
   
   get_sync_status
   internal_loop_connector
   
   export_capabilities_json
   internal_loop_connector
   - 模組: 服務骨幹模組

---

### Flow 177

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: protocol_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   create_http_adapter
   protocol_adapter
   
   send_request
   protocol_adapter
   
   handle_response
   protocol_adapter
   
   __init__
   protocol_adapter
   
   _adapt_request_data
   protocol_adapter
   
   _adapt_response_data
   protocol_adapter
   - 模組: 服務骨幹模組

---

### Flow 178

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 179

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   
   __init__
   message_broker
   
   connect
   message_broker
   
   _declare_exchanges
   message_broker
   
   publish_message
   message_broker
   
   subscribe
   message_broker
   
   create_rpc_client
   message_broker
   
   get_rpc_client
   message_broker
   
   disconnect
   message_broker
   
   setup
   message_broker
   
   _on_response
   message_broker
   
   call
   message_broker
   
   is_expired
   message_broker
   
   can_retry
   message_broker
   
   matches
   message_broker
   
   _match_pattern
   message_broker
   
   start_event_system
   message_broker
   
   stop_event_system
   message_broker
   
   publish_event
   message_broker
   
   subscribe_event
   message_broker
   
   unsubscribe_event
   message_broker
   
   _process_events
   message_broker
   
   _handle_event
   message_broker
   
   get_event_statistics
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 180

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: result_collector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   result_collector
   
   start
   result_collector
   
   _subscribe_scan_results
   result_collector
   
   _subscribe_function_results
   result_collector
   
   _subscribe_task_updates
   result_collector
   
   _subscribe_findings
   result_collector
   
   _handle_scan_result
   result_collector
   
   _handle_function_result
   result_collector
   
   _handle_task_update
   result_collector
   
   _handle_finding
   result_collector
   
   _store_result
   result_collector
   
   _trigger_handlers
   result_collector
   
   register_handler
   result_collector
   
   unregister_handler
   result_collector
   
   _set_pending_result
   result_collector
   
   wait_for_result
   result_collector
   
   get_recent_results
   result_collector
   
   get_statistics
   result_collector
   - 模組: 服務骨幹模組

---

### Flow 183

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 185

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 186

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scan_module_interface
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

---

### Flow 187

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: two_phase_scan_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   two_phase_scan_orchestrator
   
   execute_two_phase_scan
   two_phase_scan_orchestrator
   
   _execute_phase0
   two_phase_scan_orchestrator
   
   _execute_phase1
   two_phase_scan_orchestrator
   
   _analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   _select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 188

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: to_functions
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   to_function_message
   to_functions
   - 模組: 服務骨幹模組

---

### Flow 189

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scan_result_processor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

---

### Flow 190

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: skill_graph
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   skill_graph
   
   build_graph
   skill_graph
   
   _extract_success_rate
   skill_graph
   
   _extract_usage_count
   skill_graph
   
   _build_node_metadata
   skill_graph
   
   _create_skill_nodes
   skill_graph
   
   _analyze_relationships
   skill_graph
   
   _analyze_prerequisite_relationships
   skill_graph
   
   _analyze_tag_similarity_relationships
   skill_graph
   
   _analyze_language_ecosystem_relationships
   skill_graph
   
   _analyze_topic_relationships
   skill_graph
   
   _check_io_compatibility
   skill_graph
   
   _analyze_io_relationships
   skill_graph
   
   _is_compatible_io
   skill_graph
   
   _build_networkx_graph
   skill_graph
   
   find_optimal_path
   skill_graph
   
   _find_goal_capabilities
   skill_graph
   
   _create_skill_path
   skill_graph
   
   get_capability_recommendations
   skill_graph
   
   analyze_capability_centrality
   skill_graph
   
   initialize
   skill_graph
   
   rebuild_if_needed
   skill_graph
   
   find_execution_path
   skill_graph
   
   get_recommendations
   skill_graph
   
   analyze_centrality
   skill_graph
   
   get_graph_statistics
   skill_graph
   - 模組: 服務骨幹模組

---

### Flow 192

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: cli_integration_example
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   storage_manager
   
   initialize
   storage_manager
   
   _get_database_config
   storage_manager
   
   _create_backend
   storage_manager
   
   get_path
   storage_manager
   
   get_statistics
   storage_manager
   
   save_experience_sample
   storage_manager
   
   save_unified_experience_sample
   storage_manager
   
   get_experience_samples
   storage_manager
   
   save_trace
   storage_manager
   
   get_traces_by_session
   storage_manager
   
   save_training_session
   storage_manager
   
   save_command_execution
   storage_manager
   
   get_command_history
   storage_manager
   
   get_command_statistics
   storage_manager
   
   get_popular_capabilities
   storage_manager
   
   get_slow_executions
   storage_manager
   
   get_dir_size
   storage_manager
   - 模組: 認知核心模組

5. **程式組件**
   example_usage
   cli_integration_example
   
   __init__
   cli_integration_example
   
   _load_flows_data
   cli_integration_example
   
   execute_capability
   cli_integration_example
   
   _select_flow_path
   cli_integration_example
   
   _execute_flow
   cli_integration_example
   - 模組: 服務骨幹模組

---

### Flow 193

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: cli_integration_example
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   example_usage
   cli_integration_example
   
   __init__
   cli_integration_example
   
   _load_flows_data
   cli_integration_example
   
   execute_capability
   cli_integration_example
   
   _select_flow_path
   cli_integration_example
   
   _execute_flow
   cli_integration_example
   - 模組: 服務骨幹模組

---

### Flow 194

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: dynamic_strategy_adjustment
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   dynamic_strategy_adjustment
   
   adjust
   dynamic_strategy_adjustment
   
   learn_from_result
   dynamic_strategy_adjustment
   
   _adjust_for_waf
   dynamic_strategy_adjustment
   
   _adjust_based_on_success_rate
   dynamic_strategy_adjustment
   
   _adjust_for_tech_stack
   dynamic_strategy_adjustment
   
   _adjust_for_findings
   dynamic_strategy_adjustment
   - 模組: 服務骨幹模組

---

### Flow 195

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   dynamic_strategy_adjustment
   
   adjust
   dynamic_strategy_adjustment
   
   learn_from_result
   dynamic_strategy_adjustment
   
   _adjust_for_waf
   dynamic_strategy_adjustment
   
   _adjust_based_on_success_rate
   dynamic_strategy_adjustment
   
   _adjust_for_tech_stack
   dynamic_strategy_adjustment
   
   _adjust_for_findings
   dynamic_strategy_adjustment
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 196

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: risk_assessment_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   risk_assessment_engine
   
   assess_risk
   risk_assessment_engine
   
   _calculate_base_score
   risk_assessment_engine
   
   _assess_phase_i_specific_risk
   risk_assessment_engine
   
   _adjust_by_threat_intel
   risk_assessment_engine
   
   _adjust_by_exploitability
   risk_assessment_engine
   
   _adjust_by_asset_criticality
   risk_assessment_engine
   
   get_risk_level
   risk_assessment_engine
   
   batch_assess
   risk_assessment_engine
   - 模組: 服務骨幹模組

---

### Flow 202

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: aiva_flow_classifier_final
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

---

### Flow 203

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: check_flow_details
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   check_flow_details
   check_flow_details
   - 模組: 服務骨幹模組

---

### Flow 204

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: find_testable_flows
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   find_testable_flows
   find_testable_flows
   - 模組: 服務骨幹模組

---

### Flow 205

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: verify_classification
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   verify_classification
   verify_classification
   - 模組: 服務骨幹模組

---

### Flow 207

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: execution_status_monitor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

---

### Flow 209

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 210

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 212

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 213

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 215

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 217

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: weight_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   
   __init__
   weight_manager
   
   save_model_weights
   weight_manager
   
   load_model_weights
   weight_manager
   
   list_available_weights
   weight_manager
   
   _list_model_versions
   weight_manager
   
   _extract_version_info
   weight_manager
   
   _list_all_models
   weight_manager
   
   delete_weights
   weight_manager
   
   _find_weight_file
   weight_manager
   
   _calculate_file_hash
   weight_manager
   
   _save_metadata
   weight_manager
   
   _load_and_verify_metadata
   weight_manager
   
   _verify_model_compatibility
   weight_manager
   
   _create_backup
   weight_manager
   
   _cleanup_old_backups
   weight_manager
   - 模組: 服務骨幹模組

---

### Flow 218

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

5. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

---

### Flow 219

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 221

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 225

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: attack_validator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   attack_validator
   
   _load_false_positive_patterns
   attack_validator
   
   validate_result
   attack_validator
   
   _basic_validation
   attack_validator
   
   _default_validation
   attack_validator
   
   _validate_sql_injection
   attack_validator
   
   _validate_xss
   attack_validator
   
   _validate_command_injection
   attack_validator
   
   _check_false_positive
   attack_validator
   
   batch_validate
   attack_validator
   
   get_statistics
   attack_validator
   - 模組: 服務骨幹模組

---

### Flow 227

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: exploit_manager_legacy
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   exploit_manager_legacy
   
   _initialize_exploits
   exploit_manager_legacy
   
   _register_exploit
   exploit_manager_legacy
   
   register_exploit
   exploit_manager_legacy
   
   get_exploits_by_type
   exploit_manager_legacy
   
   get_exploit
   exploit_manager_legacy
   
   execute_exploit
   exploit_manager_legacy
   
   _execute_exploit_by_type
   exploit_manager_legacy
   
   _test_idor_vulnerability
   exploit_manager_legacy
   
   _test_sql_injection
   exploit_manager_legacy
   
   _test_xss_vulnerability
   exploit_manager_legacy
   
   _test_auth_bypass
   exploit_manager_legacy
   
   _test_jwt_attack
   exploit_manager_legacy
   
   _test_graphql_injection
   exploit_manager_legacy
   
   get_statistics
   exploit_manager_legacy
   
   _count_by_type
   exploit_manager_legacy
   - 模組: 服務骨幹模組

---

### Flow 229

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: exploit_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   exploit_manager_legacy
   
   _initialize_exploits
   exploit_manager_legacy
   
   _register_exploit
   exploit_manager_legacy
   
   register_exploit
   exploit_manager_legacy
   
   get_exploits_by_type
   exploit_manager_legacy
   
   get_exploit
   exploit_manager_legacy
   
   execute_exploit
   exploit_manager_legacy
   
   _execute_exploit_by_type
   exploit_manager_legacy
   
   _test_idor_vulnerability
   exploit_manager_legacy
   
   _test_sql_injection
   exploit_manager_legacy
   
   _test_xss_vulnerability
   exploit_manager_legacy
   
   _test_auth_bypass
   exploit_manager_legacy
   
   _test_jwt_attack
   exploit_manager_legacy
   
   _test_graphql_injection
   exploit_manager_legacy
   
   get_statistics
   exploit_manager_legacy
   
   _count_by_type
   exploit_manager_legacy
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   exploit_orchestrator
   
   _initialize_exploits
   exploit_orchestrator
   
   _register_exploit
   exploit_orchestrator
   
   get_exploit
   exploit_orchestrator
   
   get_all_exploits
   exploit_orchestrator
   
   get_exploits_by_type
   exploit_orchestrator
   
   get_exploits_by_severity
   exploit_orchestrator
   
   orchestrate_exploit
   exploit_orchestrator
   
   analyze_results
   exploit_orchestrator
   
   _calculate_risk_score
   exploit_orchestrator
   
   _get_timestamp
   exploit_orchestrator
   
   get_execution_history
   exploit_orchestrator
   
   get_statistics
   exploit_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 230

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: exploit_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   exploit_orchestrator
   
   _initialize_exploits
   exploit_orchestrator
   
   _register_exploit
   exploit_orchestrator
   
   get_exploit
   exploit_orchestrator
   
   get_all_exploits
   exploit_orchestrator
   
   get_exploits_by_type
   exploit_orchestrator
   
   get_exploits_by_severity
   exploit_orchestrator
   
   orchestrate_exploit
   exploit_orchestrator
   
   analyze_results
   exploit_orchestrator
   
   _calculate_risk_score
   exploit_orchestrator
   
   _get_timestamp
   exploit_orchestrator
   
   get_execution_history
   exploit_orchestrator
   
   get_statistics
   exploit_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 231

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: payload_generator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   payload_generator
   
   _load_templates
   payload_generator
   
   generate_with_target_analysis
   payload_generator
   
   _analyze_target_environment
   payload_generator
   
   _select_payload_templates
   payload_generator
   
   _is_template_suitable
   payload_generator
   
   _customize_payloads
   payload_generator
   
   _validate_payloads
   payload_generator
   
   _validate_single_payload
   payload_generator
   
   _format_output
   payload_generator
   
   _generate_usage_recommendations
   payload_generator
   
   generate
   payload_generator
   
   _encode_payload
   payload_generator
   
   generate_fuzzing_payloads
   payload_generator
   
   get_statistics
   payload_generator
   - 模組: 服務骨幹模組

---

### Flow 232

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 233

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: scan_module_interface
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

---

### Flow 234

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 235

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: two_phase_scan_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   two_phase_scan_orchestrator
   
   execute_two_phase_scan
   two_phase_scan_orchestrator
   
   _execute_phase0
   two_phase_scan_orchestrator
   
   _execute_phase1
   two_phase_scan_orchestrator
   
   _analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   _select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 237

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: scan_result_processor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   two_phase_scan_orchestrator
   
   execute_two_phase_scan
   two_phase_scan_orchestrator
   
   _execute_phase0
   two_phase_scan_orchestrator
   
   _execute_phase1
   two_phase_scan_orchestrator
   
   _analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   _select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

---

### Flow 239

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_memory_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **AI組件**
   __init__
   ai_summary_plugin
   
   register_capability
   ai_summary_plugin
   
   discover_and_register
   ai_summary_plugin
   
   _process_module_path
   ai_summary_plugin
   
   _try_register_function
   ai_summary_plugin
   
   execute_capability
   ai_summary_plugin
   
   _update_avg_execution_time
   ai_summary_plugin
   
   list_capabilities
   ai_summary_plugin
   
   get_registry_stats
   ai_summary_plugin
   
   is_enabled
   ai_summary_plugin
   
   enable
   ai_summary_plugin
   
   disable
   ai_summary_plugin
   
   get_status
   ai_summary_plugin
   
   generate_summary
   ai_summary_plugin
   
   _build_summary_prompt
   ai_summary_plugin
   
   _classify_request_type
   ai_summary_plugin
   
   _get_complexity_level
   ai_summary_plugin
   
   _calculate_efficiency_score
   ai_summary_plugin
   
   _extract_recommendations
   ai_summary_plugin
   
   _identify_learning_points
   ai_summary_plugin
   
   _create_brief_summary
   ai_summary_plugin
   
   _enhance_detailed_summary
   ai_summary_plugin
   
   _extract_processing_steps
   ai_summary_plugin
   
   _estimate_resource_usage
   ai_summary_plugin
   
   _assess_improvement_potential
   ai_summary_plugin
   
   _record_summary_history
   ai_summary_plugin
   
   get_statistics
   ai_summary_plugin
   
   configure
   ai_summary_plugin
   
   reset
   ai_summary_plugin
   
   unload
   ai_summary_plugin
   - 模組: 認知核心模組

5. **程式組件**
   __init__
   unified_memory_manager
   
   _generate_cache_key
   unified_memory_manager
   
   get_cached_prediction
   unified_memory_manager
   
   cache_prediction
   unified_memory_manager
   
   _evict_oldest_cache_entry
   unified_memory_manager
   
   clear_cache
   unified_memory_manager
   
   create_component_pool
   unified_memory_manager
   
   get_component_pool
   unified_memory_manager
   
   register_weak_ref
   unified_memory_manager
   
   start_monitoring
   unified_memory_manager
   
   stop_monitoring
   unified_memory_manager
   
   _monitor_memory
   unified_memory_manager
   
   _force_cleanup
   unified_memory_manager
   
   _cleanup_expired_cache
   unified_memory_manager
   
   process_batch
   unified_memory_manager
   
   process_large_dataset
   unified_memory_manager
   
   _get_memory_usage_mb
   unified_memory_manager
   
   _record_memory_usage
   unified_memory_manager
   
   optimize_memory
   unified_memory_manager
   
   get_comprehensive_stats
   unified_memory_manager
   
   _get_cache_stats
   unified_memory_manager
   
   _get_memory_stats
   unified_memory_manager
   
   _get_pool_stats
   unified_memory_manager
   
   get_component
   unified_memory_manager
   
   get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

---

### Flow 240

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: scan_result_processor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

---

### Flow 241

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 242

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: result_collector
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   scan_result_processor
   
   stage_1_ingest_data
   scan_result_processor
   
   stage_2_analyze_surface
   scan_result_processor
   
   stage_3_generate_strategy
   scan_result_processor
   
   stage_4_adjust_strategy
   scan_result_processor
   
   stage_5_generate_tasks
   scan_result_processor
   
   stage_6_dispatch_tasks
   scan_result_processor
   
   stage_7_monitor_execution
   scan_result_processor
   
   process
   scan_result_processor
   
   process_phase0
   scan_result_processor
   
   _analyze_phase0_and_decide
   scan_result_processor
   
   _select_engines_for_phase1
   scan_result_processor
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   result_collector
   
   start
   result_collector
   
   _subscribe_scan_results
   result_collector
   
   _subscribe_function_results
   result_collector
   
   _subscribe_task_updates
   result_collector
   
   _subscribe_findings
   result_collector
   
   _handle_scan_result
   result_collector
   
   _handle_function_result
   result_collector
   
   _handle_task_update
   result_collector
   
   _handle_finding
   result_collector
   
   _store_result
   result_collector
   
   _trigger_handlers
   result_collector
   
   register_handler
   result_collector
   
   unregister_handler
   result_collector
   
   _set_pending_result
   result_collector
   
   wait_for_result
   result_collector
   
   get_recent_results
   result_collector
   
   get_statistics
   result_collector
   - 模組: 服務骨幹模組

---

### Flow 243

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: anti_hallucination_module
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   demo_anti_hallucination
   anti_hallucination_module
   
   __init__
   anti_hallucination_module
   
   _check_knowledge_base_health
   anti_hallucination_module
   
   _fallback_knowledge_validation
   anti_hallucination_module
   
   _get_technique_category
   anti_hallucination_module
   
   _validate_technique_consistency
   anti_hallucination_module
   
   _setup_logger
   anti_hallucination_module
   
   validate_attack_plan
   anti_hallucination_module
   
   _validate_single_step
   anti_hallucination_module
   
   _validate_with_knowledge_base_fallback
   anti_hallucination_module
   
   _validate_step_sequence
   anti_hallucination_module
   
   _is_known_technique
   anti_hallucination_module
   
   _extract_relevance_score
   anti_hallucination_module
   
   _validate_with_knowledge_base
   anti_hallucination_module
   
   _validate_step_logic
   anti_hallucination_module
   
   get_validation_stats
   anti_hallucination_module
   
   export_validation_report
   anti_hallucination_module
   
   reset_knowledge_base
   anti_hallucination_module
   - 模組: 服務骨幹模組

---

### Flow 247

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: skill_graph
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   skill_graph
   
   build_graph
   skill_graph
   
   _extract_success_rate
   skill_graph
   
   _extract_usage_count
   skill_graph
   
   _build_node_metadata
   skill_graph
   
   _create_skill_nodes
   skill_graph
   
   _analyze_relationships
   skill_graph
   
   _analyze_prerequisite_relationships
   skill_graph
   
   _analyze_tag_similarity_relationships
   skill_graph
   
   _analyze_language_ecosystem_relationships
   skill_graph
   
   _analyze_topic_relationships
   skill_graph
   
   _check_io_compatibility
   skill_graph
   
   _analyze_io_relationships
   skill_graph
   
   _is_compatible_io
   skill_graph
   
   _build_networkx_graph
   skill_graph
   
   find_optimal_path
   skill_graph
   
   _find_goal_capabilities
   skill_graph
   
   _create_skill_path
   skill_graph
   
   get_capability_recommendations
   skill_graph
   
   analyze_capability_centrality
   skill_graph
   
   initialize
   skill_graph
   
   rebuild_if_needed
   skill_graph
   
   find_execution_path
   skill_graph
   
   get_recommendations
   skill_graph
   
   analyze_centrality
   skill_graph
   
   get_graph_statistics
   skill_graph
   - 模組: 服務骨幹模組

---

### Flow 250

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: weight_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   
   __init__
   weight_manager
   
   save_model_weights
   weight_manager
   
   load_model_weights
   weight_manager
   
   list_available_weights
   weight_manager
   
   _list_model_versions
   weight_manager
   
   _extract_version_info
   weight_manager
   
   _list_all_models
   weight_manager
   
   delete_weights
   weight_manager
   
   _find_weight_file
   weight_manager
   
   _calculate_file_hash
   weight_manager
   
   _save_metadata
   weight_manager
   
   _load_and_verify_metadata
   weight_manager
   
   _verify_model_compatibility
   weight_manager
   
   _create_backup
   weight_manager
   
   _cleanup_old_backups
   weight_manager
   - 模組: 服務骨幹模組

---

### Flow 251

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: knowledge_base
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   add_document
   knowledge_base
   
   search
   knowledge_base
   
   __init__
   knowledge_base
   
   add_knowledge
   knowledge_base
   
   index_codebase
   knowledge_base
   
   get_stats
   knowledge_base
   - 模組: 服務骨幹模組

---

### Flow 252

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: assistant
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   add_document
   knowledge_base
   
   search
   knowledge_base
   
   __init__
   knowledge_base
   
   add_knowledge
   knowledge_base
   
   index_codebase
   knowledge_base
   
   get_stats
   knowledge_base
   - 模組: 服務骨幹模組

5. **程式組件**
   get_dialog_assistant
   assistant
   
   classify_command
   assistant
   
   __init__
   assistant
   
   _ensure_initialized
   assistant
   
   _get_rag_kb
   assistant
   
   _get_function_caller
   assistant
   
   process_user_input
   assistant
   
   _handle_intent
   assistant
   
   _handle_list_capabilities
   assistant
   
   _handle_explain_capability
   assistant
   
   _handle_run_scan
   assistant
   
   _handle_compare_capabilities
   assistant
   
   _handle_generate_cli
   assistant
   
   _handle_system_status
   assistant
   
   _add_conversation_entry
   assistant
   
   get_conversation_history
   assistant
   
   clear_conversation_history
   assistant
   
   __getattr__
   assistant
   
   __call__
   assistant
   - 模組: 服務骨幹模組

---

### Flow 253

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   add_document
   knowledge_base
   
   search
   knowledge_base
   
   __init__
   knowledge_base
   
   add_knowledge
   knowledge_base
   
   index_codebase
   knowledge_base
   
   get_stats
   knowledge_base
   - 模組: 服務骨幹模組

5. **程式組件**
   create_unified_vector_store
   unified_vector_store
   
   __init__
   unified_vector_store
   
   initialize
   unified_vector_store
   
   _migrate_from_legacy
   unified_vector_store
   
   _get_embedding_model
   unified_vector_store
   
   _simple_embedding
   unified_vector_store
   
   add_document
   unified_vector_store
   
   add_batch
   unified_vector_store
   
   search
   unified_vector_store
   
   delete_document
   unified_vector_store
   
   get_document
   unified_vector_store
   
   get_statistics
   unified_vector_store
   
   close
   unified_vector_store
   - 模組: 服務骨幹模組

---

### Flow 254

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   add_document
   knowledge_base
   
   search
   knowledge_base
   
   __init__
   knowledge_base
   
   add_knowledge
   knowledge_base
   
   index_codebase
   knowledge_base
   
   get_stats
   knowledge_base
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

---

### Flow 256

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

---

### Flow 258

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 259

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 260

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_models
   
   forward
   rl_models
   
   select_action
   rl_models
   
   _get_activation
   rl_models
   
   _init_weights
   rl_models
   
   evaluate_actions
   rl_models
   
   _build_feature_extractor
   rl_models
   
   _build_actor
   rl_models
   
   _build_critic
   rl_models
   
   push
   rl_models
   
   sample
   rl_models
   
   __len__
   rl_models
   
   get
   rl_models
   
   compute_returns
   rl_models
   
   clear
   rl_models
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 264

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 266

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **混合組件**
   quick_plan_and_execute
   capability_orchestrator
   
   __init__
   capability_orchestrator
   
   plan
   capability_orchestrator
   
   _query_relevant_capabilities
   capability_orchestrator
   
   _fallback_capability_search
   capability_orchestrator
   
   _filter_available_capabilities
   capability_orchestrator
   
   _select_best_capabilities
   capability_orchestrator
   
   _calculate_capability_score
   capability_orchestrator
   
   _generate_execution_sequence
   capability_orchestrator
   
   _order_scan_sequence
   capability_orchestrator
   
   _order_attack_sequence
   capability_orchestrator
   
   _order_comprehensive_sequence
   capability_orchestrator
   
   _capabilities_to_commands
   capability_orchestrator
   
   _build_command_from_capability
   capability_orchestrator
   
   _map_capability_to_command_type
   capability_orchestrator
   
   _build_command_payload
   capability_orchestrator
   
   _get_target_module
   capability_orchestrator
   
   _assess_risk_level
   capability_orchestrator
   
   _generate_reasoning
   capability_orchestrator
   
   execute
   capability_orchestrator
   
   _extract_issues
   capability_orchestrator
   
   learn_from_execution
   capability_orchestrator
   - 模組: 核心能力模組

5. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 267

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: aiva_flow_classifier_final
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

---

### Flow 270

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: event_listener
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   main
   event_listener
   
   __init__
   event_listener
   
   broker
   event_listener
   
   connector
   event_listener
   
   start_listening
   event_listener
   
   stop_listening
   event_listener
   
   _on_task_completed_wrapper
   event_listener
   
   _on_task_completed
   event_listener
   
   _process_learning
   event_listener
   
   get_status
   event_listener
   - 模組: 服務骨幹模組

---

### Flow 271

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   get_unified_caller
   unified_function_caller
   
   call_any_function
   unified_function_caller
   
   call_sqli_detection
   unified_function_caller
   
   call_xss_detection
   unified_function_caller
   
   call_idor_detection
   unified_function_caller
   
   call_go_ssrf_detection
   unified_function_caller
   
   call_typescript_frontend_scan
   unified_function_caller
   
   __init__
   unified_function_caller
   
   _init_endpoints
   unified_function_caller
   
   call_python
   unified_function_caller
   
   call_http
   unified_function_caller
   
   call_grpc
   unified_function_caller
   
   call_function
   unified_function_caller
   
   _call_python_module
   unified_function_caller
   
   _call_http_module
   unified_function_caller
   
   _call_grpc_module
   unified_function_caller
   
   list_all_functions
   unified_function_caller
   
   get_module_info
   unified_function_caller
   
   test_unified_caller
   unified_function_caller
   - 模組: 服務骨幹模組

---

### Flow 273

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: cli_integration_example
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   example_usage
   cli_integration_example
   
   __init__
   cli_integration_example
   
   _load_flows_data
   cli_integration_example
   
   execute_capability
   cli_integration_example
   
   _select_flow_path
   cli_integration_example
   
   _execute_flow
   cli_integration_example
   - 模組: 服務骨幹模組

---

### Flow 275

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: postgresql_vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   aiva_flow_classifier_final
   
   __init__
   aiva_flow_classifier_final
   
   load_flow_data
   aiva_flow_classifier_final
   
   _extract_script_name
   aiva_flow_classifier_final
   
   _get_script_description
   aiva_flow_classifier_final
   
   _classify_module
   aiva_flow_classifier_final
   
   _classify_component_type
   aiva_flow_classifier_final
   
   classify_flows
   aiva_flow_classifier_final
   
   analyze_multi_path_endpoints
   aiva_flow_classifier_final
   
   generate_reports
   aiva_flow_classifier_final
   
   _generate_classification_report
   aiva_flow_classifier_final
   
   _generate_complete_flow_details
   aiva_flow_classifier_final
   
   _generate_multi_path_report
   aiva_flow_classifier_final
   
   _generate_json_export
   aiva_flow_classifier_final
   
   run
   aiva_flow_classifier_final
   - 模組: 服務骨幹模組

5. **程式組件**
   demo_postgresql_vector_store
   postgresql_vector_store
   
   __init__
   postgresql_vector_store
   
   initialize
   postgresql_vector_store
   
   add_document
   postgresql_vector_store
   
   search
   postgresql_vector_store
   
   get_document
   postgresql_vector_store
   
   delete_document
   postgresql_vector_store
   
   get_statistics
   postgresql_vector_store
   
   execute_unified_query
   postgresql_vector_store
   
   close
   postgresql_vector_store
   - 模組: 服務骨幹模組

---

### Flow 276

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: check_flow_details
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   check_flow_details
   check_flow_details
   - 模組: 服務骨幹模組

---

### Flow 277

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: find_testable_flows
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   find_testable_flows
   find_testable_flows
   - 模組: 服務骨幹模組

---

### Flow 278

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: verify_classification
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   verify_classification
   verify_classification
   - 模組: 服務骨幹模組

---

### Flow 280

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: execution_status_monitor
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

---

### Flow 281

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

5. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

### Flow 282

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: unified_memory_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   to_dict
   execution_status_monitor
   
   from_dict
   execution_status_monitor
   
   __init__
   execution_status_monitor
   
   record_worker_heartbeat
   execution_status_monitor
   
   record_task_start
   execution_status_monitor
   
   record_task_completion
   execution_status_monitor
   
   get_system_health
   execution_status_monitor
   
   check_sla_violations
   execution_status_monitor
   
   _get_recent_alerts
   execution_status_monitor
   
   add_alert
   execution_status_monitor
   
   start_monitoring
   execution_status_monitor
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   unified_memory_manager
   
   _generate_cache_key
   unified_memory_manager
   
   get_cached_prediction
   unified_memory_manager
   
   cache_prediction
   unified_memory_manager
   
   _evict_oldest_cache_entry
   unified_memory_manager
   
   clear_cache
   unified_memory_manager
   
   create_component_pool
   unified_memory_manager
   
   get_component_pool
   unified_memory_manager
   
   register_weak_ref
   unified_memory_manager
   
   start_monitoring
   unified_memory_manager
   
   stop_monitoring
   unified_memory_manager
   
   _monitor_memory
   unified_memory_manager
   
   _force_cleanup
   unified_memory_manager
   
   _cleanup_expired_cache
   unified_memory_manager
   
   process_batch
   unified_memory_manager
   
   process_large_dataset
   unified_memory_manager
   
   _get_memory_usage_mb
   unified_memory_manager
   
   _record_memory_usage
   unified_memory_manager
   
   optimize_memory
   unified_memory_manager
   
   get_comprehensive_stats
   unified_memory_manager
   
   _get_cache_stats
   unified_memory_manager
   
   _get_memory_stats
   unified_memory_manager
   
   _get_pool_stats
   unified_memory_manager
   
   get_component
   unified_memory_manager
   
   get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

---

### Flow 284

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: payload_generator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   get_execution_planner
   execution_planner
   
   __init__
   execution_planner
   
   create_execution_plan
   execution_planner
   
   execute_plan
   execution_planner
   
   _check_resources
   execution_planner
   
   _execute_step
   execution_planner
   
   _validate_input
   execution_planner
   
   _execute_simple_command
   execution_planner
   
   _format_output
   execution_planner
   
   _execute_ai_task
   execution_planner
   
   _execute_rust_scan
   execution_planner
   
   _generate_report
   execution_planner
   
   _execute_generic_step
   execution_planner
   
   _aggregate_results
   execution_planner
   
   get_plan_status
   execution_planner
   
   cancel_plan
   execution_planner
   
   get_execution_stats
   execution_planner
   - 模組: 任務規劃模組

5. **程式組件**
   __init__
   payload_generator
   
   _load_templates
   payload_generator
   
   generate_with_target_analysis
   payload_generator
   
   _analyze_target_environment
   payload_generator
   
   _select_payload_templates
   payload_generator
   
   _is_template_suitable
   payload_generator
   
   _customize_payloads
   payload_generator
   
   _validate_payloads
   payload_generator
   
   _validate_single_payload
   payload_generator
   
   _format_output
   payload_generator
   
   _generate_usage_recommendations
   payload_generator
   
   generate
   payload_generator
   
   _encode_payload
   payload_generator
   
   generate_fuzzing_payloads
   payload_generator
   
   get_statistics
   payload_generator
   - 模組: 服務骨幹模組

---

### Flow 285

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 286

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 291

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 292

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: analysis_engine
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   analysis_engine
   
   _load_cache_index
   analysis_engine
   
   get_file_hash
   analysis_engine
   
   is_cached
   analysis_engine
   
   update_cache
   analysis_engine
   
   save_cache_index
   analysis_engine
   
   __post_init__
   analysis_engine
   
   initialize
   analysis_engine
   
   _extract_code_features
   analysis_engine
   
   _calculate_cyclomatic_complexity
   analysis_engine
   
   _calculate_nesting_depth
   analysis_engine
   
   _extract_security_features
   analysis_engine
   
   _extract_semantic_features
   analysis_engine
   
   analyze_code
   analysis_engine
   
   index_codebase
   analysis_engine
   
   _collect_python_files
   analysis_engine
   
   _filter_files_for_indexing
   analysis_engine
   
   _batch_index_files
   analysis_engine
   
   _process_file_batch
   analysis_engine
   
   _safe_index_file
   analysis_engine
   
   _index_file_content
   analysis_engine
   
   _extract_chunks_from_ast
   analysis_engine
   
   _extract_node_content
   analysis_engine
   
   _extract_by_line_numbers
   analysis_engine
   
   _handle_unparseable_file
   analysis_engine
   
   _add_code_chunk
   analysis_engine
   
   _extract_analysis_keywords
   analysis_engine
   
   search_code_chunks
   analysis_engine
   
   _extract_query_keywords
   analysis_engine
   
   _calculate_chunk_scores
   analysis_engine
   
   _apply_exact_matches
   analysis_engine
   
   _apply_partial_matches
   analysis_engine
   
   _format_search_results
   analysis_engine
   
   _get_indexing_stats
   analysis_engine
   
   _create_failed_results
   analysis_engine
   
   _perform_ai_analysis
   analysis_engine
   
   _generate_findings
   analysis_engine
   
   _generate_recommendations
   analysis_engine
   
   _calculate_risk_level
   analysis_engine
   
   _generate_explanation
   analysis_engine
   
   get_analysis_summary
   analysis_engine
   
   visit_node
   analysis_engine
   - 模組: 服務骨幹模組

---

### Flow 294

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 296

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: weight_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   
   __init__
   weight_manager
   
   save_model_weights
   weight_manager
   
   load_model_weights
   weight_manager
   
   list_available_weights
   weight_manager
   
   _list_model_versions
   weight_manager
   
   _extract_version_info
   weight_manager
   
   _list_all_models
   weight_manager
   
   delete_weights
   weight_manager
   
   _find_weight_file
   weight_manager
   
   _calculate_file_hash
   weight_manager
   
   _save_metadata
   weight_manager
   
   _load_and_verify_metadata
   weight_manager
   
   _verify_model_compatibility
   weight_manager
   
   _create_backup
   weight_manager
   
   _cleanup_old_backups
   weight_manager
   - 模組: 服務骨幹模組

---

### Flow 297

- **長度**: 4 步
- **起點**: scalable_bio_trainer
- **終點**: vector_store
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

---

### Flow 299

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   save_experience_sample
   backends
   
   get_experience_samples
   backends
   
   save_trace
   backends
   
   get_traces_by_session
   backends
   
   save_training_session
   backends
   
   get_statistics
   backends
   
   __init__
   backends
   
   save_unified_experience_sample
   backends
   
   model_dump
   backends
   - 模組: 服務骨幹模組

---

### Flow 300

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   __init__
   command_repository
   
   save_command_execution
   command_repository
   
   get_command_history
   command_repository
   
   get_command_statistics
   command_repository
   
   get_popular_capabilities
   command_repository
   
   get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 301

- **長度**: 5 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **AI組件**
   relu
   neural_network
   
   relu_derivative
   neural_network
   
   sigmoid
   neural_network
   
   sigmoid_derivative
   neural_network
   
   tanh
   neural_network
   
   tanh_derivative
   neural_network
   
   softmax
   neural_network
   
   __init__
   neural_network
   
   forward
   neural_network
   
   backward
   neural_network
   
   update_weights
   neural_network
   
   predict
   neural_network
   
   reset_hidden_state
   neural_network
   
   reset_states
   neural_network
   
   create_classifier
   neural_network
   
   create_regressor
   neural_network
   - 模組: 認知核心模組

3. **AI組件**
   __init__
   rl_trainers
   
   select_action
   rl_trainers
   
   train_step
   rl_trainers
   
   get_metrics
   rl_trainers
   
   save
   rl_trainers
   
   load
   rl_trainers
   
   store_transition
   rl_trainers
   
   update
   rl_trainers
   - 模組: 外學模組

4. **程式組件**
   __init__
   vector_store
   
   _initialize_backend
   vector_store
   
   _get_embedding_model
   vector_store
   
   _simple_embedding
   vector_store
   
   add_document
   vector_store
   
   add_batch
   vector_store
   
   search
   vector_store
   
   delete_document
   vector_store
   
   get_document
   vector_store
   
   save
   vector_store
   
   load
   vector_store
   
   count
   vector_store
   
   get_statistics
   vector_store
   - 模組: 服務骨幹模組

5. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 304

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: real_bio_net_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   __init__
   scalable_bio_trainer
   
   train
   scalable_bio_trainer
   
   _train_epoch
   scalable_bio_trainer
   
   _validate
   scalable_bio_trainer
   
   _compute_loss
   scalable_bio_trainer
   
   _count_correct_predictions
   scalable_bio_trainer
   
   get_training_history
   scalable_bio_trainer
   
   save_model
   scalable_bio_trainer
   
   load_model
   scalable_bio_trainer
   
   mean
   scalable_bio_trainer
   
   sum
   scalable_bio_trainer
   
   abs
   scalable_bio_trainer
   - 模組: 外學模組

2. **程式組件**
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   
   __init__
   real_bio_net_adapter
   
   _load_or_initialize_weights
   real_bio_net_adapter
   
   forward
   real_bio_net_adapter
   
   _fallback_forward
   real_bio_net_adapter
   
   _softmax
   real_bio_net_adapter
   
   save_weights
   real_bio_net_adapter
   
   generate
   real_bio_net_adapter
   
   _create_real_input_vector
   real_bio_net_adapter
   - 模組: 服務骨幹模組

---

### Flow 306

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: nlg_system
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   __init__
   nlg_system
   
   _init_response_templates
   nlg_system
   
   _init_context_analyzers
   nlg_system
   
   generate_response
   nlg_system
   
   _analyze_context
   nlg_system
   
   _detect_intent
   nlg_system
   
   _extract_entities
   nlg_system
   
   _analyze_sentiment
   nlg_system
   
   _extract_technical_details
   nlg_system
   
   _determine_response_type
   nlg_system
   
   _select_template
   nlg_system
   
   _fill_template
   nlg_system
   
   _generate_result_detail
   nlg_system
   
   _extract_filename
   nlg_system
   
   _post_process_response
   nlg_system
   - 模組: 服務骨幹模組

---

### Flow 307

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 308

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: matrix_visualizer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   main
   matrix_visualizer
   
   make_subplots
   matrix_visualizer
   
   __init__
   matrix_visualizer
   
   generate_heatmap
   matrix_visualizer
   
   generate_coverage_chart
   matrix_visualizer
   
   generate_role_comparison_chart
   matrix_visualizer
   
   generate_html_report
   matrix_visualizer
   
   _generate_all_charts
   matrix_visualizer
   
   _get_analysis_data
   matrix_visualizer
   
   _get_html_template
   matrix_visualizer
   
   _render_html_template
   matrix_visualizer
   
   export_to_csv
   matrix_visualizer
   
   add_trace
   matrix_visualizer
   
   update_layout
   matrix_visualizer
   
   to_html
   matrix_visualizer
   
   write_html
   matrix_visualizer
   - 模組: 服務骨幹模組

---

### Flow 309

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: permission_matrix
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

---

### Flow 310

- **長度**: 3 步
- **起點**: logging_formatter
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

3. **程式組件**
   main
   authz_mapper
   
   __init__
   authz_mapper
   
   assign_role_to_user
   authz_mapper
   
   revoke_role_from_user
   authz_mapper
   
   set_user_attribute
   authz_mapper
   
   get_user_roles
   authz_mapper
   
   check_user_permission
   authz_mapper
   
   get_user_all_permissions
   authz_mapper
   
   detect_permission_conflicts
   authz_mapper
   
   analyze_role_overlap
   authz_mapper
   
   simulate_role_removal
   authz_mapper
   
   recommend_role_consolidation
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 311

- **長度**: 3 步
- **起點**: logging_formatter
- **終點**: matrix_visualizer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   main
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   
   __init__
   permission_matrix
   
   add_role
   permission_matrix
   
   add_resource
   permission_matrix
   
   add_permission
   permission_matrix
   
   grant_permission
   permission_matrix
   
   revoke_permission
   permission_matrix
   
   check_permission
   permission_matrix
   
   _evaluate_condition
   permission_matrix
   
   get_role_permissions
   permission_matrix
   
   get_resource_permissions
   permission_matrix
   
   to_dataframe
   permission_matrix
   
   to_numpy_matrix
   permission_matrix
   
   analyze_coverage
   permission_matrix
   
   find_over_privileged_roles
   permission_matrix
   
   export_to_dict
   permission_matrix
   
   __post_init__
   permission_matrix
   
   _check_risk_level
   permission_matrix
   
   _check_environment_limits
   permission_matrix
   
   _check_attack_tags
   permission_matrix
   
   _production_safety_check
   permission_matrix
   
   get_allowed_operations
   permission_matrix
   
   __len__
   permission_matrix
   
   to_dict
   permission_matrix
   
   to_json
   permission_matrix
   
   empty
   permission_matrix
   - 模組: 服務骨幹模組

3. **程式組件**
   main
   matrix_visualizer
   
   make_subplots
   matrix_visualizer
   
   __init__
   matrix_visualizer
   
   generate_heatmap
   matrix_visualizer
   
   generate_coverage_chart
   matrix_visualizer
   
   generate_role_comparison_chart
   matrix_visualizer
   
   generate_html_report
   matrix_visualizer
   
   _generate_all_charts
   matrix_visualizer
   
   _get_analysis_data
   matrix_visualizer
   
   _get_html_template
   matrix_visualizer
   
   _render_html_template
   matrix_visualizer
   
   export_to_csv
   matrix_visualizer
   
   add_trace
   matrix_visualizer
   
   update_layout
   matrix_visualizer
   
   to_html
   matrix_visualizer
   
   write_html
   matrix_visualizer
   - 模組: 服務骨幹模組

---

### Flow 312

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: payload_generator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   
   __init__
   logging_formatter
   
   format
   logging_formatter
   
   get_logger
   logging_formatter
   
   log_with_context
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   __init__
   payload_generator
   
   _load_templates
   payload_generator
   
   generate_with_target_analysis
   payload_generator
   
   _analyze_target_environment
   payload_generator
   
   _select_payload_templates
   payload_generator
   
   _is_template_suitable
   payload_generator
   
   _customize_payloads
   payload_generator
   
   _validate_payloads
   payload_generator
   
   _validate_single_payload
   payload_generator
   
   _format_output
   payload_generator
   
   _generate_usage_recommendations
   payload_generator
   
   generate
   payload_generator
   
   _encode_payload
   payload_generator
   
   generate_fuzzing_payloads
   payload_generator
   
   get_statistics
   payload_generator
   - 模組: 服務骨幹模組

---

### Flow 313

- **長度**: 2 步
- **起點**: strategy_generator
- **終點**: scenario_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   __init__
   strategy_generator
   
   generate
   strategy_generator
   
   generate_from_intent
   strategy_generator
   
   _generate_vulnerability_test_strategy
   strategy_generator
   
   _generate_surface_scan_strategy
   strategy_generator
   
   _generate_exploit_strategy
   strategy_generator
   
   _generate_analysis_strategy
   strategy_generator
   
   _generate_default_strategy
   strategy_generator
   
   _generate_sqli_tasks_from_intent
   strategy_generator
   
   _generate_xss_tasks_from_intent
   strategy_generator
   
   _generate_ssrf_tasks_from_intent
   strategy_generator
   
   _generate_xss_tasks
   strategy_generator
   
   _generate_sqli_tasks
   strategy_generator
   
   _generate_ssrf_tasks
   strategy_generator
   
   _generate_idor_tasks
   strategy_generator
   
   _calculate_priority
   strategy_generator
   
   _prioritize_tasks
   strategy_generator
   
   _estimate_duration
   strategy_generator
   - 模組: 服務骨幹模組

2. **程式組件**
   __init__
   scenario_manager
   
   create_scenario
   scenario_manager
   
   save_scenario
   scenario_manager
   
   load_scenario
   scenario_manager
   
   list_scenarios
   scenario_manager
   
   _load_all_scenarios
   scenario_manager
   
   validate_scenario
   scenario_manager
   
   check_target_health
   scenario_manager
   
   _estimate_duration
   scenario_manager
   
   create_owasp_webgoat_scenarios
   scenario_manager
   
   create_juice_shop_scenarios
   scenario_manager
   
   _create_sql_injection_plan_easy
   scenario_manager
   
   _create_sql_injection_plan_medium
   scenario_manager
   
   _create_xss_plan_easy
   scenario_manager
   
   _create_ssrf_plan_medium
   scenario_manager
   
   _create_juice_shop_sql_login_plan
   scenario_manager
   
   _create_juice_shop_xss_dom_plan
   scenario_manager
   
   get_training_curriculum
   scenario_manager
   
   export_scenarios
   scenario_manager
   
   get_statistics
   scenario_manager
   - 模組: 服務骨幹模組

---

### Flow 314

- **長度**: 2 步
- **起點**: monitoring
- **終點**: optimized_core
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   monitor_performance
   monitoring
   
   __init__
   monitoring
   
   record_duration
   monitoring
   
   increment_counter
   monitoring
   
   set_gauge
   monitoring
   
   _make_key
   monitoring
   
   get_metrics_summary
   monitoring
   
   update_component_health
   monitoring
   
   get_system_health_status
   monitoring
   
   check_component_freshness
   monitoring
   
   decorator
   monitoring
   
   wrapper
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

---

### Flow 315

- **長度**: 3 步
- **起點**: monitoring
- **終點**: train_classifier
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   monitor_performance
   monitoring
   
   __init__
   monitoring
   
   record_duration
   monitoring
   
   increment_counter
   monitoring
   
   set_gauge
   monitoring
   
   _make_key
   monitoring
   
   get_metrics_summary
   monitoring
   
   update_component_health
   monitoring
   
   get_system_health_status
   monitoring
   
   check_component_freshness
   monitoring
   
   decorator
   monitoring
   
   wrapper
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   optimized_process_scan_results
   optimized_core
   
   optimized_ai_prediction
   optimized_core
   
   startup
   optimized_core
   
   get_metrics
   optimized_core
   
   health_check
   optimized_core
   
   prove_aiva_independence
   optimized_core
   
   __init__
   optimized_core
   
   predict
   optimized_core
   
   predict_batch
   optimized_core
   
   _compute_prediction
   optimized_core
   
   _get_cache_key
   optimized_core
   
   clear_cache
   optimized_core
   
   get_cache_stats
   optimized_core
   
   analyze_current_capabilities
   optimized_core
   
   compare_with_gpt4
   optimized_core
   
   demonstrate_self_sufficiency
   optimized_core
   
   final_verdict
   optimized_core
   - 模組: 服務骨幹模組

3. **程式組件**
   create_database
   train_classifier
   
   load_data
   train_classifier
   
   train_and_save_model
   train_classifier
   
   load_model
   train_classifier
   
   has_sklearn
   train_classifier
   
   require_sklearn
   train_classifier
   - 模組: 服務骨幹模組

---

### Flow 319

- **長度**: 2 步
- **起點**: initial_surface
- **終點**: exploit_orchestrator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   analyze
   initial_surface
   
   _summarize_asset
   initial_surface
   
   _calculate_risk_score
   initial_surface
   
   _detect_xss_candidates
   initial_surface
   
   _detect_sqli_candidates
   initial_surface
   
   _detect_ssrf_candidates
   initial_surface
   
   _detect_idor_candidates
   initial_surface
   
   _evaluate_parameter
   initial_surface
   - 模組: 服務骨幹模組

2. **程式組件**
   __init__
   exploit_orchestrator
   
   _initialize_exploits
   exploit_orchestrator
   
   _register_exploit
   exploit_orchestrator
   
   get_exploit
   exploit_orchestrator
   
   get_all_exploits
   exploit_orchestrator
   
   get_exploits_by_type
   exploit_orchestrator
   
   get_exploits_by_severity
   exploit_orchestrator
   
   orchestrate_exploit
   exploit_orchestrator
   
   analyze_results
   exploit_orchestrator
   
   _calculate_risk_score
   exploit_orchestrator
   
   _get_timestamp
   exploit_orchestrator
   
   get_execution_history
   exploit_orchestrator
   
   get_statistics
   exploit_orchestrator
   - 模組: 服務骨幹模組

---

### Flow 320

- **長度**: 2 步
- **起點**: initial_surface
- **終點**: scan_module_interface
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   analyze
   initial_surface
   
   _summarize_asset
   initial_surface
   
   _calculate_risk_score
   initial_surface
   
   _detect_xss_candidates
   initial_surface
   
   _detect_sqli_candidates
   initial_surface
   
   _detect_ssrf_candidates
   initial_surface
   
   _detect_idor_candidates
   initial_surface
   
   _evaluate_parameter
   initial_surface
   - 模組: 服務骨幹模組

2. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

---

### Flow 321

- **長度**: 3 步
- **起點**: initial_surface
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   analyze
   initial_surface
   
   _summarize_asset
   initial_surface
   
   _calculate_risk_score
   initial_surface
   
   _detect_xss_candidates
   initial_surface
   
   _detect_sqli_candidates
   initial_surface
   
   _detect_ssrf_candidates
   initial_surface
   
   _detect_idor_candidates
   initial_surface
   
   _evaluate_parameter
   initial_surface
   - 模組: 服務骨幹模組

2. **程式組件**
   process_scan_data
   scan_module_interface
   
   _process_assets
   scan_module_interface
   
   _process_fingerprints
   scan_module_interface
   
   _calculate_risk_score
   scan_module_interface
   
   _categorize_asset
   scan_module_interface
   
   send_phase0_command
   scan_module_interface
   
   send_phase1_command
   scan_module_interface
   
   process_phase0_result
   scan_module_interface
   - 模組: 服務骨幹模組

3. **程式組件**
   _count_tasks_by_type
   app
   
   startup
   app
   
   shutdown
   app
   
   health_check
   app
   
   get_scan_status
   app
   
   _process_single_scan_with_retry
   app
   
   process_phase0_results
   app
   
   process_scan_results
   app
   
   process_function_results
   app
   
   monitor_execution_status
   app
   - 模組: 服務骨幹模組

---

