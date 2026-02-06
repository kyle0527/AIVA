# 完整數據流詳細列表

生成時間: 2026-01-31 01:04:39
總數據流數量: 399

---

## 認知核心模組 (cognitive_core)

包含 69 條數據流

### Flow 5

- **長度**: 2 步
- **起點**: adaptive_weight_manager
- **終點**: adaptive_weight_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   WeightConfig
   adaptive_weight_manager
   
   WeightConfig.to_dict
   adaptive_weight_manager
   
   WeightConfig.normalize
   adaptive_weight_manager
   
   ContextFactors
   adaptive_weight_manager
   
   PerformanceRecord
   adaptive_weight_manager
   
   AdaptiveWeightManager
   adaptive_weight_manager
   
   AdaptiveWeightManager.__init__
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_weights
   adaptive_weight_manager
   
   AdaptiveWeightManager._apply_factor
   adaptive_weight_manager
   
   AdaptiveWeightManager.update_performance
   adaptive_weight_manager
   
   AdaptiveWeightManager._update_learned_offset
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_source_success_rates
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_statistics
   adaptive_weight_manager
   
   AdaptiveWeightManager._calculate_recent_success_rate
   adaptive_weight_manager
   
   AdaptiveWeightManager.reset_learning
   adaptive_weight_manager
   
   AdaptiveWeightManager.export_state
   adaptive_weight_manager
   
   AdaptiveWeightManager.import_state
   adaptive_weight_manager
   
   get_adaptive_weight_manager
   adaptive_weight_manager
   
   get_dynamic_weights
   adaptive_weight_manager
   - 模組: 認知核心模組

2. **程式組件**
   WeightConfig
   adaptive_weight_manager
   
   WeightConfig.to_dict
   adaptive_weight_manager
   
   WeightConfig.normalize
   adaptive_weight_manager
   
   ContextFactors
   adaptive_weight_manager
   
   PerformanceRecord
   adaptive_weight_manager
   
   AdaptiveWeightManager
   adaptive_weight_manager
   
   AdaptiveWeightManager.__init__
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_weights
   adaptive_weight_manager
   
   AdaptiveWeightManager._apply_factor
   adaptive_weight_manager
   
   AdaptiveWeightManager.update_performance
   adaptive_weight_manager
   
   AdaptiveWeightManager._update_learned_offset
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_source_success_rates
   adaptive_weight_manager
   
   AdaptiveWeightManager.get_statistics
   adaptive_weight_manager
   
   AdaptiveWeightManager._calculate_recent_success_rate
   adaptive_weight_manager
   
   AdaptiveWeightManager.reset_learning
   adaptive_weight_manager
   
   AdaptiveWeightManager.export_state
   adaptive_weight_manager
   
   AdaptiveWeightManager.import_state
   adaptive_weight_manager
   
   get_adaptive_weight_manager
   adaptive_weight_manager
   
   get_dynamic_weights
   adaptive_weight_manager
   - 模組: 認知核心模組

---

### Flow 8

- **長度**: 2 步
- **起點**: scan_result_processor
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   ScanResultProcessor
   scan_result_processor
   
   ScanResultProcessor.__init__
   scan_result_processor
   
   ScanResultProcessor.stage_1_ingest_data
   scan_result_processor
   
   ScanResultProcessor.stage_2_analyze_surface
   scan_result_processor
   
   ScanResultProcessor.stage_3_generate_strategy
   scan_result_processor
   
   ScanResultProcessor.stage_4_adjust_strategy
   scan_result_processor
   
   ScanResultProcessor.stage_5_generate_tasks
   scan_result_processor
   
   ScanResultProcessor.stage_6_dispatch_tasks
   scan_result_processor
   
   ScanResultProcessor.stage_7_monitor_execution
   scan_result_processor
   
   ScanResultProcessor.process
   scan_result_processor
   
   ScanResultProcessor.process_phase0
   scan_result_processor
   
   ScanResultProcessor._analyze_phase0_and_decide
   scan_result_processor
   
   ScanResultProcessor._fallback_rule_decision
   scan_result_processor
   
   ScanResultProcessor._select_engines_for_phase1
   scan_result_processor
   - 模組: 核心能力模組

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 16

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: capability_orchestrator
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

---

### Flow 18

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: execution_orchestrator
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   ExecutionResult
   execution_orchestrator
   
   ExecutionResult.__init__
   execution_orchestrator
   
   ExecutionOrchestrator
   execution_orchestrator
   
   ExecutionOrchestrator.__init__
   execution_orchestrator
   
   ExecutionOrchestrator.execute_plan
   execution_orchestrator
   
   ExecutionOrchestrator._build_cli_command
   execution_orchestrator
   
   ExecutionOrchestrator._check_dependencies
   execution_orchestrator
   
   ExecutionOrchestrator.get_execution_status
   execution_orchestrator
   
   ExecutionOrchestrator.list_active_executions
   execution_orchestrator
   - 模組: 認知核心模組

---

### Flow 20

- **長度**: 2 步
- **起點**: vector_store
- **終點**: capability_encoder
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

2. **程式組件**
   EncodingConfig
   capability_encoder
   
   EncodingConfig.validate
   capability_encoder
   
   CapabilityEncoder
   capability_encoder
   
   CapabilityEncoder.__init__
   capability_encoder
   
   CapabilityEncoder.encode
   capability_encoder
   
   CapabilityEncoder.encode_batch
   capability_encoder
   
   CapabilityEncoder._encode_module
   capability_encoder
   
   CapabilityEncoder._encode_component_type
   capability_encoder
   
   CapabilityEncoder._encode_parameters
   capability_encoder
   
   CapabilityEncoder._encode_tags
   capability_encoder
   
   CapabilityEncoder._encode_structure
   capability_encoder
   
   CapabilityEncoder.similarity
   capability_encoder
   
   CapabilityEncoder.find_similar
   capability_encoder
   
   encode_capability
   capability_encoder
   
   encode_capabilities
   capability_encoder
   - 模組: 認知核心模組

---

### Flow 23

- **長度**: 2 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: web_architecture
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   ArchitectureType
   web_architecture
   
   AuthScheme
   web_architecture
   
   ArchitectureFingerprint
   web_architecture
   
   ArchitectureFingerprint.to_dict
   web_architecture
   
   JWTAnalysis
   web_architecture
   
   JWTAnalysis.to_dict
   web_architecture
   
   GraphQLSchema
   web_architecture
   
   GraphQLSchema.to_dict
   web_architecture
   
   WebArchitectureAnalyzer
   web_architecture
   
   WebArchitectureAnalyzer.detect_graphql_introspection
   web_architecture
   
   WebArchitectureAnalyzer.parse_graphql_schema
   web_architecture
   
   WebArchitectureAnalyzer.analyze_jwt
   web_architecture
   
   WebArchitectureAnalyzer.generate_jwt_attack_payloads
   web_architecture
   
   WebArchitectureAnalyzer.check_bola
   web_architecture
   
   WebArchitectureAnalyzer._calculate_response_similarity
   web_architecture
   
   WebArchitectureAnalyzer.check_mass_assignment
   web_architecture
   
   WebArchitectureAnalyzer.check_websocket_security
   web_architecture
   
   WebArchitectureAnalyzer.identify_architecture
   web_architecture
   - 模組: 認知核心模組

---

### Flow 24

- **長度**: 3 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: cve_identification
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   SignalTier
   cve_identification
   
   CVESignature
   cve_identification
   
   CVESignature.to_dict
   cve_identification
   
   CVEMatch
   cve_identification
   
   CVEMatch.to_dict
   cve_identification
   
   CVEMatch.is_exploitable
   cve_identification
   
   CVEIdentifier
   cve_identification
   
   CVEIdentifier.identify
   cve_identification
   
   CVEIdentifier.identify_by_fingerprint
   cve_identification
   
   CVEIdentifier.get_exploit_payloads
   cve_identification
   
   CVEIdentifier._check_single_cve
   cve_identification
   
   CVEIdentifier._build_exploit_recommendations
   cve_identification
   
   CVEIdentifier.register_cve
   cve_identification
   
   CVEIdentifier.get_all_cve_ids
   cve_identification
   
   CVEIdentifier.get_cve_by_severity
   cve_identification
   - 模組: 認知核心模組

3. **程式組件**
   SignalTier
   cve_identification
   
   CVESignature
   cve_identification
   
   CVESignature.to_dict
   cve_identification
   
   CVEMatch
   cve_identification
   
   CVEMatch.to_dict
   cve_identification
   
   CVEMatch.is_exploitable
   cve_identification
   
   CVEIdentifier
   cve_identification
   
   CVEIdentifier.identify
   cve_identification
   
   CVEIdentifier.identify_by_fingerprint
   cve_identification
   
   CVEIdentifier.get_exploit_payloads
   cve_identification
   
   CVEIdentifier._check_single_cve
   cve_identification
   
   CVEIdentifier._build_exploit_recommendations
   cve_identification
   
   CVEIdentifier.register_cve
   cve_identification
   
   CVEIdentifier.get_all_cve_ids
   cve_identification
   
   CVEIdentifier.get_cve_by_severity
   cve_identification
   - 模組: 認知核心模組

---

### Flow 25

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 26

- **長度**: 2 步
- **起點**: skill_graph
- **終點**: skill_graph
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   SkillNode
   skill_graph
   
   SkillEdge
   skill_graph
   
   SkillPath
   skill_graph
   
   SkillGraphBuilder
   skill_graph
   
   SkillGraphBuilder.__init__
   skill_graph
   
   SkillGraphBuilder.build_graph
   skill_graph
   
   SkillGraphBuilder._extract_success_rate
   skill_graph
   
   SkillGraphBuilder._extract_usage_count
   skill_graph
   
   SkillGraphBuilder._build_node_metadata
   skill_graph
   
   SkillGraphBuilder._create_skill_nodes
   skill_graph
   
   SkillGraphBuilder._analyze_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_prerequisite_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_tag_similarity_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_language_ecosystem_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_topic_relationships
   skill_graph
   
   SkillGraphBuilder._check_io_compatibility
   skill_graph
   
   SkillGraphBuilder._analyze_io_relationships
   skill_graph
   
   SkillGraphBuilder._is_compatible_io
   skill_graph
   
   SkillGraphBuilder._build_networkx_graph
   skill_graph
   
   SkillGraphAnalyzer
   skill_graph
   
   SkillGraphAnalyzer.__init__
   skill_graph
   
   SkillGraphAnalyzer.find_optimal_path
   skill_graph
   
   SkillGraphAnalyzer._find_goal_capabilities
   skill_graph
   
   SkillGraphAnalyzer._create_skill_path
   skill_graph
   
   SkillGraphAnalyzer.get_capability_recommendations
   skill_graph
   
   SkillGraphAnalyzer.analyze_capability_centrality
   skill_graph
   
   AIVASkillGraph
   skill_graph
   
   AIVASkillGraph.__init__
   skill_graph
   
   AIVASkillGraph.initialize
   skill_graph
   
   AIVASkillGraph.rebuild_if_needed
   skill_graph
   
   AIVASkillGraph.find_execution_path
   skill_graph
   
   AIVASkillGraph.get_recommendations
   skill_graph
   
   AIVASkillGraph.analyze_centrality
   skill_graph
   
   AIVASkillGraph.get_graph_statistics
   skill_graph
   - 模組: 認知核心模組

2. **程式組件**
   SkillNode
   skill_graph
   
   SkillEdge
   skill_graph
   
   SkillPath
   skill_graph
   
   SkillGraphBuilder
   skill_graph
   
   SkillGraphBuilder.__init__
   skill_graph
   
   SkillGraphBuilder.build_graph
   skill_graph
   
   SkillGraphBuilder._extract_success_rate
   skill_graph
   
   SkillGraphBuilder._extract_usage_count
   skill_graph
   
   SkillGraphBuilder._build_node_metadata
   skill_graph
   
   SkillGraphBuilder._create_skill_nodes
   skill_graph
   
   SkillGraphBuilder._analyze_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_prerequisite_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_tag_similarity_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_language_ecosystem_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_topic_relationships
   skill_graph
   
   SkillGraphBuilder._check_io_compatibility
   skill_graph
   
   SkillGraphBuilder._analyze_io_relationships
   skill_graph
   
   SkillGraphBuilder._is_compatible_io
   skill_graph
   
   SkillGraphBuilder._build_networkx_graph
   skill_graph
   
   SkillGraphAnalyzer
   skill_graph
   
   SkillGraphAnalyzer.__init__
   skill_graph
   
   SkillGraphAnalyzer.find_optimal_path
   skill_graph
   
   SkillGraphAnalyzer._find_goal_capabilities
   skill_graph
   
   SkillGraphAnalyzer._create_skill_path
   skill_graph
   
   SkillGraphAnalyzer.get_capability_recommendations
   skill_graph
   
   SkillGraphAnalyzer.analyze_capability_centrality
   skill_graph
   
   AIVASkillGraph
   skill_graph
   
   AIVASkillGraph.__init__
   skill_graph
   
   AIVASkillGraph.initialize
   skill_graph
   
   AIVASkillGraph.rebuild_if_needed
   skill_graph
   
   AIVASkillGraph.find_execution_path
   skill_graph
   
   AIVASkillGraph.get_recommendations
   skill_graph
   
   AIVASkillGraph.analyze_centrality
   skill_graph
   
   AIVASkillGraph.get_graph_statistics
   skill_graph
   - 模組: 認知核心模組

---

### Flow 34

- **長度**: 2 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: INTEGRATION_EXAMPLE
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

---

### Flow 35

- **長度**: 2 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 36

- **長度**: 4 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: waf_bypass
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

3. **程式組件**
   BypassCategory
   waf_bypass
   
   BypassTechnique
   waf_bypass
   
   BypassTechnique.to_dict
   waf_bypass
   
   WAFBypassEngine
   waf_bypass
   
   WAFBypassEngine.detect_waf
   waf_bypass
   
   WAFBypassEngine.get_bypass_techniques
   waf_bypass
   
   WAFBypassEngine.mutate_payload
   waf_bypass
   
   WAFBypassEngine.encode_ibm037
   waf_bypass
   
   WAFBypassEngine.generate_chunked_body
   waf_bypass
   
   WAFBypassEngine.generate_padding
   waf_bypass
   
   WAFBypassEngine.get_spoof_headers
   waf_bypass
   
   WAFBypassEngine.get_browser_headers
   waf_bypass
   
   WAFBypassEngine.register_technique
   waf_bypass
   
   WAFBypassEngine.register_waf_signature
   waf_bypass
   - 模組: 認知核心模組

4. **程式組件**
   BypassCategory
   waf_bypass
   
   BypassTechnique
   waf_bypass
   
   BypassTechnique.to_dict
   waf_bypass
   
   WAFBypassEngine
   waf_bypass
   
   WAFBypassEngine.detect_waf
   waf_bypass
   
   WAFBypassEngine.get_bypass_techniques
   waf_bypass
   
   WAFBypassEngine.mutate_payload
   waf_bypass
   
   WAFBypassEngine.encode_ibm037
   waf_bypass
   
   WAFBypassEngine.generate_chunked_body
   waf_bypass
   
   WAFBypassEngine.generate_padding
   waf_bypass
   
   WAFBypassEngine.get_spoof_headers
   waf_bypass
   
   WAFBypassEngine.get_browser_headers
   waf_bypass
   
   WAFBypassEngine.register_technique
   waf_bypass
   
   WAFBypassEngine.register_waf_signature
   waf_bypass
   - 模組: 認知核心模組

---

### Flow 45

- **長度**: 2 步
- **起點**: multilang_coordinator
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   log_cross_language_call
   multilang_coordinator
   
   MultiLanguageAICoordinator
   multilang_coordinator
   
   MultiLanguageAICoordinator.__init__
   multilang_coordinator
   
   MultiLanguageAICoordinator.initialize
   multilang_coordinator
   
   MultiLanguageAICoordinator.check_module_availability
   multilang_coordinator
   
   MultiLanguageAICoordinator.execute_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._execute_python_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._select_best_language
   multilang_coordinator
   
   MultiLanguageAICoordinator.get_status
   multilang_coordinator
   
   MultiLanguageAICoordinator.enable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator.disable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_rust_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_go_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_typescript_service
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_rust_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_go_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_typescript_ai
   multilang_coordinator
   - 模組: 核心能力模組

2. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 47

- **長度**: 2 步
- **起點**: capability_orchestrator
- **終點**: internal_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

2. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

---

### Flow 55

- **長度**: 2 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   ConfidenceLevel
   base
   
   ConfidenceLevel.to_score
   base
   
   VulnerabilityType
   base
   
   DatabaseType
   base
   
   WAFVendor
   base
   
   CloudProvider
   base
   
   DetectionResult
   base
   
   DetectionResult.to_dict
   base
   
   DetectionResult.should_exploit
   base
   
   AttackContext
   base
   
   ResponseAnalysis
   base
   
   PayloadResult
   base
   
   KnowledgeRegistry
   base
   
   KnowledgeRegistry.__new__
   base
   
   KnowledgeRegistry._initialize
   base
   
   KnowledgeRegistry.register_sqli_fingerprint
   base
   
   KnowledgeRegistry.get_sqli_fingerprints
   base
   
   KnowledgeRegistry.register_xss_payload
   base
   
   KnowledgeRegistry.get_xss_payloads
   base
   
   KnowledgeRegistry.register_waf_signature
   base
   
   KnowledgeRegistry.register_cve_pattern
   base
   
   KnowledgeRegistry.get_cve_pattern
   base
   
   KnowledgeRegistry.register_detector
   base
   
   KnowledgeRegistry.run_custom_detector
   base
   - 模組: 認知核心模組

---

### Flow 56

- **長度**: 2 步
- **起點**: anti_hallucination_module
- **終點**: anti_hallucination_module
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   KnowledgeBaseUnavailableError
   anti_hallucination_module
   
   AntiHallucinationModule
   anti_hallucination_module
   
   AntiHallucinationModule.__init__
   anti_hallucination_module
   
   AntiHallucinationModule._require_knowledge_base
   anti_hallucination_module
   
   AntiHallucinationModule._get_technique_category
   anti_hallucination_module
   
   AntiHallucinationModule._validate_technique_consistency
   anti_hallucination_module
   
   AntiHallucinationModule._setup_logger
   anti_hallucination_module
   
   AntiHallucinationModule.validate_attack_plan
   anti_hallucination_module
   
   AntiHallucinationModule._validate_single_step
   anti_hallucination_module
   
   AntiHallucinationModule._validate_step_sequence
   anti_hallucination_module
   
   AntiHallucinationModule._is_known_technique
   anti_hallucination_module
   
   AntiHallucinationModule._extract_relevance_score
   anti_hallucination_module
   
   AntiHallucinationModule._validate_with_knowledge_base
   anti_hallucination_module
   
   AntiHallucinationModule._validate_step_logic
   anti_hallucination_module
   
   AntiHallucinationModule.get_validation_stats
   anti_hallucination_module
   
   AntiHallucinationModule.export_validation_report
   anti_hallucination_module
   
   AntiHallucinationModule.reset_knowledge_base
   anti_hallucination_module
   - 模組: 認知核心模組

2. **程式組件**
   KnowledgeBaseUnavailableError
   anti_hallucination_module
   
   AntiHallucinationModule
   anti_hallucination_module
   
   AntiHallucinationModule.__init__
   anti_hallucination_module
   
   AntiHallucinationModule._require_knowledge_base
   anti_hallucination_module
   
   AntiHallucinationModule._get_technique_category
   anti_hallucination_module
   
   AntiHallucinationModule._validate_technique_consistency
   anti_hallucination_module
   
   AntiHallucinationModule._setup_logger
   anti_hallucination_module
   
   AntiHallucinationModule.validate_attack_plan
   anti_hallucination_module
   
   AntiHallucinationModule._validate_single_step
   anti_hallucination_module
   
   AntiHallucinationModule._validate_step_sequence
   anti_hallucination_module
   
   AntiHallucinationModule._is_known_technique
   anti_hallucination_module
   
   AntiHallucinationModule._extract_relevance_score
   anti_hallucination_module
   
   AntiHallucinationModule._validate_with_knowledge_base
   anti_hallucination_module
   
   AntiHallucinationModule._validate_step_logic
   anti_hallucination_module
   
   AntiHallucinationModule.get_validation_stats
   anti_hallucination_module
   
   AntiHallucinationModule.export_validation_report
   anti_hallucination_module
   
   AntiHallucinationModule.reset_knowledge_base
   anti_hallucination_module
   - 模組: 認知核心模組

---

### Flow 62

- **長度**: 2 步
- **起點**: real_bio_net_adapter
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   RealScalableBioNet
   real_bio_net_adapter
   
   RealScalableBioNet.__init__
   real_bio_net_adapter
   
   RealScalableBioNet._load_or_initialize_weights
   real_bio_net_adapter
   
   RealScalableBioNet.forward
   real_bio_net_adapter
   
   RealScalableBioNet._softmax
   real_bio_net_adapter
   
   RealScalableBioNet.save_weights
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent.__init__
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent.generate
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent._create_real_input_vector
   real_bio_net_adapter
   
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   - 模組: 認知核心模組

2. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 90

- **長度**: 2 步
- **起點**: INTEGRATION_EXAMPLE
- **終點**: vulnerability_detection
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnhancedDecisionWithKnowledge
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.__init__
   INTEGRATION_EXAMPLE
   
   EnhancedDecisionWithKnowledge.decide_with_vuln_check
   INTEGRATION_EXAMPLE
   
   identify_high_risk_target
   INTEGRATION_EXAMPLE
   
   bypass_waf_automatically
   INTEGRATION_EXAMPLE
   
   analyze_graphql_api
   INTEGRATION_EXAMPLE
   
   full_decision_pipeline_example
   INTEGRATION_EXAMPLE
   - 模組: 認知核心模組

2. **程式組件**
   VulnerabilityDetector
   vulnerability_detection
   
   VulnerabilityDetector.check_sqli
   vulnerability_detection
   
   VulnerabilityDetector.check_xss
   vulnerability_detection
   
   VulnerabilityDetector.check_ssrf
   vulnerability_detection
   
   VulnerabilityDetector.check_idor
   vulnerability_detection
   
   VulnerabilityDetector._detect_waf
   vulnerability_detection
   
   VulnerabilityDetector._check_sqli_false_positive
   vulnerability_detection
   
   VulnerabilityDetector._build_sqli_recommendations
   vulnerability_detection
   
   VulnerabilityDetector._calculate_response_similarity
   vulnerability_detection
   
   VulnerabilityDetector.get_sqli_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_xss_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_ssrf_targets
   vulnerability_detection
   - 模組: 認知核心模組

---

### Flow 98

- **長度**: 2 步
- **起點**: real_neural_core
- **終點**: aiva_embedding
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

2. **程式組件**
   AIVAEmbedding
   aiva_embedding
   
   AIVAEmbedding.__init__
   aiva_embedding
   
   AIVAEmbedding._mean_pooling
   aiva_embedding
   
   AIVAEmbedding.forward
   aiva_embedding
   
   AIVAEmbedding.encode
   aiva_embedding
   
   AIVAEmbedding.similarity
   aiva_embedding
   
   AIVAEmbedding.save
   aiva_embedding
   
   AIVAEmbedding.load
   aiva_embedding
   
   SentenceTransformer
   aiva_embedding
   - 模組: 認知核心模組

---

### Flow 99

- **長度**: 2 步
- **起點**: capability_orchestrator
- **終點**: capability_orchestrator
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

2. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

---

### Flow 101

- **長度**: 2 步
- **起點**: capability_registry
- **終點**: unified_vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

2. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

---

### Flow 102

- **長度**: 2 步
- **起點**: capability_registry
- **終點**: knowledge_base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

2. **程式組件**
   VectorStoreProtocol
   knowledge_base
   
   VectorStoreProtocol.add_document
   knowledge_base
   
   VectorStoreProtocol.search
   knowledge_base
   
   VectorStoreProtocol.add_capability_from_registry
   knowledge_base
   
   VectorStoreProtocol.search_by_environment
   knowledge_base
   
   KnowledgeBase
   knowledge_base
   
   KnowledgeBase.__init__
   knowledge_base
   
   KnowledgeBase.search
   knowledge_base
   
   KnowledgeBase.query
   knowledge_base
   
   KnowledgeBase.add_knowledge
   knowledge_base
   
   KnowledgeBase.index_codebase
   knowledge_base
   
   KnowledgeBase.get_stats
   knowledge_base
   - 模組: 認知核心模組

---

### Flow 103

- **長度**: 2 步
- **起點**: capability_registry
- **終點**: internal_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

2. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

---

### Flow 108

- **長度**: 2 步
- **起點**: internal_loop_connector
- **終點**: internal_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

2. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

---

### Flow 113

- **長度**: 2 步
- **起點**: event_listener
- **終點**: external_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   ExternalLearningListener
   event_listener
   
   ExternalLearningListener.__init__
   event_listener
   
   ExternalLearningListener.broker
   event_listener
   
   ExternalLearningListener.connector
   event_listener
   
   ExternalLearningListener.knowledge_manager
   event_listener
   
   ExternalLearningListener.start_listening
   event_listener
   
   ExternalLearningListener.stop_listening
   event_listener
   
   ExternalLearningListener._on_result_received
   event_listener
   
   ExternalLearningListener._process_finding
   event_listener
   
   ExternalLearningListener.get_statistics
   event_listener
   
   main
   event_listener
   - 模組: 認知核心模組(學習子系統)

2. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

---

### Flow 120

- **長度**: 2 步
- **起點**: capability_orchestrator
- **終點**: execution_orchestrator
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

2. **程式組件**
   ExecutionResult
   execution_orchestrator
   
   ExecutionResult.__init__
   execution_orchestrator
   
   ExecutionOrchestrator
   execution_orchestrator
   
   ExecutionOrchestrator.__init__
   execution_orchestrator
   
   ExecutionOrchestrator.execute_plan
   execution_orchestrator
   
   ExecutionOrchestrator._build_cli_command
   execution_orchestrator
   
   ExecutionOrchestrator._check_dependencies
   execution_orchestrator
   
   ExecutionOrchestrator.get_execution_status
   execution_orchestrator
   
   ExecutionOrchestrator.list_active_executions
   execution_orchestrator
   - 模組: 認知核心模組

---

### Flow 121

- **長度**: 2 步
- **起點**: dispatcher
- **終點**: dispatcher
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   CognitiveDispatcher
   dispatcher
   
   CognitiveDispatcher.__init__
   dispatcher
   
   CognitiveDispatcher.broker
   dispatcher
   
   CognitiveDispatcher._build_message
   dispatcher
   
   CognitiveDispatcher.request_plan
   dispatcher
   
   CognitiveDispatcher.execute_capability
   dispatcher
   
   CognitiveDispatcher.trigger_learning
   dispatcher
   
   CognitiveDispatcher.notify_decision
   dispatcher
   
   CognitiveDispatcher.store_result
   dispatcher
   
   CognitiveDispatcher.call_task_planning_sync
   dispatcher
   
   CognitiveDispatcher.call_core_capabilities_sync
   dispatcher
   
   CognitiveDispatcher.call_external_learning_sync
   dispatcher
   
   CognitiveDispatcher.execute_and_notify
   dispatcher
   
   CognitiveDispatcher.get_dispatch_stats
   dispatcher
   
   get_dispatcher
   dispatcher
   
   dispatch_to_task_planning
   dispatcher
   
   dispatch_to_core_capabilities
   dispatcher
   
   dispatch_to_external_learning
   dispatcher
   
   PlanningDispatcher
   dispatcher
   
   PlanningDispatcher.__init__
   dispatcher
   
   PlanningDispatcher.broker
   dispatcher
   
   PlanningDispatcher._build_message
   dispatcher
   
   PlanningDispatcher.execute_plan_step
   dispatcher
   
   PlanningDispatcher.confirm_decision
   dispatcher
   
   PlanningDispatcher.query_resource
   dispatcher
   
   PlanningDispatcher.request_analysis
   dispatcher
   
   PlanningDispatcher.notify_plan_status
   dispatcher
   
   PlanningDispatcher.execute_attack_sync
   dispatcher
   
   PlanningDispatcher.execute_scan_sync
   dispatcher
   
   PlanningDispatcher.call_cognitive_sync
   dispatcher
   
   PlanningDispatcher.call_exploration_sync
   dispatcher
   
   PlanningDispatcher.execute_plan
   dispatcher
   
   PlanningDispatcher.execute_with_confirmation
   dispatcher
   
   PlanningDispatcher.get_dispatch_stats
   dispatcher
   
   dispatch_to_cognitive_core
   dispatcher
   
   execute_attack
   dispatcher
   
   execute_scan
   dispatcher
   - 模組: 任務規劃模組

2. **程式組件**
   CognitiveDispatcher
   dispatcher
   
   CognitiveDispatcher.__init__
   dispatcher
   
   CognitiveDispatcher.broker
   dispatcher
   
   CognitiveDispatcher._build_message
   dispatcher
   
   CognitiveDispatcher.request_plan
   dispatcher
   
   CognitiveDispatcher.execute_capability
   dispatcher
   
   CognitiveDispatcher.trigger_learning
   dispatcher
   
   CognitiveDispatcher.notify_decision
   dispatcher
   
   CognitiveDispatcher.store_result
   dispatcher
   
   CognitiveDispatcher.call_task_planning_sync
   dispatcher
   
   CognitiveDispatcher.call_core_capabilities_sync
   dispatcher
   
   CognitiveDispatcher.call_external_learning_sync
   dispatcher
   
   CognitiveDispatcher.execute_and_notify
   dispatcher
   
   CognitiveDispatcher.get_dispatch_stats
   dispatcher
   
   get_dispatcher
   dispatcher
   
   dispatch_to_task_planning
   dispatcher
   
   dispatch_to_core_capabilities
   dispatcher
   
   dispatch_to_external_learning
   dispatcher
   
   PlanningDispatcher
   dispatcher
   
   PlanningDispatcher.__init__
   dispatcher
   
   PlanningDispatcher.broker
   dispatcher
   
   PlanningDispatcher._build_message
   dispatcher
   
   PlanningDispatcher.execute_plan_step
   dispatcher
   
   PlanningDispatcher.confirm_decision
   dispatcher
   
   PlanningDispatcher.query_resource
   dispatcher
   
   PlanningDispatcher.request_analysis
   dispatcher
   
   PlanningDispatcher.notify_plan_status
   dispatcher
   
   PlanningDispatcher.execute_attack_sync
   dispatcher
   
   PlanningDispatcher.execute_scan_sync
   dispatcher
   
   PlanningDispatcher.call_cognitive_sync
   dispatcher
   
   PlanningDispatcher.call_exploration_sync
   dispatcher
   
   PlanningDispatcher.execute_plan
   dispatcher
   
   PlanningDispatcher.execute_with_confirmation
   dispatcher
   
   PlanningDispatcher.get_dispatch_stats
   dispatcher
   
   dispatch_to_cognitive_core
   dispatcher
   
   execute_attack
   dispatcher
   
   execute_scan
   dispatcher
   - 模組: 認知核心模組

---

### Flow 128

- **長度**: 2 步
- **起點**: unified_vector_store
- **終點**: vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

2. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

---

### Flow 131

- **長度**: 2 步
- **起點**: unified_vector_store
- **終點**: unified_vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

2. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

---

### Flow 133

- **長度**: 2 步
- **起點**: weight_manager
- **終點**: weight_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   WeightMetadata
   weight_manager
   
   AIWeightManager
   weight_manager
   
   AIWeightManager.__init__
   weight_manager
   
   AIWeightManager.save_model_weights
   weight_manager
   
   AIWeightManager.load_model_weights
   weight_manager
   
   AIWeightManager.list_available_weights
   weight_manager
   
   AIWeightManager._list_model_versions
   weight_manager
   
   AIWeightManager._extract_version_info
   weight_manager
   
   AIWeightManager._list_all_models
   weight_manager
   
   AIWeightManager.delete_weights
   weight_manager
   
   AIWeightManager._find_weight_file
   weight_manager
   
   AIWeightManager._calculate_file_hash
   weight_manager
   
   AIWeightManager._save_metadata
   weight_manager
   
   AIWeightManager._load_and_verify_metadata
   weight_manager
   
   AIWeightManager._verify_model_compatibility
   weight_manager
   
   AIWeightManager._create_backup
   weight_manager
   
   AIWeightManager._cleanup_old_backups
   weight_manager
   
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   - 模組: 認知核心模組

2. **程式組件**
   WeightMetadata
   weight_manager
   
   AIWeightManager
   weight_manager
   
   AIWeightManager.__init__
   weight_manager
   
   AIWeightManager.save_model_weights
   weight_manager
   
   AIWeightManager.load_model_weights
   weight_manager
   
   AIWeightManager.list_available_weights
   weight_manager
   
   AIWeightManager._list_model_versions
   weight_manager
   
   AIWeightManager._extract_version_info
   weight_manager
   
   AIWeightManager._list_all_models
   weight_manager
   
   AIWeightManager.delete_weights
   weight_manager
   
   AIWeightManager._find_weight_file
   weight_manager
   
   AIWeightManager._calculate_file_hash
   weight_manager
   
   AIWeightManager._save_metadata
   weight_manager
   
   AIWeightManager._load_and_verify_metadata
   weight_manager
   
   AIWeightManager._verify_model_compatibility
   weight_manager
   
   AIWeightManager._create_backup
   weight_manager
   
   AIWeightManager._cleanup_old_backups
   weight_manager
   
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   - 模組: 認知核心模組

---

### Flow 138

- **長度**: 2 步
- **起點**: unified_vector_store
- **終點**: aiva_embedding
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

2. **程式組件**
   AIVAEmbedding
   aiva_embedding
   
   AIVAEmbedding.__init__
   aiva_embedding
   
   AIVAEmbedding._mean_pooling
   aiva_embedding
   
   AIVAEmbedding.forward
   aiva_embedding
   
   AIVAEmbedding.encode
   aiva_embedding
   
   AIVAEmbedding.similarity
   aiva_embedding
   
   AIVAEmbedding.save
   aiva_embedding
   
   AIVAEmbedding.load
   aiva_embedding
   
   SentenceTransformer
   aiva_embedding
   - 模組: 認知核心模組

---

### Flow 139

- **長度**: 2 步
- **起點**: capability_encoder
- **終點**: capability_encoder
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EncodingConfig
   capability_encoder
   
   EncodingConfig.validate
   capability_encoder
   
   CapabilityEncoder
   capability_encoder
   
   CapabilityEncoder.__init__
   capability_encoder
   
   CapabilityEncoder.encode
   capability_encoder
   
   CapabilityEncoder.encode_batch
   capability_encoder
   
   CapabilityEncoder._encode_module
   capability_encoder
   
   CapabilityEncoder._encode_component_type
   capability_encoder
   
   CapabilityEncoder._encode_parameters
   capability_encoder
   
   CapabilityEncoder._encode_tags
   capability_encoder
   
   CapabilityEncoder._encode_structure
   capability_encoder
   
   CapabilityEncoder.similarity
   capability_encoder
   
   CapabilityEncoder.find_similar
   capability_encoder
   
   encode_capability
   capability_encoder
   
   encode_capabilities
   capability_encoder
   - 模組: 認知核心模組

2. **程式組件**
   EncodingConfig
   capability_encoder
   
   EncodingConfig.validate
   capability_encoder
   
   CapabilityEncoder
   capability_encoder
   
   CapabilityEncoder.__init__
   capability_encoder
   
   CapabilityEncoder.encode
   capability_encoder
   
   CapabilityEncoder.encode_batch
   capability_encoder
   
   CapabilityEncoder._encode_module
   capability_encoder
   
   CapabilityEncoder._encode_component_type
   capability_encoder
   
   CapabilityEncoder._encode_parameters
   capability_encoder
   
   CapabilityEncoder._encode_tags
   capability_encoder
   
   CapabilityEncoder._encode_structure
   capability_encoder
   
   CapabilityEncoder.similarity
   capability_encoder
   
   CapabilityEncoder.find_similar
   capability_encoder
   
   encode_capability
   capability_encoder
   
   encode_capabilities
   capability_encoder
   - 模組: 認知核心模組

---

### Flow 141

- **長度**: 2 步
- **起點**: ai_capability_query
- **終點**: knowledge_base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

2. **程式組件**
   VectorStoreProtocol
   knowledge_base
   
   VectorStoreProtocol.add_document
   knowledge_base
   
   VectorStoreProtocol.search
   knowledge_base
   
   VectorStoreProtocol.add_capability_from_registry
   knowledge_base
   
   VectorStoreProtocol.search_by_environment
   knowledge_base
   
   KnowledgeBase
   knowledge_base
   
   KnowledgeBase.__init__
   knowledge_base
   
   KnowledgeBase.search
   knowledge_base
   
   KnowledgeBase.query
   knowledge_base
   
   KnowledgeBase.add_knowledge
   knowledge_base
   
   KnowledgeBase.index_codebase
   knowledge_base
   
   KnowledgeBase.get_stats
   knowledge_base
   - 模組: 認知核心模組

---

### Flow 154

- **長度**: 2 步
- **起點**: vulnerability_detection
- **終點**: base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   VulnerabilityDetector
   vulnerability_detection
   
   VulnerabilityDetector.check_sqli
   vulnerability_detection
   
   VulnerabilityDetector.check_xss
   vulnerability_detection
   
   VulnerabilityDetector.check_ssrf
   vulnerability_detection
   
   VulnerabilityDetector.check_idor
   vulnerability_detection
   
   VulnerabilityDetector._detect_waf
   vulnerability_detection
   
   VulnerabilityDetector._check_sqli_false_positive
   vulnerability_detection
   
   VulnerabilityDetector._build_sqli_recommendations
   vulnerability_detection
   
   VulnerabilityDetector._calculate_response_similarity
   vulnerability_detection
   
   VulnerabilityDetector.get_sqli_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_xss_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_ssrf_targets
   vulnerability_detection
   - 模組: 認知核心模組

2. **程式組件**
   ConfidenceLevel
   base
   
   ConfidenceLevel.to_score
   base
   
   VulnerabilityType
   base
   
   DatabaseType
   base
   
   WAFVendor
   base
   
   CloudProvider
   base
   
   DetectionResult
   base
   
   DetectionResult.to_dict
   base
   
   DetectionResult.should_exploit
   base
   
   AttackContext
   base
   
   ResponseAnalysis
   base
   
   PayloadResult
   base
   
   KnowledgeRegistry
   base
   
   KnowledgeRegistry.__new__
   base
   
   KnowledgeRegistry._initialize
   base
   
   KnowledgeRegistry.register_sqli_fingerprint
   base
   
   KnowledgeRegistry.get_sqli_fingerprints
   base
   
   KnowledgeRegistry.register_xss_payload
   base
   
   KnowledgeRegistry.get_xss_payloads
   base
   
   KnowledgeRegistry.register_waf_signature
   base
   
   KnowledgeRegistry.register_cve_pattern
   base
   
   KnowledgeRegistry.get_cve_pattern
   base
   
   KnowledgeRegistry.register_detector
   base
   
   KnowledgeRegistry.run_custom_detector
   base
   - 模組: 認知核心模組

---

### Flow 155

- **長度**: 2 步
- **起點**: postgresql_vector_store
- **終點**: postgresql_vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   PostgreSQLVectorStore
   postgresql_vector_store
   
   PostgreSQLVectorStore.__init__
   postgresql_vector_store
   
   PostgreSQLVectorStore.initialize
   postgresql_vector_store
   
   PostgreSQLVectorStore.add_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.search
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.delete_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_statistics
   postgresql_vector_store
   
   PostgreSQLVectorStore.execute_unified_query
   postgresql_vector_store
   
   PostgreSQLVectorStore.close
   postgresql_vector_store
   
   demo_postgresql_vector_store
   postgresql_vector_store
   - 模組: 認知核心模組

2. **程式組件**
   PostgreSQLVectorStore
   postgresql_vector_store
   
   PostgreSQLVectorStore.__init__
   postgresql_vector_store
   
   PostgreSQLVectorStore.initialize
   postgresql_vector_store
   
   PostgreSQLVectorStore.add_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.search
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.delete_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_statistics
   postgresql_vector_store
   
   PostgreSQLVectorStore.execute_unified_query
   postgresql_vector_store
   
   PostgreSQLVectorStore.close
   postgresql_vector_store
   
   demo_postgresql_vector_store
   postgresql_vector_store
   - 模組: 認知核心模組

---

### Flow 194

- **長度**: 3 步
- **起點**: sync_experiences
- **終點**: vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

3. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

---

### Flow 197

- **長度**: 2 步
- **起點**: app
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   ScanRequest
   app
   
   ScanResponse
   app
   
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
   
   start_scan
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

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 199

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   ConfidenceLevel
   base
   
   ConfidenceLevel.to_score
   base
   
   VulnerabilityType
   base
   
   DatabaseType
   base
   
   WAFVendor
   base
   
   CloudProvider
   base
   
   DetectionResult
   base
   
   DetectionResult.to_dict
   base
   
   DetectionResult.should_exploit
   base
   
   AttackContext
   base
   
   ResponseAnalysis
   base
   
   PayloadResult
   base
   
   KnowledgeRegistry
   base
   
   KnowledgeRegistry.__new__
   base
   
   KnowledgeRegistry._initialize
   base
   
   KnowledgeRegistry.register_sqli_fingerprint
   base
   
   KnowledgeRegistry.get_sqli_fingerprints
   base
   
   KnowledgeRegistry.register_xss_payload
   base
   
   KnowledgeRegistry.get_xss_payloads
   base
   
   KnowledgeRegistry.register_waf_signature
   base
   
   KnowledgeRegistry.register_cve_pattern
   base
   
   KnowledgeRegistry.get_cve_pattern
   base
   
   KnowledgeRegistry.register_detector
   base
   
   KnowledgeRegistry.run_custom_detector
   base
   - 模組: 認知核心模組

---

### Flow 200

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: waf_bypass
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   BypassCategory
   waf_bypass
   
   BypassTechnique
   waf_bypass
   
   BypassTechnique.to_dict
   waf_bypass
   
   WAFBypassEngine
   waf_bypass
   
   WAFBypassEngine.detect_waf
   waf_bypass
   
   WAFBypassEngine.get_bypass_techniques
   waf_bypass
   
   WAFBypassEngine.mutate_payload
   waf_bypass
   
   WAFBypassEngine.encode_ibm037
   waf_bypass
   
   WAFBypassEngine.generate_chunked_body
   waf_bypass
   
   WAFBypassEngine.generate_padding
   waf_bypass
   
   WAFBypassEngine.get_spoof_headers
   waf_bypass
   
   WAFBypassEngine.get_browser_headers
   waf_bypass
   
   WAFBypassEngine.register_technique
   waf_bypass
   
   WAFBypassEngine.register_waf_signature
   waf_bypass
   - 模組: 認知核心模組

---

### Flow 202

- **長度**: 2 步
- **起點**: external_loop_connector
- **終點**: weight_manager
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

2. **程式組件**
   WeightMetadata
   weight_manager
   
   AIWeightManager
   weight_manager
   
   AIWeightManager.__init__
   weight_manager
   
   AIWeightManager.save_model_weights
   weight_manager
   
   AIWeightManager.load_model_weights
   weight_manager
   
   AIWeightManager.list_available_weights
   weight_manager
   
   AIWeightManager._list_model_versions
   weight_manager
   
   AIWeightManager._extract_version_info
   weight_manager
   
   AIWeightManager._list_all_models
   weight_manager
   
   AIWeightManager.delete_weights
   weight_manager
   
   AIWeightManager._find_weight_file
   weight_manager
   
   AIWeightManager._calculate_file_hash
   weight_manager
   
   AIWeightManager._save_metadata
   weight_manager
   
   AIWeightManager._load_and_verify_metadata
   weight_manager
   
   AIWeightManager._verify_model_compatibility
   weight_manager
   
   AIWeightManager._create_backup
   weight_manager
   
   AIWeightManager._cleanup_old_backups
   weight_manager
   
   get_weight_manager
   weight_manager
   
   initialize_weight_manager
   weight_manager
   - 模組: 認知核心模組

---

### Flow 206

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

---

### Flow 207

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: knowledge_base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   VectorStoreProtocol
   knowledge_base
   
   VectorStoreProtocol.add_document
   knowledge_base
   
   VectorStoreProtocol.search
   knowledge_base
   
   VectorStoreProtocol.add_capability_from_registry
   knowledge_base
   
   VectorStoreProtocol.search_by_environment
   knowledge_base
   
   KnowledgeBase
   knowledge_base
   
   KnowledgeBase.__init__
   knowledge_base
   
   KnowledgeBase.search
   knowledge_base
   
   KnowledgeBase.query
   knowledge_base
   
   KnowledgeBase.add_knowledge
   knowledge_base
   
   KnowledgeBase.index_codebase
   knowledge_base
   
   KnowledgeBase.get_stats
   knowledge_base
   - 模組: 認知核心模組

---

### Flow 208

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: rag_engine
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **AI組件**
   KnowledgeType
   rag_engine
   
   QueryCache
   rag_engine
   
   QueryCache.__init__
   rag_engine
   
   QueryCache._make_key
   rag_engine
   
   QueryCache.get
   rag_engine
   
   QueryCache.set
   rag_engine
   
   QueryCache._cleanup
   rag_engine
   
   QueryCache.clear
   rag_engine
   
   QueryCache.stats
   rag_engine
   
   RAGEngine
   rag_engine
   
   RAGEngine.__init__
   rag_engine
   
   RAGEngine._search_with_cache
   rag_engine
   
   RAGEngine.enhance_attack_plan
   rag_engine
   
   RAGEngine.suggest_next_step
   rag_engine
   
   RAGEngine.analyze_failure
   rag_engine
   
   RAGEngine.get_relevant_payloads
   rag_engine
   
   RAGEngine.learn_from_experience
   rag_engine
   
   RAGEngine._extract_successful_pattern
   rag_engine
   
   RAGEngine.retrieve_similar_cases
   rag_engine
   
   RAGEngine.search_capabilities_by_environment
   rag_engine
   
   RAGEngine.load_capabilities_from_registry
   rag_engine
   
   RAGEngine.index_new_experience
   rag_engine
   
   RAGEngine.save_knowledge
   rag_engine
   
   RAGEngine.get_statistics
   rag_engine
   - 模組: 認知核心模組

---

### Flow 215

- **長度**: 2 步
- **起點**: ai_capability_query
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

2. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 236

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: rag_engine
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **AI組件**
   KnowledgeType
   rag_engine
   
   QueryCache
   rag_engine
   
   QueryCache.__init__
   rag_engine
   
   QueryCache._make_key
   rag_engine
   
   QueryCache.get
   rag_engine
   
   QueryCache.set
   rag_engine
   
   QueryCache._cleanup
   rag_engine
   
   QueryCache.clear
   rag_engine
   
   QueryCache.stats
   rag_engine
   
   RAGEngine
   rag_engine
   
   RAGEngine.__init__
   rag_engine
   
   RAGEngine._search_with_cache
   rag_engine
   
   RAGEngine.enhance_attack_plan
   rag_engine
   
   RAGEngine.suggest_next_step
   rag_engine
   
   RAGEngine.analyze_failure
   rag_engine
   
   RAGEngine.get_relevant_payloads
   rag_engine
   
   RAGEngine.learn_from_experience
   rag_engine
   
   RAGEngine._extract_successful_pattern
   rag_engine
   
   RAGEngine.retrieve_similar_cases
   rag_engine
   
   RAGEngine.search_capabilities_by_environment
   rag_engine
   
   RAGEngine.load_capabilities_from_registry
   rag_engine
   
   RAGEngine.index_new_experience
   rag_engine
   
   RAGEngine.save_knowledge
   rag_engine
   
   RAGEngine.get_statistics
   rag_engine
   - 模組: 認知核心模組

---

### Flow 237

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 238

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: internal_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

---

### Flow 239

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: external_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

---

### Flow 240

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: vulnerability_detection
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   VulnerabilityDetector
   vulnerability_detection
   
   VulnerabilityDetector.check_sqli
   vulnerability_detection
   
   VulnerabilityDetector.check_xss
   vulnerability_detection
   
   VulnerabilityDetector.check_ssrf
   vulnerability_detection
   
   VulnerabilityDetector.check_idor
   vulnerability_detection
   
   VulnerabilityDetector._detect_waf
   vulnerability_detection
   
   VulnerabilityDetector._check_sqli_false_positive
   vulnerability_detection
   
   VulnerabilityDetector._build_sqli_recommendations
   vulnerability_detection
   
   VulnerabilityDetector._calculate_response_similarity
   vulnerability_detection
   
   VulnerabilityDetector.get_sqli_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_xss_payloads
   vulnerability_detection
   
   VulnerabilityDetector.get_ssrf_targets
   vulnerability_detection
   - 模組: 認知核心模組

---

### Flow 241

- **長度**: 3 步
- **起點**: enhanced_decision_agent
- **終點**: cve_identification
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   SignalTier
   cve_identification
   
   CVESignature
   cve_identification
   
   CVESignature.to_dict
   cve_identification
   
   CVEMatch
   cve_identification
   
   CVEMatch.to_dict
   cve_identification
   
   CVEMatch.is_exploitable
   cve_identification
   
   CVEIdentifier
   cve_identification
   
   CVEIdentifier.identify
   cve_identification
   
   CVEIdentifier.identify_by_fingerprint
   cve_identification
   
   CVEIdentifier.get_exploit_payloads
   cve_identification
   
   CVEIdentifier._check_single_cve
   cve_identification
   
   CVEIdentifier._build_exploit_recommendations
   cve_identification
   
   CVEIdentifier.register_cve
   cve_identification
   
   CVEIdentifier.get_all_cve_ids
   cve_identification
   
   CVEIdentifier.get_cve_by_severity
   cve_identification
   - 模組: 認知核心模組

3. **程式組件**
   SignalTier
   cve_identification
   
   CVESignature
   cve_identification
   
   CVESignature.to_dict
   cve_identification
   
   CVEMatch
   cve_identification
   
   CVEMatch.to_dict
   cve_identification
   
   CVEMatch.is_exploitable
   cve_identification
   
   CVEIdentifier
   cve_identification
   
   CVEIdentifier.identify
   cve_identification
   
   CVEIdentifier.identify_by_fingerprint
   cve_identification
   
   CVEIdentifier.get_exploit_payloads
   cve_identification
   
   CVEIdentifier._check_single_cve
   cve_identification
   
   CVEIdentifier._build_exploit_recommendations
   cve_identification
   
   CVEIdentifier.register_cve
   cve_identification
   
   CVEIdentifier.get_all_cve_ids
   cve_identification
   
   CVEIdentifier.get_cve_by_severity
   cve_identification
   - 模組: 認知核心模組

---

### Flow 243

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: web_architecture
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   ArchitectureType
   web_architecture
   
   AuthScheme
   web_architecture
   
   ArchitectureFingerprint
   web_architecture
   
   ArchitectureFingerprint.to_dict
   web_architecture
   
   JWTAnalysis
   web_architecture
   
   JWTAnalysis.to_dict
   web_architecture
   
   GraphQLSchema
   web_architecture
   
   GraphQLSchema.to_dict
   web_architecture
   
   WebArchitectureAnalyzer
   web_architecture
   
   WebArchitectureAnalyzer.detect_graphql_introspection
   web_architecture
   
   WebArchitectureAnalyzer.parse_graphql_schema
   web_architecture
   
   WebArchitectureAnalyzer.analyze_jwt
   web_architecture
   
   WebArchitectureAnalyzer.generate_jwt_attack_payloads
   web_architecture
   
   WebArchitectureAnalyzer.check_bola
   web_architecture
   
   WebArchitectureAnalyzer._calculate_response_similarity
   web_architecture
   
   WebArchitectureAnalyzer.check_mass_assignment
   web_architecture
   
   WebArchitectureAnalyzer.check_websocket_security
   web_architecture
   
   WebArchitectureAnalyzer.identify_architecture
   web_architecture
   - 模組: 認知核心模組

---

### Flow 256

- **長度**: 2 步
- **起點**: two_phase_scan_orchestrator
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   TwoPhaseOrchestratorError
   two_phase_scan_orchestrator
   
   Phase0TimeoutError
   two_phase_scan_orchestrator
   
   Phase1TimeoutError
   two_phase_scan_orchestrator
   
   AIDecisionError
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.__init__
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.execute_scan_with_context
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.execute_two_phase_scan
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._execute_phase0
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._execute_phase1
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._fallback_decision_rules
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 核心能力模組

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

### Flow 268

- **長度**: 2 步
- **起點**: real_neural_core
- **終點**: real_neural_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

2. **AI組件**
   RealAICore
   real_neural_core
   
   RealAICore.__init__
   real_neural_core
   
   RealAICore._build_5m_network
   real_neural_core
   
   RealAICore._initialize_weights
   real_neural_core
   
   RealAICore._build_legacy_network
   real_neural_core
   
   RealAICore.forward
   real_neural_core
   
   RealAICore.forward_with_aux
   real_neural_core
   
   RealAICore.save_weights
   real_neural_core
   
   RealAICore.load_weights
   real_neural_core
   
   RealAICore._validate_weight_filepath
   real_neural_core
   
   RealAICore._extract_state_dict
   real_neural_core
   
   RealAICore._try_direct_load
   real_neural_core
   
   RealAICore._apply_key_mapping
   real_neural_core
   
   RealAICore._load_with_partial_match
   real_neural_core
   
   RealAICore._log_weight_info
   real_neural_core
   
   RealDecisionEngine
   real_neural_core
   
   RealDecisionEngine.__init__
   real_neural_core
   
   RealDecisionEngine.encode_input
   real_neural_core
   
   RealDecisionEngine._enhance_bug_bounty_context
   real_neural_core
   
   RealDecisionEngine._extract_bug_bounty_features
   real_neural_core
   
   RealDecisionEngine._extract_attack_intent_features
   real_neural_core
   
   RealDecisionEngine._extract_target_features
   real_neural_core
   
   RealDecisionEngine._extract_tool_features
   real_neural_core
   
   RealDecisionEngine._extract_context_features
   real_neural_core
   
   RealDecisionEngine.generate_decision
   real_neural_core
   
   RealDecisionEngine._prepare_decision_input
   real_neural_core
   
   RealDecisionEngine._calculate_enhanced_confidence
   real_neural_core
   
   RealDecisionEngine._analyze_decision_output
   real_neural_core
   
   RealDecisionEngine._analyze_bug_bounty_decision
   real_neural_core
   
   RealDecisionEngine.decide
   real_neural_core
   
   RealDecisionEngine.train_step
   real_neural_core
   
   RealDecisionEngine._compute_training_loss
   real_neural_core
   
   RealDecisionEngine._compute_dual_output_loss
   real_neural_core
   
   RealDecisionEngine._compute_single_output_loss
   real_neural_core
   
   RealDecisionEngine._perform_backward_pass
   real_neural_core
   
   RealDecisionEngine._update_training_statistics
   real_neural_core
   
   RealDecisionEngine._calculate_gradient_norm
   real_neural_core
   
   RealDecisionEngine.save_model
   real_neural_core
   
   create_real_ai_replacement
   real_neural_core
   - 模組: 認知核心模組

---

### Flow 271

- **長度**: 2 步
- **起點**: ai_decision_core
- **終點**: ai_decision_core
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   UserConstraints
   ai_decision_core
   
   ScanStrategy
   ai_decision_core
   
   AIDecisionCore
   ai_decision_core
   
   AIDecisionCore.__init__
   ai_decision_core
   
   AIDecisionCore.initialize
   ai_decision_core
   
   AIDecisionCore.decide_scan_strategy
   ai_decision_core
   
   AIDecisionCore._filter_capabilities
   ai_decision_core
   
   AIDecisionCore._search_rag_suggestions
   ai_decision_core
   
   AIDecisionCore._generate_strategy
   ai_decision_core
   
   AIDecisionCore.get_flow_execution_order
   ai_decision_core
   
   AIDecisionCore.generate_attack_plan
   ai_decision_core
   
   quick_decision
   ai_decision_core
   - 模組: 認知核心模組

2. **AI組件**
   UserConstraints
   ai_decision_core
   
   ScanStrategy
   ai_decision_core
   
   AIDecisionCore
   ai_decision_core
   
   AIDecisionCore.__init__
   ai_decision_core
   
   AIDecisionCore.initialize
   ai_decision_core
   
   AIDecisionCore.decide_scan_strategy
   ai_decision_core
   
   AIDecisionCore._filter_capabilities
   ai_decision_core
   
   AIDecisionCore._search_rag_suggestions
   ai_decision_core
   
   AIDecisionCore._generate_strategy
   ai_decision_core
   
   AIDecisionCore.get_flow_execution_order
   ai_decision_core
   
   AIDecisionCore.generate_attack_plan
   ai_decision_core
   
   quick_decision
   ai_decision_core
   - 模組: 認知核心模組

---

### Flow 273

- **長度**: 2 步
- **起點**: execution_planner
- **終點**: execution_planner
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionPlanner
   execution_planner
   
   ExecutionPlanner.__init__
   execution_planner
   
   ExecutionPlanner.create_execution_plan
   execution_planner
   
   ExecutionPlanner.execute_plan
   execution_planner
   
   ExecutionPlanner._check_resources
   execution_planner
   
   ExecutionPlanner._execute_step
   execution_planner
   
   ExecutionPlanner._validate_input
   execution_planner
   
   ExecutionPlanner._execute_simple_command
   execution_planner
   
   ExecutionPlanner._format_output
   execution_planner
   
   ExecutionPlanner._execute_ai_task
   execution_planner
   
   ExecutionPlanner._execute_rust_scan
   execution_planner
   
   ExecutionPlanner._generate_report
   execution_planner
   
   ExecutionPlanner._execute_generic_step
   execution_planner
   
   ExecutionPlanner._aggregate_results
   execution_planner
   
   ExecutionPlanner.get_plan_status
   execution_planner
   
   ExecutionPlanner.cancel_plan
   execution_planner
   
   ExecutionPlanner.get_execution_stats
   execution_planner
   
   get_execution_planner
   execution_planner
   
   ScanStrategy
   execution_planner
   
   ExecutionStep
   execution_planner
   
   ExecutionStep.__init__
   execution_planner
   
   ExecutionPlan
   execution_planner
   
   ExecutionPlan.__init__
   execution_planner
   
   NextPhaseDecision
   execution_planner
   
   NextPhaseDecision.__init__
   execution_planner
   
   ExecutionPlanner.generate_plan
   execution_planner
   
   ExecutionPlanner._generate_initial_scan_plan
   execution_planner
   
   ExecutionPlanner._generate_informed_scan_plan
   execution_planner
   
   ExecutionPlanner._build_scan_scope
   execution_planner
   
   ExecutionPlanner._create_typescript_scan_step
   execution_planner
   
   ExecutionPlanner._create_go_scan_step
   execution_planner
   
   ExecutionPlanner._create_python_crawl_step
   execution_planner
   
   ExecutionPlanner.decide_next_phase
   execution_planner
   
   ExecutionPlanner._analyze_rust_result
   execution_planner
   
   ExecutionPlanner._identify_missing_info
   execution_planner
   - 模組: 認知核心模組

2. **程式組件**
   ExecutionPlanner
   execution_planner
   
   ExecutionPlanner.__init__
   execution_planner
   
   ExecutionPlanner.create_execution_plan
   execution_planner
   
   ExecutionPlanner.execute_plan
   execution_planner
   
   ExecutionPlanner._check_resources
   execution_planner
   
   ExecutionPlanner._execute_step
   execution_planner
   
   ExecutionPlanner._validate_input
   execution_planner
   
   ExecutionPlanner._execute_simple_command
   execution_planner
   
   ExecutionPlanner._format_output
   execution_planner
   
   ExecutionPlanner._execute_ai_task
   execution_planner
   
   ExecutionPlanner._execute_rust_scan
   execution_planner
   
   ExecutionPlanner._generate_report
   execution_planner
   
   ExecutionPlanner._execute_generic_step
   execution_planner
   
   ExecutionPlanner._aggregate_results
   execution_planner
   
   ExecutionPlanner.get_plan_status
   execution_planner
   
   ExecutionPlanner.cancel_plan
   execution_planner
   
   ExecutionPlanner.get_execution_stats
   execution_planner
   
   get_execution_planner
   execution_planner
   
   ScanStrategy
   execution_planner
   
   ExecutionStep
   execution_planner
   
   ExecutionStep.__init__
   execution_planner
   
   ExecutionPlan
   execution_planner
   
   ExecutionPlan.__init__
   execution_planner
   
   NextPhaseDecision
   execution_planner
   
   NextPhaseDecision.__init__
   execution_planner
   
   ExecutionPlanner.generate_plan
   execution_planner
   
   ExecutionPlanner._generate_initial_scan_plan
   execution_planner
   
   ExecutionPlanner._generate_informed_scan_plan
   execution_planner
   
   ExecutionPlanner._build_scan_scope
   execution_planner
   
   ExecutionPlanner._create_typescript_scan_step
   execution_planner
   
   ExecutionPlanner._create_go_scan_step
   execution_planner
   
   ExecutionPlanner._create_python_crawl_step
   execution_planner
   
   ExecutionPlanner.decide_next_phase
   execution_planner
   
   ExecutionPlanner._analyze_rust_result
   execution_planner
   
   ExecutionPlanner._identify_missing_info
   execution_planner
   - 模組: 認知核心模組

---

### Flow 275

- **長度**: 2 步
- **起點**: assistant
- **終點**: vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

2. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

---

### Flow 276

- **長度**: 2 步
- **起點**: assistant
- **終點**: knowledge_base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

2. **程式組件**
   VectorStoreProtocol
   knowledge_base
   
   VectorStoreProtocol.add_document
   knowledge_base
   
   VectorStoreProtocol.search
   knowledge_base
   
   VectorStoreProtocol.add_capability_from_registry
   knowledge_base
   
   VectorStoreProtocol.search_by_environment
   knowledge_base
   
   KnowledgeBase
   knowledge_base
   
   KnowledgeBase.__init__
   knowledge_base
   
   KnowledgeBase.search
   knowledge_base
   
   KnowledgeBase.query
   knowledge_base
   
   KnowledgeBase.add_knowledge
   knowledge_base
   
   KnowledgeBase.index_codebase
   knowledge_base
   
   KnowledgeBase.get_stats
   knowledge_base
   - 模組: 認知核心模組

---

### Flow 286

- **長度**: 2 步
- **起點**: cve_identification
- **終點**: base
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   SignalTier
   cve_identification
   
   CVESignature
   cve_identification
   
   CVESignature.to_dict
   cve_identification
   
   CVEMatch
   cve_identification
   
   CVEMatch.to_dict
   cve_identification
   
   CVEMatch.is_exploitable
   cve_identification
   
   CVEIdentifier
   cve_identification
   
   CVEIdentifier.identify
   cve_identification
   
   CVEIdentifier.identify_by_fingerprint
   cve_identification
   
   CVEIdentifier.get_exploit_payloads
   cve_identification
   
   CVEIdentifier._check_single_cve
   cve_identification
   
   CVEIdentifier._build_exploit_recommendations
   cve_identification
   
   CVEIdentifier.register_cve
   cve_identification
   
   CVEIdentifier.get_all_cve_ids
   cve_identification
   
   CVEIdentifier.get_cve_by_severity
   cve_identification
   - 模組: 認知核心模組

2. **程式組件**
   ConfidenceLevel
   base
   
   ConfidenceLevel.to_score
   base
   
   VulnerabilityType
   base
   
   DatabaseType
   base
   
   WAFVendor
   base
   
   CloudProvider
   base
   
   DetectionResult
   base
   
   DetectionResult.to_dict
   base
   
   DetectionResult.should_exploit
   base
   
   AttackContext
   base
   
   ResponseAnalysis
   base
   
   PayloadResult
   base
   
   KnowledgeRegistry
   base
   
   KnowledgeRegistry.__new__
   base
   
   KnowledgeRegistry._initialize
   base
   
   KnowledgeRegistry.register_sqli_fingerprint
   base
   
   KnowledgeRegistry.get_sqli_fingerprints
   base
   
   KnowledgeRegistry.register_xss_payload
   base
   
   KnowledgeRegistry.get_xss_payloads
   base
   
   KnowledgeRegistry.register_waf_signature
   base
   
   KnowledgeRegistry.register_cve_pattern
   base
   
   KnowledgeRegistry.get_cve_pattern
   base
   
   KnowledgeRegistry.register_detector
   base
   
   KnowledgeRegistry.run_custom_detector
   base
   - 模組: 認知核心模組

---

### Flow 291

- **長度**: 2 步
- **起點**: vector_store
- **終點**: aiva_embedding
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

2. **程式組件**
   AIVAEmbedding
   aiva_embedding
   
   AIVAEmbedding.__init__
   aiva_embedding
   
   AIVAEmbedding._mean_pooling
   aiva_embedding
   
   AIVAEmbedding.forward
   aiva_embedding
   
   AIVAEmbedding.encode
   aiva_embedding
   
   AIVAEmbedding.similarity
   aiva_embedding
   
   AIVAEmbedding.save
   aiva_embedding
   
   AIVAEmbedding.load
   aiva_embedding
   
   SentenceTransformer
   aiva_embedding
   - 模組: 認知核心模組

---

### Flow 294

- **長度**: 2 步
- **起點**: unified_vector_store
- **終點**: postgresql_vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   UnifiedVectorStore
   unified_vector_store
   
   UnifiedVectorStore.__init__
   unified_vector_store
   
   UnifiedVectorStore.initialize
   unified_vector_store
   
   UnifiedVectorStore._migrate_from_legacy
   unified_vector_store
   
   UnifiedVectorStore._get_embedding_model
   unified_vector_store
   
   UnifiedVectorStore._simple_embedding
   unified_vector_store
   
   UnifiedVectorStore.add_document
   unified_vector_store
   
   UnifiedVectorStore.add_batch
   unified_vector_store
   
   UnifiedVectorStore.search
   unified_vector_store
   
   UnifiedVectorStore.delete_document
   unified_vector_store
   
   UnifiedVectorStore.get_document
   unified_vector_store
   
   UnifiedVectorStore.get_statistics
   unified_vector_store
   
   UnifiedVectorStore.close
   unified_vector_store
   
   UnifiedVectorStore.add_capability_from_registry
   unified_vector_store
   
   UnifiedVectorStore.search_by_environment
   unified_vector_store
   
   create_unified_vector_store
   unified_vector_store
   - 模組: 認知核心模組

2. **程式組件**
   PostgreSQLVectorStore
   postgresql_vector_store
   
   PostgreSQLVectorStore.__init__
   postgresql_vector_store
   
   PostgreSQLVectorStore.initialize
   postgresql_vector_store
   
   PostgreSQLVectorStore.add_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.search
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.delete_document
   postgresql_vector_store
   
   PostgreSQLVectorStore.get_statistics
   postgresql_vector_store
   
   PostgreSQLVectorStore.execute_unified_query
   postgresql_vector_store
   
   PostgreSQLVectorStore.close
   postgresql_vector_store
   
   demo_postgresql_vector_store
   postgresql_vector_store
   - 模組: 認知核心模組

---

### Flow 313

- **長度**: 2 步
- **起點**: analysis_engine
- **終點**: real_bio_net_adapter
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisType
   analysis_engine
   
   IndexingConfig
   analysis_engine
   
   CacheManager
   analysis_engine
   
   CacheManager.__init__
   analysis_engine
   
   CacheManager._load_cache_index
   analysis_engine
   
   CacheManager.get_file_hash
   analysis_engine
   
   CacheManager.is_cached
   analysis_engine
   
   CacheManager.update_cache
   analysis_engine
   
   CacheManager.save_cache_index
   analysis_engine
   
   AIAnalysisResult
   analysis_engine
   
   CodeChunk
   analysis_engine
   
   CodeChunk.__post_init__
   analysis_engine
   
   AIAnalysisEngine
   analysis_engine
   
   AIAnalysisEngine.__init__
   analysis_engine
   
   AIAnalysisEngine.initialize
   analysis_engine
   
   AIAnalysisEngine._extract_code_features
   analysis_engine
   
   AIAnalysisEngine._calculate_cyclomatic_complexity
   analysis_engine
   
   AIAnalysisEngine._calculate_nesting_depth
   analysis_engine
   
   AIAnalysisEngine._extract_security_features
   analysis_engine
   
   AIAnalysisEngine._extract_semantic_features
   analysis_engine
   
   AIAnalysisEngine.analyze_code
   analysis_engine
   
   AIAnalysisEngine.index_codebase
   analysis_engine
   
   AIAnalysisEngine._collect_python_files
   analysis_engine
   
   AIAnalysisEngine._filter_files_for_indexing
   analysis_engine
   
   AIAnalysisEngine._batch_index_files
   analysis_engine
   
   AIAnalysisEngine._process_file_batch
   analysis_engine
   
   AIAnalysisEngine._safe_index_file
   analysis_engine
   
   AIAnalysisEngine._index_file_content
   analysis_engine
   
   AIAnalysisEngine._extract_chunks_from_ast
   analysis_engine
   
   AIAnalysisEngine._extract_node_content
   analysis_engine
   
   AIAnalysisEngine._extract_by_line_numbers
   analysis_engine
   
   AIAnalysisEngine._handle_unparseable_file
   analysis_engine
   
   AIAnalysisEngine._add_code_chunk
   analysis_engine
   
   AIAnalysisEngine._extract_analysis_keywords
   analysis_engine
   
   AIAnalysisEngine.search_code_chunks
   analysis_engine
   
   AIAnalysisEngine._extract_query_keywords
   analysis_engine
   
   AIAnalysisEngine._calculate_chunk_scores
   analysis_engine
   
   AIAnalysisEngine._apply_exact_matches
   analysis_engine
   
   AIAnalysisEngine._apply_partial_matches
   analysis_engine
   
   AIAnalysisEngine._format_search_results
   analysis_engine
   
   AIAnalysisEngine._get_indexing_stats
   analysis_engine
   
   AIAnalysisEngine._create_failed_results
   analysis_engine
   
   AIAnalysisEngine._perform_ai_analysis
   analysis_engine
   
   AIAnalysisEngine._generate_findings
   analysis_engine
   
   AIAnalysisEngine._generate_recommendations
   analysis_engine
   
   AIAnalysisEngine._calculate_risk_level
   analysis_engine
   
   AIAnalysisEngine._generate_explanation
   analysis_engine
   
   AIAnalysisEngine.get_analysis_summary
   analysis_engine
   - 模組: 核心能力模組

2. **程式組件**
   RealScalableBioNet
   real_bio_net_adapter
   
   RealScalableBioNet.__init__
   real_bio_net_adapter
   
   RealScalableBioNet._load_or_initialize_weights
   real_bio_net_adapter
   
   RealScalableBioNet.forward
   real_bio_net_adapter
   
   RealScalableBioNet._softmax
   real_bio_net_adapter
   
   RealScalableBioNet.save_weights
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent.__init__
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent.generate
   real_bio_net_adapter
   
   RealBioNeuronRAGAgent._create_real_input_vector
   real_bio_net_adapter
   
   create_real_scalable_bionet
   real_bio_net_adapter
   
   create_real_rag_agent
   real_bio_net_adapter
   - 模組: 認知核心模組

---

### Flow 322

- **長度**: 2 步
- **起點**: ai_menu
- **終點**: ai_capability_query
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **AI組件**
   AIVAIntelligentMenu
   ai_menu
   
   AIVAIntelligentMenu.__init__
   ai_menu
   
   AIVAIntelligentMenu.print_banner
   ai_menu
   
   AIVAIntelligentMenu.show_main_menu
   ai_menu
   
   AIVAIntelligentMenu.handle_ai_conversation
   ai_menu
   
   AIVAIntelligentMenu.handle_capability_search
   ai_menu
   
   AIVAIntelligentMenu.handle_one_click_attack
   ai_menu
   
   AIVAIntelligentMenu.handle_workflow_recommendation
   ai_menu
   
   AIVAIntelligentMenu.handle_system_stats
   ai_menu
   
   AIVAIntelligentMenu.handle_health_check
   ai_menu
   
   AIVAIntelligentMenu.handle_sync_rag
   ai_menu
   
   AIVAIntelligentMenu._execute_capability
   ai_menu
   
   AIVAIntelligentMenu._execute_workflow
   ai_menu
   
   AIVAIntelligentMenu.run
   ai_menu
   
   AIVAIntelligentMenu.show_help
   ai_menu
   
   main
   ai_menu
   - 模組: 核心能力模組

2. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

---

### Flow 330

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: execution_planner
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   ExecutionPlanner
   execution_planner
   
   ExecutionPlanner.__init__
   execution_planner
   
   ExecutionPlanner.create_execution_plan
   execution_planner
   
   ExecutionPlanner.execute_plan
   execution_planner
   
   ExecutionPlanner._check_resources
   execution_planner
   
   ExecutionPlanner._execute_step
   execution_planner
   
   ExecutionPlanner._validate_input
   execution_planner
   
   ExecutionPlanner._execute_simple_command
   execution_planner
   
   ExecutionPlanner._format_output
   execution_planner
   
   ExecutionPlanner._execute_ai_task
   execution_planner
   
   ExecutionPlanner._execute_rust_scan
   execution_planner
   
   ExecutionPlanner._generate_report
   execution_planner
   
   ExecutionPlanner._execute_generic_step
   execution_planner
   
   ExecutionPlanner._aggregate_results
   execution_planner
   
   ExecutionPlanner.get_plan_status
   execution_planner
   
   ExecutionPlanner.cancel_plan
   execution_planner
   
   ExecutionPlanner.get_execution_stats
   execution_planner
   
   get_execution_planner
   execution_planner
   
   ScanStrategy
   execution_planner
   
   ExecutionStep
   execution_planner
   
   ExecutionStep.__init__
   execution_planner
   
   ExecutionPlan
   execution_planner
   
   ExecutionPlan.__init__
   execution_planner
   
   NextPhaseDecision
   execution_planner
   
   NextPhaseDecision.__init__
   execution_planner
   
   ExecutionPlanner.generate_plan
   execution_planner
   
   ExecutionPlanner._generate_initial_scan_plan
   execution_planner
   
   ExecutionPlanner._generate_informed_scan_plan
   execution_planner
   
   ExecutionPlanner._build_scan_scope
   execution_planner
   
   ExecutionPlanner._create_typescript_scan_step
   execution_planner
   
   ExecutionPlanner._create_go_scan_step
   execution_planner
   
   ExecutionPlanner._create_python_crawl_step
   execution_planner
   
   ExecutionPlanner.decide_next_phase
   execution_planner
   
   ExecutionPlanner._analyze_rust_result
   execution_planner
   
   ExecutionPlanner._identify_missing_info
   execution_planner
   - 模組: 認知核心模組

---

### Flow 354

- **長度**: 2 步
- **起點**: ai_capability_query
- **終點**: internal_loop_connector
- **主要模組**: 認知核心模組
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

2. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

---

### Flow 364

- **長度**: 2 步
- **起點**: rag_engine
- **終點**: rag_engine
- **主要模組**: 認知核心模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   KnowledgeType
   rag_engine
   
   QueryCache
   rag_engine
   
   QueryCache.__init__
   rag_engine
   
   QueryCache._make_key
   rag_engine
   
   QueryCache.get
   rag_engine
   
   QueryCache.set
   rag_engine
   
   QueryCache._cleanup
   rag_engine
   
   QueryCache.clear
   rag_engine
   
   QueryCache.stats
   rag_engine
   
   RAGEngine
   rag_engine
   
   RAGEngine.__init__
   rag_engine
   
   RAGEngine._search_with_cache
   rag_engine
   
   RAGEngine.enhance_attack_plan
   rag_engine
   
   RAGEngine.suggest_next_step
   rag_engine
   
   RAGEngine.analyze_failure
   rag_engine
   
   RAGEngine.get_relevant_payloads
   rag_engine
   
   RAGEngine.learn_from_experience
   rag_engine
   
   RAGEngine._extract_successful_pattern
   rag_engine
   
   RAGEngine.retrieve_similar_cases
   rag_engine
   
   RAGEngine.search_capabilities_by_environment
   rag_engine
   
   RAGEngine.load_capabilities_from_registry
   rag_engine
   
   RAGEngine.index_new_experience
   rag_engine
   
   RAGEngine.save_knowledge
   rag_engine
   
   RAGEngine.get_statistics
   rag_engine
   - 模組: 認知核心模組

2. **AI組件**
   KnowledgeType
   rag_engine
   
   QueryCache
   rag_engine
   
   QueryCache.__init__
   rag_engine
   
   QueryCache._make_key
   rag_engine
   
   QueryCache.get
   rag_engine
   
   QueryCache.set
   rag_engine
   
   QueryCache._cleanup
   rag_engine
   
   QueryCache.clear
   rag_engine
   
   QueryCache.stats
   rag_engine
   
   RAGEngine
   rag_engine
   
   RAGEngine.__init__
   rag_engine
   
   RAGEngine._search_with_cache
   rag_engine
   
   RAGEngine.enhance_attack_plan
   rag_engine
   
   RAGEngine.suggest_next_step
   rag_engine
   
   RAGEngine.analyze_failure
   rag_engine
   
   RAGEngine.get_relevant_payloads
   rag_engine
   
   RAGEngine.learn_from_experience
   rag_engine
   
   RAGEngine._extract_successful_pattern
   rag_engine
   
   RAGEngine.retrieve_similar_cases
   rag_engine
   
   RAGEngine.search_capabilities_by_environment
   rag_engine
   
   RAGEngine.load_capabilities_from_registry
   rag_engine
   
   RAGEngine.index_new_experience
   rag_engine
   
   RAGEngine.save_knowledge
   rag_engine
   
   RAGEngine.get_statistics
   rag_engine
   - 模組: 認知核心模組

---

### Flow 376

- **長度**: 3 步
- **起點**: core_service_coordinator
- **終點**: execution_planner
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

2. **程式組件**
   ExecutionPlanner
   execution_planner
   
   ExecutionPlanner.__init__
   execution_planner
   
   ExecutionPlanner.create_execution_plan
   execution_planner
   
   ExecutionPlanner.execute_plan
   execution_planner
   
   ExecutionPlanner._check_resources
   execution_planner
   
   ExecutionPlanner._execute_step
   execution_planner
   
   ExecutionPlanner._validate_input
   execution_planner
   
   ExecutionPlanner._execute_simple_command
   execution_planner
   
   ExecutionPlanner._format_output
   execution_planner
   
   ExecutionPlanner._execute_ai_task
   execution_planner
   
   ExecutionPlanner._execute_rust_scan
   execution_planner
   
   ExecutionPlanner._generate_report
   execution_planner
   
   ExecutionPlanner._execute_generic_step
   execution_planner
   
   ExecutionPlanner._aggregate_results
   execution_planner
   
   ExecutionPlanner.get_plan_status
   execution_planner
   
   ExecutionPlanner.cancel_plan
   execution_planner
   
   ExecutionPlanner.get_execution_stats
   execution_planner
   
   get_execution_planner
   execution_planner
   
   ScanStrategy
   execution_planner
   
   ExecutionStep
   execution_planner
   
   ExecutionStep.__init__
   execution_planner
   
   ExecutionPlan
   execution_planner
   
   ExecutionPlan.__init__
   execution_planner
   
   NextPhaseDecision
   execution_planner
   
   NextPhaseDecision.__init__
   execution_planner
   
   ExecutionPlanner.generate_plan
   execution_planner
   
   ExecutionPlanner._generate_initial_scan_plan
   execution_planner
   
   ExecutionPlanner._generate_informed_scan_plan
   execution_planner
   
   ExecutionPlanner._build_scan_scope
   execution_planner
   
   ExecutionPlanner._create_typescript_scan_step
   execution_planner
   
   ExecutionPlanner._create_go_scan_step
   execution_planner
   
   ExecutionPlanner._create_python_crawl_step
   execution_planner
   
   ExecutionPlanner.decide_next_phase
   execution_planner
   
   ExecutionPlanner._analyze_rust_result
   execution_planner
   
   ExecutionPlanner._identify_missing_info
   execution_planner
   - 模組: 任務規劃模組

3. **程式組件**
   ExecutionPlanner
   execution_planner
   
   ExecutionPlanner.__init__
   execution_planner
   
   ExecutionPlanner.create_execution_plan
   execution_planner
   
   ExecutionPlanner.execute_plan
   execution_planner
   
   ExecutionPlanner._check_resources
   execution_planner
   
   ExecutionPlanner._execute_step
   execution_planner
   
   ExecutionPlanner._validate_input
   execution_planner
   
   ExecutionPlanner._execute_simple_command
   execution_planner
   
   ExecutionPlanner._format_output
   execution_planner
   
   ExecutionPlanner._execute_ai_task
   execution_planner
   
   ExecutionPlanner._execute_rust_scan
   execution_planner
   
   ExecutionPlanner._generate_report
   execution_planner
   
   ExecutionPlanner._execute_generic_step
   execution_planner
   
   ExecutionPlanner._aggregate_results
   execution_planner
   
   ExecutionPlanner.get_plan_status
   execution_planner
   
   ExecutionPlanner.cancel_plan
   execution_planner
   
   ExecutionPlanner.get_execution_stats
   execution_planner
   
   get_execution_planner
   execution_planner
   
   ScanStrategy
   execution_planner
   
   ExecutionStep
   execution_planner
   
   ExecutionStep.__init__
   execution_planner
   
   ExecutionPlan
   execution_planner
   
   ExecutionPlan.__init__
   execution_planner
   
   NextPhaseDecision
   execution_planner
   
   NextPhaseDecision.__init__
   execution_planner
   
   ExecutionPlanner.generate_plan
   execution_planner
   
   ExecutionPlanner._generate_initial_scan_plan
   execution_planner
   
   ExecutionPlanner._generate_informed_scan_plan
   execution_planner
   
   ExecutionPlanner._build_scan_scope
   execution_planner
   
   ExecutionPlanner._create_typescript_scan_step
   execution_planner
   
   ExecutionPlanner._create_go_scan_step
   execution_planner
   
   ExecutionPlanner._create_python_crawl_step
   execution_planner
   
   ExecutionPlanner.decide_next_phase
   execution_planner
   
   ExecutionPlanner._analyze_rust_result
   execution_planner
   
   ExecutionPlanner._identify_missing_info
   execution_planner
   - 模組: 認知核心模組

---

### Flow 389

- **長度**: 2 步
- **起點**: execution_orchestrator
- **終點**: execution_orchestrator
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   execution_orchestrator
   
   ExecutionResult.__init__
   execution_orchestrator
   
   ExecutionOrchestrator
   execution_orchestrator
   
   ExecutionOrchestrator.__init__
   execution_orchestrator
   
   ExecutionOrchestrator.execute_plan
   execution_orchestrator
   
   ExecutionOrchestrator._build_cli_command
   execution_orchestrator
   
   ExecutionOrchestrator._check_dependencies
   execution_orchestrator
   
   ExecutionOrchestrator.get_execution_status
   execution_orchestrator
   
   ExecutionOrchestrator.list_active_executions
   execution_orchestrator
   - 模組: 認知核心模組

2. **程式組件**
   ExecutionResult
   execution_orchestrator
   
   ExecutionResult.__init__
   execution_orchestrator
   
   ExecutionOrchestrator
   execution_orchestrator
   
   ExecutionOrchestrator.__init__
   execution_orchestrator
   
   ExecutionOrchestrator.execute_plan
   execution_orchestrator
   
   ExecutionOrchestrator._build_cli_command
   execution_orchestrator
   
   ExecutionOrchestrator._check_dependencies
   execution_orchestrator
   
   ExecutionOrchestrator.get_execution_status
   execution_orchestrator
   
   ExecutionOrchestrator.list_active_executions
   execution_orchestrator
   - 模組: 認知核心模組

---

### Flow 393

- **長度**: 2 步
- **起點**: ai_capability_query
- **終點**: vector_store
- **主要模組**: 認知核心模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   AICapabilityQuery
   ai_capability_query
   
   AICapabilityQuery.__init__
   ai_capability_query
   
   AICapabilityQuery.vector_store
   ai_capability_query
   
   AICapabilityQuery.kb
   ai_capability_query
   
   AICapabilityQuery.connector
   ai_capability_query
   
   AICapabilityQuery.query
   ai_capability_query
   
   AICapabilityQuery.display_results
   ai_capability_query
   
   AICapabilityQuery._display_results_rich
   ai_capability_query
   
   AICapabilityQuery._display_results_plain
   ai_capability_query
   
   AICapabilityQuery.show_statistics
   ai_capability_query
   
   AICapabilityQuery._display_statistics_rich
   ai_capability_query
   
   AICapabilityQuery._display_statistics_plain
   ai_capability_query
   
   AICapabilityQuery.get_workflow_recommendation
   ai_capability_query
   
   AICapabilityQuery.query_by_module
   ai_capability_query
   
   AICapabilityQuery.query_by_language
   ai_capability_query
   
   AICapabilityQuery.query_with_filters
   ai_capability_query
   
   AICapabilityQuery.get_classification_report
   ai_capability_query
   
   AICapabilityQuery._empty_classification_report
   ai_capability_query
   
   AICapabilityQuery.display_classification_report
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_rich
   ai_capability_query
   
   AICapabilityQuery._display_classification_report_plain
   ai_capability_query
   
   AICapabilityQuery.save_classification_report
   ai_capability_query
   
   quick_query
   ai_capability_query
   
   quick_stats
   ai_capability_query
   - 模組: 認知核心模組

2. **程式組件**
   VectorStore
   vector_store
   
   VectorStore.__init__
   vector_store
   
   VectorStore._initialize_backend
   vector_store
   
   VectorStore._get_embedding_model
   vector_store
   
   VectorStore._simple_embedding
   vector_store
   
   VectorStore._encode_rag_trigger
   vector_store
   
   VectorStore.add_capability_from_registry
   vector_store
   
   VectorStore.search_by_environment
   vector_store
   
   VectorStore.add_capability
   vector_store
   
   VectorStore.add_capabilities_batch
   vector_store
   
   VectorStore.search_capabilities
   vector_store
   
   VectorStore.add_document
   vector_store
   
   VectorStore.add_batch
   vector_store
   
   VectorStore.search
   vector_store
   
   VectorStore.delete_document
   vector_store
   
   VectorStore.get_document
   vector_store
   
   VectorStore.save
   vector_store
   
   VectorStore.load
   vector_store
   
   VectorStore.count
   vector_store
   
   VectorStore.get_statistics
   vector_store
   - 模組: 認知核心模組

---

### Flow 398

- **長度**: 2 步
- **起點**: attack_coordinator
- **終點**: enhanced_decision_agent
- **主要模組**: 認知核心模組
- **主要組件類型**: AI對外能力

**執行路徑**:

1. **程式組件**
   AttackCoordinator
   attack_coordinator
   
   AttackCoordinator.__init__
   attack_coordinator
   
   AttackCoordinator._init_cli_executor
   attack_coordinator
   
   AttackCoordinator._execute_cli_command
   attack_coordinator
   
   AttackCoordinator.detect_vulnerabilities
   attack_coordinator
   
   AttackCoordinator.coordinate_multilang
   attack_coordinator
   
   AttackCoordinator.execute_attack
   attack_coordinator
   
   AttackCoordinator.execute_two_phase_scan
   attack_coordinator
   
   AttackCoordinator.query_capabilities
   attack_coordinator
   
   AttackCoordinator.unified_attack
   attack_coordinator
   
   AttackCoordinator.process_scan_command
   attack_coordinator
   - 模組: 任務規劃模組

2. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

---

## 內探模組 (internal_exploration)

包含 43 條數據流

### Flow 28

- **長度**: 2 步
- **起點**: enhanced_classifier_processor
- **終點**: enhanced_classifier_processor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ProcessingStage
   enhanced_classifier_processor
   
   ProcessingCheckpoint
   enhanced_classifier_processor
   
   EnhancedCapabilityDescription
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.__init__
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._init_llm_templates
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.load_from_classifier_data
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.check_recovery_point
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.save_checkpoint
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_data_hash
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_1_raw_analysis
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_module_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_component_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_ai_indicators
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_2_semantic_enhancement
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_semantic_name
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._categorize_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_complexity
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_interaction_pattern
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_related_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_semantic_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._cluster_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_3_llm_augmentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_purpose
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_scenarios
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_risk_assessment
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_examples
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_llm_insights
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_4_standardization
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._group_by_module_priority
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_capability_scope
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_ai_readable_group_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_usage_context
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_expected_outcome
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_selection_criteria
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_common_path_steps
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._analyze_path_differences
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._recommend_optimal_path
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_standardized_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_usage_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_enhanced_cli_integration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_friendly_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_input_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_output_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._identify_side_effects
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_duration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_dependencies
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._map_ai_capability_level
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_quality_score
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_capability_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_cli_integration_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_markdown_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_5_completion
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.run_full_pipeline
   enhanced_classifier_processor
   
   main
   enhanced_classifier_processor
   - 模組: 內探模組

2. **程式組件**
   ProcessingStage
   enhanced_classifier_processor
   
   ProcessingCheckpoint
   enhanced_classifier_processor
   
   EnhancedCapabilityDescription
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.__init__
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._init_llm_templates
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.load_from_classifier_data
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.check_recovery_point
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.save_checkpoint
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_data_hash
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_1_raw_analysis
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_module_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_component_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_ai_indicators
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_2_semantic_enhancement
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_semantic_name
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._categorize_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_complexity
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_interaction_pattern
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_related_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_semantic_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._cluster_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_3_llm_augmentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_purpose
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_scenarios
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_risk_assessment
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_examples
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_llm_insights
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_4_standardization
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._group_by_module_priority
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_capability_scope
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_ai_readable_group_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_usage_context
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_expected_outcome
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_selection_criteria
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_common_path_steps
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._analyze_path_differences
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._recommend_optimal_path
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_standardized_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_usage_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_enhanced_cli_integration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_friendly_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_input_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_output_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._identify_side_effects
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_duration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_dependencies
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._map_ai_capability_level
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_quality_score
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_capability_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_cli_integration_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_markdown_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_5_completion
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.run_full_pipeline
   enhanced_classifier_processor
   
   main
   enhanced_classifier_processor
   - 模組: 內探模組

---

### Flow 30

- **長度**: 2 步
- **起點**: analyze_dataflow_breakpoints
- **終點**: analyze_dataflow_breakpoints
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   BreakpointIssue
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.__init__
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer._load_analysis_results
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.build_flow_graph
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_breakpoints
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_dead_ends
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_isolated_islands
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_bottlenecks
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_circular_dependencies
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.analyze_missing_connections
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.generate_report
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.run_full_analysis
   analyze_dataflow_breakpoints
   
   main
   analyze_dataflow_breakpoints
   - 模組: 內探模組

2. **程式組件**
   BreakpointIssue
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.__init__
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer._load_analysis_results
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.build_flow_graph
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_breakpoints
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_dead_ends
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_isolated_islands
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_bottlenecks
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_circular_dependencies
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.analyze_missing_connections
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.generate_report
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.run_full_analysis
   analyze_dataflow_breakpoints
   
   main
   analyze_dataflow_breakpoints
   - 模組: 內探模組

---

### Flow 32

- **長度**: 2 步
- **起點**: system_self_explorer
- **終點**: system_self_explorer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   SystemCapability
   system_self_explorer
   
   EngineInfo
   system_self_explorer
   
   SystemSelfExplorer
   system_self_explorer
   
   SystemSelfExplorer.__init__
   system_self_explorer
   
   SystemSelfExplorer.initialize
   system_self_explorer
   
   SystemSelfExplorer._load_flows_data
   system_self_explorer
   
   SystemSelfExplorer._parse_capabilities
   system_self_explorer
   
   SystemSelfExplorer._parse_engines
   system_self_explorer
   
   SystemSelfExplorer._get_capabilities_for_flows
   system_self_explorer
   
   SystemSelfExplorer.get_available_attacks
   system_self_explorer
   
   SystemSelfExplorer.get_available_engines
   system_self_explorer
   
   SystemSelfExplorer.get_capability_by_type
   system_self_explorer
   
   SystemSelfExplorer.check_capability_available
   system_self_explorer
   
   SystemSelfExplorer.get_all_capabilities
   system_self_explorer
   
   SystemSelfExplorer.get_system_summary
   system_self_explorer
   
   quick_explore
   system_self_explorer
   - 模組: 內探模組

2. **程式組件**
   SystemCapability
   system_self_explorer
   
   EngineInfo
   system_self_explorer
   
   SystemSelfExplorer
   system_self_explorer
   
   SystemSelfExplorer.__init__
   system_self_explorer
   
   SystemSelfExplorer.initialize
   system_self_explorer
   
   SystemSelfExplorer._load_flows_data
   system_self_explorer
   
   SystemSelfExplorer._parse_capabilities
   system_self_explorer
   
   SystemSelfExplorer._parse_engines
   system_self_explorer
   
   SystemSelfExplorer._get_capabilities_for_flows
   system_self_explorer
   
   SystemSelfExplorer.get_available_attacks
   system_self_explorer
   
   SystemSelfExplorer.get_available_engines
   system_self_explorer
   
   SystemSelfExplorer.get_capability_by_type
   system_self_explorer
   
   SystemSelfExplorer.check_capability_available
   system_self_explorer
   
   SystemSelfExplorer.get_all_capabilities
   system_self_explorer
   
   SystemSelfExplorer.get_system_summary
   system_self_explorer
   
   quick_explore
   system_self_explorer
   - 模組: 內探模組

---

### Flow 44

- **長度**: 2 步
- **起點**: run_analysis
- **終點**: analyze_dataflow_breakpoints
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

2. **程式組件**
   BreakpointIssue
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.__init__
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer._load_analysis_results
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.build_flow_graph
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_breakpoints
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_dead_ends
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_isolated_islands
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_bottlenecks
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_circular_dependencies
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.analyze_missing_connections
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.generate_report
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.run_full_analysis
   analyze_dataflow_breakpoints
   
   main
   analyze_dataflow_breakpoints
   - 模組: 內探模組

---

### Flow 65

- **長度**: 2 步
- **起點**: ai_executor_interface
- **終點**: unified_executor_controller
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   SimpleExecutionResult
   ai_executor_interface
   
   AIExecutorInterface
   ai_executor_interface
   
   AIExecutorInterface.__init__
   ai_executor_interface
   
   AIExecutorInterface._ensure_initialized
   ai_executor_interface
   
   AIExecutorInterface.execute
   ai_executor_interface
   
   AIExecutorInterface.execute_batch
   ai_executor_interface
   
   AIExecutorInterface.get_available_capabilities
   ai_executor_interface
   
   AIExecutorInterface.get_execution_status
   ai_executor_interface
   
   AIExecutorInterface.get_execution_history
   ai_executor_interface
   
   AIExecutorInterface.clear_history
   ai_executor_interface
   
   get_executor
   ai_executor_interface
   
   quick_execute
   ai_executor_interface
   
   list_capabilities
   ai_executor_interface
   - 模組: 任務規劃模組

2. **程式組件**
   ExecutionResult
   unified_executor_controller
   
   UnifiedExecutorController
   unified_executor_controller
   
   UnifiedExecutorController.__init__
   unified_executor_controller
   
   UnifiedExecutorController.initialize
   unified_executor_controller
   
   UnifiedExecutorController._load_capability_mappings
   unified_executor_controller
   
   UnifiedExecutorController.execute_capability
   unified_executor_controller
   
   UnifiedExecutorController._execute_internal
   unified_executor_controller
   
   UnifiedExecutorController._execute_external
   unified_executor_controller
   
   UnifiedExecutorController.list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController.show_menu
   unified_executor_controller
   
   UnifiedExecutorController._menu_external_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_internal_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_capability_details
   unified_executor_controller
   
   main
   unified_executor_controller
   - 模組: 內探模組

---

### Flow 67

- **長度**: 2 步
- **起點**: enhanced_classifier_processor
- **終點**: aiva_internal_classifier
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ProcessingStage
   enhanced_classifier_processor
   
   ProcessingCheckpoint
   enhanced_classifier_processor
   
   EnhancedCapabilityDescription
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.__init__
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._init_llm_templates
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.load_from_classifier_data
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.check_recovery_point
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.save_checkpoint
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_data_hash
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_1_raw_analysis
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_module_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_component_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_ai_indicators
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_2_semantic_enhancement
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_semantic_name
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._categorize_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_complexity
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_interaction_pattern
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_related_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_semantic_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._cluster_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_3_llm_augmentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_purpose
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_scenarios
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_risk_assessment
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_examples
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_llm_insights
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_4_standardization
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._group_by_module_priority
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_capability_scope
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_ai_readable_group_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_usage_context
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_expected_outcome
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_selection_criteria
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_common_path_steps
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._analyze_path_differences
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._recommend_optimal_path
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_standardized_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_usage_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_enhanced_cli_integration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_friendly_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_input_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_output_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._identify_side_effects
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_duration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_dependencies
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._map_ai_capability_level
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_quality_score
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_capability_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_cli_integration_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_markdown_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_5_completion
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.run_full_pipeline
   enhanced_classifier_processor
   
   main
   enhanced_classifier_processor
   - 模組: 內探模組

2. **程式組件**
   AIVAFlowClassifier
   aiva_internal_classifier
   
   AIVAFlowClassifier.__init__
   aiva_internal_classifier
   
   AIVAFlowClassifier.load_module_config
   aiva_internal_classifier
   
   AIVAFlowClassifier.get_default_config
   aiva_internal_classifier
   
   AIVAFlowClassifier.load_flow_data
   aiva_internal_classifier
   
   AIVAFlowClassifier._extract_script_name
   aiva_internal_classifier
   
   AIVAFlowClassifier._get_script_description
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_script_by_name
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_module_from_path
   aiva_internal_classifier
   
   AIVAFlowClassifier._merge_duplicate_flows
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_ai_capability_type
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_module
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_component_type
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_loop_type
   aiva_internal_classifier
   
   AIVAFlowClassifier.classify_flows
   aiva_internal_classifier
   
   AIVAFlowClassifier.analyze_multi_path_endpoints
   aiva_internal_classifier
   
   AIVAFlowClassifier.generate_reports
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_classification_report
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_complete_flow_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_flow_details_header
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_flows_by_module
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_module_section
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_single_flow_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_classification_step
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_function_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_multi_path_report
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_multi_path_header
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_single_endpoint_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_endpoint_summary
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_path_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_path_difference_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_script_comparison
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_usage_scenario_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_json_export
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_cli_command
   aiva_internal_classifier
   
   AIVAFlowClassifier._get_endpoint_function_info
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_structured_tags
   aiva_internal_classifier
   
   AIVAFlowClassifier.run
   aiva_internal_classifier
   
   main
   aiva_internal_classifier
   - 模組: 內探模組

---

### Flow 68

- **長度**: 2 步
- **起點**: practical_analyzer
- **終點**: practical_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   Issue
   practical_analyzer
   
   PracticalAnalyzer
   practical_analyzer
   
   PracticalAnalyzer.__init__
   practical_analyzer
   
   PracticalAnalyzer.analyze_report
   practical_analyzer
   
   PracticalAnalyzer._parse_definition_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_call_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_potential_missing
   practical_analyzer
   
   PracticalAnalyzer._deduplicate_issues
   practical_analyzer
   
   PracticalAnalyzer._classify_issue
   practical_analyzer
   
   PracticalAnalyzer._write_statistics
   practical_analyzer
   
   PracticalAnalyzer._write_critical_issues
   practical_analyzer
   
   PracticalAnalyzer._write_high_issues
   practical_analyzer
   
   PracticalAnalyzer._write_medium_low_issues
   practical_analyzer
   
   PracticalAnalyzer._write_action_plan
   practical_analyzer
   
   PracticalAnalyzer._write_quick_start_guide
   practical_analyzer
   
   PracticalAnalyzer.generate_practical_report
   practical_analyzer
   
   PracticalAnalyzer.generate_quick_fix_list
   practical_analyzer
   
   PracticalAnalyzer._get_timestamp
   practical_analyzer
   
   main
   practical_analyzer
   - 模組: 內探模組

2. **程式組件**
   Issue
   practical_analyzer
   
   PracticalAnalyzer
   practical_analyzer
   
   PracticalAnalyzer.__init__
   practical_analyzer
   
   PracticalAnalyzer.analyze_report
   practical_analyzer
   
   PracticalAnalyzer._parse_definition_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_call_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_potential_missing
   practical_analyzer
   
   PracticalAnalyzer._deduplicate_issues
   practical_analyzer
   
   PracticalAnalyzer._classify_issue
   practical_analyzer
   
   PracticalAnalyzer._write_statistics
   practical_analyzer
   
   PracticalAnalyzer._write_critical_issues
   practical_analyzer
   
   PracticalAnalyzer._write_high_issues
   practical_analyzer
   
   PracticalAnalyzer._write_medium_low_issues
   practical_analyzer
   
   PracticalAnalyzer._write_action_plan
   practical_analyzer
   
   PracticalAnalyzer._write_quick_start_guide
   practical_analyzer
   
   PracticalAnalyzer.generate_practical_report
   practical_analyzer
   
   PracticalAnalyzer.generate_quick_fix_list
   practical_analyzer
   
   PracticalAnalyzer._get_timestamp
   practical_analyzer
   
   main
   practical_analyzer
   - 模組: 內探模組

---

### Flow 69

- **長度**: 2 步
- **起點**: enhanced_capability_integrator
- **終點**: enhanced_capability_integrator
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   CapabilityReference
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.__init__
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.load_capabilities_data
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator._build_indexes
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capability_by_flow_id
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capabilities_by_module
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.search_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_recommended_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.generate_usage_report
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.export_for_cli_integration
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.create_enhanced_executor_integration
   enhanced_capability_integrator
   
   main
   enhanced_capability_integrator
   - 模組: 內探模組

2. **程式組件**
   CapabilityReference
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.__init__
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.load_capabilities_data
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator._build_indexes
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capability_by_flow_id
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capabilities_by_module
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.search_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_recommended_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.generate_usage_report
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.export_for_cli_integration
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.create_enhanced_executor_integration
   enhanced_capability_integrator
   
   main
   enhanced_capability_integrator
   - 模組: 內探模組

---

### Flow 70

- **長度**: 2 步
- **起點**: run_analysis
- **終點**: analyze_connection_recommendations
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

2. **程式組件**
   FunctionInfo
   analyze_connection_recommendations
   
   ConnectionRecommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.__init__
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.extract_all_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._should_skip_file
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_from_ast
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._parse_function
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_imports
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.analyze_missing_connections
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_orphaned_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_potential_callers
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_connection_confidence
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._are_files_related
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_name_similarity
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_semantic_match
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_parameter_compatibility
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_call_chain_logic
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_docstring_hints
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._evaluate_impact
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._suggest_location
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._generate_code_example
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._impact_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._confidence_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.generate_report
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._write_recommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._get_timestamp
   analyze_connection_recommendations
   
   main
   analyze_connection_recommendations
   - 模組: 內探模組

---

### Flow 73

- **長度**: 2 步
- **起點**: run_analysis
- **終點**: core_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

2. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

---

### Flow 76

- **長度**: 2 步
- **起點**: analyze_results
- **終點**: analyze_results
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   load_analysis_data
   analyze_results
   
   print_basic_statistics
   analyze_results
   
   build_connection_graph
   analyze_results
   
   calculate_connection_stats
   analyze_results
   
   classify_connection_status
   analyze_results
   
   group_by_status
   analyze_results
   
   print_connection_analysis
   analyze_results
   
   print_typical_cases
   analyze_results
   
   print_over_connected_modules
   analyze_results
   
   print_under_connected_modules
   analyze_results
   
   print_isolated_modules
   analyze_results
   
   verify_design_principles
   analyze_results
   
   generate_improvement_suggestions
   analyze_results
   
   print_improvement_suggestions
   analyze_results
   
   export_detailed_data
   analyze_results
   
   analyze_report_quality
   analyze_results
   - 模組: 內探模組

2. **程式組件**
   load_analysis_data
   analyze_results
   
   print_basic_statistics
   analyze_results
   
   build_connection_graph
   analyze_results
   
   calculate_connection_stats
   analyze_results
   
   classify_connection_status
   analyze_results
   
   group_by_status
   analyze_results
   
   print_connection_analysis
   analyze_results
   
   print_typical_cases
   analyze_results
   
   print_over_connected_modules
   analyze_results
   
   print_under_connected_modules
   analyze_results
   
   print_isolated_modules
   analyze_results
   
   verify_design_principles
   analyze_results
   
   generate_improvement_suggestions
   analyze_results
   
   print_improvement_suggestions
   analyze_results
   
   export_detailed_data
   analyze_results
   
   analyze_report_quality
   analyze_results
   - 模組: 內探模組

---

### Flow 84

- **長度**: 4 步
- **起點**: analyze_results
- **終點**: core_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   load_analysis_data
   analyze_results
   
   print_basic_statistics
   analyze_results
   
   build_connection_graph
   analyze_results
   
   calculate_connection_stats
   analyze_results
   
   classify_connection_status
   analyze_results
   
   group_by_status
   analyze_results
   
   print_connection_analysis
   analyze_results
   
   print_typical_cases
   analyze_results
   
   print_over_connected_modules
   analyze_results
   
   print_under_connected_modules
   analyze_results
   
   print_isolated_modules
   analyze_results
   
   verify_design_principles
   analyze_results
   
   generate_improvement_suggestions
   analyze_results
   
   print_improvement_suggestions
   analyze_results
   
   export_detailed_data
   analyze_results
   
   analyze_report_quality
   analyze_results
   - 模組: 內探模組

2. **程式組件**
   load_analysis_data
   analyze_results
   
   print_basic_statistics
   analyze_results
   
   build_connection_graph
   analyze_results
   
   calculate_connection_stats
   analyze_results
   
   classify_connection_status
   analyze_results
   
   group_by_status
   analyze_results
   
   print_connection_analysis
   analyze_results
   
   print_typical_cases
   analyze_results
   
   print_over_connected_modules
   analyze_results
   
   print_under_connected_modules
   analyze_results
   
   print_isolated_modules
   analyze_results
   
   verify_design_principles
   analyze_results
   
   generate_improvement_suggestions
   analyze_results
   
   print_improvement_suggestions
   analyze_results
   
   export_detailed_data
   analyze_results
   
   analyze_report_quality
   analyze_results
   - 模組: 內探模組

3. **程式組件**
   load_analysis_data
   analyze_results
   
   print_basic_statistics
   analyze_results
   
   build_connection_graph
   analyze_results
   
   calculate_connection_stats
   analyze_results
   
   classify_connection_status
   analyze_results
   
   group_by_status
   analyze_results
   
   print_connection_analysis
   analyze_results
   
   print_typical_cases
   analyze_results
   
   print_over_connected_modules
   analyze_results
   
   print_under_connected_modules
   analyze_results
   
   print_isolated_modules
   analyze_results
   
   verify_design_principles
   analyze_results
   
   generate_improvement_suggestions
   analyze_results
   
   print_improvement_suggestions
   analyze_results
   
   export_detailed_data
   analyze_results
   
   analyze_report_quality
   analyze_results
   - 模組: 內探模組

4. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

---

### Flow 106

- **長度**: 2 步
- **起點**: internal_loop_connector
- **終點**: aiva_internal_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI內部能力**
   CapabilityScopeClassifier
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_scope
   internal_loop_connector
   
   CapabilityScopeClassifier.classify_access_level
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_available_in
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_cli_info
   internal_loop_connector
   
   CapabilityScopeClassifier._verify_cli_content
   internal_loop_connector
   
   CapabilityScopeClassifier._infer_cli_command
   internal_loop_connector
   
   CapabilityScopeClassifier.detect_service_dependencies
   internal_loop_connector
   
   InternalLoopConnector
   internal_loop_connector
   
   InternalLoopConnector.__init__
   internal_loop_connector
   
   InternalLoopConnector.module_explorer
   internal_loop_connector
   
   InternalLoopConnector.capability_analyzer
   internal_loop_connector
   
   InternalLoopConnector.query_capabilities
   internal_loop_connector
   
   InternalLoopConnector.sync_capabilities_to_rag
   internal_loop_connector
   
   InternalLoopConnector._load_capabilities_from_analysis_data
   internal_loop_connector
   
   InternalLoopConnector._convert_flow_to_capability
   internal_loop_connector
   
   InternalLoopConnector._infer_category_from_module
   internal_loop_connector
   
   InternalLoopConnector._scan_flows_structured
   internal_loop_connector
   
   InternalLoopConnector._snake_to_camel
   internal_loop_connector
   
   InternalLoopConnector._enhance_capabilities
   internal_loop_connector
   
   InternalLoopConnector._match_sub_category
   internal_loop_connector
   
   InternalLoopConnector._classify_aiva_module
   internal_loop_connector
   
   InternalLoopConnector._matches_module_path
   internal_loop_connector
   
   InternalLoopConnector._classify_sub_module
   internal_loop_connector
   
   InternalLoopConnector._infer_module_from_name
   internal_loop_connector
   
   InternalLoopConnector._categorize_capability
   internal_loop_connector
   
   InternalLoopConnector._assess_complexity
   internal_loop_connector
   
   InternalLoopConnector._generate_tags
   internal_loop_connector
   
   InternalLoopConnector._build_invocation_metadata
   internal_loop_connector
   
   InternalLoopConnector._build_parameter_definitions
   internal_loop_connector
   
   InternalLoopConnector._generate_param_example
   internal_loop_connector
   
   InternalLoopConnector._build_return_definition
   internal_loop_connector
   
   InternalLoopConnector._generate_usage_examples
   internal_loop_connector
   
   InternalLoopConnector._convert_to_capability_model
   internal_loop_connector
   
   InternalLoopConnector._build_basic_info_section
   internal_loop_connector
   
   InternalLoopConnector._build_parameters_section
   internal_loop_connector
   
   InternalLoopConnector._build_examples_section
   internal_loop_connector
   
   InternalLoopConnector._build_health_section
   internal_loop_connector
   
   InternalLoopConnector._build_dependencies_section
   internal_loop_connector
   
   InternalLoopConnector._convert_to_documents
   internal_loop_connector
   
   InternalLoopConnector._inject_to_rag
   internal_loop_connector
   
   InternalLoopConnector.query_self_awareness
   internal_loop_connector
   
   InternalLoopConnector.report_issue
   internal_loop_connector
   
   InternalLoopConnector.search_solution
   internal_loop_connector
   
   InternalLoopConnector.get_sync_status
   internal_loop_connector
   
   InternalLoopConnector.export_capabilities_json
   internal_loop_connector
   - 模組: 認知核心模組

2. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

---

### Flow 110

- **長度**: 2 步
- **起點**: analyze_missing_function_connections
- **終點**: analyze_missing_function_connections
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

2. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

---

### Flow 112

- **長度**: 2 步
- **起點**: aiva_cli
- **終點**: aiva_internal_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   load_flow_definitions
   aiva_cli
   
   create_flow_command
   aiva_cli
   
   register_all_flow_commands
   aiva_cli
   
   aiva
   aiva_cli
   
   run
   aiva_cli
   
   query
   aiva_cli
   
   train
   aiva_cli
   
   scan
   aiva_cli
   
   status
   aiva_cli
   
   health
   aiva_cli
   
   list_flows
   aiva_cli
   
   show_flow_statistics
   aiva_cli
   
   show_flows_by_endpoint_module
   aiva_cli
   - 模組: 核心能力模組

2. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

---

### Flow 123

- **長度**: 2 步
- **起點**: run_analysis
- **終點**: practical_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

2. **程式組件**
   Issue
   practical_analyzer
   
   PracticalAnalyzer
   practical_analyzer
   
   PracticalAnalyzer.__init__
   practical_analyzer
   
   PracticalAnalyzer.analyze_report
   practical_analyzer
   
   PracticalAnalyzer._parse_definition_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_call_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_potential_missing
   practical_analyzer
   
   PracticalAnalyzer._deduplicate_issues
   practical_analyzer
   
   PracticalAnalyzer._classify_issue
   practical_analyzer
   
   PracticalAnalyzer._write_statistics
   practical_analyzer
   
   PracticalAnalyzer._write_critical_issues
   practical_analyzer
   
   PracticalAnalyzer._write_high_issues
   practical_analyzer
   
   PracticalAnalyzer._write_medium_low_issues
   practical_analyzer
   
   PracticalAnalyzer._write_action_plan
   practical_analyzer
   
   PracticalAnalyzer._write_quick_start_guide
   practical_analyzer
   
   PracticalAnalyzer.generate_practical_report
   practical_analyzer
   
   PracticalAnalyzer.generate_quick_fix_list
   practical_analyzer
   
   PracticalAnalyzer._get_timestamp
   practical_analyzer
   
   main
   practical_analyzer
   - 模組: 內探模組

---

### Flow 153

- **長度**: 2 步
- **起點**: aiva_internal_executor
- **終點**: aiva_internal_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

2. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

---

### Flow 157

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: aiva_external_classifier
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   MultiLanguageClassifier
   aiva_external_classifier
   
   MultiLanguageClassifier.__init__
   aiva_external_classifier
   
   MultiLanguageClassifier._setup_input_paths
   aiva_external_classifier
   
   MultiLanguageClassifier.scan_all_modules
   aiva_external_classifier
   
   MultiLanguageClassifier._detect_module_language
   aiva_external_classifier
   
   MultiLanguageClassifier.process_module
   aiva_external_classifier
   
   MultiLanguageClassifier._normalize_flow
   aiva_external_classifier
   
   MultiLanguageClassifier._is_operable_capability
   aiva_external_classifier
   
   MultiLanguageClassifier._get_operability_reason
   aiva_external_classifier
   
   MultiLanguageClassifier._extract_parameters
   aiva_external_classifier
   
   MultiLanguageClassifier._infer_use_case
   aiva_external_classifier
   
   MultiLanguageClassifier._extract_entry_points_from_function_details
   aiva_external_classifier
   
   MultiLanguageClassifier._convert_struct_to_flows
   aiva_external_classifier
   
   MultiLanguageClassifier._convert_graphs_to_flows
   aiva_external_classifier
   
   MultiLanguageClassifier._convert_flow_chains
   aiva_external_classifier
   
   MultiLanguageClassifier._infer_module_info
   aiva_external_classifier
   
   MultiLanguageClassifier.generate_classification_data
   aiva_external_classifier
   
   MultiLanguageClassifier.generate_summary_report
   aiva_external_classifier
   
   MultiLanguageClassifier.run
   aiva_external_classifier
   
   main
   aiva_external_classifier
   - 模組: 內探模組

---

### Flow 158

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: aiva_external_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   MultiLangExecutor
   aiva_external_executor
   
   MultiLangExecutor.__init__
   aiva_external_executor
   
   MultiLangExecutor._load_all_capabilities
   aiva_external_executor
   
   MultiLangExecutor._load_python_capabilities
   aiva_external_executor
   
   MultiLangExecutor._load_compiled_lang_capabilities
   aiva_external_executor
   
   MultiLangExecutor._extract_python_params
   aiva_external_executor
   
   MultiLangExecutor._extract_lang_params
   aiva_external_executor
   
   MultiLangExecutor.list_capabilities
   aiva_external_executor
   
   MultiLangExecutor.execute_python
   aiva_external_executor
   
   MultiLangExecutor._execute_python_top_level_function
   aiva_external_executor
   
   MultiLangExecutor._execute_python_class_method
   aiva_external_executor
   
   MultiLangExecutor.execute_rust
   aiva_external_executor
   
   MultiLangExecutor.execute_go
   aiva_external_executor
   
   MultiLangExecutor.execute_typescript
   aiva_external_executor
   
   MultiLangExecutor.generate_reference_docs
   aiva_external_executor
   
   MultiLangExecutor._generate_markdown_reference
   aiva_external_executor
   
   MultiLangExecutor._generate_json_database
   aiva_external_executor
   
   InteractiveMenu
   aiva_external_executor
   
   InteractiveMenu.__init__
   aiva_external_executor
   
   InteractiveMenu._build_indexes
   aiva_external_executor
   
   InteractiveMenu.clear
   aiva_external_executor
   
   InteractiveMenu.run
   aiva_external_executor
   
   InteractiveMenu._show_module_menu
   aiva_external_executor
   
   InteractiveMenu._show_capability_menu
   aiva_external_executor
   
   InteractiveMenu._show_variant_menu
   aiva_external_executor
   
   main
   aiva_external_executor
   - 模組: 內探模組

---

### Flow 159

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: aiva_internal_classifier
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   AIVAFlowClassifier
   aiva_internal_classifier
   
   AIVAFlowClassifier.__init__
   aiva_internal_classifier
   
   AIVAFlowClassifier.load_module_config
   aiva_internal_classifier
   
   AIVAFlowClassifier.get_default_config
   aiva_internal_classifier
   
   AIVAFlowClassifier.load_flow_data
   aiva_internal_classifier
   
   AIVAFlowClassifier._extract_script_name
   aiva_internal_classifier
   
   AIVAFlowClassifier._get_script_description
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_script_by_name
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_module_from_path
   aiva_internal_classifier
   
   AIVAFlowClassifier._merge_duplicate_flows
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_ai_capability_type
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_module
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_component_type
   aiva_internal_classifier
   
   AIVAFlowClassifier._classify_loop_type
   aiva_internal_classifier
   
   AIVAFlowClassifier.classify_flows
   aiva_internal_classifier
   
   AIVAFlowClassifier.analyze_multi_path_endpoints
   aiva_internal_classifier
   
   AIVAFlowClassifier.generate_reports
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_classification_report
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_complete_flow_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_flow_details_header
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_flows_by_module
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_module_section
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_single_flow_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_classification_step
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_function_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_multi_path_report
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_multi_path_header
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_single_endpoint_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_endpoint_summary
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_path_details
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_path_difference_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_script_comparison
   aiva_internal_classifier
   
   AIVAFlowClassifier._write_usage_scenario_analysis
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_json_export
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_cli_command
   aiva_internal_classifier
   
   AIVAFlowClassifier._get_endpoint_function_info
   aiva_internal_classifier
   
   AIVAFlowClassifier._generate_structured_tags
   aiva_internal_classifier
   
   AIVAFlowClassifier.run
   aiva_internal_classifier
   
   main
   aiva_internal_classifier
   - 模組: 內探模組

---

### Flow 160

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: aiva_internal_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

---

### Flow 161

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: enhanced_capability_integrator
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   CapabilityReference
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.__init__
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.load_capabilities_data
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator._build_indexes
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capability_by_flow_id
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_capabilities_by_module
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.search_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.get_recommended_capabilities
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.generate_usage_report
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.export_for_cli_integration
   enhanced_capability_integrator
   
   EnhancedCapabilityIntegrator.create_enhanced_executor_integration
   enhanced_capability_integrator
   
   main
   enhanced_capability_integrator
   - 模組: 內探模組

---

### Flow 162

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: enhanced_classifier_processor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   ProcessingStage
   enhanced_classifier_processor
   
   ProcessingCheckpoint
   enhanced_classifier_processor
   
   EnhancedCapabilityDescription
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.__init__
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._init_llm_templates
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.load_from_classifier_data
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.check_recovery_point
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.save_checkpoint
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_data_hash
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_1_raw_analysis
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_module_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_component_from_flow
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_ai_indicators
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_2_semantic_enhancement
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_semantic_name
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._categorize_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_complexity
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._detect_interaction_pattern
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_related_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_semantic_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._cluster_capabilities
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_3_llm_augmentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_purpose
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_scenarios
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_risk_assessment
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._simulate_llm_examples
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_llm_insights
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_4_standardization
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._group_by_module_priority
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._classify_capability_scope
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_ai_readable_group_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_usage_context
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_expected_outcome
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_selection_criteria
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._find_common_path_steps
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._analyze_path_differences
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._recommend_optimal_path
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._create_standardized_capability
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_usage_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_enhanced_cli_integration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_ai_friendly_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_input_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_output_description
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._identify_side_effects
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._estimate_duration
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._extract_dependencies
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._map_ai_capability_level
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._calculate_quality_score
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._build_capability_taxonomy
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_cli_integration_guide
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor._generate_markdown_documentation
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.stage_5_completion
   enhanced_classifier_processor
   
   EnhancedClassifierProcessor.run_full_pipeline
   enhanced_classifier_processor
   
   main
   enhanced_classifier_processor
   - 模組: 內探模組

---

### Flow 163

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: unified_executor_controller
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   ExecutionResult
   unified_executor_controller
   
   UnifiedExecutorController
   unified_executor_controller
   
   UnifiedExecutorController.__init__
   unified_executor_controller
   
   UnifiedExecutorController.initialize
   unified_executor_controller
   
   UnifiedExecutorController._load_capability_mappings
   unified_executor_controller
   
   UnifiedExecutorController.execute_capability
   unified_executor_controller
   
   UnifiedExecutorController._execute_internal
   unified_executor_controller
   
   UnifiedExecutorController._execute_external
   unified_executor_controller
   
   UnifiedExecutorController.list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController.show_menu
   unified_executor_controller
   
   UnifiedExecutorController._menu_external_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_internal_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_capability_details
   unified_executor_controller
   
   main
   unified_executor_controller
   - 模組: 內探模組

---

### Flow 174

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: analyze_connection_recommendations
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   FunctionInfo
   analyze_connection_recommendations
   
   ConnectionRecommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.__init__
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.extract_all_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._should_skip_file
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_from_ast
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._parse_function
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_imports
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.analyze_missing_connections
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_orphaned_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_potential_callers
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_connection_confidence
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._are_files_related
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_name_similarity
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_semantic_match
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_parameter_compatibility
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_call_chain_logic
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_docstring_hints
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._evaluate_impact
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._suggest_location
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._generate_code_example
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._impact_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._confidence_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.generate_report
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._write_recommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._get_timestamp
   analyze_connection_recommendations
   
   main
   analyze_connection_recommendations
   - 模組: 內探模組

---

### Flow 175

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: analyze_dataflow_breakpoints
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   BreakpointIssue
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.__init__
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer._load_analysis_results
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.build_flow_graph
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_breakpoints
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_dead_ends
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_isolated_islands
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_bottlenecks
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_circular_dependencies
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.analyze_missing_connections
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.generate_report
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.run_full_analysis
   analyze_dataflow_breakpoints
   
   main
   analyze_dataflow_breakpoints
   - 模組: 內探模組

---

### Flow 176

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: analyze_missing_function_connections
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

---

### Flow 177

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: core_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

---

### Flow 178

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: practical_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   Issue
   practical_analyzer
   
   PracticalAnalyzer
   practical_analyzer
   
   PracticalAnalyzer.__init__
   practical_analyzer
   
   PracticalAnalyzer.analyze_report
   practical_analyzer
   
   PracticalAnalyzer._parse_definition_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_call_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_potential_missing
   practical_analyzer
   
   PracticalAnalyzer._deduplicate_issues
   practical_analyzer
   
   PracticalAnalyzer._classify_issue
   practical_analyzer
   
   PracticalAnalyzer._write_statistics
   practical_analyzer
   
   PracticalAnalyzer._write_critical_issues
   practical_analyzer
   
   PracticalAnalyzer._write_high_issues
   practical_analyzer
   
   PracticalAnalyzer._write_medium_low_issues
   practical_analyzer
   
   PracticalAnalyzer._write_action_plan
   practical_analyzer
   
   PracticalAnalyzer._write_quick_start_guide
   practical_analyzer
   
   PracticalAnalyzer.generate_practical_report
   practical_analyzer
   
   PracticalAnalyzer.generate_quick_fix_list
   practical_analyzer
   
   PracticalAnalyzer._get_timestamp
   practical_analyzer
   
   main
   practical_analyzer
   - 模組: 內探模組

---

### Flow 179

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: run_analysis
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

---

### Flow 180

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: standalone_cli_validator
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   check_dependencies
   standalone_cli_validator
   
   load_module_directly
   standalone_cli_validator
   
   load_ai_capabilities
   standalone_cli_validator
   
   load_classification_data
   standalone_cli_validator
   
   load_analysis_results
   standalone_cli_validator
   
   validate_flow_parameters
   standalone_cli_validator
   
   list_all_capabilities
   standalone_cli_validator
   
   validate_single_flow
   standalone_cli_validator
   
   validate_all_flows
   standalone_cli_validator
   
   test_ai_capability
   standalone_cli_validator
   
   main
   standalone_cli_validator
   - 模組: 內探模組

---

### Flow 214

- **長度**: 2 步
- **起點**: run_analysis
- **終點**: analyze_missing_function_connections
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisRunner
   run_analysis
   
   AnalysisRunner.__init__
   run_analysis
   
   AnalysisRunner.print_banner
   run_analysis
   
   AnalysisRunner._get_output_dir
   run_analysis
   
   AnalysisRunner._get_analysis_json
   run_analysis
   
   AnalysisRunner.run_full_analysis
   run_analysis
   
   AnalysisRunner.run_quick_scan
   run_analysis
   
   AnalysisRunner.run_dataflow_analysis
   run_analysis
   
   AnalysisRunner.run_missing_connection_analysis
   run_analysis
   
   AnalysisRunner.run_practical_analysis
   run_analysis
   
   AnalysisRunner.run_connection_recommendations
   run_analysis
   
   AnalysisRunner.run_batch_analysis
   run_analysis
   
   AnalysisRunner._print_report_summary
   run_analysis
   
   AnalysisRunner.save_json_report
   run_analysis
   
   main
   run_analysis
   - 模組: 內探模組

2. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

---

### Flow 251

- **長度**: 2 步
- **起點**: analyze_connection_recommendations
- **終點**: analyze_connection_recommendations
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionInfo
   analyze_connection_recommendations
   
   ConnectionRecommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.__init__
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.extract_all_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._should_skip_file
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_from_ast
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._parse_function
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_imports
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.analyze_missing_connections
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_orphaned_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_potential_callers
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_connection_confidence
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._are_files_related
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_name_similarity
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_semantic_match
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_parameter_compatibility
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_call_chain_logic
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_docstring_hints
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._evaluate_impact
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._suggest_location
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._generate_code_example
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._impact_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._confidence_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.generate_report
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._write_recommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._get_timestamp
   analyze_connection_recommendations
   
   main
   analyze_connection_recommendations
   - 模組: 內探模組

2. **程式組件**
   FunctionInfo
   analyze_connection_recommendations
   
   ConnectionRecommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.__init__
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.extract_all_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._should_skip_file
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_from_ast
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._parse_function
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._extract_imports
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.analyze_missing_connections
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_orphaned_functions
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._find_potential_callers
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_connection_confidence
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._are_files_related
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._calculate_name_similarity
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_semantic_match
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_parameter_compatibility
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_call_chain_logic
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._check_docstring_hints
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._evaluate_impact
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._suggest_location
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._generate_code_example
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._impact_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._confidence_score
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer.generate_report
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._write_recommendation
   analyze_connection_recommendations
   
   ConnectionRecommendationAnalyzer._get_timestamp
   analyze_connection_recommendations
   
   main
   analyze_connection_recommendations
   - 模組: 內探模組

---

### Flow 252

- **長度**: 2 步
- **起點**: aiva_flow_analyzer
- **終點**: aiva_flow_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   Node
   aiva_flow_analyzer
   
   Node.__init__
   aiva_flow_analyzer
   
   Node._sanitize_id
   aiva_flow_analyzer
   
   Node.to_dict
   aiva_flow_analyzer
   
   Graph
   aiva_flow_analyzer
   
   Graph.__init__
   aiva_flow_analyzer
   
   Graph.add
   aiva_flow_analyzer
   
   Graph.link
   aiva_flow_analyzer
   
   Graph.to_dict
   aiva_flow_analyzer
   
   ParameterExtractor
   aiva_flow_analyzer
   
   ParameterExtractor.extract_parameters
   aiva_flow_analyzer
   
   ParameterExtractor._ast_to_value
   aiva_flow_analyzer
   
   ParameterExtractor._annotation_to_string
   aiva_flow_analyzer
   
   FlowBuilder
   aiva_flow_analyzer
   
   FlowBuilder.__init__
   aiva_flow_analyzer
   
   FlowBuilder.build
   aiva_flow_analyzer
   
   FlowBuilder._visit_block
   aiva_flow_analyzer
   
   FlowBuilder.visit_Call
   aiva_flow_analyzer
   
   FlowBuilder._resolve_name
   aiva_flow_analyzer
   
   FlowBuilder.visit_If
   aiva_flow_analyzer
   
   FlowBuilder.visit_While
   aiva_flow_analyzer
   
   FlowBuilder.visit_For
   aiva_flow_analyzer
   
   FlowBuilder.visit_Return
   aiva_flow_analyzer
   
   FlowBuilder.visit_FunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_ClassDef
   aiva_flow_analyzer
   
   FlowStitcher
   aiva_flow_analyzer
   
   FlowStitcher.__init__
   aiva_flow_analyzer
   
   FlowStitcher.register_graph
   aiva_flow_analyzer
   
   FlowStitcher.stitch
   aiva_flow_analyzer
   
   FlowStitcher.build_flow_chains
   aiva_flow_analyzer
   
   FlowStitcher._build_paths_recursive
   aiva_flow_analyzer
   
   FlowStitcher.build_function_details
   aiva_flow_analyzer
   
   ReadmeExtractor
   aiva_flow_analyzer
   
   ReadmeExtractor.__init__
   aiva_flow_analyzer
   
   ReadmeExtractor.get_readme_content
   aiva_flow_analyzer
   
   ReadmeExtractor._extract_summary
   aiva_flow_analyzer
   
   analyze_and_generate
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.__init__
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.analyze_directory
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.save_results
   aiva_flow_analyzer
   - 模組: 內探模組

2. **程式組件**
   Node
   aiva_flow_analyzer
   
   Node.__init__
   aiva_flow_analyzer
   
   Node._sanitize_id
   aiva_flow_analyzer
   
   Node.to_dict
   aiva_flow_analyzer
   
   Graph
   aiva_flow_analyzer
   
   Graph.__init__
   aiva_flow_analyzer
   
   Graph.add
   aiva_flow_analyzer
   
   Graph.link
   aiva_flow_analyzer
   
   Graph.to_dict
   aiva_flow_analyzer
   
   ParameterExtractor
   aiva_flow_analyzer
   
   ParameterExtractor.extract_parameters
   aiva_flow_analyzer
   
   ParameterExtractor._ast_to_value
   aiva_flow_analyzer
   
   ParameterExtractor._annotation_to_string
   aiva_flow_analyzer
   
   FlowBuilder
   aiva_flow_analyzer
   
   FlowBuilder.__init__
   aiva_flow_analyzer
   
   FlowBuilder.build
   aiva_flow_analyzer
   
   FlowBuilder._visit_block
   aiva_flow_analyzer
   
   FlowBuilder.visit_Call
   aiva_flow_analyzer
   
   FlowBuilder._resolve_name
   aiva_flow_analyzer
   
   FlowBuilder.visit_If
   aiva_flow_analyzer
   
   FlowBuilder.visit_While
   aiva_flow_analyzer
   
   FlowBuilder.visit_For
   aiva_flow_analyzer
   
   FlowBuilder.visit_Return
   aiva_flow_analyzer
   
   FlowBuilder.visit_FunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_ClassDef
   aiva_flow_analyzer
   
   FlowStitcher
   aiva_flow_analyzer
   
   FlowStitcher.__init__
   aiva_flow_analyzer
   
   FlowStitcher.register_graph
   aiva_flow_analyzer
   
   FlowStitcher.stitch
   aiva_flow_analyzer
   
   FlowStitcher.build_flow_chains
   aiva_flow_analyzer
   
   FlowStitcher._build_paths_recursive
   aiva_flow_analyzer
   
   FlowStitcher.build_function_details
   aiva_flow_analyzer
   
   ReadmeExtractor
   aiva_flow_analyzer
   
   ReadmeExtractor.__init__
   aiva_flow_analyzer
   
   ReadmeExtractor.get_readme_content
   aiva_flow_analyzer
   
   ReadmeExtractor._extract_summary
   aiva_flow_analyzer
   
   analyze_and_generate
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.__init__
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.analyze_directory
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.save_results
   aiva_flow_analyzer
   - 模組: 內探模組

---

### Flow 284

- **長度**: 2 步
- **起點**: analyze_missing_function_connections
- **終點**: core_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

2. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

---

### Flow 311

- **長度**: 2 步
- **起點**: core_analyzer
- **終點**: core_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

2. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

---

### Flow 337

- **長度**: 2 步
- **起點**: core_analyzer
- **終點**: aiva_flow_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

2. **程式組件**
   Node
   aiva_flow_analyzer
   
   Node.__init__
   aiva_flow_analyzer
   
   Node._sanitize_id
   aiva_flow_analyzer
   
   Node.to_dict
   aiva_flow_analyzer
   
   Graph
   aiva_flow_analyzer
   
   Graph.__init__
   aiva_flow_analyzer
   
   Graph.add
   aiva_flow_analyzer
   
   Graph.link
   aiva_flow_analyzer
   
   Graph.to_dict
   aiva_flow_analyzer
   
   ParameterExtractor
   aiva_flow_analyzer
   
   ParameterExtractor.extract_parameters
   aiva_flow_analyzer
   
   ParameterExtractor._ast_to_value
   aiva_flow_analyzer
   
   ParameterExtractor._annotation_to_string
   aiva_flow_analyzer
   
   FlowBuilder
   aiva_flow_analyzer
   
   FlowBuilder.__init__
   aiva_flow_analyzer
   
   FlowBuilder.build
   aiva_flow_analyzer
   
   FlowBuilder._visit_block
   aiva_flow_analyzer
   
   FlowBuilder.visit_Call
   aiva_flow_analyzer
   
   FlowBuilder._resolve_name
   aiva_flow_analyzer
   
   FlowBuilder.visit_If
   aiva_flow_analyzer
   
   FlowBuilder.visit_While
   aiva_flow_analyzer
   
   FlowBuilder.visit_For
   aiva_flow_analyzer
   
   FlowBuilder.visit_Return
   aiva_flow_analyzer
   
   FlowBuilder.visit_FunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_AsyncFunctionDef
   aiva_flow_analyzer
   
   FlowBuilder.visit_ClassDef
   aiva_flow_analyzer
   
   FlowStitcher
   aiva_flow_analyzer
   
   FlowStitcher.__init__
   aiva_flow_analyzer
   
   FlowStitcher.register_graph
   aiva_flow_analyzer
   
   FlowStitcher.stitch
   aiva_flow_analyzer
   
   FlowStitcher.build_flow_chains
   aiva_flow_analyzer
   
   FlowStitcher._build_paths_recursive
   aiva_flow_analyzer
   
   FlowStitcher.build_function_details
   aiva_flow_analyzer
   
   ReadmeExtractor
   aiva_flow_analyzer
   
   ReadmeExtractor.__init__
   aiva_flow_analyzer
   
   ReadmeExtractor.get_readme_content
   aiva_flow_analyzer
   
   ReadmeExtractor._extract_summary
   aiva_flow_analyzer
   
   analyze_and_generate
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.__init__
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.analyze_directory
   aiva_flow_analyzer
   
   AIVAFlowAnalyzer.save_results
   aiva_flow_analyzer
   - 模組: 內探模組

---

### Flow 338

- **長度**: 2 步
- **起點**: core_analyzer
- **終點**: analyze_dataflow_breakpoints
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

2. **程式組件**
   BreakpointIssue
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.__init__
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer._load_analysis_results
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.build_flow_graph
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_breakpoints
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_dead_ends
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_isolated_islands
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_bottlenecks
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.detect_circular_dependencies
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.analyze_missing_connections
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.generate_report
   analyze_dataflow_breakpoints
   
   DataFlowBreakpointAnalyzer.run_full_analysis
   analyze_dataflow_breakpoints
   
   main
   analyze_dataflow_breakpoints
   - 模組: 內探模組

---

### Flow 339

- **長度**: 2 步
- **起點**: core_analyzer
- **終點**: analyze_missing_function_connections
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

2. **程式組件**
   FunctionSignature
   analyze_missing_function_connections
   
   MissingConnection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.__init__
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._load_analysis_results
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.extract_function_signatures
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._analyze_file_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_signature
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_parameters
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_return_info
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._extract_function_calls
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._get_function_name
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_missing_definitions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_orphaned_with_entry_exit
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_potential_missing_links
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.find_unused_returns
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._is_likely_external
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._should_ignore_connection
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._find_similar_functions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._levenshtein_distance
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._group_connections
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_summary_section
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._format_connection_item
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_detailed_list
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_priority_suggestions
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_statistics
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer._write_file_ranking
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.generate_report
   analyze_missing_function_connections
   
   MissingConnectionAnalyzer.run_full_analysis
   analyze_missing_function_connections
   
   main
   analyze_missing_function_connections
   - 模組: 內探模組

---

### Flow 340

- **長度**: 2 步
- **起點**: core_analyzer
- **終點**: practical_analyzer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisReport
   core_analyzer
   
   AnalysisReport.to_dict
   core_analyzer
   
   AnalysisReport.save_json
   core_analyzer
   
   AnalysisReport.save_markdown
   core_analyzer
   
   AnalysisReport._write_header
   core_analyzer
   
   AnalysisReport._write_statistics
   core_analyzer
   
   AnalysisReport._write_critical_issues
   core_analyzer
   
   AnalysisReport._write_high_issues
   core_analyzer
   
   AnalysisReport._write_recommendations
   core_analyzer
   
   classify_script_type
   core_analyzer
   
   CoreAnalyzer
   core_analyzer
   
   CoreAnalyzer.__init__
   core_analyzer
   
   CoreAnalyzer.full_analysis
   core_analyzer
   
   CoreAnalyzer.quick_scan
   core_analyzer
   
   CoreAnalyzer.diagnose_critical_only
   core_analyzer
   
   CoreAnalyzer._generate_recommendations
   core_analyzer
   
   CoreAnalyzer._print_summary
   core_analyzer
   
   main
   core_analyzer
   - 模組: 內探模組

2. **程式組件**
   Issue
   practical_analyzer
   
   PracticalAnalyzer
   practical_analyzer
   
   PracticalAnalyzer.__init__
   practical_analyzer
   
   PracticalAnalyzer.analyze_report
   practical_analyzer
   
   PracticalAnalyzer._parse_definition_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_call_missing
   practical_analyzer
   
   PracticalAnalyzer._parse_potential_missing
   practical_analyzer
   
   PracticalAnalyzer._deduplicate_issues
   practical_analyzer
   
   PracticalAnalyzer._classify_issue
   practical_analyzer
   
   PracticalAnalyzer._write_statistics
   practical_analyzer
   
   PracticalAnalyzer._write_critical_issues
   practical_analyzer
   
   PracticalAnalyzer._write_high_issues
   practical_analyzer
   
   PracticalAnalyzer._write_medium_low_issues
   practical_analyzer
   
   PracticalAnalyzer._write_action_plan
   practical_analyzer
   
   PracticalAnalyzer._write_quick_start_guide
   practical_analyzer
   
   PracticalAnalyzer.generate_practical_report
   practical_analyzer
   
   PracticalAnalyzer.generate_quick_fix_list
   practical_analyzer
   
   PracticalAnalyzer._get_timestamp
   practical_analyzer
   
   main
   practical_analyzer
   - 模組: 內探模組

---

### Flow 362

- **長度**: 2 步
- **起點**: unified_executor_controller
- **終點**: aiva_internal_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   unified_executor_controller
   
   UnifiedExecutorController
   unified_executor_controller
   
   UnifiedExecutorController.__init__
   unified_executor_controller
   
   UnifiedExecutorController.initialize
   unified_executor_controller
   
   UnifiedExecutorController._load_capability_mappings
   unified_executor_controller
   
   UnifiedExecutorController.execute_capability
   unified_executor_controller
   
   UnifiedExecutorController._execute_internal
   unified_executor_controller
   
   UnifiedExecutorController._execute_external
   unified_executor_controller
   
   UnifiedExecutorController.list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController.show_menu
   unified_executor_controller
   
   UnifiedExecutorController._menu_external_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_internal_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_capability_details
   unified_executor_controller
   
   main
   unified_executor_controller
   - 模組: 內探模組

2. **程式組件**
   find_latest_classification_file
   aiva_internal_executor
   
   FlowExecutor
   aiva_internal_executor
   
   FlowExecutor.__init__
   aiva_internal_executor
   
   FlowExecutor._load_data
   aiva_internal_executor
   
   FlowExecutor.get_flow_by_id
   aiva_internal_executor
   
   FlowExecutor.search_capabilities
   aiva_internal_executor
   
   FlowExecutor.show_capability_info
   aiva_internal_executor
   
   FlowExecutor._full_path_to_module
   aiva_internal_executor
   
   FlowExecutor._snake_to_camel
   aiva_internal_executor
   
   FlowExecutor._find_entry_method
   aiva_internal_executor
   
   FlowExecutor.generate_reference_docs
   aiva_internal_executor
   
   FlowExecutor.execute_flow
   aiva_internal_executor
   
   main
   aiva_internal_executor
   
   InteractiveMenu
   aiva_internal_executor
   
   InteractiveMenu.__init__
   aiva_internal_executor
   
   InteractiveMenu._build_indexes
   aiva_internal_executor
   
   InteractiveMenu.clear
   aiva_internal_executor
   
   InteractiveMenu.run
   aiva_internal_executor
   
   InteractiveMenu._show_capability_menu
   aiva_internal_executor
   
   InteractiveMenu._show_variant_menu
   aiva_internal_executor
   - 模組: 內探模組

---

### Flow 363

- **長度**: 2 步
- **起點**: unified_executor_controller
- **終點**: aiva_external_executor
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   unified_executor_controller
   
   UnifiedExecutorController
   unified_executor_controller
   
   UnifiedExecutorController.__init__
   unified_executor_controller
   
   UnifiedExecutorController.initialize
   unified_executor_controller
   
   UnifiedExecutorController._load_capability_mappings
   unified_executor_controller
   
   UnifiedExecutorController.execute_capability
   unified_executor_controller
   
   UnifiedExecutorController._execute_internal
   unified_executor_controller
   
   UnifiedExecutorController._execute_external
   unified_executor_controller
   
   UnifiedExecutorController.list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController.show_menu
   unified_executor_controller
   
   UnifiedExecutorController._menu_external_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_internal_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_list_capabilities
   unified_executor_controller
   
   UnifiedExecutorController._menu_capability_details
   unified_executor_controller
   
   main
   unified_executor_controller
   - 模組: 內探模組

2. **程式組件**
   MultiLangExecutor
   aiva_external_executor
   
   MultiLangExecutor.__init__
   aiva_external_executor
   
   MultiLangExecutor._load_all_capabilities
   aiva_external_executor
   
   MultiLangExecutor._load_python_capabilities
   aiva_external_executor
   
   MultiLangExecutor._load_compiled_lang_capabilities
   aiva_external_executor
   
   MultiLangExecutor._extract_python_params
   aiva_external_executor
   
   MultiLangExecutor._extract_lang_params
   aiva_external_executor
   
   MultiLangExecutor.list_capabilities
   aiva_external_executor
   
   MultiLangExecutor.execute_python
   aiva_external_executor
   
   MultiLangExecutor._execute_python_top_level_function
   aiva_external_executor
   
   MultiLangExecutor._execute_python_class_method
   aiva_external_executor
   
   MultiLangExecutor.execute_rust
   aiva_external_executor
   
   MultiLangExecutor.execute_go
   aiva_external_executor
   
   MultiLangExecutor.execute_typescript
   aiva_external_executor
   
   MultiLangExecutor.generate_reference_docs
   aiva_external_executor
   
   MultiLangExecutor._generate_markdown_reference
   aiva_external_executor
   
   MultiLangExecutor._generate_json_database
   aiva_external_executor
   
   InteractiveMenu
   aiva_external_executor
   
   InteractiveMenu.__init__
   aiva_external_executor
   
   InteractiveMenu._build_indexes
   aiva_external_executor
   
   InteractiveMenu.clear
   aiva_external_executor
   
   InteractiveMenu.run
   aiva_external_executor
   
   InteractiveMenu._show_module_menu
   aiva_external_executor
   
   InteractiveMenu._show_capability_menu
   aiva_external_executor
   
   InteractiveMenu._show_variant_menu
   aiva_external_executor
   
   main
   aiva_external_executor
   - 模組: 內探模組

---

### Flow 371

- **長度**: 2 步
- **起點**: ai_decision_core
- **終點**: system_self_explorer
- **主要模組**: 內探模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   UserConstraints
   ai_decision_core
   
   ScanStrategy
   ai_decision_core
   
   AIDecisionCore
   ai_decision_core
   
   AIDecisionCore.__init__
   ai_decision_core
   
   AIDecisionCore.initialize
   ai_decision_core
   
   AIDecisionCore.decide_scan_strategy
   ai_decision_core
   
   AIDecisionCore._filter_capabilities
   ai_decision_core
   
   AIDecisionCore._search_rag_suggestions
   ai_decision_core
   
   AIDecisionCore._generate_strategy
   ai_decision_core
   
   AIDecisionCore.get_flow_execution_order
   ai_decision_core
   
   AIDecisionCore.generate_attack_plan
   ai_decision_core
   
   quick_decision
   ai_decision_core
   - 模組: 認知核心模組

2. **程式組件**
   SystemCapability
   system_self_explorer
   
   EngineInfo
   system_self_explorer
   
   SystemSelfExplorer
   system_self_explorer
   
   SystemSelfExplorer.__init__
   system_self_explorer
   
   SystemSelfExplorer.initialize
   system_self_explorer
   
   SystemSelfExplorer._load_flows_data
   system_self_explorer
   
   SystemSelfExplorer._parse_capabilities
   system_self_explorer
   
   SystemSelfExplorer._parse_engines
   system_self_explorer
   
   SystemSelfExplorer._get_capabilities_for_flows
   system_self_explorer
   
   SystemSelfExplorer.get_available_attacks
   system_self_explorer
   
   SystemSelfExplorer.get_available_engines
   system_self_explorer
   
   SystemSelfExplorer.get_capability_by_type
   system_self_explorer
   
   SystemSelfExplorer.check_capability_available
   system_self_explorer
   
   SystemSelfExplorer.get_all_capabilities
   system_self_explorer
   
   SystemSelfExplorer.get_system_summary
   system_self_explorer
   
   quick_explore
   system_self_explorer
   - 模組: 內探模組

---

## 任務規劃模組 (task_planning)

包含 18 條數據流

### Flow 9

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

### Flow 12

- **長度**: 2 步
- **起點**: task_converter
- **終點**: task_converter
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   TaskPriority
   task_converter
   
   ExecutableTask
   task_converter
   
   ExecutableTask.__repr__
   task_converter
   
   TaskSequence
   task_converter
   
   TaskSequence.add_task
   task_converter
   
   TaskSequence.get_task
   task_converter
   
   TaskSequence.get_pending_tasks
   task_converter
   
   TaskSequence.get_runnable_tasks
   task_converter
   
   TaskConverter
   task_converter
   
   TaskConverter.__init__
   task_converter
   
   TaskConverter.convert
   task_converter
   
   TaskConverter._topological_sort
   task_converter
   
   TaskConverter._get_node_priority
   task_converter
   
   TaskConverter._create_task_from_node
   task_converter
   
   TaskConverter._interpolate_variables
   task_converter
   
   TaskConverter._resolve_nested_variable
   task_converter
   - 模組: 任務規劃模組

2. **程式組件**
   TaskPriority
   task_converter
   
   ExecutableTask
   task_converter
   
   ExecutableTask.__repr__
   task_converter
   
   TaskSequence
   task_converter
   
   TaskSequence.add_task
   task_converter
   
   TaskSequence.get_task
   task_converter
   
   TaskSequence.get_pending_tasks
   task_converter
   
   TaskSequence.get_runnable_tasks
   task_converter
   
   TaskConverter
   task_converter
   
   TaskConverter.__init__
   task_converter
   
   TaskConverter.convert
   task_converter
   
   TaskConverter._topological_sort
   task_converter
   
   TaskConverter._get_node_priority
   task_converter
   
   TaskConverter._create_task_from_node
   task_converter
   
   TaskConverter._interpolate_variables
   task_converter
   
   TaskConverter._resolve_nested_variable
   task_converter
   - 模組: 任務規劃模組

---

### Flow 14

- **長度**: 2 步
- **起點**: backends
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   StorageBackend
   backends
   
   StorageBackend.save_experience_sample
   backends
   
   StorageBackend.get_experience_samples
   backends
   
   StorageBackend.save_trace
   backends
   
   StorageBackend.get_traces_by_session
   backends
   
   StorageBackend.save_training_session
   backends
   
   StorageBackend.get_statistics
   backends
   
   SQLiteBackend
   backends
   
   SQLiteBackend.__init__
   backends
   
   SQLiteBackend.save_experience_sample
   backends
   
   SQLiteBackend.save_unified_experience_sample
   backends
   
   SQLiteBackend.get_experience_samples
   backends
   
   SQLiteBackend.save_trace
   backends
   
   SQLiteBackend.get_traces_by_session
   backends
   
   SQLiteBackend.save_training_session
   backends
   
   SQLiteBackend.get_statistics
   backends
   
   PostgreSQLBackend
   backends
   
   PostgreSQLBackend.__init__
   backends
   
   JSONLBackend
   backends
   
   JSONLBackend.__init__
   backends
   
   JSONLBackend.save_experience_sample
   backends
   
   JSONLBackend.get_experience_samples
   backends
   
   JSONLBackend.save_trace
   backends
   
   JSONLBackend.get_traces_by_session
   backends
   
   JSONLBackend.save_training_session
   backends
   
   JSONLBackend.get_statistics
   backends
   
   HybridBackend
   backends
   
   HybridBackend.__init__
   backends
   
   HybridBackend.save_experience_sample
   backends
   
   HybridBackend.get_experience_samples
   backends
   
   HybridBackend.save_trace
   backends
   
   HybridBackend.get_traces_by_session
   backends
   
   HybridBackend.save_training_session
   backends
   
   HybridBackend.get_statistics
   backends
   - 模組: 服務骨幹模組

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

### Flow 39

- **長度**: 2 步
- **起點**: strategy_engine
- **終點**: policy_manager
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   StrategyEngine
   strategy_engine
   
   StrategyEngine.__init__
   strategy_engine
   
   StrategyEngine.make_strategy_decision
   strategy_engine
   
   StrategyEngine.assess_risk_factors
   strategy_engine
   
   StrategyEngine._encode_situation_for_neural
   strategy_engine
   
   StrategyEngine._build_strategy_from_neural_decision
   strategy_engine
   
   StrategyEngine._adjust_confidence_by_risk
   strategy_engine
   
   StrategyEngine._get_similar_decisions
   strategy_engine
   
   StrategyEngine._calculate_historical_confidence
   strategy_engine
   
   StrategyEngine.build_strategy_decision_prompt
   strategy_engine
   - 模組: 任務規劃模組

2. **程式組件**
   RiskRule
   policy_manager
   
   RiskLevel
   policy_manager
   
   PolicyManager
   policy_manager
   
   PolicyManager.__init__
   policy_manager
   
   PolicyManager._load_policy
   policy_manager
   
   PolicyManager._use_fallback_policy
   policy_manager
   
   PolicyManager.assess_risk
   policy_manager
   
   PolicyManager._evaluate_condition
   policy_manager
   
   PolicyManager._determine_risk_level
   policy_manager
   
   PolicyManager.reload_policy
   policy_manager
   
   PolicyManager.get_policy_info
   policy_manager
   - 模組: 任務規劃模組

---

### Flow 40

- **長度**: 2 步
- **起點**: ast_parser
- **終點**: ast_parser
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   NodeType
   ast_parser
   
   AttackFlowNode
   ast_parser
   
   AttackFlowNode.__repr__
   ast_parser
   
   AttackFlowEdge
   ast_parser
   
   AttackFlowEdge.__repr__
   ast_parser
   
   AttackFlowGraph
   ast_parser
   
   AttackFlowGraph.add_node
   ast_parser
   
   AttackFlowGraph.add_edge
   ast_parser
   
   AttackFlowGraph.get_start_node
   ast_parser
   
   AttackFlowGraph.get_next_nodes
   ast_parser
   
   AttackFlowGraph.validate
   ast_parser
   
   ASTParser
   ast_parser
   
   ASTParser.__init__
   ast_parser
   
   ASTParser.parse_dict
   ast_parser
   
   ASTParser.parse_text
   ast_parser
   
   ASTParser.create_example_sqli_flow
   ast_parser
   - 模組: 任務規劃模組

2. **程式組件**
   NodeType
   ast_parser
   
   AttackFlowNode
   ast_parser
   
   AttackFlowNode.__repr__
   ast_parser
   
   AttackFlowEdge
   ast_parser
   
   AttackFlowEdge.__repr__
   ast_parser
   
   AttackFlowGraph
   ast_parser
   
   AttackFlowGraph.add_node
   ast_parser
   
   AttackFlowGraph.add_edge
   ast_parser
   
   AttackFlowGraph.get_start_node
   ast_parser
   
   AttackFlowGraph.get_next_nodes
   ast_parser
   
   AttackFlowGraph.validate
   ast_parser
   
   ASTParser
   ast_parser
   
   ASTParser.__init__
   ast_parser
   
   ASTParser.parse_dict
   ast_parser
   
   ASTParser.parse_text
   ast_parser
   
   ASTParser.create_example_sqli_flow
   ast_parser
   - 模組: 任務規劃模組

---

### Flow 54

- **長度**: 2 步
- **起點**: core_service_coordinator
- **終點**: command_router
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

2. **程式組件**
   CommandType
   command_router
   
   ExecutionMode
   command_router
   
   CommandContext
   command_router
   
   CommandContext.__post_init__
   command_router
   
   ExecutionResult
   command_router
   
   ExecutionResult.__post_init__
   command_router
   
   CommandRouter
   command_router
   
   CommandRouter.__init__
   command_router
   
   CommandRouter._initialize_intelligent_routes
   command_router
   
   CommandRouter._initialize_ai_keywords
   command_router
   
   CommandRouter._initialize_complexity_patterns
   command_router
   
   CommandRouter._analyze_command_complexity
   command_router
   
   CommandRouter._requires_ai_analysis
   command_router
   
   CommandRouter.route_command
   command_router
   
   CommandRouter.get_command_stats
   command_router
   
   CommandRouter.update_route
   command_router
   
   CommandRouter.get_available_commands
   command_router
   
   get_command_router
   command_router
   - 模組: 任務規劃模組

---

### Flow 61

- **長度**: 2 步
- **起點**: scenario_manager
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ScenarioManager
   scenario_manager
   
   ScenarioManager.__init__
   scenario_manager
   
   ScenarioManager.create_scenario
   scenario_manager
   
   ScenarioManager.save_scenario
   scenario_manager
   
   ScenarioManager.load_scenario
   scenario_manager
   
   ScenarioManager.list_scenarios
   scenario_manager
   
   ScenarioManager._load_all_scenarios
   scenario_manager
   
   ScenarioManager.validate_scenario
   scenario_manager
   
   ScenarioManager.check_target_health
   scenario_manager
   
   ScenarioManager._estimate_duration
   scenario_manager
   
   ScenarioManager.create_owasp_webgoat_scenarios
   scenario_manager
   
   ScenarioManager.create_juice_shop_scenarios
   scenario_manager
   
   ScenarioManager._create_sql_injection_plan_easy
   scenario_manager
   
   ScenarioManager._create_sql_injection_plan_medium
   scenario_manager
   
   ScenarioManager._create_xss_plan_easy
   scenario_manager
   
   ScenarioManager._create_ssrf_plan_medium
   scenario_manager
   
   ScenarioManager._create_juice_shop_sql_login_plan
   scenario_manager
   
   ScenarioManager._create_juice_shop_xss_dom_plan
   scenario_manager
   
   ScenarioManager.get_training_curriculum
   scenario_manager
   
   ScenarioManager.export_scenarios
   scenario_manager
   
   ScenarioManager.get_statistics
   scenario_manager
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

### Flow 63

- **長度**: 2 步
- **起點**: learning_adapter
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   LearningAdapter
   learning_adapter
   
   LearningAdapter.__init__
   learning_adapter
   
   LearningAdapter.learn_from_experience
   learning_adapter
   
   LearningAdapter.train_model
   learning_adapter
   
   LearningAdapter.retrieve_knowledge
   learning_adapter
   
   LearningAdapter.run_training_session
   learning_adapter
   
   LearningAdapter.get_learning_status
   learning_adapter
   
   LearningAdapter.get_statistics
   learning_adapter
   
   LearningAdapter.save_state
   learning_adapter
   - 模組: 任務規劃模組

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

### Flow 72

- **長度**: 2 步
- **起點**: tool_selector
- **終點**: tool_selector
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ServiceType
   tool_selector
   
   ToolDecision
   tool_selector
   
   ToolDecision.__repr__
   tool_selector
   
   ToolSelector
   tool_selector
   
   ToolSelector.__init__
   tool_selector
   
   ToolSelector.select_tool
   tool_selector
   
   ToolSelector._select_service_type
   tool_selector
   
   ToolSelector._determine_endpoint_and_function
   tool_selector
   
   ToolSelector._prepare_parameters
   tool_selector
   
   ToolSelector._determine_routing_key
   tool_selector
   - 模組: 任務規劃模組

2. **程式組件**
   ServiceType
   tool_selector
   
   ToolDecision
   tool_selector
   
   ToolDecision.__repr__
   tool_selector
   
   ToolSelector
   tool_selector
   
   ToolSelector.__init__
   tool_selector
   
   ToolSelector.select_tool
   tool_selector
   
   ToolSelector._select_service_type
   tool_selector
   
   ToolSelector._determine_endpoint_and_function
   tool_selector
   
   ToolSelector._prepare_parameters
   tool_selector
   
   ToolSelector._determine_routing_key
   tool_selector
   - 模組: 任務規劃模組

---

### Flow 96

- **長度**: 2 步
- **起點**: policy_manager
- **終點**: policy_manager
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   RiskRule
   policy_manager
   
   RiskLevel
   policy_manager
   
   PolicyManager
   policy_manager
   
   PolicyManager.__init__
   policy_manager
   
   PolicyManager._load_policy
   policy_manager
   
   PolicyManager._use_fallback_policy
   policy_manager
   
   PolicyManager.assess_risk
   policy_manager
   
   PolicyManager._evaluate_condition
   policy_manager
   
   PolicyManager._determine_risk_level
   policy_manager
   
   PolicyManager.reload_policy
   policy_manager
   
   PolicyManager.get_policy_info
   policy_manager
   - 模組: 任務規劃模組

2. **程式組件**
   RiskRule
   policy_manager
   
   RiskLevel
   policy_manager
   
   PolicyManager
   policy_manager
   
   PolicyManager.__init__
   policy_manager
   
   PolicyManager._load_policy
   policy_manager
   
   PolicyManager._use_fallback_policy
   policy_manager
   
   PolicyManager.assess_risk
   policy_manager
   
   PolicyManager._evaluate_condition
   policy_manager
   
   PolicyManager._determine_risk_level
   policy_manager
   
   PolicyManager.reload_policy
   policy_manager
   
   PolicyManager.get_policy_info
   policy_manager
   - 模組: 任務規劃模組

---

### Flow 132

- **長度**: 2 步
- **起點**: mode_manager
- **終點**: execution_status_monitor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ModeManager
   mode_manager
   
   ModeManager.__init__
   mode_manager
   
   ModeManager.get_mode
   mode_manager
   
   ModeManager.set_mode
   mode_manager
   
   ModeManager._load_from_config
   mode_manager
   
   ModeManager._save_to_config
   mode_manager
   
   ModeManager.reset
   mode_manager
   
   get_mode_manager
   mode_manager
   - 模組: 任務規劃模組

2. **程式組件**
   EnvironmentType
   execution_status_monitor
   
   ExecutionContext
   execution_status_monitor
   
   ExecutionContext.__post_init__
   execution_status_monitor
   
   ExecutionContext.to_dict
   execution_status_monitor
   
   ExecutionContext.from_dict
   execution_status_monitor
   
   ExecutionMonitor
   execution_status_monitor
   
   ExecutionMonitor.__init__
   execution_status_monitor
   
   ExecutionMonitor.start_task_execution
   execution_status_monitor
   
   ExecutionMonitor.complete_task_execution
   execution_status_monitor
   
   ExecutionMonitor.record_error
   execution_status_monitor
   
   ExecutionMonitor.record_step
   execution_status_monitor
   
   ExecutionMonitor.record_decision_point
   execution_status_monitor
   
   ExecutionMonitor.record_tool_invocation
   execution_status_monitor
   
   ExecutionMonitor.get_task_traces
   execution_status_monitor
   
   ExecutionMonitor.get_task_errors
   execution_status_monitor
   
   ExecutionStatusMonitor
   execution_status_monitor
   
   ExecutionStatusMonitor.__init__
   execution_status_monitor
   
   ExecutionStatusMonitor.record_worker_heartbeat
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_start
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_completion
   execution_status_monitor
   
   ExecutionStatusMonitor.get_system_health
   execution_status_monitor
   
   ExecutionStatusMonitor.check_sla_violations
   execution_status_monitor
   
   ExecutionStatusMonitor._get_recent_alerts
   execution_status_monitor
   
   ExecutionStatusMonitor.add_alert
   execution_status_monitor
   
   ExecutionStatusMonitor.start_monitoring
   execution_status_monitor
   - 模組: 任務規劃模組

---

### Flow 210

- **長度**: 2 步
- **起點**: task_executor
- **終點**: execution_status_monitor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   task_executor
   
   TaskExecutor
   task_executor
   
   TaskExecutor.__init__
   task_executor
   
   TaskExecutor.execute_task
   task_executor
   
   TaskExecutor._execute_by_service_type
   task_executor
   
   TaskExecutor._execute_scan_service
   task_executor
   
   TaskExecutor._call_capability_dynamically
   task_executor
   
   TaskExecutor._execute_function_service
   task_executor
   
   TaskExecutor._execute_integration_service
   task_executor
   
   TaskExecutor._execute_core_service
   task_executor
   
   TaskExecutor._infer_capability_name
   task_executor
   - 模組: 任務規劃模組

2. **程式組件**
   EnvironmentType
   execution_status_monitor
   
   ExecutionContext
   execution_status_monitor
   
   ExecutionContext.__post_init__
   execution_status_monitor
   
   ExecutionContext.to_dict
   execution_status_monitor
   
   ExecutionContext.from_dict
   execution_status_monitor
   
   ExecutionMonitor
   execution_status_monitor
   
   ExecutionMonitor.__init__
   execution_status_monitor
   
   ExecutionMonitor.start_task_execution
   execution_status_monitor
   
   ExecutionMonitor.complete_task_execution
   execution_status_monitor
   
   ExecutionMonitor.record_error
   execution_status_monitor
   
   ExecutionMonitor.record_step
   execution_status_monitor
   
   ExecutionMonitor.record_decision_point
   execution_status_monitor
   
   ExecutionMonitor.record_tool_invocation
   execution_status_monitor
   
   ExecutionMonitor.get_task_traces
   execution_status_monitor
   
   ExecutionMonitor.get_task_errors
   execution_status_monitor
   
   ExecutionStatusMonitor
   execution_status_monitor
   
   ExecutionStatusMonitor.__init__
   execution_status_monitor
   
   ExecutionStatusMonitor.record_worker_heartbeat
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_start
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_completion
   execution_status_monitor
   
   ExecutionStatusMonitor.get_system_health
   execution_status_monitor
   
   ExecutionStatusMonitor.check_sla_violations
   execution_status_monitor
   
   ExecutionStatusMonitor._get_recent_alerts
   execution_status_monitor
   
   ExecutionStatusMonitor.add_alert
   execution_status_monitor
   
   ExecutionStatusMonitor.start_monitoring
   execution_status_monitor
   - 模組: 任務規劃模組

---

### Flow 213

- **長度**: 2 步
- **起點**: ai_executor_interface
- **終點**: ai_executor_interface
- **主要模組**: 任務規劃模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   SimpleExecutionResult
   ai_executor_interface
   
   AIExecutorInterface
   ai_executor_interface
   
   AIExecutorInterface.__init__
   ai_executor_interface
   
   AIExecutorInterface._ensure_initialized
   ai_executor_interface
   
   AIExecutorInterface.execute
   ai_executor_interface
   
   AIExecutorInterface.execute_batch
   ai_executor_interface
   
   AIExecutorInterface.get_available_capabilities
   ai_executor_interface
   
   AIExecutorInterface.get_execution_status
   ai_executor_interface
   
   AIExecutorInterface.get_execution_history
   ai_executor_interface
   
   AIExecutorInterface.clear_history
   ai_executor_interface
   
   get_executor
   ai_executor_interface
   
   quick_execute
   ai_executor_interface
   
   list_capabilities
   ai_executor_interface
   - 模組: 任務規劃模組

2. **AI組件**
   SimpleExecutionResult
   ai_executor_interface
   
   AIExecutorInterface
   ai_executor_interface
   
   AIExecutorInterface.__init__
   ai_executor_interface
   
   AIExecutorInterface._ensure_initialized
   ai_executor_interface
   
   AIExecutorInterface.execute
   ai_executor_interface
   
   AIExecutorInterface.execute_batch
   ai_executor_interface
   
   AIExecutorInterface.get_available_capabilities
   ai_executor_interface
   
   AIExecutorInterface.get_execution_status
   ai_executor_interface
   
   AIExecutorInterface.get_execution_history
   ai_executor_interface
   
   AIExecutorInterface.clear_history
   ai_executor_interface
   
   get_executor
   ai_executor_interface
   
   quick_execute
   ai_executor_interface
   
   list_capabilities
   ai_executor_interface
   - 模組: 任務規劃模組

---

### Flow 262

- **長度**: 2 步
- **起點**: mode_manager
- **終點**: mode_manager
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ModeManager
   mode_manager
   
   ModeManager.__init__
   mode_manager
   
   ModeManager.get_mode
   mode_manager
   
   ModeManager.set_mode
   mode_manager
   
   ModeManager._load_from_config
   mode_manager
   
   ModeManager._save_to_config
   mode_manager
   
   ModeManager.reset
   mode_manager
   
   get_mode_manager
   mode_manager
   - 模組: 任務規劃模組

2. **程式組件**
   ModeManager
   mode_manager
   
   ModeManager.__init__
   mode_manager
   
   ModeManager.get_mode
   mode_manager
   
   ModeManager.set_mode
   mode_manager
   
   ModeManager._load_from_config
   mode_manager
   
   ModeManager._save_to_config
   mode_manager
   
   ModeManager.reset
   mode_manager
   
   get_mode_manager
   mode_manager
   - 模組: 任務規劃模組

---

### Flow 264

- **長度**: 2 步
- **起點**: command_builder
- **終點**: command_builder
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   CommandBuildError
   command_builder
   
   CommandBuildError.__init__
   command_builder
   
   CommandBuilder
   command_builder
   
   CommandBuilder.__init__
   command_builder
   
   CommandBuilder.build_command
   command_builder
   
   CommandBuilder.preview_parameters
   command_builder
   - 模組: 任務規劃模組

2. **程式組件**
   CommandBuildError
   command_builder
   
   CommandBuildError.__init__
   command_builder
   
   CommandBuilder
   command_builder
   
   CommandBuilder.__init__
   command_builder
   
   CommandBuilder.build_command
   command_builder
   
   CommandBuilder.preview_parameters
   command_builder
   - 模組: 任務規劃模組

---

### Flow 266

- **長度**: 2 步
- **起點**: external_loop_connector
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

### Flow 373

- **長度**: 2 步
- **起點**: plan_builder
- **終點**: capability_matcher
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   PlanBuilder
   plan_builder
   
   PlanBuilder.__init__
   plan_builder
   
   PlanBuilder.plan_attack
   plan_builder
   
   PlanBuilder._encode_target_for_neural
   plan_builder
   
   PlanBuilder._build_plan_from_neural_decision
   plan_builder
   
   PlanBuilder._calculate_plan_confidence
   plan_builder
   
   PlanBuilder.build_plan_generation_prompt
   plan_builder
   
   PlanBuilder._build_base_prompt
   plan_builder
   
   PlanBuilder._build_rag_section
   plan_builder
   
   PlanBuilder._build_historical_section
   plan_builder
   
   PlanBuilder._categorize_experiences
   plan_builder
   
   PlanBuilder._build_experience_stats
   plan_builder
   
   PlanBuilder._build_success_cases
   plan_builder
   
   PlanBuilder._build_failure_lessons
   plan_builder
   
   PlanBuilder._build_constraints_section
   plan_builder
   
   PlanBuilder._build_output_structure
   plan_builder
   
   PlanBuilder._analyze_feedback_for_planning
   plan_builder
   
   PlanBuilder._get_default_feedback_insights
   plan_builder
   
   PlanBuilder._find_similar_target_feedbacks
   plan_builder
   
   PlanBuilder._extract_domain
   plan_builder
   
   PlanBuilder._calculate_strategy_success_rates
   plan_builder
   
   PlanBuilder._identify_success_patterns
   plan_builder
   
   PlanBuilder._identify_failure_patterns
   plan_builder
   
   PlanBuilder._calculate_waf_risk
   plan_builder
   
   PlanBuilder._calculate_error_probability
   plan_builder
   
   PlanBuilder._calculate_avg_success_rate
   plan_builder
   
   PlanBuilder._calculate_avg_execution_time
   plan_builder
   
   PlanBuilder._recommend_best_strategy
   plan_builder
   
   PlanBuilder._generate_planning_adjustments
   plan_builder
   
   PlanBuilder._define_success_criteria
   plan_builder
   
   PlanBuilder._get_default_success_criteria
   plan_builder
   - 模組: 任務規劃模組

2. **程式組件**
   CapabilityMatcher
   capability_matcher
   
   CapabilityMatcher.__init__
   capability_matcher
   
   CapabilityMatcher._load_capabilities
   capability_matcher
   
   CapabilityMatcher.match_intent
   capability_matcher
   
   CapabilityMatcher.format_command
   capability_matcher
   
   CapabilityMatcher.get_flow_by_id
   capability_matcher
   - 模組: 任務規劃模組

---

### Flow 391

- **長度**: 2 步
- **起點**: plan_builder
- **終點**: unified_executor
- **主要模組**: 任務規劃模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   PlanBuilder
   plan_builder
   
   PlanBuilder.__init__
   plan_builder
   
   PlanBuilder.plan_attack
   plan_builder
   
   PlanBuilder._encode_target_for_neural
   plan_builder
   
   PlanBuilder._build_plan_from_neural_decision
   plan_builder
   
   PlanBuilder._calculate_plan_confidence
   plan_builder
   
   PlanBuilder.build_plan_generation_prompt
   plan_builder
   
   PlanBuilder._build_base_prompt
   plan_builder
   
   PlanBuilder._build_rag_section
   plan_builder
   
   PlanBuilder._build_historical_section
   plan_builder
   
   PlanBuilder._categorize_experiences
   plan_builder
   
   PlanBuilder._build_experience_stats
   plan_builder
   
   PlanBuilder._build_success_cases
   plan_builder
   
   PlanBuilder._build_failure_lessons
   plan_builder
   
   PlanBuilder._build_constraints_section
   plan_builder
   
   PlanBuilder._build_output_structure
   plan_builder
   
   PlanBuilder._analyze_feedback_for_planning
   plan_builder
   
   PlanBuilder._get_default_feedback_insights
   plan_builder
   
   PlanBuilder._find_similar_target_feedbacks
   plan_builder
   
   PlanBuilder._extract_domain
   plan_builder
   
   PlanBuilder._calculate_strategy_success_rates
   plan_builder
   
   PlanBuilder._identify_success_patterns
   plan_builder
   
   PlanBuilder._identify_failure_patterns
   plan_builder
   
   PlanBuilder._calculate_waf_risk
   plan_builder
   
   PlanBuilder._calculate_error_probability
   plan_builder
   
   PlanBuilder._calculate_avg_success_rate
   plan_builder
   
   PlanBuilder._calculate_avg_execution_time
   plan_builder
   
   PlanBuilder._recommend_best_strategy
   plan_builder
   
   PlanBuilder._generate_planning_adjustments
   plan_builder
   
   PlanBuilder._define_success_criteria
   plan_builder
   
   PlanBuilder._get_default_success_criteria
   plan_builder
   - 模組: 任務規劃模組

2. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

---

## 核心能力模組 (core_capabilities)

包含 14 條數據流

### Flow 48

- **長度**: 3 步
- **起點**: capability_orchestrator
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: 混合組件

**執行路徑**:

1. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

2. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

3. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 49

- **長度**: 2 步
- **起點**: capability_orchestrator
- **終點**: risk_policy_manager
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   TaskRequirement
   capability_orchestrator
   
   CapabilityPlan
   capability_orchestrator
   
   ExecutionResult
   capability_orchestrator
   
   CapabilityOrchestrator
   capability_orchestrator
   
   CapabilityOrchestrator.__init__
   capability_orchestrator
   
   CapabilityOrchestrator.plan
   capability_orchestrator
   
   CapabilityOrchestrator._query_relevant_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._filter_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator.group_capabilities_by_aiva_module
   capability_orchestrator
   
   CapabilityOrchestrator._fallback_capability_search
   capability_orchestrator
   
   CapabilityOrchestrator._filter_available_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._select_best_capabilities
   capability_orchestrator
   
   CapabilityOrchestrator._calculate_capability_score
   capability_orchestrator
   
   CapabilityOrchestrator._generate_execution_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_scan_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_attack_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._order_comprehensive_sequence
   capability_orchestrator
   
   CapabilityOrchestrator._capabilities_to_cli_commands
   capability_orchestrator
   
   CapabilityOrchestrator._assess_risk_level
   capability_orchestrator
   
   CapabilityOrchestrator._generate_reasoning
   capability_orchestrator
   
   CapabilityOrchestrator.execute
   capability_orchestrator
   
   CapabilityOrchestrator._extract_issues_from_outputs
   capability_orchestrator
   
   CapabilityOrchestrator.learn_from_execution
   capability_orchestrator
   
   quick_plan_and_execute
   capability_orchestrator
   - 模組: 認知核心模組

2. **程式組件**
   RiskPolicyManager
   risk_policy_manager
   
   RiskPolicyManager.__init__
   risk_policy_manager
   
   RiskPolicyManager._load_policy
   risk_policy_manager
   
   RiskPolicyManager.assess_risk
   risk_policy_manager
   
   RiskPolicyManager._evaluate_condition
   risk_policy_manager
   
   RiskPolicyManager._determine_risk_level
   risk_policy_manager
   
   RiskPolicyManager.get_mitigation_actions
   risk_policy_manager
   - 模組: 核心能力模組

---

### Flow 94

- **長度**: 2 步
- **起點**: analysis_engine
- **終點**: analysis_engine
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AnalysisType
   analysis_engine
   
   IndexingConfig
   analysis_engine
   
   CacheManager
   analysis_engine
   
   CacheManager.__init__
   analysis_engine
   
   CacheManager._load_cache_index
   analysis_engine
   
   CacheManager.get_file_hash
   analysis_engine
   
   CacheManager.is_cached
   analysis_engine
   
   CacheManager.update_cache
   analysis_engine
   
   CacheManager.save_cache_index
   analysis_engine
   
   AIAnalysisResult
   analysis_engine
   
   CodeChunk
   analysis_engine
   
   CodeChunk.__post_init__
   analysis_engine
   
   AIAnalysisEngine
   analysis_engine
   
   AIAnalysisEngine.__init__
   analysis_engine
   
   AIAnalysisEngine.initialize
   analysis_engine
   
   AIAnalysisEngine._extract_code_features
   analysis_engine
   
   AIAnalysisEngine._calculate_cyclomatic_complexity
   analysis_engine
   
   AIAnalysisEngine._calculate_nesting_depth
   analysis_engine
   
   AIAnalysisEngine._extract_security_features
   analysis_engine
   
   AIAnalysisEngine._extract_semantic_features
   analysis_engine
   
   AIAnalysisEngine.analyze_code
   analysis_engine
   
   AIAnalysisEngine.index_codebase
   analysis_engine
   
   AIAnalysisEngine._collect_python_files
   analysis_engine
   
   AIAnalysisEngine._filter_files_for_indexing
   analysis_engine
   
   AIAnalysisEngine._batch_index_files
   analysis_engine
   
   AIAnalysisEngine._process_file_batch
   analysis_engine
   
   AIAnalysisEngine._safe_index_file
   analysis_engine
   
   AIAnalysisEngine._index_file_content
   analysis_engine
   
   AIAnalysisEngine._extract_chunks_from_ast
   analysis_engine
   
   AIAnalysisEngine._extract_node_content
   analysis_engine
   
   AIAnalysisEngine._extract_by_line_numbers
   analysis_engine
   
   AIAnalysisEngine._handle_unparseable_file
   analysis_engine
   
   AIAnalysisEngine._add_code_chunk
   analysis_engine
   
   AIAnalysisEngine._extract_analysis_keywords
   analysis_engine
   
   AIAnalysisEngine.search_code_chunks
   analysis_engine
   
   AIAnalysisEngine._extract_query_keywords
   analysis_engine
   
   AIAnalysisEngine._calculate_chunk_scores
   analysis_engine
   
   AIAnalysisEngine._apply_exact_matches
   analysis_engine
   
   AIAnalysisEngine._apply_partial_matches
   analysis_engine
   
   AIAnalysisEngine._format_search_results
   analysis_engine
   
   AIAnalysisEngine._get_indexing_stats
   analysis_engine
   
   AIAnalysisEngine._create_failed_results
   analysis_engine
   
   AIAnalysisEngine._perform_ai_analysis
   analysis_engine
   
   AIAnalysisEngine._generate_findings
   analysis_engine
   
   AIAnalysisEngine._generate_recommendations
   analysis_engine
   
   AIAnalysisEngine._calculate_risk_level
   analysis_engine
   
   AIAnalysisEngine._generate_explanation
   analysis_engine
   
   AIAnalysisEngine.get_analysis_summary
   analysis_engine
   - 模組: 核心能力模組

2. **程式組件**
   AnalysisType
   analysis_engine
   
   IndexingConfig
   analysis_engine
   
   CacheManager
   analysis_engine
   
   CacheManager.__init__
   analysis_engine
   
   CacheManager._load_cache_index
   analysis_engine
   
   CacheManager.get_file_hash
   analysis_engine
   
   CacheManager.is_cached
   analysis_engine
   
   CacheManager.update_cache
   analysis_engine
   
   CacheManager.save_cache_index
   analysis_engine
   
   AIAnalysisResult
   analysis_engine
   
   CodeChunk
   analysis_engine
   
   CodeChunk.__post_init__
   analysis_engine
   
   AIAnalysisEngine
   analysis_engine
   
   AIAnalysisEngine.__init__
   analysis_engine
   
   AIAnalysisEngine.initialize
   analysis_engine
   
   AIAnalysisEngine._extract_code_features
   analysis_engine
   
   AIAnalysisEngine._calculate_cyclomatic_complexity
   analysis_engine
   
   AIAnalysisEngine._calculate_nesting_depth
   analysis_engine
   
   AIAnalysisEngine._extract_security_features
   analysis_engine
   
   AIAnalysisEngine._extract_semantic_features
   analysis_engine
   
   AIAnalysisEngine.analyze_code
   analysis_engine
   
   AIAnalysisEngine.index_codebase
   analysis_engine
   
   AIAnalysisEngine._collect_python_files
   analysis_engine
   
   AIAnalysisEngine._filter_files_for_indexing
   analysis_engine
   
   AIAnalysisEngine._batch_index_files
   analysis_engine
   
   AIAnalysisEngine._process_file_batch
   analysis_engine
   
   AIAnalysisEngine._safe_index_file
   analysis_engine
   
   AIAnalysisEngine._index_file_content
   analysis_engine
   
   AIAnalysisEngine._extract_chunks_from_ast
   analysis_engine
   
   AIAnalysisEngine._extract_node_content
   analysis_engine
   
   AIAnalysisEngine._extract_by_line_numbers
   analysis_engine
   
   AIAnalysisEngine._handle_unparseable_file
   analysis_engine
   
   AIAnalysisEngine._add_code_chunk
   analysis_engine
   
   AIAnalysisEngine._extract_analysis_keywords
   analysis_engine
   
   AIAnalysisEngine.search_code_chunks
   analysis_engine
   
   AIAnalysisEngine._extract_query_keywords
   analysis_engine
   
   AIAnalysisEngine._calculate_chunk_scores
   analysis_engine
   
   AIAnalysisEngine._apply_exact_matches
   analysis_engine
   
   AIAnalysisEngine._apply_partial_matches
   analysis_engine
   
   AIAnalysisEngine._format_search_results
   analysis_engine
   
   AIAnalysisEngine._get_indexing_stats
   analysis_engine
   
   AIAnalysisEngine._create_failed_results
   analysis_engine
   
   AIAnalysisEngine._perform_ai_analysis
   analysis_engine
   
   AIAnalysisEngine._generate_findings
   analysis_engine
   
   AIAnalysisEngine._generate_recommendations
   analysis_engine
   
   AIAnalysisEngine._calculate_risk_level
   analysis_engine
   
   AIAnalysisEngine._generate_explanation
   analysis_engine
   
   AIAnalysisEngine.get_analysis_summary
   analysis_engine
   - 模組: 核心能力模組

---

### Flow 100

- **長度**: 2 步
- **起點**: capability_registry
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: 混合組件

**執行路徑**:

1. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

2. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 187

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: bizlogic_scanner
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   _build_scan_options
   bizlogic_scanner
   
   _print_scan_header
   bizlogic_scanner
   
   _print_vulnerability_details
   bizlogic_scanner
   
   scan_target
   bizlogic_scanner
   
   _check_target_availability
   bizlogic_scanner
   
   _detect_available_targets
   bizlogic_scanner
   
   _save_scan_report
   bizlogic_scanner
   
   main
   bizlogic_scanner
   - 模組: 核心能力模組

---

### Flow 192

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: ai_menu
- **主要模組**: 核心能力模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **AI組件**
   AIVAIntelligentMenu
   ai_menu
   
   AIVAIntelligentMenu.__init__
   ai_menu
   
   AIVAIntelligentMenu.print_banner
   ai_menu
   
   AIVAIntelligentMenu.show_main_menu
   ai_menu
   
   AIVAIntelligentMenu.handle_ai_conversation
   ai_menu
   
   AIVAIntelligentMenu.handle_capability_search
   ai_menu
   
   AIVAIntelligentMenu.handle_one_click_attack
   ai_menu
   
   AIVAIntelligentMenu.handle_workflow_recommendation
   ai_menu
   
   AIVAIntelligentMenu.handle_system_stats
   ai_menu
   
   AIVAIntelligentMenu.handle_health_check
   ai_menu
   
   AIVAIntelligentMenu.handle_sync_rag
   ai_menu
   
   AIVAIntelligentMenu._execute_capability
   ai_menu
   
   AIVAIntelligentMenu._execute_workflow
   ai_menu
   
   AIVAIntelligentMenu.run
   ai_menu
   
   AIVAIntelligentMenu.show_help
   ai_menu
   
   main
   ai_menu
   - 模組: 核心能力模組

---

### Flow 209

- **長度**: 2 步
- **起點**: scan_result_processor
- **終點**: to_functions
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ScanResultProcessor
   scan_result_processor
   
   ScanResultProcessor.__init__
   scan_result_processor
   
   ScanResultProcessor.stage_1_ingest_data
   scan_result_processor
   
   ScanResultProcessor.stage_2_analyze_surface
   scan_result_processor
   
   ScanResultProcessor.stage_3_generate_strategy
   scan_result_processor
   
   ScanResultProcessor.stage_4_adjust_strategy
   scan_result_processor
   
   ScanResultProcessor.stage_5_generate_tasks
   scan_result_processor
   
   ScanResultProcessor.stage_6_dispatch_tasks
   scan_result_processor
   
   ScanResultProcessor.stage_7_monitor_execution
   scan_result_processor
   
   ScanResultProcessor.process
   scan_result_processor
   
   ScanResultProcessor.process_phase0
   scan_result_processor
   
   ScanResultProcessor._analyze_phase0_and_decide
   scan_result_processor
   
   ScanResultProcessor._fallback_rule_decision
   scan_result_processor
   
   ScanResultProcessor._select_engines_for_phase1
   scan_result_processor
   - 模組: 核心能力模組

2. **程式組件**
   to_function_message
   to_functions
   - 模組: 核心能力模組

---

### Flow 211

- **長度**: 3 步
- **起點**: task_executor
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: 混合組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   task_executor
   
   TaskExecutor
   task_executor
   
   TaskExecutor.__init__
   task_executor
   
   TaskExecutor.execute_task
   task_executor
   
   TaskExecutor._execute_by_service_type
   task_executor
   
   TaskExecutor._execute_scan_service
   task_executor
   
   TaskExecutor._call_capability_dynamically
   task_executor
   
   TaskExecutor._execute_function_service
   task_executor
   
   TaskExecutor._execute_integration_service
   task_executor
   
   TaskExecutor._execute_core_service
   task_executor
   
   TaskExecutor._infer_capability_name
   task_executor
   - 模組: 任務規劃模組

2. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

3. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 226

- **長度**: 2 步
- **起點**: aiva_cli
- **終點**: aiva_cli
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   load_flow_definitions
   aiva_cli
   
   create_flow_command
   aiva_cli
   
   register_all_flow_commands
   aiva_cli
   
   aiva
   aiva_cli
   
   run
   aiva_cli
   
   query
   aiva_cli
   
   train
   aiva_cli
   
   scan
   aiva_cli
   
   status
   aiva_cli
   
   health
   aiva_cli
   
   list_flows
   aiva_cli
   
   show_flow_statistics
   aiva_cli
   
   show_flows_by_endpoint_module
   aiva_cli
   - 模組: 核心能力模組

2. **程式組件**
   load_flow_definitions
   aiva_cli
   
   create_flow_command
   aiva_cli
   
   register_all_flow_commands
   aiva_cli
   
   aiva
   aiva_cli
   
   run
   aiva_cli
   
   query
   aiva_cli
   
   train
   aiva_cli
   
   scan
   aiva_cli
   
   status
   aiva_cli
   
   health
   aiva_cli
   
   list_flows
   aiva_cli
   
   show_flow_statistics
   aiva_cli
   
   show_flows_by_endpoint_module
   aiva_cli
   - 模組: 核心能力模組

---

### Flow 250

- **長度**: 2 步
- **起點**: attack_coordinator
- **終點**: two_phase_scan_orchestrator
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackCoordinator
   attack_coordinator
   
   AttackCoordinator.__init__
   attack_coordinator
   
   AttackCoordinator._init_cli_executor
   attack_coordinator
   
   AttackCoordinator._execute_cli_command
   attack_coordinator
   
   AttackCoordinator.detect_vulnerabilities
   attack_coordinator
   
   AttackCoordinator.coordinate_multilang
   attack_coordinator
   
   AttackCoordinator.execute_attack
   attack_coordinator
   
   AttackCoordinator.execute_two_phase_scan
   attack_coordinator
   
   AttackCoordinator.query_capabilities
   attack_coordinator
   
   AttackCoordinator.unified_attack
   attack_coordinator
   
   AttackCoordinator.process_scan_command
   attack_coordinator
   - 模組: 任務規劃模組

2. **程式組件**
   TwoPhaseOrchestratorError
   two_phase_scan_orchestrator
   
   Phase0TimeoutError
   two_phase_scan_orchestrator
   
   Phase1TimeoutError
   two_phase_scan_orchestrator
   
   AIDecisionError
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.__init__
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.execute_scan_with_context
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator.execute_two_phase_scan
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._execute_phase0
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._execute_phase1
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._analyze_phase0_and_decide
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._fallback_decision_rules
   two_phase_scan_orchestrator
   
   TwoPhaseScanOrchestrator._select_engines_for_phase1
   two_phase_scan_orchestrator
   - 模組: 核心能力模組

---

### Flow 279

- **長度**: 2 步
- **起點**: assistant
- **終點**: assistant
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

2. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

---

### Flow 298

- **長度**: 2 步
- **起點**: skill_graph
- **終點**: capability_registry
- **主要模組**: 核心能力模組
- **主要組件類型**: 混合組件

**執行路徑**:

1. **程式組件**
   SkillNode
   skill_graph
   
   SkillEdge
   skill_graph
   
   SkillPath
   skill_graph
   
   SkillGraphBuilder
   skill_graph
   
   SkillGraphBuilder.__init__
   skill_graph
   
   SkillGraphBuilder.build_graph
   skill_graph
   
   SkillGraphBuilder._extract_success_rate
   skill_graph
   
   SkillGraphBuilder._extract_usage_count
   skill_graph
   
   SkillGraphBuilder._build_node_metadata
   skill_graph
   
   SkillGraphBuilder._create_skill_nodes
   skill_graph
   
   SkillGraphBuilder._analyze_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_prerequisite_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_tag_similarity_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_language_ecosystem_relationships
   skill_graph
   
   SkillGraphBuilder._analyze_topic_relationships
   skill_graph
   
   SkillGraphBuilder._check_io_compatibility
   skill_graph
   
   SkillGraphBuilder._analyze_io_relationships
   skill_graph
   
   SkillGraphBuilder._is_compatible_io
   skill_graph
   
   SkillGraphBuilder._build_networkx_graph
   skill_graph
   
   SkillGraphAnalyzer
   skill_graph
   
   SkillGraphAnalyzer.__init__
   skill_graph
   
   SkillGraphAnalyzer.find_optimal_path
   skill_graph
   
   SkillGraphAnalyzer._find_goal_capabilities
   skill_graph
   
   SkillGraphAnalyzer._create_skill_path
   skill_graph
   
   SkillGraphAnalyzer.get_capability_recommendations
   skill_graph
   
   SkillGraphAnalyzer.analyze_capability_centrality
   skill_graph
   
   AIVASkillGraph
   skill_graph
   
   AIVASkillGraph.__init__
   skill_graph
   
   AIVASkillGraph.initialize
   skill_graph
   
   AIVASkillGraph.rebuild_if_needed
   skill_graph
   
   AIVASkillGraph.find_execution_path
   skill_graph
   
   AIVASkillGraph.get_recommendations
   skill_graph
   
   AIVASkillGraph.analyze_centrality
   skill_graph
   
   AIVASkillGraph.get_graph_statistics
   skill_graph
   - 模組: 認知核心模組

2. **混合組件**
   CapabilityInfo
   capability_registry
   
   CapabilityInfo.__init__
   capability_registry
   
   CapabilityInfo.to_dict
   capability_registry
   
   CapabilityInfo.from_capability_record
   capability_registry
   
   CapabilityRegistry
   capability_registry
   
   CapabilityRegistry.__new__
   capability_registry
   
   CapabilityRegistry.__init__
   capability_registry
   
   CapabilityRegistry.load_from_exploration
   capability_registry
   
   CapabilityRegistry.register_capability
   capability_registry
   
   CapabilityRegistry.get_capability
   capability_registry
   
   CapabilityRegistry.list_capabilities
   capability_registry
   
   CapabilityRegistry.list_capabilities_async
   capability_registry
   
   CapabilityRegistry.list_modules
   capability_registry
   
   CapabilityRegistry.search_capabilities
   capability_registry
   
   CapabilityRegistry.get_statistics
   capability_registry
   
   CapabilityRegistry.clear
   capability_registry
   
   get_capability_registry
   capability_registry
   
   initialize_capability_registry
   capability_registry
   - 模組: 核心能力模組

---

### Flow 321

- **長度**: 3 步
- **起點**: ai_menu
- **終點**: assistant
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   AIVAIntelligentMenu
   ai_menu
   
   AIVAIntelligentMenu.__init__
   ai_menu
   
   AIVAIntelligentMenu.print_banner
   ai_menu
   
   AIVAIntelligentMenu.show_main_menu
   ai_menu
   
   AIVAIntelligentMenu.handle_ai_conversation
   ai_menu
   
   AIVAIntelligentMenu.handle_capability_search
   ai_menu
   
   AIVAIntelligentMenu.handle_one_click_attack
   ai_menu
   
   AIVAIntelligentMenu.handle_workflow_recommendation
   ai_menu
   
   AIVAIntelligentMenu.handle_system_stats
   ai_menu
   
   AIVAIntelligentMenu.handle_health_check
   ai_menu
   
   AIVAIntelligentMenu.handle_sync_rag
   ai_menu
   
   AIVAIntelligentMenu._execute_capability
   ai_menu
   
   AIVAIntelligentMenu._execute_workflow
   ai_menu
   
   AIVAIntelligentMenu.run
   ai_menu
   
   AIVAIntelligentMenu.show_help
   ai_menu
   
   main
   ai_menu
   - 模組: 核心能力模組

2. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

3. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

---

### Flow 395

- **長度**: 3 步
- **起點**: attack_coordinator
- **終點**: task_context
- **主要模組**: 核心能力模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackCoordinator
   attack_coordinator
   
   AttackCoordinator.__init__
   attack_coordinator
   
   AttackCoordinator._init_cli_executor
   attack_coordinator
   
   AttackCoordinator._execute_cli_command
   attack_coordinator
   
   AttackCoordinator.detect_vulnerabilities
   attack_coordinator
   
   AttackCoordinator.coordinate_multilang
   attack_coordinator
   
   AttackCoordinator.execute_attack
   attack_coordinator
   
   AttackCoordinator.execute_two_phase_scan
   attack_coordinator
   
   AttackCoordinator.query_capabilities
   attack_coordinator
   
   AttackCoordinator.unified_attack
   attack_coordinator
   
   AttackCoordinator.process_scan_command
   attack_coordinator
   - 模組: 任務規劃模組

2. **程式組件**
   TaskIntent
   task_context
   
   TaskPhase
   task_context
   
   StealthLevel
   task_context
   
   TaskConstraints
   task_context
   
   AIDecisionParams
   task_context
   
   TaskContext
   task_context
   
   ScanTaskContext
   task_context
   
   AttackTaskContext
   task_context
   
   AnalysisTaskContext
   task_context
   
   create_scan_context
   task_context
   
   create_attack_context
   task_context
   
   parse_user_input_to_context
   task_context
   - 模組: 核心能力模組

3. **程式組件**
   TaskIntent
   task_context
   
   TaskPhase
   task_context
   
   StealthLevel
   task_context
   
   TaskConstraints
   task_context
   
   AIDecisionParams
   task_context
   
   TaskContext
   task_context
   
   ScanTaskContext
   task_context
   
   AttackTaskContext
   task_context
   
   AnalysisTaskContext
   task_context
   
   create_scan_context
   task_context
   
   create_attack_context
   task_context
   
   parse_user_input_to_context
   task_context
   - 模組: 核心能力模組

---

## 服務骨幹模組 (service_backbone)

包含 39 條數據流

### Flow 6

- **長度**: 2 步
- **起點**: backends
- **終點**: models
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   StorageBackend
   backends
   
   StorageBackend.save_experience_sample
   backends
   
   StorageBackend.get_experience_samples
   backends
   
   StorageBackend.save_trace
   backends
   
   StorageBackend.get_traces_by_session
   backends
   
   StorageBackend.save_training_session
   backends
   
   StorageBackend.get_statistics
   backends
   
   SQLiteBackend
   backends
   
   SQLiteBackend.__init__
   backends
   
   SQLiteBackend.save_experience_sample
   backends
   
   SQLiteBackend.save_unified_experience_sample
   backends
   
   SQLiteBackend.get_experience_samples
   backends
   
   SQLiteBackend.save_trace
   backends
   
   SQLiteBackend.get_traces_by_session
   backends
   
   SQLiteBackend.save_training_session
   backends
   
   SQLiteBackend.get_statistics
   backends
   
   PostgreSQLBackend
   backends
   
   PostgreSQLBackend.__init__
   backends
   
   JSONLBackend
   backends
   
   JSONLBackend.__init__
   backends
   
   JSONLBackend.save_experience_sample
   backends
   
   JSONLBackend.get_experience_samples
   backends
   
   JSONLBackend.save_trace
   backends
   
   JSONLBackend.get_traces_by_session
   backends
   
   JSONLBackend.save_training_session
   backends
   
   JSONLBackend.get_statistics
   backends
   
   HybridBackend
   backends
   
   HybridBackend.__init__
   backends
   
   HybridBackend.save_experience_sample
   backends
   
   HybridBackend.get_experience_samples
   backends
   
   HybridBackend.save_trace
   backends
   
   HybridBackend.get_traces_by_session
   backends
   
   HybridBackend.save_training_session
   backends
   
   HybridBackend.get_statistics
   backends
   - 模組: 服務骨幹模組

2. **AI組件**
   EnhancedVulnerability
   models
   
   EnhancedFindingPayload
   models
   
   RiskFactor
   models
   
   RiskAssessmentContext
   models
   
   RiskAssessmentResult
   models
   
   EnhancedRiskAssessment
   models
   
   RiskTrendAnalysis
   models
   
   AttackPathNode
   models
   
   AttackPathEdge
   models
   
   AttackPathPayload
   models
   
   AttackPathRecommendation
   models
   
   EnhancedAttackPathNode
   models
   
   EnhancedAttackPath
   models
   
   VulnerabilityCorrelation
   models
   
   EnhancedVulnerabilityCorrelation
   models
   
   CodeLevelRootCause
   models
   
   TaskDependency
   models
   
   EnhancedTaskExecution
   models
   
   EnhancedTaskExecution.validate_task_id
   models
   
   TaskQueue
   models
   
   TestStrategy
   models
   
   EnhancedModuleStatus
   models
   
   SystemOrchestration
   models
   
   ExperienceSampleModel
   models
   
   TraceRecordModel
   models
   
   TrainingSessionModel
   models
   
   ModelCheckpointModel
   models
   
   KnowledgeEntryModel
   models
   
   ScenarioModel
   models
   
   CommandExecutionModel
   models
   - 模組: 服務骨幹模組

---

### Flow 11

- **長度**: 2 步
- **起點**: app
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ScanRequest
   app
   
   ScanResponse
   app
   
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
   
   start_scan
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

2. **程式組件**
   ScanRequest
   app
   
   ScanResponse
   app
   
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
   
   start_scan
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

### Flow 13

- **長度**: 2 步
- **起點**: enhanced_unified_caller
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionCallResult
   enhanced_unified_caller
   
   ModuleEndpoint
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.__init__
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.initialize
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._setup_protocol_adapters
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._init_endpoints
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_function
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_multiple_functions
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.health_check
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.cleanup
   enhanced_unified_caller
   
   get_unified_caller
   enhanced_unified_caller
   - 模組: 服務骨幹模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

---

### Flow 22

- **長度**: 2 步
- **起點**: message_broker
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

2. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 29

- **長度**: 2 步
- **起點**: monitoring
- **終點**: monitoring
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ComponentHealth
   monitoring
   
   Metric
   monitoring
   
   MetricsCollector
   monitoring
   
   MetricsCollector.__init__
   monitoring
   
   MetricsCollector.record_duration
   monitoring
   
   MetricsCollector.increment_counter
   monitoring
   
   MetricsCollector.set_gauge
   monitoring
   
   MetricsCollector._make_key
   monitoring
   
   MetricsCollector.get_metrics_summary
   monitoring
   
   MetricsCollector.update_component_health
   monitoring
   
   MetricsCollector.get_system_health_status
   monitoring
   
   MetricsCollector.check_component_freshness
   monitoring
   
   monitor_performance
   monitoring
   - 模組: 服務骨幹模組

2. **程式組件**
   ComponentHealth
   monitoring
   
   Metric
   monitoring
   
   MetricsCollector
   monitoring
   
   MetricsCollector.__init__
   monitoring
   
   MetricsCollector.record_duration
   monitoring
   
   MetricsCollector.increment_counter
   monitoring
   
   MetricsCollector.set_gauge
   monitoring
   
   MetricsCollector._make_key
   monitoring
   
   MetricsCollector.get_metrics_summary
   monitoring
   
   MetricsCollector.update_component_health
   monitoring
   
   MetricsCollector.get_system_health_status
   monitoring
   
   MetricsCollector.check_component_freshness
   monitoring
   
   monitor_performance
   monitoring
   - 模組: 服務骨幹模組

---

### Flow 33

- **長度**: 2 步
- **起點**: command_repository
- **終點**: models
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   CommandRepository
   command_repository
   
   CommandRepository.__init__
   command_repository
   
   CommandRepository.save_command_execution
   command_repository
   
   CommandRepository.get_command_history
   command_repository
   
   CommandRepository.get_command_statistics
   command_repository
   
   CommandRepository.get_popular_capabilities
   command_repository
   
   CommandRepository.get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

2. **AI組件**
   EnhancedVulnerability
   models
   
   EnhancedFindingPayload
   models
   
   RiskFactor
   models
   
   RiskAssessmentContext
   models
   
   RiskAssessmentResult
   models
   
   EnhancedRiskAssessment
   models
   
   RiskTrendAnalysis
   models
   
   AttackPathNode
   models
   
   AttackPathEdge
   models
   
   AttackPathPayload
   models
   
   AttackPathRecommendation
   models
   
   EnhancedAttackPathNode
   models
   
   EnhancedAttackPath
   models
   
   VulnerabilityCorrelation
   models
   
   EnhancedVulnerabilityCorrelation
   models
   
   CodeLevelRootCause
   models
   
   TaskDependency
   models
   
   EnhancedTaskExecution
   models
   
   EnhancedTaskExecution.validate_task_id
   models
   
   TaskQueue
   models
   
   TestStrategy
   models
   
   EnhancedModuleStatus
   models
   
   SystemOrchestration
   models
   
   ExperienceSampleModel
   models
   
   TraceRecordModel
   models
   
   TrainingSessionModel
   models
   
   ModelCheckpointModel
   models
   
   KnowledgeEntryModel
   models
   
   ScenarioModel
   models
   
   CommandExecutionModel
   models
   - 模組: 服務骨幹模組

---

### Flow 57

- **長度**: 2 步
- **起點**: enhanced_unified_caller
- **終點**: protocol_adapter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionCallResult
   enhanced_unified_caller
   
   ModuleEndpoint
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.__init__
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.initialize
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._setup_protocol_adapters
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._init_endpoints
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_function
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_multiple_functions
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.health_check
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.cleanup
   enhanced_unified_caller
   
   get_unified_caller
   enhanced_unified_caller
   - 模組: 服務骨幹模組

2. **程式組件**
   ProtocolAdapter
   protocol_adapter
   
   ProtocolAdapter.send_request
   protocol_adapter
   
   ProtocolAdapter.handle_response
   protocol_adapter
   
   HttpProtocolAdapter
   protocol_adapter
   
   HttpProtocolAdapter.__init__
   protocol_adapter
   
   HttpProtocolAdapter.send_request
   protocol_adapter
   
   HttpProtocolAdapter.handle_response
   protocol_adapter
   
   HttpProtocolAdapter._adapt_request_data
   protocol_adapter
   
   HttpProtocolAdapter._adapt_response_data
   protocol_adapter
   
   create_http_adapter
   protocol_adapter
   - 模組: 服務骨幹模組

---

### Flow 58

- **長度**: 2 步
- **起點**: ai_controller
- **終點**: ai_controller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   AISubsystemController
   ai_controller
   
   AISubsystemController.__init__
   ai_controller
   
   AISubsystemController.master_ai
   ai_controller
   
   AISubsystemController.process_specialized_request
   ai_controller
   
   AISubsystemController._analyze_task_complexity
   ai_controller
   
   AISubsystemController._direct_processing
   ai_controller
   
   AISubsystemController._coordinated_code_fixing
   ai_controller
   
   AISubsystemController._execute_code_fixing
   ai_controller
   
   AISubsystemController._coordinated_detection
   ai_controller
   
   AISubsystemController._multi_ai_coordination
   ai_controller
   
   AISubsystemController.get_summary_plugin_status
   ai_controller
   
   AISubsystemController.enable_summary_plugin
   ai_controller
   
   AISubsystemController.disable_summary_plugin
   ai_controller
   
   AISubsystemController.configure_summary_plugin
   ai_controller
   
   AISubsystemController.get_summary_statistics
   ai_controller
   
   AISubsystemController.reset_summary_plugin
   ai_controller
   
   AISubsystemController.unload_summary_plugin
   ai_controller
   
   AISubsystemController._record_unified_decision
   ai_controller
   
   AISubsystemController.get_control_statistics
   ai_controller
   
   AISubsystemController._classify_request_type
   ai_controller
   
   AISubsystemController._get_complexity_level
   ai_controller
   
   AISubsystemController._calculate_efficiency_score
   ai_controller
   
   AISubsystemController._extract_recommendations
   ai_controller
   
   AISubsystemController._identify_learning_points
   ai_controller
   
   AISubsystemController._create_brief_summary
   ai_controller
   
   AISubsystemController._enhance_detailed_summary
   ai_controller
   
   AISubsystemController._extract_processing_steps
   ai_controller
   
   AISubsystemController._estimate_resource_usage
   ai_controller
   
   AISubsystemController._assess_improvement_potential
   ai_controller
   
   AISubsystemController._record_summary_history
   ai_controller
   
   AISubsystemController.get_ai_summary_statistics
   ai_controller
   
   AISubsystemController._generate_summary_recommendations
   ai_controller
   
   AISubsystemController.configure_summary_settings
   ai_controller
   
   AISubsystemController.get_latest_summaries
   ai_controller
   
   AISubsystemController.export_summary_report
   ai_controller
   
   AISubsystemController.generate_comprehensive_summary
   ai_controller
   
   AISubsystemController._perform_quantitative_analysis
   ai_controller
   
   AISubsystemController._analyze_trends
   ai_controller
   
   AISubsystemController._generate_comprehensive_recommendations
   ai_controller
   
   demonstrate_unified_control
   ai_controller
   - 模組: 服務骨幹模組

2. **AI組件**
   AISubsystemController
   ai_controller
   
   AISubsystemController.__init__
   ai_controller
   
   AISubsystemController.master_ai
   ai_controller
   
   AISubsystemController.process_specialized_request
   ai_controller
   
   AISubsystemController._analyze_task_complexity
   ai_controller
   
   AISubsystemController._direct_processing
   ai_controller
   
   AISubsystemController._coordinated_code_fixing
   ai_controller
   
   AISubsystemController._execute_code_fixing
   ai_controller
   
   AISubsystemController._coordinated_detection
   ai_controller
   
   AISubsystemController._multi_ai_coordination
   ai_controller
   
   AISubsystemController.get_summary_plugin_status
   ai_controller
   
   AISubsystemController.enable_summary_plugin
   ai_controller
   
   AISubsystemController.disable_summary_plugin
   ai_controller
   
   AISubsystemController.configure_summary_plugin
   ai_controller
   
   AISubsystemController.get_summary_statistics
   ai_controller
   
   AISubsystemController.reset_summary_plugin
   ai_controller
   
   AISubsystemController.unload_summary_plugin
   ai_controller
   
   AISubsystemController._record_unified_decision
   ai_controller
   
   AISubsystemController.get_control_statistics
   ai_controller
   
   AISubsystemController._classify_request_type
   ai_controller
   
   AISubsystemController._get_complexity_level
   ai_controller
   
   AISubsystemController._calculate_efficiency_score
   ai_controller
   
   AISubsystemController._extract_recommendations
   ai_controller
   
   AISubsystemController._identify_learning_points
   ai_controller
   
   AISubsystemController._create_brief_summary
   ai_controller
   
   AISubsystemController._enhance_detailed_summary
   ai_controller
   
   AISubsystemController._extract_processing_steps
   ai_controller
   
   AISubsystemController._estimate_resource_usage
   ai_controller
   
   AISubsystemController._assess_improvement_potential
   ai_controller
   
   AISubsystemController._record_summary_history
   ai_controller
   
   AISubsystemController.get_ai_summary_statistics
   ai_controller
   
   AISubsystemController._generate_summary_recommendations
   ai_controller
   
   AISubsystemController.configure_summary_settings
   ai_controller
   
   AISubsystemController.get_latest_summaries
   ai_controller
   
   AISubsystemController.export_summary_report
   ai_controller
   
   AISubsystemController.generate_comprehensive_summary
   ai_controller
   
   AISubsystemController._perform_quantitative_analysis
   ai_controller
   
   AISubsystemController._analyze_trends
   ai_controller
   
   AISubsystemController._generate_comprehensive_recommendations
   ai_controller
   
   demonstrate_unified_control
   ai_controller
   - 模組: 服務骨幹模組

---

### Flow 71

- **長度**: 2 步
- **起點**: assistant
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   DialogIntent
   assistant
   
   DialogIntent.classify_command
   assistant
   
   AIVACommandProcessor
   assistant
   
   AIVACommandProcessor.__init__
   assistant
   
   AIVACommandProcessor._ensure_initialized
   assistant
   
   AIVACommandProcessor._get_rag_kb
   assistant
   
   AIVACommandProcessor._get_function_caller
   assistant
   
   AIVACommandProcessor._add_command_entry
   assistant
   
   AIVACommandProcessor._handle_command
   assistant
   
   AIVACommandProcessor.process_user_input
   assistant
   
   AIVACommandProcessor._handle_intent
   assistant
   
   AIVACommandProcessor._handle_list_capabilities
   assistant
   
   AIVACommandProcessor._handle_explain_capability
   assistant
   
   AIVACommandProcessor._extract_targets
   assistant
   
   AIVACommandProcessor._parse_constraints
   assistant
   
   AIVACommandProcessor._determine_scan_strategy
   assistant
   
   AIVACommandProcessor._execute_multi_engine_scan
   assistant
   
   AIVACommandProcessor._build_scan_response
   assistant
   
   AIVACommandProcessor._search_capability_via_rag
   assistant
   
   AIVACommandProcessor._execute_capability
   assistant
   
   AIVACommandProcessor._handle_run_scan
   assistant
   
   AIVACommandProcessor._handle_compare_capabilities
   assistant
   
   AIVACommandProcessor._build_command_with_params
   assistant
   
   AIVACommandProcessor._format_cli_commands
   assistant
   
   AIVACommandProcessor._handle_generate_cli
   assistant
   
   AIVACommandProcessor._handle_system_status
   assistant
   
   AIVACommandProcessor._add_conversation_entry
   assistant
   
   AIVACommandProcessor.get_conversation_history
   assistant
   
   AIVACommandProcessor.clear_conversation_history
   assistant
   
   get_dialog_assistant
   assistant
   
   _LazyDialogAssistant
   assistant
   
   _LazyDialogAssistant.__getattr__
   assistant
   
   _LazyDialogAssistant.__call__
   assistant
   - 模組: 核心能力模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

---

### Flow 97

- **長度**: 2 步
- **起點**: policy_manager
- **終點**: permission_matrix
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   RiskRule
   policy_manager
   
   RiskLevel
   policy_manager
   
   PolicyManager
   policy_manager
   
   PolicyManager.__init__
   policy_manager
   
   PolicyManager._load_policy
   policy_manager
   
   PolicyManager._use_fallback_policy
   policy_manager
   
   PolicyManager.assess_risk
   policy_manager
   
   PolicyManager._evaluate_condition
   policy_manager
   
   PolicyManager._determine_risk_level
   policy_manager
   
   PolicyManager.reload_policy
   policy_manager
   
   PolicyManager.get_policy_info
   policy_manager
   - 模組: 任務規劃模組

2. **程式組件**
   PermissionMatrix
   permission_matrix
   
   PermissionMatrix.__init__
   permission_matrix
   
   PermissionMatrix.add_role
   permission_matrix
   
   PermissionMatrix.add_resource
   permission_matrix
   
   PermissionMatrix.add_permission
   permission_matrix
   
   PermissionMatrix.grant_permission
   permission_matrix
   
   PermissionMatrix.revoke_permission
   permission_matrix
   
   PermissionMatrix.check_permission
   permission_matrix
   
   PermissionMatrix._evaluate_condition
   permission_matrix
   
   PermissionMatrix.get_role_permissions
   permission_matrix
   
   PermissionMatrix.get_resource_permissions
   permission_matrix
   
   PermissionMatrix.to_dataframe
   permission_matrix
   
   PermissionMatrix.to_numpy_matrix
   permission_matrix
   
   PermissionMatrix.analyze_coverage
   permission_matrix
   
   PermissionMatrix.find_over_privileged_roles
   permission_matrix
   
   PermissionMatrix.export_to_dict
   permission_matrix
   
   main
   permission_matrix
   
   RiskLevel
   permission_matrix
   
   OperationContext
   permission_matrix
   
   OperationContext.__post_init__
   permission_matrix
   
   RiskGuard
   permission_matrix
   
   RiskGuard.__init__
   permission_matrix
   
   RiskGuard.authorize_operation
   permission_matrix
   
   RiskGuard._check_risk_level
   permission_matrix
   
   RiskGuard._check_environment_limits
   permission_matrix
   
   RiskGuard._check_attack_tags
   permission_matrix
   
   RiskGuard._production_safety_check
   permission_matrix
   
   RiskGuard.get_allowed_operations
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   - 模組: 服務骨幹模組

---

### Flow 115

- **長度**: 2 步
- **起點**: storage_manager
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   StorageManager
   storage_manager
   
   StorageManager.__init__
   storage_manager
   
   StorageManager.initialize
   storage_manager
   
   StorageManager._get_database_config
   storage_manager
   
   StorageManager._create_backend
   storage_manager
   
   StorageManager.get_path
   storage_manager
   
   StorageManager.get_statistics
   storage_manager
   
   StorageManager.save_experience_sample
   storage_manager
   
   StorageManager.save_unified_experience_sample
   storage_manager
   
   StorageManager.get_experience_samples
   storage_manager
   
   StorageManager.save_trace
   storage_manager
   
   StorageManager.get_traces_by_session
   storage_manager
   
   StorageManager.save_training_session
   storage_manager
   
   StorageManager.save_command_execution
   storage_manager
   
   StorageManager.get_command_history
   storage_manager
   
   StorageManager.get_command_statistics
   storage_manager
   
   StorageManager.get_popular_capabilities
   storage_manager
   
   StorageManager.get_slow_executions
   storage_manager
   - 模組: 服務骨幹模組

2. **程式組件**
   StorageBackend
   backends
   
   StorageBackend.save_experience_sample
   backends
   
   StorageBackend.get_experience_samples
   backends
   
   StorageBackend.save_trace
   backends
   
   StorageBackend.get_traces_by_session
   backends
   
   StorageBackend.save_training_session
   backends
   
   StorageBackend.get_statistics
   backends
   
   SQLiteBackend
   backends
   
   SQLiteBackend.__init__
   backends
   
   SQLiteBackend.save_experience_sample
   backends
   
   SQLiteBackend.save_unified_experience_sample
   backends
   
   SQLiteBackend.get_experience_samples
   backends
   
   SQLiteBackend.save_trace
   backends
   
   SQLiteBackend.get_traces_by_session
   backends
   
   SQLiteBackend.save_training_session
   backends
   
   SQLiteBackend.get_statistics
   backends
   
   PostgreSQLBackend
   backends
   
   PostgreSQLBackend.__init__
   backends
   
   JSONLBackend
   backends
   
   JSONLBackend.__init__
   backends
   
   JSONLBackend.save_experience_sample
   backends
   
   JSONLBackend.get_experience_samples
   backends
   
   JSONLBackend.save_trace
   backends
   
   JSONLBackend.get_traces_by_session
   backends
   
   JSONLBackend.save_training_session
   backends
   
   JSONLBackend.get_statistics
   backends
   
   HybridBackend
   backends
   
   HybridBackend.__init__
   backends
   
   HybridBackend.save_experience_sample
   backends
   
   HybridBackend.get_experience_samples
   backends
   
   HybridBackend.save_trace
   backends
   
   HybridBackend.get_traces_by_session
   backends
   
   HybridBackend.save_training_session
   backends
   
   HybridBackend.get_statistics
   backends
   - 模組: 服務骨幹模組

---

### Flow 117

- **長度**: 2 步
- **起點**: plan_executor
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   PlanExecutor
   plan_executor
   
   PlanExecutor.__init__
   plan_executor
   
   PlanExecutor.execute_plan
   plan_executor
   
   PlanExecutor._publish_completion_event
   plan_executor
   
   PlanExecutor._execute_step
   plan_executor
   
   PlanExecutor._prepare_task_payload
   plan_executor
   
   PlanExecutor._send_task
   plan_executor
   
   PlanExecutor._wait_for_result
   plan_executor
   
   PlanExecutor._on_task_completed
   plan_executor
   
   PlanExecutor._check_dependencies
   plan_executor
   
   PlanExecutor._should_continue
   plan_executor
   
   PlanExecutor._record_skipped_step
   plan_executor
   
   PlanExecutor._calculate_metrics
   plan_executor
   
   PlanExecutor._calculate_sequence_accuracy
   plan_executor
   
   PlanExecutor._generate_recommendations
   plan_executor
   
   PlanExecutor._persist_result
   plan_executor
   
   PlanExecutor.get_session
   plan_executor
   
   PlanExecutor.abort_session
   plan_executor
   - 模組: 任務規劃模組

2. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 124

- **長度**: 3 步
- **起點**: unified_function_caller
- **終點**: enhanced_unified_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

3. **程式組件**
   FunctionCallResult
   enhanced_unified_caller
   
   ModuleEndpoint
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.__init__
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.initialize
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._setup_protocol_adapters
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._init_endpoints
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_function
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_multiple_functions
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.health_check
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.cleanup
   enhanced_unified_caller
   
   get_unified_caller
   enhanced_unified_caller
   - 模組: 服務骨幹模組

---

### Flow 125

- **長度**: 3 步
- **起點**: unified_function_caller
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

3. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

---

### Flow 145

- **長度**: 2 步
- **起點**: logging_formatter
- **終點**: logging_formatter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AIVALogFormatter
   logging_formatter
   
   AIVALogFormatter.__init__
   logging_formatter
   
   AIVALogFormatter.format
   logging_formatter
   
   CrossLanguageLogManager
   logging_formatter
   
   CrossLanguageLogManager.__init__
   logging_formatter
   
   CrossLanguageLogManager.get_logger
   logging_formatter
   
   CrossLanguageLogManager.log_with_context
   logging_formatter
   
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   - 模組: 服務骨幹模組

2. **程式組件**
   AIVALogFormatter
   logging_formatter
   
   AIVALogFormatter.__init__
   logging_formatter
   
   AIVALogFormatter.format
   logging_formatter
   
   CrossLanguageLogManager
   logging_formatter
   
   CrossLanguageLogManager.__init__
   logging_formatter
   
   CrossLanguageLogManager.get_logger
   logging_formatter
   
   CrossLanguageLogManager.log_with_context
   logging_formatter
   
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   - 模組: 服務骨幹模組

---

### Flow 146

- **長度**: 2 步
- **起點**: event_listener
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExternalLearningListener
   event_listener
   
   ExternalLearningListener.__init__
   event_listener
   
   ExternalLearningListener.broker
   event_listener
   
   ExternalLearningListener.connector
   event_listener
   
   ExternalLearningListener.knowledge_manager
   event_listener
   
   ExternalLearningListener.start_listening
   event_listener
   
   ExternalLearningListener.stop_listening
   event_listener
   
   ExternalLearningListener._on_result_received
   event_listener
   
   ExternalLearningListener._process_finding
   event_listener
   
   ExternalLearningListener.get_statistics
   event_listener
   
   main
   event_listener
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 164

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: ai_service
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **AI組件**
   signal_handler
   ai_service
   
   AIService
   ai_service
   
   AIService.__init__
   ai_service
   
   AIService.start
   ai_service
   
   AIService.stop
   ai_service
   
   AIService.run_api_mode
   ai_service
   
   AIService.run_monitor_mode
   ai_service
   
   AIService.run_interactive_mode
   ai_service
   
   AIService.run_daemon_mode
   ai_service
   
   main
   ai_service
   - 模組: 服務骨幹模組

---

### Flow 165

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: permission_matrix
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   PermissionMatrix
   permission_matrix
   
   PermissionMatrix.__init__
   permission_matrix
   
   PermissionMatrix.add_role
   permission_matrix
   
   PermissionMatrix.add_resource
   permission_matrix
   
   PermissionMatrix.add_permission
   permission_matrix
   
   PermissionMatrix.grant_permission
   permission_matrix
   
   PermissionMatrix.revoke_permission
   permission_matrix
   
   PermissionMatrix.check_permission
   permission_matrix
   
   PermissionMatrix._evaluate_condition
   permission_matrix
   
   PermissionMatrix.get_role_permissions
   permission_matrix
   
   PermissionMatrix.get_resource_permissions
   permission_matrix
   
   PermissionMatrix.to_dataframe
   permission_matrix
   
   PermissionMatrix.to_numpy_matrix
   permission_matrix
   
   PermissionMatrix.analyze_coverage
   permission_matrix
   
   PermissionMatrix.find_over_privileged_roles
   permission_matrix
   
   PermissionMatrix.export_to_dict
   permission_matrix
   
   main
   permission_matrix
   
   RiskLevel
   permission_matrix
   
   OperationContext
   permission_matrix
   
   OperationContext.__post_init__
   permission_matrix
   
   RiskGuard
   permission_matrix
   
   RiskGuard.__init__
   permission_matrix
   
   RiskGuard.authorize_operation
   permission_matrix
   
   RiskGuard._check_risk_level
   permission_matrix
   
   RiskGuard._check_environment_limits
   permission_matrix
   
   RiskGuard._check_attack_tags
   permission_matrix
   
   RiskGuard._production_safety_check
   permission_matrix
   
   RiskGuard.get_allowed_operations
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   - 模組: 服務骨幹模組

---

### Flow 166

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: authz_mapper
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   AuthZMapper
   authz_mapper
   
   AuthZMapper.__init__
   authz_mapper
   
   AuthZMapper.assign_role_to_user
   authz_mapper
   
   AuthZMapper.revoke_role_from_user
   authz_mapper
   
   AuthZMapper.set_user_attribute
   authz_mapper
   
   AuthZMapper.get_user_roles
   authz_mapper
   
   AuthZMapper.check_user_permission
   authz_mapper
   
   AuthZMapper.get_user_all_permissions
   authz_mapper
   
   AuthZMapper.detect_permission_conflicts
   authz_mapper
   
   AuthZMapper.analyze_role_overlap
   authz_mapper
   
   AuthZMapper.simulate_role_removal
   authz_mapper
   
   AuthZMapper.recommend_role_consolidation
   authz_mapper
   
   main
   authz_mapper
   - 模組: 服務骨幹模組

---

### Flow 167

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: matrix_visualizer
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   MatrixVisualizer
   matrix_visualizer
   
   MatrixVisualizer.__init__
   matrix_visualizer
   
   MatrixVisualizer.generate_heatmap
   matrix_visualizer
   
   MatrixVisualizer.generate_coverage_chart
   matrix_visualizer
   
   MatrixVisualizer.generate_role_comparison_chart
   matrix_visualizer
   
   MatrixVisualizer.generate_html_report
   matrix_visualizer
   
   MatrixVisualizer._generate_all_charts
   matrix_visualizer
   
   MatrixVisualizer._get_analysis_data
   matrix_visualizer
   
   MatrixVisualizer._get_html_template
   matrix_visualizer
   
   MatrixVisualizer._render_html_template
   matrix_visualizer
   
   MatrixVisualizer.export_to_csv
   matrix_visualizer
   
   main
   matrix_visualizer
   - 模組: 服務骨幹模組

---

### Flow 168

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: ai_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **AI組件**
   ComponentStatus
   ai_manager
   
   ComponentHealth
   ai_manager
   
   SystemMetrics
   ai_manager
   
   AIComponentManager
   ai_manager
   
   AIComponentManager.__init__
   ai_manager
   
   AIComponentManager.setup_logging
   ai_manager
   
   AIComponentManager.load_sot_configuration
   ai_manager
   
   AIComponentManager.setup_signal_handlers
   ai_manager
   
   AIComponentManager.start_continuous_operation
   ai_manager
   
   AIComponentManager.start_all_components
   ai_manager
   
   AIComponentManager.start_component
   ai_manager
   
   AIComponentManager.start_monitoring_threads
   ai_manager
   
   AIComponentManager.component_monitor_loop
   ai_manager
   
   AIComponentManager.metrics_collection_loop
   ai_manager
   
   AIComponentManager.check_all_components_health
   ai_manager
   
   AIComponentManager.check_component_health
   ai_manager
   
   AIComponentManager.restart_component
   ai_manager
   
   AIComponentManager.collect_system_metrics
   ai_manager
   
   AIComponentManager.main_management_loop
   ai_manager
   
   AIComponentManager.generate_status_report
   ai_manager
   
   AIComponentManager.stop_component
   ai_manager
   
   AIComponentManager.stop_all_components
   ai_manager
   
   AIComponentManager.shutdown
   ai_manager
   
   main
   ai_manager
   - 模組: 服務骨幹模組

---

### Flow 169

- **長度**: 3 步
- **起點**: sync_experiences
- **終點**: diagnose
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   print_header
   diagnose
   
   check_engines
   diagnose
   
   check_docker
   diagnose
   
   check_http
   diagnose
   
   full_diagnosis
   diagnose
   
   main
   diagnose
   - 模組: 服務骨幹模組

3. **程式組件**
   print_header
   diagnose
   
   check_engines
   diagnose
   
   check_docker
   diagnose
   
   check_http
   diagnose
   
   full_diagnosis
   diagnose
   
   main
   diagnose
   - 模組: 服務骨幹模組

---

### Flow 173

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: repair_tool
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   AIVASystemRepair
   repair_tool
   
   AIVASystemRepair.__init__
   repair_tool
   
   AIVASystemRepair.log_repair
   repair_tool
   
   AIVASystemRepair.repair_go_dependencies
   repair_tool
   
   AIVASystemRepair.fix_ssrf_unused_variable
   repair_tool
   
   AIVASystemRepair.repair_rust_compilation
   repair_tool
   
   AIVASystemRepair.repair_python_imports
   repair_tool
   
   AIVASystemRepair.check_system_connectivity
   repair_tool
   
   AIVASystemRepair.verify_target_range_connection
   repair_tool
   
   AIVASystemRepair.generate_repair_report
   repair_tool
   
   AIVASystemRepair.print_repair_summary
   repair_tool
   
   AIVASystemRepair.run_full_repair
   repair_tool
   
   main
   repair_tool
   - 模組: 服務骨幹模組

---

### Flow 196

- **長度**: 2 步
- **起點**: app
- **終點**: core_service_coordinator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ScanRequest
   app
   
   ScanResponse
   app
   
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
   
   start_scan
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

2. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

---

### Flow 212

- **長度**: 2 步
- **起點**: task_executor
- **終點**: unified_function_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   task_executor
   
   TaskExecutor
   task_executor
   
   TaskExecutor.__init__
   task_executor
   
   TaskExecutor.execute_task
   task_executor
   
   TaskExecutor._execute_by_service_type
   task_executor
   
   TaskExecutor._execute_scan_service
   task_executor
   
   TaskExecutor._call_capability_dynamically
   task_executor
   
   TaskExecutor._execute_function_service
   task_executor
   
   TaskExecutor._execute_integration_service
   task_executor
   
   TaskExecutor._execute_core_service
   task_executor
   
   TaskExecutor._infer_capability_name
   task_executor
   - 模組: 任務規劃模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

---

### Flow 220

- **長度**: 2 步
- **起點**: multilang_coordinator
- **終點**: logging_formatter
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   log_cross_language_call
   multilang_coordinator
   
   MultiLanguageAICoordinator
   multilang_coordinator
   
   MultiLanguageAICoordinator.__init__
   multilang_coordinator
   
   MultiLanguageAICoordinator.initialize
   multilang_coordinator
   
   MultiLanguageAICoordinator.check_module_availability
   multilang_coordinator
   
   MultiLanguageAICoordinator.execute_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._execute_python_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._select_best_language
   multilang_coordinator
   
   MultiLanguageAICoordinator.get_status
   multilang_coordinator
   
   MultiLanguageAICoordinator.enable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator.disable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_rust_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_go_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_typescript_service
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_rust_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_go_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_typescript_ai
   multilang_coordinator
   - 模組: 核心能力模組

2. **程式組件**
   AIVALogFormatter
   logging_formatter
   
   AIVALogFormatter.__init__
   logging_formatter
   
   AIVALogFormatter.format
   logging_formatter
   
   CrossLanguageLogManager
   logging_formatter
   
   CrossLanguageLogManager.__init__
   logging_formatter
   
   CrossLanguageLogManager.get_logger
   logging_formatter
   
   CrossLanguageLogManager.log_with_context
   logging_formatter
   
   create_unified_logger
   logging_formatter
   
   log_ai_decision
   logging_formatter
   
   log_cross_language_call
   logging_formatter
   
   get_aiva_logger
   logging_formatter
   - 模組: 服務骨幹模組

---

### Flow 224

- **長度**: 2 步
- **起點**: unified_memory_manager
- **終點**: unified_memory_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   UnifiedMemoryManager
   unified_memory_manager
   
   UnifiedMemoryManager.__init__
   unified_memory_manager
   
   UnifiedMemoryManager._generate_cache_key
   unified_memory_manager
   
   UnifiedMemoryManager.get_cached_prediction
   unified_memory_manager
   
   UnifiedMemoryManager.cache_prediction
   unified_memory_manager
   
   UnifiedMemoryManager._evict_oldest_cache_entry
   unified_memory_manager
   
   UnifiedMemoryManager.clear_cache
   unified_memory_manager
   
   UnifiedMemoryManager.create_component_pool
   unified_memory_manager
   
   UnifiedMemoryManager.get_component_pool
   unified_memory_manager
   
   UnifiedMemoryManager.register_weak_ref
   unified_memory_manager
   
   UnifiedMemoryManager.start_monitoring
   unified_memory_manager
   
   UnifiedMemoryManager.stop_monitoring
   unified_memory_manager
   
   UnifiedMemoryManager._monitor_memory
   unified_memory_manager
   
   UnifiedMemoryManager._force_cleanup
   unified_memory_manager
   
   UnifiedMemoryManager._cleanup_expired_cache
   unified_memory_manager
   
   UnifiedMemoryManager.process_batch
   unified_memory_manager
   
   UnifiedMemoryManager.process_large_dataset
   unified_memory_manager
   
   UnifiedMemoryManager._get_memory_usage_mb
   unified_memory_manager
   
   UnifiedMemoryManager._record_memory_usage
   unified_memory_manager
   
   UnifiedMemoryManager.optimize_memory
   unified_memory_manager
   
   UnifiedMemoryManager.get_comprehensive_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_cache_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_memory_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_pool_stats
   unified_memory_manager
   
   ComponentPool
   unified_memory_manager
   
   ComponentPool.__init__
   unified_memory_manager
   
   ComponentPool.get_component
   unified_memory_manager
   
   ComponentPool.get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

2. **程式組件**
   UnifiedMemoryManager
   unified_memory_manager
   
   UnifiedMemoryManager.__init__
   unified_memory_manager
   
   UnifiedMemoryManager._generate_cache_key
   unified_memory_manager
   
   UnifiedMemoryManager.get_cached_prediction
   unified_memory_manager
   
   UnifiedMemoryManager.cache_prediction
   unified_memory_manager
   
   UnifiedMemoryManager._evict_oldest_cache_entry
   unified_memory_manager
   
   UnifiedMemoryManager.clear_cache
   unified_memory_manager
   
   UnifiedMemoryManager.create_component_pool
   unified_memory_manager
   
   UnifiedMemoryManager.get_component_pool
   unified_memory_manager
   
   UnifiedMemoryManager.register_weak_ref
   unified_memory_manager
   
   UnifiedMemoryManager.start_monitoring
   unified_memory_manager
   
   UnifiedMemoryManager.stop_monitoring
   unified_memory_manager
   
   UnifiedMemoryManager._monitor_memory
   unified_memory_manager
   
   UnifiedMemoryManager._force_cleanup
   unified_memory_manager
   
   UnifiedMemoryManager._cleanup_expired_cache
   unified_memory_manager
   
   UnifiedMemoryManager.process_batch
   unified_memory_manager
   
   UnifiedMemoryManager.process_large_dataset
   unified_memory_manager
   
   UnifiedMemoryManager._get_memory_usage_mb
   unified_memory_manager
   
   UnifiedMemoryManager._record_memory_usage
   unified_memory_manager
   
   UnifiedMemoryManager.optimize_memory
   unified_memory_manager
   
   UnifiedMemoryManager.get_comprehensive_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_cache_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_memory_stats
   unified_memory_manager
   
   UnifiedMemoryManager._get_pool_stats
   unified_memory_manager
   
   ComponentPool
   unified_memory_manager
   
   ComponentPool.__init__
   unified_memory_manager
   
   ComponentPool.get_component
   unified_memory_manager
   
   ComponentPool.get_pool_stats
   unified_memory_manager
   - 模組: 服務骨幹模組

---

### Flow 246

- **長度**: 3 步
- **起點**: task_executor
- **終點**: enhanced_unified_caller
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExecutionResult
   task_executor
   
   TaskExecutor
   task_executor
   
   TaskExecutor.__init__
   task_executor
   
   TaskExecutor.execute_task
   task_executor
   
   TaskExecutor._execute_by_service_type
   task_executor
   
   TaskExecutor._execute_scan_service
   task_executor
   
   TaskExecutor._call_capability_dynamically
   task_executor
   
   TaskExecutor._execute_function_service
   task_executor
   
   TaskExecutor._execute_integration_service
   task_executor
   
   TaskExecutor._execute_core_service
   task_executor
   
   TaskExecutor._infer_capability_name
   task_executor
   - 模組: 任務規劃模組

2. **程式組件**
   FunctionCallResult
   unified_function_caller
   
   ModuleEndpoint
   unified_function_caller
   
   UnifiedFunctionCaller
   unified_function_caller
   
   UnifiedFunctionCaller.__init__
   unified_function_caller
   
   UnifiedFunctionCaller._init_endpoints
   unified_function_caller
   
   UnifiedFunctionCaller.call_python
   unified_function_caller
   
   UnifiedFunctionCaller.call_http
   unified_function_caller
   
   UnifiedFunctionCaller.call_grpc
   unified_function_caller
   
   UnifiedFunctionCaller.call_function
   unified_function_caller
   
   UnifiedFunctionCaller._call_python_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_http_module
   unified_function_caller
   
   UnifiedFunctionCaller._call_grpc_module
   unified_function_caller
   
   UnifiedFunctionCaller.list_all_functions
   unified_function_caller
   
   UnifiedFunctionCaller.get_module_info
   unified_function_caller
   
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
   - 模組: 服務骨幹模組

3. **程式組件**
   FunctionCallResult
   enhanced_unified_caller
   
   ModuleEndpoint
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.__init__
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.initialize
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._setup_protocol_adapters
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller._init_endpoints
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_function
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.call_multiple_functions
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.health_check
   enhanced_unified_caller
   
   EnhancedUnifiedFunctionCaller.cleanup
   enhanced_unified_caller
   
   get_unified_caller
   enhanced_unified_caller
   - 模組: 服務骨幹模組

---

### Flow 258

- **長度**: 3 步
- **起點**: permission_matrix
- **終點**: permission_matrix
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   PermissionMatrix
   permission_matrix
   
   PermissionMatrix.__init__
   permission_matrix
   
   PermissionMatrix.add_role
   permission_matrix
   
   PermissionMatrix.add_resource
   permission_matrix
   
   PermissionMatrix.add_permission
   permission_matrix
   
   PermissionMatrix.grant_permission
   permission_matrix
   
   PermissionMatrix.revoke_permission
   permission_matrix
   
   PermissionMatrix.check_permission
   permission_matrix
   
   PermissionMatrix._evaluate_condition
   permission_matrix
   
   PermissionMatrix.get_role_permissions
   permission_matrix
   
   PermissionMatrix.get_resource_permissions
   permission_matrix
   
   PermissionMatrix.to_dataframe
   permission_matrix
   
   PermissionMatrix.to_numpy_matrix
   permission_matrix
   
   PermissionMatrix.analyze_coverage
   permission_matrix
   
   PermissionMatrix.find_over_privileged_roles
   permission_matrix
   
   PermissionMatrix.export_to_dict
   permission_matrix
   
   main
   permission_matrix
   
   RiskLevel
   permission_matrix
   
   OperationContext
   permission_matrix
   
   OperationContext.__post_init__
   permission_matrix
   
   RiskGuard
   permission_matrix
   
   RiskGuard.__init__
   permission_matrix
   
   RiskGuard.authorize_operation
   permission_matrix
   
   RiskGuard._check_risk_level
   permission_matrix
   
   RiskGuard._check_environment_limits
   permission_matrix
   
   RiskGuard._check_attack_tags
   permission_matrix
   
   RiskGuard._production_safety_check
   permission_matrix
   
   RiskGuard.get_allowed_operations
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   - 模組: 服務骨幹模組

2. **程式組件**
   PermissionMatrix
   permission_matrix
   
   PermissionMatrix.__init__
   permission_matrix
   
   PermissionMatrix.add_role
   permission_matrix
   
   PermissionMatrix.add_resource
   permission_matrix
   
   PermissionMatrix.add_permission
   permission_matrix
   
   PermissionMatrix.grant_permission
   permission_matrix
   
   PermissionMatrix.revoke_permission
   permission_matrix
   
   PermissionMatrix.check_permission
   permission_matrix
   
   PermissionMatrix._evaluate_condition
   permission_matrix
   
   PermissionMatrix.get_role_permissions
   permission_matrix
   
   PermissionMatrix.get_resource_permissions
   permission_matrix
   
   PermissionMatrix.to_dataframe
   permission_matrix
   
   PermissionMatrix.to_numpy_matrix
   permission_matrix
   
   PermissionMatrix.analyze_coverage
   permission_matrix
   
   PermissionMatrix.find_over_privileged_roles
   permission_matrix
   
   PermissionMatrix.export_to_dict
   permission_matrix
   
   main
   permission_matrix
   
   RiskLevel
   permission_matrix
   
   OperationContext
   permission_matrix
   
   OperationContext.__post_init__
   permission_matrix
   
   RiskGuard
   permission_matrix
   
   RiskGuard.__init__
   permission_matrix
   
   RiskGuard.authorize_operation
   permission_matrix
   
   RiskGuard._check_risk_level
   permission_matrix
   
   RiskGuard._check_environment_limits
   permission_matrix
   
   RiskGuard._check_attack_tags
   permission_matrix
   
   RiskGuard._production_safety_check
   permission_matrix
   
   RiskGuard.get_allowed_operations
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   - 模組: 服務骨幹模組

3. **程式組件**
   PermissionMatrix
   permission_matrix
   
   PermissionMatrix.__init__
   permission_matrix
   
   PermissionMatrix.add_role
   permission_matrix
   
   PermissionMatrix.add_resource
   permission_matrix
   
   PermissionMatrix.add_permission
   permission_matrix
   
   PermissionMatrix.grant_permission
   permission_matrix
   
   PermissionMatrix.revoke_permission
   permission_matrix
   
   PermissionMatrix.check_permission
   permission_matrix
   
   PermissionMatrix._evaluate_condition
   permission_matrix
   
   PermissionMatrix.get_role_permissions
   permission_matrix
   
   PermissionMatrix.get_resource_permissions
   permission_matrix
   
   PermissionMatrix.to_dataframe
   permission_matrix
   
   PermissionMatrix.to_numpy_matrix
   permission_matrix
   
   PermissionMatrix.analyze_coverage
   permission_matrix
   
   PermissionMatrix.find_over_privileged_roles
   permission_matrix
   
   PermissionMatrix.export_to_dict
   permission_matrix
   
   main
   permission_matrix
   
   RiskLevel
   permission_matrix
   
   OperationContext
   permission_matrix
   
   OperationContext.__post_init__
   permission_matrix
   
   RiskGuard
   permission_matrix
   
   RiskGuard.__init__
   permission_matrix
   
   RiskGuard.authorize_operation
   permission_matrix
   
   RiskGuard._check_risk_level
   permission_matrix
   
   RiskGuard._check_environment_limits
   permission_matrix
   
   RiskGuard._check_attack_tags
   permission_matrix
   
   RiskGuard._production_safety_check
   permission_matrix
   
   RiskGuard.get_allowed_operations
   permission_matrix
   
   get_risk_guard
   permission_matrix
   
   authorize_operation
   permission_matrix
   - 模組: 服務骨幹模組

---

### Flow 260

- **長度**: 2 步
- **起點**: backends
- **終點**: backends
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   StorageBackend
   backends
   
   StorageBackend.save_experience_sample
   backends
   
   StorageBackend.get_experience_samples
   backends
   
   StorageBackend.save_trace
   backends
   
   StorageBackend.get_traces_by_session
   backends
   
   StorageBackend.save_training_session
   backends
   
   StorageBackend.get_statistics
   backends
   
   SQLiteBackend
   backends
   
   SQLiteBackend.__init__
   backends
   
   SQLiteBackend.save_experience_sample
   backends
   
   SQLiteBackend.save_unified_experience_sample
   backends
   
   SQLiteBackend.get_experience_samples
   backends
   
   SQLiteBackend.save_trace
   backends
   
   SQLiteBackend.get_traces_by_session
   backends
   
   SQLiteBackend.save_training_session
   backends
   
   SQLiteBackend.get_statistics
   backends
   
   PostgreSQLBackend
   backends
   
   PostgreSQLBackend.__init__
   backends
   
   JSONLBackend
   backends
   
   JSONLBackend.__init__
   backends
   
   JSONLBackend.save_experience_sample
   backends
   
   JSONLBackend.get_experience_samples
   backends
   
   JSONLBackend.save_trace
   backends
   
   JSONLBackend.get_traces_by_session
   backends
   
   JSONLBackend.save_training_session
   backends
   
   JSONLBackend.get_statistics
   backends
   
   HybridBackend
   backends
   
   HybridBackend.__init__
   backends
   
   HybridBackend.save_experience_sample
   backends
   
   HybridBackend.get_experience_samples
   backends
   
   HybridBackend.save_trace
   backends
   
   HybridBackend.get_traces_by_session
   backends
   
   HybridBackend.save_training_session
   backends
   
   HybridBackend.get_statistics
   backends
   - 模組: 服務骨幹模組

2. **程式組件**
   StorageBackend
   backends
   
   StorageBackend.save_experience_sample
   backends
   
   StorageBackend.get_experience_samples
   backends
   
   StorageBackend.save_trace
   backends
   
   StorageBackend.get_traces_by_session
   backends
   
   StorageBackend.save_training_session
   backends
   
   StorageBackend.get_statistics
   backends
   
   SQLiteBackend
   backends
   
   SQLiteBackend.__init__
   backends
   
   SQLiteBackend.save_experience_sample
   backends
   
   SQLiteBackend.save_unified_experience_sample
   backends
   
   SQLiteBackend.get_experience_samples
   backends
   
   SQLiteBackend.save_trace
   backends
   
   SQLiteBackend.get_traces_by_session
   backends
   
   SQLiteBackend.save_training_session
   backends
   
   SQLiteBackend.get_statistics
   backends
   
   PostgreSQLBackend
   backends
   
   PostgreSQLBackend.__init__
   backends
   
   JSONLBackend
   backends
   
   JSONLBackend.__init__
   backends
   
   JSONLBackend.save_experience_sample
   backends
   
   JSONLBackend.get_experience_samples
   backends
   
   JSONLBackend.save_trace
   backends
   
   JSONLBackend.get_traces_by_session
   backends
   
   JSONLBackend.save_training_session
   backends
   
   JSONLBackend.get_statistics
   backends
   
   HybridBackend
   backends
   
   HybridBackend.__init__
   backends
   
   HybridBackend.save_experience_sample
   backends
   
   HybridBackend.get_experience_samples
   backends
   
   HybridBackend.save_trace
   backends
   
   HybridBackend.get_traces_by_session
   backends
   
   HybridBackend.save_training_session
   backends
   
   HybridBackend.get_statistics
   backends
   - 模組: 服務骨幹模組

---

### Flow 278

- **長度**: 2 步
- **起點**: storage_manager
- **終點**: command_repository
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   StorageManager
   storage_manager
   
   StorageManager.__init__
   storage_manager
   
   StorageManager.initialize
   storage_manager
   
   StorageManager._get_database_config
   storage_manager
   
   StorageManager._create_backend
   storage_manager
   
   StorageManager.get_path
   storage_manager
   
   StorageManager.get_statistics
   storage_manager
   
   StorageManager.save_experience_sample
   storage_manager
   
   StorageManager.save_unified_experience_sample
   storage_manager
   
   StorageManager.get_experience_samples
   storage_manager
   
   StorageManager.save_trace
   storage_manager
   
   StorageManager.get_traces_by_session
   storage_manager
   
   StorageManager.save_training_session
   storage_manager
   
   StorageManager.save_command_execution
   storage_manager
   
   StorageManager.get_command_history
   storage_manager
   
   StorageManager.get_command_statistics
   storage_manager
   
   StorageManager.get_popular_capabilities
   storage_manager
   
   StorageManager.get_slow_executions
   storage_manager
   - 模組: 服務骨幹模組

2. **程式組件**
   CommandRepository
   command_repository
   
   CommandRepository.__init__
   command_repository
   
   CommandRepository.save_command_execution
   command_repository
   
   CommandRepository.get_command_history
   command_repository
   
   CommandRepository.get_command_statistics
   command_repository
   
   CommandRepository.get_popular_capabilities
   command_repository
   
   CommandRepository.get_slow_executions
   command_repository
   - 模組: 服務骨幹模組

---

### Flow 282

- **長度**: 3 步
- **起點**: core_service_coordinator
- **終點**: core_service_coordinator
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

2. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

3. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

---

### Flow 287

- **長度**: 2 步
- **起點**: dispatcher
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   CognitiveDispatcher
   dispatcher
   
   CognitiveDispatcher.__init__
   dispatcher
   
   CognitiveDispatcher.broker
   dispatcher
   
   CognitiveDispatcher._build_message
   dispatcher
   
   CognitiveDispatcher.request_plan
   dispatcher
   
   CognitiveDispatcher.execute_capability
   dispatcher
   
   CognitiveDispatcher.trigger_learning
   dispatcher
   
   CognitiveDispatcher.notify_decision
   dispatcher
   
   CognitiveDispatcher.store_result
   dispatcher
   
   CognitiveDispatcher.call_task_planning_sync
   dispatcher
   
   CognitiveDispatcher.call_core_capabilities_sync
   dispatcher
   
   CognitiveDispatcher.call_external_learning_sync
   dispatcher
   
   CognitiveDispatcher.execute_and_notify
   dispatcher
   
   CognitiveDispatcher.get_dispatch_stats
   dispatcher
   
   get_dispatcher
   dispatcher
   
   dispatch_to_task_planning
   dispatcher
   
   dispatch_to_core_capabilities
   dispatcher
   
   dispatch_to_external_learning
   dispatcher
   
   PlanningDispatcher
   dispatcher
   
   PlanningDispatcher.__init__
   dispatcher
   
   PlanningDispatcher.broker
   dispatcher
   
   PlanningDispatcher._build_message
   dispatcher
   
   PlanningDispatcher.execute_plan_step
   dispatcher
   
   PlanningDispatcher.confirm_decision
   dispatcher
   
   PlanningDispatcher.query_resource
   dispatcher
   
   PlanningDispatcher.request_analysis
   dispatcher
   
   PlanningDispatcher.notify_plan_status
   dispatcher
   
   PlanningDispatcher.execute_attack_sync
   dispatcher
   
   PlanningDispatcher.execute_scan_sync
   dispatcher
   
   PlanningDispatcher.call_cognitive_sync
   dispatcher
   
   PlanningDispatcher.call_exploration_sync
   dispatcher
   
   PlanningDispatcher.execute_plan
   dispatcher
   
   PlanningDispatcher.execute_with_confirmation
   dispatcher
   
   PlanningDispatcher.get_dispatch_stats
   dispatcher
   
   dispatch_to_cognitive_core
   dispatcher
   
   execute_attack
   dispatcher
   
   execute_scan
   dispatcher
   - 模組: 認知核心模組

2. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 302

- **長度**: 2 步
- **起點**: multilang_coordinator
- **終點**: app
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   log_cross_language_call
   multilang_coordinator
   
   MultiLanguageAICoordinator
   multilang_coordinator
   
   MultiLanguageAICoordinator.__init__
   multilang_coordinator
   
   MultiLanguageAICoordinator.initialize
   multilang_coordinator
   
   MultiLanguageAICoordinator.check_module_availability
   multilang_coordinator
   
   MultiLanguageAICoordinator.execute_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._execute_python_task
   multilang_coordinator
   
   MultiLanguageAICoordinator._select_best_language
   multilang_coordinator
   
   MultiLanguageAICoordinator.get_status
   multilang_coordinator
   
   MultiLanguageAICoordinator.enable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator.disable_module
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_rust_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_go_service
   multilang_coordinator
   
   MultiLanguageAICoordinator._check_typescript_service
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_rust_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_go_ai
   multilang_coordinator
   
   MultiLanguageAICoordinator.call_typescript_ai
   multilang_coordinator
   - 模組: 核心能力模組

2. **程式組件**
   ScanRequest
   app
   
   ScanResponse
   app
   
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
   
   start_scan
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

### Flow 304

- **長度**: 2 步
- **起點**: ai_service
- **終點**: ai_service
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   signal_handler
   ai_service
   
   AIService
   ai_service
   
   AIService.__init__
   ai_service
   
   AIService.start
   ai_service
   
   AIService.stop
   ai_service
   
   AIService.run_api_mode
   ai_service
   
   AIService.run_monitor_mode
   ai_service
   
   AIService.run_interactive_mode
   ai_service
   
   AIService.run_daemon_mode
   ai_service
   
   main
   ai_service
   - 模組: 服務骨幹模組

2. **AI組件**
   signal_handler
   ai_service
   
   AIService
   ai_service
   
   AIService.__init__
   ai_service
   
   AIService.start
   ai_service
   
   AIService.stop
   ai_service
   
   AIService.run_api_mode
   ai_service
   
   AIService.run_monitor_mode
   ai_service
   
   AIService.run_interactive_mode
   ai_service
   
   AIService.run_daemon_mode
   ai_service
   
   main
   ai_service
   - 模組: 服務骨幹模組

---

### Flow 341

- **長度**: 2 步
- **起點**: dispatcher_base
- **終點**: message_broker
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   BaseDispatcher
   dispatcher_base
   
   BaseDispatcher.__init__
   dispatcher_base
   
   BaseDispatcher.broker
   dispatcher_base
   
   BaseDispatcher._build_message
   dispatcher_base
   
   BaseDispatcher.send_message
   dispatcher_base
   
   BaseDispatcher.broadcast
   dispatcher_base
   
   BaseDispatcher.request_task
   dispatcher_base
   
   BaseDispatcher.report_result
   dispatcher_base
   
   BaseDispatcher.call_module
   dispatcher_base
   
   BaseDispatcher.call_cli
   dispatcher_base
   
   BaseDispatcher.call_with_json
   dispatcher_base
   
   BaseDispatcher.call_rust
   dispatcher_base
   
   BaseDispatcher.call_go
   dispatcher_base
   
   BaseDispatcher.call_node
   dispatcher_base
   
   BaseDispatcher.call_docker
   dispatcher_base
   
   BaseDispatcher.get_dispatch_stats
   dispatcher_base
   
   BaseDispatcher.health_check
   dispatcher_base
   
   MessageFormats
   dispatcher_base
   
   MessageFormats.task_request
   dispatcher_base
   
   MessageFormats.event_notification
   dispatcher_base
   
   MessageFormats.result_report
   dispatcher_base
   
   Exchanges
   dispatcher_base
   
   RoutingKeys
   dispatcher_base
   - 模組: 服務骨幹模組

2. **程式組件**
   MessageBroker
   message_broker
   
   MessageBroker.__init__
   message_broker
   
   MessageBroker.connect
   message_broker
   
   MessageBroker._declare_exchanges
   message_broker
   
   MessageBroker.publish_message
   message_broker
   
   MessageBroker.subscribe
   message_broker
   
   MessageBroker.create_rpc_client
   message_broker
   
   MessageBroker.get_rpc_client
   message_broker
   
   MessageBroker.disconnect
   message_broker
   
   RPCClient
   message_broker
   
   RPCClient.__init__
   message_broker
   
   RPCClient.setup
   message_broker
   
   RPCClient._on_response
   message_broker
   
   RPCClient.call
   message_broker
   
   EventPriority
   message_broker
   
   AIVAEvent
   message_broker
   
   AIVAEvent.is_expired
   message_broker
   
   AIVAEvent.can_retry
   message_broker
   
   EventSubscription
   message_broker
   
   EventSubscription.matches
   message_broker
   
   EventSubscription._match_pattern
   message_broker
   
   EnhancedMessageBroker
   message_broker
   
   EnhancedMessageBroker.__init__
   message_broker
   
   EnhancedMessageBroker.start_event_system
   message_broker
   
   EnhancedMessageBroker.stop_event_system
   message_broker
   
   EnhancedMessageBroker.publish_event
   message_broker
   
   EnhancedMessageBroker.subscribe_event
   message_broker
   
   EnhancedMessageBroker.unsubscribe_event
   message_broker
   
   EnhancedMessageBroker._process_events
   message_broker
   
   EnhancedMessageBroker._handle_event
   message_broker
   
   EnhancedMessageBroker.get_event_statistics
   message_broker
   
   get_enhanced_message_broker
   message_broker
   
   publish_aiva_event
   message_broker
   
   subscribe_aiva_events
   message_broker
   - 模組: 服務骨幹模組

---

### Flow 375

- **長度**: 3 步
- **起點**: core_service_coordinator
- **終點**: context_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AIVACoreServiceCoordinator
   core_service_coordinator
   
   AIVACoreServiceCoordinator.__init__
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._initialize_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._setup_monitoring_and_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._apply_initial_config
   core_service_coordinator
   
   AIVACoreServiceCoordinator._configure_security_middleware
   core_service_coordinator
   
   AIVACoreServiceCoordinator._on_config_changed
   core_service_coordinator
   
   AIVACoreServiceCoordinator.start
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._start_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator.stop
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_core_components
   core_service_coordinator
   
   AIVACoreServiceCoordinator._stop_shared_services
   core_service_coordinator
   
   AIVACoreServiceCoordinator._cleanup_on_failure
   core_service_coordinator
   
   AIVACoreServiceCoordinator.process_command
   core_service_coordinator
   
   AIVACoreServiceCoordinator.get_service_status
   core_service_coordinator
   
   AIVACoreServiceCoordinator.health_check
   core_service_coordinator
   
   get_core_service_coordinator
   core_service_coordinator
   
   process_command
   core_service_coordinator
   
   initialize_core_module
   core_service_coordinator
   
   shutdown_core_module
   core_service_coordinator
   - 模組: 服務骨幹模組

2. **程式組件**
   ContextManager
   context_manager
   
   ContextManager.__init__
   context_manager
   
   ContextManager.create_context
   context_manager
   
   ContextManager._update_session
   context_manager
   
   ContextManager.get_context
   context_manager
   
   ContextManager.update_context
   context_manager
   
   ContextManager.set_variable
   context_manager
   
   ContextManager.get_variable
   context_manager
   
   ContextManager.add_history
   context_manager
   
   ContextManager.get_context_history
   context_manager
   
   ContextManager.get_session_contexts
   context_manager
   
   ContextManager.get_session_info
   context_manager
   
   ContextManager.cleanup_context
   context_manager
   
   ContextManager.cleanup_session
   context_manager
   
   ContextManager.cleanup_expired_contexts
   context_manager
   
   ContextManager.cleanup_expired_sessions
   context_manager
   
   ContextManager.get_context_stats
   context_manager
   
   get_context_manager
   context_manager
   - 模組: 服務骨幹模組

3. **程式組件**
   ContextManager
   context_manager
   
   ContextManager.__init__
   context_manager
   
   ContextManager.create_context
   context_manager
   
   ContextManager._update_session
   context_manager
   
   ContextManager.get_context
   context_manager
   
   ContextManager.update_context
   context_manager
   
   ContextManager.set_variable
   context_manager
   
   ContextManager.get_variable
   context_manager
   
   ContextManager.add_history
   context_manager
   
   ContextManager.get_context_history
   context_manager
   
   ContextManager.get_session_contexts
   context_manager
   
   ContextManager.get_session_info
   context_manager
   
   ContextManager.cleanup_context
   context_manager
   
   ContextManager.cleanup_session
   context_manager
   
   ContextManager.cleanup_expired_contexts
   context_manager
   
   ContextManager.cleanup_expired_sessions
   context_manager
   
   ContextManager.get_context_stats
   context_manager
   
   get_context_manager
   context_manager
   - 模組: 服務骨幹模組

---

### Flow 386

- **長度**: 2 步
- **起點**: ai_manager
- **終點**: monitoring
- **主要模組**: 服務骨幹模組
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   ComponentStatus
   ai_manager
   
   ComponentHealth
   ai_manager
   
   SystemMetrics
   ai_manager
   
   AIComponentManager
   ai_manager
   
   AIComponentManager.__init__
   ai_manager
   
   AIComponentManager.setup_logging
   ai_manager
   
   AIComponentManager.load_sot_configuration
   ai_manager
   
   AIComponentManager.setup_signal_handlers
   ai_manager
   
   AIComponentManager.start_continuous_operation
   ai_manager
   
   AIComponentManager.start_all_components
   ai_manager
   
   AIComponentManager.start_component
   ai_manager
   
   AIComponentManager.start_monitoring_threads
   ai_manager
   
   AIComponentManager.component_monitor_loop
   ai_manager
   
   AIComponentManager.metrics_collection_loop
   ai_manager
   
   AIComponentManager.check_all_components_health
   ai_manager
   
   AIComponentManager.check_component_health
   ai_manager
   
   AIComponentManager.restart_component
   ai_manager
   
   AIComponentManager.collect_system_metrics
   ai_manager
   
   AIComponentManager.main_management_loop
   ai_manager
   
   AIComponentManager.generate_status_report
   ai_manager
   
   AIComponentManager.stop_component
   ai_manager
   
   AIComponentManager.stop_all_components
   ai_manager
   
   AIComponentManager.shutdown
   ai_manager
   
   main
   ai_manager
   - 模組: 服務骨幹模組

2. **程式組件**
   ComponentHealth
   monitoring
   
   Metric
   monitoring
   
   MetricsCollector
   monitoring
   
   MetricsCollector.__init__
   monitoring
   
   MetricsCollector.record_duration
   monitoring
   
   MetricsCollector.increment_counter
   monitoring
   
   MetricsCollector.set_gauge
   monitoring
   
   MetricsCollector._make_key
   monitoring
   
   MetricsCollector.get_metrics_summary
   monitoring
   
   MetricsCollector.update_component_health
   monitoring
   
   MetricsCollector.get_system_health_status
   monitoring
   
   MetricsCollector.check_component_freshness
   monitoring
   
   monitor_performance
   monitoring
   - 模組: 服務骨幹模組

---

### Flow 387

- **長度**: 2 步
- **起點**: ai_manager
- **終點**: ai_manager
- **主要模組**: 服務骨幹模組
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   ComponentStatus
   ai_manager
   
   ComponentHealth
   ai_manager
   
   SystemMetrics
   ai_manager
   
   AIComponentManager
   ai_manager
   
   AIComponentManager.__init__
   ai_manager
   
   AIComponentManager.setup_logging
   ai_manager
   
   AIComponentManager.load_sot_configuration
   ai_manager
   
   AIComponentManager.setup_signal_handlers
   ai_manager
   
   AIComponentManager.start_continuous_operation
   ai_manager
   
   AIComponentManager.start_all_components
   ai_manager
   
   AIComponentManager.start_component
   ai_manager
   
   AIComponentManager.start_monitoring_threads
   ai_manager
   
   AIComponentManager.component_monitor_loop
   ai_manager
   
   AIComponentManager.metrics_collection_loop
   ai_manager
   
   AIComponentManager.check_all_components_health
   ai_manager
   
   AIComponentManager.check_component_health
   ai_manager
   
   AIComponentManager.restart_component
   ai_manager
   
   AIComponentManager.collect_system_metrics
   ai_manager
   
   AIComponentManager.main_management_loop
   ai_manager
   
   AIComponentManager.generate_status_report
   ai_manager
   
   AIComponentManager.stop_component
   ai_manager
   
   AIComponentManager.stop_all_components
   ai_manager
   
   AIComponentManager.shutdown
   ai_manager
   
   main
   ai_manager
   - 模組: 服務骨幹模組

2. **AI組件**
   ComponentStatus
   ai_manager
   
   ComponentHealth
   ai_manager
   
   SystemMetrics
   ai_manager
   
   AIComponentManager
   ai_manager
   
   AIComponentManager.__init__
   ai_manager
   
   AIComponentManager.setup_logging
   ai_manager
   
   AIComponentManager.load_sot_configuration
   ai_manager
   
   AIComponentManager.setup_signal_handlers
   ai_manager
   
   AIComponentManager.start_continuous_operation
   ai_manager
   
   AIComponentManager.start_all_components
   ai_manager
   
   AIComponentManager.start_component
   ai_manager
   
   AIComponentManager.start_monitoring_threads
   ai_manager
   
   AIComponentManager.component_monitor_loop
   ai_manager
   
   AIComponentManager.metrics_collection_loop
   ai_manager
   
   AIComponentManager.check_all_components_health
   ai_manager
   
   AIComponentManager.check_component_health
   ai_manager
   
   AIComponentManager.restart_component
   ai_manager
   
   AIComponentManager.collect_system_metrics
   ai_manager
   
   AIComponentManager.main_management_loop
   ai_manager
   
   AIComponentManager.generate_status_report
   ai_manager
   
   AIComponentManager.stop_component
   ai_manager
   
   AIComponentManager.stop_all_components
   ai_manager
   
   AIComponentManager.shutdown
   ai_manager
   
   main
   ai_manager
   - 模組: 服務骨幹模組

---

## 認知核心模組(學習子系統) (learning_system)

包含 29 條數據流

### Flow 1

- **長度**: 2 步
- **起點**: notification_system
- **終點**: notification_system
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   NotificationLevel
   notification_system
   
   NotificationType
   notification_system
   
   UserNotification
   notification_system
   
   UserNotification.__init__
   notification_system
   
   UserNotification.to_dict
   notification_system
   
   UserNotification.to_log_message
   notification_system
   
   NotificationSystem
   notification_system
   
   NotificationSystem.__init__
   notification_system
   
   NotificationSystem.register_callback
   notification_system
   
   NotificationSystem.notify
   notification_system
   
   NotificationSystem.notify_unknown_situation
   notification_system
   
   NotificationSystem.notify_rag_triggered
   notification_system
   
   NotificationSystem.notify_rag_completed
   notification_system
   
   NotificationSystem.notify_rag_failed
   notification_system
   
   NotificationSystem.notify_learning_started
   notification_system
   
   NotificationSystem.notify_learning_completed
   notification_system
   
   NotificationSystem.get_notification_history
   notification_system
   
   NotificationSystem.clear_history
   notification_system
   
   get_notification_system
   notification_system
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   NotificationLevel
   notification_system
   
   NotificationType
   notification_system
   
   UserNotification
   notification_system
   
   UserNotification.__init__
   notification_system
   
   UserNotification.to_dict
   notification_system
   
   UserNotification.to_log_message
   notification_system
   
   NotificationSystem
   notification_system
   
   NotificationSystem.__init__
   notification_system
   
   NotificationSystem.register_callback
   notification_system
   
   NotificationSystem.notify
   notification_system
   
   NotificationSystem.notify_unknown_situation
   notification_system
   
   NotificationSystem.notify_rag_triggered
   notification_system
   
   NotificationSystem.notify_rag_completed
   notification_system
   
   NotificationSystem.notify_rag_failed
   notification_system
   
   NotificationSystem.notify_learning_started
   notification_system
   
   NotificationSystem.notify_learning_completed
   notification_system
   
   NotificationSystem.get_notification_history
   notification_system
   
   NotificationSystem.clear_history
   notification_system
   
   get_notification_system
   notification_system
   - 模組: 認知核心模組(學習子系統)

---

### Flow 2

- **長度**: 2 步
- **起點**: model_trainer
- **終點**: rl_trainers
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   ModelTrainer
   model_trainer
   
   ModelTrainer.__init__
   model_trainer
   
   ModelTrainer.train
   model_trainer
   
   ModelTrainer.train_supervised
   model_trainer
   
   ModelTrainer.train_reinforcement
   model_trainer
   
   ModelTrainer.train_dqn
   model_trainer
   
   ModelTrainer.train_ppo
   model_trainer
   
   ModelTrainer._prepare_supervised_data
   model_trainer
   
   ModelTrainer._extract_features
   model_trainer
   
   ModelTrainer._prepare_rl_data
   model_trainer
   
   ModelTrainer._build_state_vector
   model_trainer
   
   ModelTrainer._encode_attack_type
   model_trainer
   
   ModelTrainer._encode_action
   model_trainer
   
   ModelTrainer._calculate_step_reward
   model_trainer
   
   ModelTrainer._train_model_supervised
   model_trainer
   
   ModelTrainer._train_model_rl
   model_trainer
   
   ModelTrainer._evaluate_model
   model_trainer
   
   ModelTrainer._save_model
   model_trainer
   
   ModelTrainer.load_model
   model_trainer
   
   ModelTrainer.test_on_scenario
   model_trainer
   
   ModelTrainer._increment_version
   model_trainer
   
   ModelTrainer._persist_training_result
   model_trainer
   - 模組: 認知核心模組(學習子系統)

2. **AI組件**
   DQNTrainer
   rl_trainers
   
   DQNTrainer.__init__
   rl_trainers
   
   DQNTrainer.select_action
   rl_trainers
   
   DQNTrainer.train_step
   rl_trainers
   
   DQNTrainer.get_metrics
   rl_trainers
   
   DQNTrainer.save
   rl_trainers
   
   DQNTrainer.load
   rl_trainers
   
   PPOTrainer
   rl_trainers
   
   PPOTrainer.__init__
   rl_trainers
   
   PPOTrainer.select_action
   rl_trainers
   
   PPOTrainer.store_transition
   rl_trainers
   
   PPOTrainer.update
   rl_trainers
   
   PPOTrainer.get_metrics
   rl_trainers
   
   PPOTrainer.save
   rl_trainers
   
   PPOTrainer.load
   rl_trainers
   - 模組: 認知核心模組(學習子系統)

---

### Flow 4

- **長度**: 2 步
- **起點**: rag_trigger
- **終點**: rag_trigger
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   UnknownSituationAlert
   rag_trigger
   
   UnknownSituationAlert.__init__
   rag_trigger
   
   UnknownSituationAlert.to_dict
   rag_trigger
   
   RAGTrigger
   rag_trigger
   
   RAGTrigger.__init__
   rag_trigger
   
   RAGTrigger.calculate_similarity
   rag_trigger
   
   RAGTrigger._extract_features
   rag_trigger
   
   RAGTrigger.check_if_known_situation
   rag_trigger
   
   RAGTrigger.trigger_rag_if_needed
   rag_trigger
   
   RAGTrigger._generate_search_query
   rag_trigger
   
   RAGTrigger._perform_rag_search
   rag_trigger
   
   RAGTrigger._search_internal_vector_store
   rag_trigger
   
   RAGTrigger._search_external_resources
   rag_trigger
   
   RAGTrigger._search_cve_database
   rag_trigger
   
   RAGTrigger._search_exploit_db
   rag_trigger
   
   RAGTrigger._search_google
   rag_trigger
   
   RAGTrigger._search_github_advisory
   rag_trigger
   
   RAGTrigger.get_alert_history
   rag_trigger
   
   RAGTrigger.clear_alert_history
   rag_trigger
   
   SelfOptimizationTrigger
   rag_trigger
   
   SelfOptimizationTrigger.__init__
   rag_trigger
   
   SelfOptimizationTrigger.trigger_internal_analysis
   rag_trigger
   
   SelfOptimizationTrigger.trigger_external_feedback
   rag_trigger
   
   SelfOptimizationTrigger.generate_optimization_decisions
   rag_trigger
   
   SelfOptimizationTrigger._collect_system_health
   rag_trigger
   
   SelfOptimizationTrigger._analyze_capabilities
   rag_trigger
   
   SelfOptimizationTrigger._assess_code_quality
   rag_trigger
   
   SelfOptimizationTrigger._collect_performance_metrics
   rag_trigger
   
   SelfOptimizationTrigger._analyze_attack_effectiveness
   rag_trigger
   
   SelfOptimizationTrigger._identify_target_patterns
   rag_trigger
   
   SelfOptimizationTrigger._detect_defense_mechanisms
   rag_trigger
   
   SelfOptimizationTrigger._extract_internal_priorities
   rag_trigger
   
   SelfOptimizationTrigger._extract_external_priorities
   rag_trigger
   
   SelfOptimizationTrigger._generate_action_recommendations
   rag_trigger
   
   SelfOptimizationTrigger._estimate_optimization_impact
   rag_trigger
   
   trigger_internal_optimization
   rag_trigger
   
   trigger_external_optimization
   rag_trigger
   
   generate_ai_optimization_plan
   rag_trigger
   - 模組: 認知核心模組(學習子系統)

2. **AI組件**
   UnknownSituationAlert
   rag_trigger
   
   UnknownSituationAlert.__init__
   rag_trigger
   
   UnknownSituationAlert.to_dict
   rag_trigger
   
   RAGTrigger
   rag_trigger
   
   RAGTrigger.__init__
   rag_trigger
   
   RAGTrigger.calculate_similarity
   rag_trigger
   
   RAGTrigger._extract_features
   rag_trigger
   
   RAGTrigger.check_if_known_situation
   rag_trigger
   
   RAGTrigger.trigger_rag_if_needed
   rag_trigger
   
   RAGTrigger._generate_search_query
   rag_trigger
   
   RAGTrigger._perform_rag_search
   rag_trigger
   
   RAGTrigger._search_internal_vector_store
   rag_trigger
   
   RAGTrigger._search_external_resources
   rag_trigger
   
   RAGTrigger._search_cve_database
   rag_trigger
   
   RAGTrigger._search_exploit_db
   rag_trigger
   
   RAGTrigger._search_google
   rag_trigger
   
   RAGTrigger._search_github_advisory
   rag_trigger
   
   RAGTrigger.get_alert_history
   rag_trigger
   
   RAGTrigger.clear_alert_history
   rag_trigger
   
   SelfOptimizationTrigger
   rag_trigger
   
   SelfOptimizationTrigger.__init__
   rag_trigger
   
   SelfOptimizationTrigger.trigger_internal_analysis
   rag_trigger
   
   SelfOptimizationTrigger.trigger_external_feedback
   rag_trigger
   
   SelfOptimizationTrigger.generate_optimization_decisions
   rag_trigger
   
   SelfOptimizationTrigger._collect_system_health
   rag_trigger
   
   SelfOptimizationTrigger._analyze_capabilities
   rag_trigger
   
   SelfOptimizationTrigger._assess_code_quality
   rag_trigger
   
   SelfOptimizationTrigger._collect_performance_metrics
   rag_trigger
   
   SelfOptimizationTrigger._analyze_attack_effectiveness
   rag_trigger
   
   SelfOptimizationTrigger._identify_target_patterns
   rag_trigger
   
   SelfOptimizationTrigger._detect_defense_mechanisms
   rag_trigger
   
   SelfOptimizationTrigger._extract_internal_priorities
   rag_trigger
   
   SelfOptimizationTrigger._extract_external_priorities
   rag_trigger
   
   SelfOptimizationTrigger._generate_action_recommendations
   rag_trigger
   
   SelfOptimizationTrigger._estimate_optimization_impact
   rag_trigger
   
   trigger_internal_optimization
   rag_trigger
   
   trigger_external_optimization
   rag_trigger
   
   generate_ai_optimization_plan
   rag_trigger
   - 模組: 認知核心模組(學習子系統)

---

### Flow 7

- **長度**: 2 步
- **起點**: ai_model_manager
- **終點**: scalable_bio_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI組件**
   AIModelManager
   ai_model_manager
   
   AIModelManager.__init__
   ai_model_manager
   
   AIModelManager.initialize_models
   ai_model_manager
   
   AIModelManager.train_models
   ai_model_manager
   
   AIModelManager._prepare_training_data
   ai_model_manager
   
   AIModelManager._create_no_data_result
   ai_model_manager
   
   AIModelManager._setup_training_config
   ai_model_manager
   
   AIModelManager._execute_training
   ai_model_manager
   
   AIModelManager._prepare_training_arrays
   ai_model_manager
   
   AIModelManager._has_real_sample_data
   ai_model_manager
   
   AIModelManager._extract_real_data_arrays
   ai_model_manager
   
   AIModelManager._generate_synthetic_data_arrays
   ai_model_manager
   
   AIModelManager._update_model_state
   ai_model_manager
   
   AIModelManager._create_success_result
   ai_model_manager
   
   AIModelManager._create_failure_result
   ai_model_manager
   
   AIModelManager.make_decision
   ai_model_manager
   
   AIModelManager._validate_decision_with_scalable_net
   ai_model_manager
   
   AIModelManager._merge_dual_outputs
   ai_model_manager
   
   AIModelManager.get_model_status
   ai_model_manager
   
   AIModelManager.update_from_experience
   ai_model_manager
   
   AIModelManager._save_model
   ai_model_manager
   
   AIModelManager.load_model
   ai_model_manager
   
   AIModelManager.predict_batch
   ai_model_manager
   
   AIModelManager._create_experience_adapter
   ai_model_manager
   - 模組: 認知核心模組

2. **AI內部能力**
   ScalableBioTrainingConfig
   scalable_bio_trainer
   
   ScalableBioTrainer
   scalable_bio_trainer
   
   ScalableBioTrainer.__init__
   scalable_bio_trainer
   
   ScalableBioTrainer.train
   scalable_bio_trainer
   
   ScalableBioTrainer._train_epoch
   scalable_bio_trainer
   
   ScalableBioTrainer._validate
   scalable_bio_trainer
   
   ScalableBioTrainer._compute_loss
   scalable_bio_trainer
   
   ScalableBioTrainer._count_correct_predictions
   scalable_bio_trainer
   
   ScalableBioTrainer.get_training_history
   scalable_bio_trainer
   
   ScalableBioTrainer.save_model
   scalable_bio_trainer
   
   ScalableBioTrainer.load_model
   scalable_bio_trainer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 19

- **長度**: 2 步
- **起點**: unified_tracer
- **終點**: unified_tracer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 27

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: experience_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 95

- **長度**: 2 步
- **起點**: ast_trace_comparator
- **終點**: ast_trace_comparator
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ComparisonMetrics
   ast_trace_comparator
   
   ComparisonMetrics.to_dict
   ast_trace_comparator
   
   StepComparison
   ast_trace_comparator
   
   ASTTraceComparator
   ast_trace_comparator
   
   ASTTraceComparator.__init__
   ast_trace_comparator
   
   ASTTraceComparator.compare
   ast_trace_comparator
   
   ASTTraceComparator._extract_expected_steps
   ast_trace_comparator
   
   ASTTraceComparator._extract_actual_steps
   ast_trace_comparator
   
   ASTTraceComparator._calculate_completion
   ast_trace_comparator
   
   ASTTraceComparator._calculate_sequence_match
   ast_trace_comparator
   
   ASTTraceComparator._longest_common_subsequence
   ast_trace_comparator
   
   ASTTraceComparator._find_extra_steps
   ast_trace_comparator
   
   ASTTraceComparator._count_success_failure
   ast_trace_comparator
   
   ASTTraceComparator._calculate_timing
   ast_trace_comparator
   
   ASTTraceComparator._calculate_overall_score
   ast_trace_comparator
   
   ASTTraceComparator.generate_feedback
   ast_trace_comparator
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   ComparisonMetrics
   ast_trace_comparator
   
   ComparisonMetrics.to_dict
   ast_trace_comparator
   
   StepComparison
   ast_trace_comparator
   
   ASTTraceComparator
   ast_trace_comparator
   
   ASTTraceComparator.__init__
   ast_trace_comparator
   
   ASTTraceComparator.compare
   ast_trace_comparator
   
   ASTTraceComparator._extract_expected_steps
   ast_trace_comparator
   
   ASTTraceComparator._extract_actual_steps
   ast_trace_comparator
   
   ASTTraceComparator._calculate_completion
   ast_trace_comparator
   
   ASTTraceComparator._calculate_sequence_match
   ast_trace_comparator
   
   ASTTraceComparator._longest_common_subsequence
   ast_trace_comparator
   
   ASTTraceComparator._find_extra_steps
   ast_trace_comparator
   
   ASTTraceComparator._count_success_failure
   ast_trace_comparator
   
   ASTTraceComparator._calculate_timing
   ast_trace_comparator
   
   ASTTraceComparator._calculate_overall_score
   ast_trace_comparator
   
   ASTTraceComparator.generate_feedback
   ast_trace_comparator
   - 模組: 認知核心模組(學習子系統)

---

### Flow 107

- **長度**: 2 步
- **起點**: experience_manager
- **終點**: experience_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 118

- **長度**: 2 步
- **起點**: plan_executor
- **終點**: unified_tracer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   PlanExecutor
   plan_executor
   
   PlanExecutor.__init__
   plan_executor
   
   PlanExecutor.execute_plan
   plan_executor
   
   PlanExecutor._publish_completion_event
   plan_executor
   
   PlanExecutor._execute_step
   plan_executor
   
   PlanExecutor._prepare_task_payload
   plan_executor
   
   PlanExecutor._send_task
   plan_executor
   
   PlanExecutor._wait_for_result
   plan_executor
   
   PlanExecutor._on_task_completed
   plan_executor
   
   PlanExecutor._check_dependencies
   plan_executor
   
   PlanExecutor._should_continue
   plan_executor
   
   PlanExecutor._record_skipped_step
   plan_executor
   
   PlanExecutor._calculate_metrics
   plan_executor
   
   PlanExecutor._calculate_sequence_accuracy
   plan_executor
   
   PlanExecutor._generate_recommendations
   plan_executor
   
   PlanExecutor._persist_result
   plan_executor
   
   PlanExecutor.get_session
   plan_executor
   
   PlanExecutor.abort_session
   plan_executor
   - 模組: 任務規劃模組

2. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 119

- **長度**: 2 步
- **起點**: rl_trainers
- **終點**: rl_models
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI組件**
   DQNTrainer
   rl_trainers
   
   DQNTrainer.__init__
   rl_trainers
   
   DQNTrainer.select_action
   rl_trainers
   
   DQNTrainer.train_step
   rl_trainers
   
   DQNTrainer.get_metrics
   rl_trainers
   
   DQNTrainer.save
   rl_trainers
   
   DQNTrainer.load
   rl_trainers
   
   PPOTrainer
   rl_trainers
   
   PPOTrainer.__init__
   rl_trainers
   
   PPOTrainer.select_action
   rl_trainers
   
   PPOTrainer.store_transition
   rl_trainers
   
   PPOTrainer.update
   rl_trainers
   
   PPOTrainer.get_metrics
   rl_trainers
   
   PPOTrainer.save
   rl_trainers
   
   PPOTrainer.load
   rl_trainers
   - 模組: 認知核心模組(學習子系統)

2. **AI內部能力**
   DQNNetwork
   rl_models
   
   DQNNetwork.__init__
   rl_models
   
   DQNNetwork.forward
   rl_models
   
   DQNNetwork.select_action
   rl_models
   
   DQNNetwork._get_activation
   rl_models
   
   DQNNetwork._init_weights
   rl_models
   
   ActorCritic
   rl_models
   
   ActorCritic.__init__
   rl_models
   
   ActorCritic.forward
   rl_models
   
   ActorCritic.select_action
   rl_models
   
   ActorCritic.evaluate_actions
   rl_models
   
   ActorCritic._build_feature_extractor
   rl_models
   
   ActorCritic._build_actor
   rl_models
   
   ActorCritic._build_critic
   rl_models
   
   ActorCritic._get_activation
   rl_models
   
   ActorCritic._init_weights
   rl_models
   
   ReplayBuffer
   rl_models
   
   ReplayBuffer.__init__
   rl_models
   
   ReplayBuffer.push
   rl_models
   
   ReplayBuffer.sample
   rl_models
   
   ReplayBuffer.__len__
   rl_models
   
   RolloutBuffer
   rl_models
   
   RolloutBuffer.__init__
   rl_models
   
   RolloutBuffer.push
   rl_models
   
   RolloutBuffer.get
   rl_models
   
   RolloutBuffer.compute_returns
   rl_models
   
   RolloutBuffer.clear
   rl_models
   
   RolloutBuffer.__len__
   rl_models
   - 模組: 認知核心模組(學習子系統)

---

### Flow 134

- **長度**: 3 步
- **起點**: experience_manager
- **終點**: notification_system
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   NotificationLevel
   notification_system
   
   NotificationType
   notification_system
   
   UserNotification
   notification_system
   
   UserNotification.__init__
   notification_system
   
   UserNotification.to_dict
   notification_system
   
   UserNotification.to_log_message
   notification_system
   
   NotificationSystem
   notification_system
   
   NotificationSystem.__init__
   notification_system
   
   NotificationSystem.register_callback
   notification_system
   
   NotificationSystem.notify
   notification_system
   
   NotificationSystem.notify_unknown_situation
   notification_system
   
   NotificationSystem.notify_rag_triggered
   notification_system
   
   NotificationSystem.notify_rag_completed
   notification_system
   
   NotificationSystem.notify_rag_failed
   notification_system
   
   NotificationSystem.notify_learning_started
   notification_system
   
   NotificationSystem.notify_learning_completed
   notification_system
   
   NotificationSystem.get_notification_history
   notification_system
   
   NotificationSystem.clear_history
   notification_system
   
   get_notification_system
   notification_system
   - 模組: 認知核心模組(學習子系統)

3. **程式組件**
   NotificationLevel
   notification_system
   
   NotificationType
   notification_system
   
   UserNotification
   notification_system
   
   UserNotification.__init__
   notification_system
   
   UserNotification.to_dict
   notification_system
   
   UserNotification.to_log_message
   notification_system
   
   NotificationSystem
   notification_system
   
   NotificationSystem.__init__
   notification_system
   
   NotificationSystem.register_callback
   notification_system
   
   NotificationSystem.notify
   notification_system
   
   NotificationSystem.notify_unknown_situation
   notification_system
   
   NotificationSystem.notify_rag_triggered
   notification_system
   
   NotificationSystem.notify_rag_completed
   notification_system
   
   NotificationSystem.notify_rag_failed
   notification_system
   
   NotificationSystem.notify_learning_started
   notification_system
   
   NotificationSystem.notify_learning_completed
   notification_system
   
   NotificationSystem.get_notification_history
   notification_system
   
   NotificationSystem.clear_history
   notification_system
   
   get_notification_system
   notification_system
   - 模組: 認知核心模組(學習子系統)

---

### Flow 135

- **長度**: 2 步
- **起點**: experience_manager
- **終點**: rag_trigger
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

2. **AI組件**
   UnknownSituationAlert
   rag_trigger
   
   UnknownSituationAlert.__init__
   rag_trigger
   
   UnknownSituationAlert.to_dict
   rag_trigger
   
   RAGTrigger
   rag_trigger
   
   RAGTrigger.__init__
   rag_trigger
   
   RAGTrigger.calculate_similarity
   rag_trigger
   
   RAGTrigger._extract_features
   rag_trigger
   
   RAGTrigger.check_if_known_situation
   rag_trigger
   
   RAGTrigger.trigger_rag_if_needed
   rag_trigger
   
   RAGTrigger._generate_search_query
   rag_trigger
   
   RAGTrigger._perform_rag_search
   rag_trigger
   
   RAGTrigger._search_internal_vector_store
   rag_trigger
   
   RAGTrigger._search_external_resources
   rag_trigger
   
   RAGTrigger._search_cve_database
   rag_trigger
   
   RAGTrigger._search_exploit_db
   rag_trigger
   
   RAGTrigger._search_google
   rag_trigger
   
   RAGTrigger._search_github_advisory
   rag_trigger
   
   RAGTrigger.get_alert_history
   rag_trigger
   
   RAGTrigger.clear_alert_history
   rag_trigger
   
   SelfOptimizationTrigger
   rag_trigger
   
   SelfOptimizationTrigger.__init__
   rag_trigger
   
   SelfOptimizationTrigger.trigger_internal_analysis
   rag_trigger
   
   SelfOptimizationTrigger.trigger_external_feedback
   rag_trigger
   
   SelfOptimizationTrigger.generate_optimization_decisions
   rag_trigger
   
   SelfOptimizationTrigger._collect_system_health
   rag_trigger
   
   SelfOptimizationTrigger._analyze_capabilities
   rag_trigger
   
   SelfOptimizationTrigger._assess_code_quality
   rag_trigger
   
   SelfOptimizationTrigger._collect_performance_metrics
   rag_trigger
   
   SelfOptimizationTrigger._analyze_attack_effectiveness
   rag_trigger
   
   SelfOptimizationTrigger._identify_target_patterns
   rag_trigger
   
   SelfOptimizationTrigger._detect_defense_mechanisms
   rag_trigger
   
   SelfOptimizationTrigger._extract_internal_priorities
   rag_trigger
   
   SelfOptimizationTrigger._extract_external_priorities
   rag_trigger
   
   SelfOptimizationTrigger._generate_action_recommendations
   rag_trigger
   
   SelfOptimizationTrigger._estimate_optimization_impact
   rag_trigger
   
   trigger_internal_optimization
   rag_trigger
   
   trigger_external_optimization
   rag_trigger
   
   generate_ai_optimization_plan
   rag_trigger
   - 模組: 認知核心模組(學習子系統)

---

### Flow 193

- **長度**: 2 步
- **起點**: sync_experiences
- **終點**: event_listener
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   sync_experiences_to_vector_store
   sync_experiences
   
   sync_knowledge_base_to_vector_store
   sync_experiences
   
   main
   sync_experiences
   - 模組: 認知核心模組

2. **程式組件**
   ExternalLearningListener
   event_listener
   
   ExternalLearningListener.__init__
   event_listener
   
   ExternalLearningListener.broker
   event_listener
   
   ExternalLearningListener.connector
   event_listener
   
   ExternalLearningListener.knowledge_manager
   event_listener
   
   ExternalLearningListener.start_listening
   event_listener
   
   ExternalLearningListener.stop_listening
   event_listener
   
   ExternalLearningListener._on_result_received
   event_listener
   
   ExternalLearningListener._process_finding
   event_listener
   
   ExternalLearningListener.get_statistics
   event_listener
   
   main
   event_listener
   - 模組: 認知核心模組(學習子系統)

---

### Flow 198

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: model_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **AI組件**
   ModelTrainer
   model_trainer
   
   ModelTrainer.__init__
   model_trainer
   
   ModelTrainer.train
   model_trainer
   
   ModelTrainer.train_supervised
   model_trainer
   
   ModelTrainer.train_reinforcement
   model_trainer
   
   ModelTrainer.train_dqn
   model_trainer
   
   ModelTrainer.train_ppo
   model_trainer
   
   ModelTrainer._prepare_supervised_data
   model_trainer
   
   ModelTrainer._extract_features
   model_trainer
   
   ModelTrainer._prepare_rl_data
   model_trainer
   
   ModelTrainer._build_state_vector
   model_trainer
   
   ModelTrainer._encode_attack_type
   model_trainer
   
   ModelTrainer._encode_action
   model_trainer
   
   ModelTrainer._calculate_step_reward
   model_trainer
   
   ModelTrainer._train_model_supervised
   model_trainer
   
   ModelTrainer._train_model_rl
   model_trainer
   
   ModelTrainer._evaluate_model
   model_trainer
   
   ModelTrainer._save_model
   model_trainer
   
   ModelTrainer.load_model
   model_trainer
   
   ModelTrainer.test_on_scenario
   model_trainer
   
   ModelTrainer._increment_version
   model_trainer
   
   ModelTrainer._persist_training_result
   model_trainer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 203

- **長度**: 2 步
- **起點**: ai_model_manager
- **終點**: model_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   AIModelManager
   ai_model_manager
   
   AIModelManager.__init__
   ai_model_manager
   
   AIModelManager.initialize_models
   ai_model_manager
   
   AIModelManager.train_models
   ai_model_manager
   
   AIModelManager._prepare_training_data
   ai_model_manager
   
   AIModelManager._create_no_data_result
   ai_model_manager
   
   AIModelManager._setup_training_config
   ai_model_manager
   
   AIModelManager._execute_training
   ai_model_manager
   
   AIModelManager._prepare_training_arrays
   ai_model_manager
   
   AIModelManager._has_real_sample_data
   ai_model_manager
   
   AIModelManager._extract_real_data_arrays
   ai_model_manager
   
   AIModelManager._generate_synthetic_data_arrays
   ai_model_manager
   
   AIModelManager._update_model_state
   ai_model_manager
   
   AIModelManager._create_success_result
   ai_model_manager
   
   AIModelManager._create_failure_result
   ai_model_manager
   
   AIModelManager.make_decision
   ai_model_manager
   
   AIModelManager._validate_decision_with_scalable_net
   ai_model_manager
   
   AIModelManager._merge_dual_outputs
   ai_model_manager
   
   AIModelManager.get_model_status
   ai_model_manager
   
   AIModelManager.update_from_experience
   ai_model_manager
   
   AIModelManager._save_model
   ai_model_manager
   
   AIModelManager.load_model
   ai_model_manager
   
   AIModelManager.predict_batch
   ai_model_manager
   
   AIModelManager._create_experience_adapter
   ai_model_manager
   - 模組: 認知核心模組

2. **AI組件**
   ModelTrainer
   model_trainer
   
   ModelTrainer.__init__
   model_trainer
   
   ModelTrainer.train
   model_trainer
   
   ModelTrainer.train_supervised
   model_trainer
   
   ModelTrainer.train_reinforcement
   model_trainer
   
   ModelTrainer.train_dqn
   model_trainer
   
   ModelTrainer.train_ppo
   model_trainer
   
   ModelTrainer._prepare_supervised_data
   model_trainer
   
   ModelTrainer._extract_features
   model_trainer
   
   ModelTrainer._prepare_rl_data
   model_trainer
   
   ModelTrainer._build_state_vector
   model_trainer
   
   ModelTrainer._encode_attack_type
   model_trainer
   
   ModelTrainer._encode_action
   model_trainer
   
   ModelTrainer._calculate_step_reward
   model_trainer
   
   ModelTrainer._train_model_supervised
   model_trainer
   
   ModelTrainer._train_model_rl
   model_trainer
   
   ModelTrainer._evaluate_model
   model_trainer
   
   ModelTrainer._save_model
   model_trainer
   
   ModelTrainer.load_model
   model_trainer
   
   ModelTrainer.test_on_scenario
   model_trainer
   
   ModelTrainer._increment_version
   model_trainer
   
   ModelTrainer._persist_training_result
   model_trainer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 204

- **長度**: 2 步
- **起點**: ai_model_manager
- **終點**: experience_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI組件**
   AIModelManager
   ai_model_manager
   
   AIModelManager.__init__
   ai_model_manager
   
   AIModelManager.initialize_models
   ai_model_manager
   
   AIModelManager.train_models
   ai_model_manager
   
   AIModelManager._prepare_training_data
   ai_model_manager
   
   AIModelManager._create_no_data_result
   ai_model_manager
   
   AIModelManager._setup_training_config
   ai_model_manager
   
   AIModelManager._execute_training
   ai_model_manager
   
   AIModelManager._prepare_training_arrays
   ai_model_manager
   
   AIModelManager._has_real_sample_data
   ai_model_manager
   
   AIModelManager._extract_real_data_arrays
   ai_model_manager
   
   AIModelManager._generate_synthetic_data_arrays
   ai_model_manager
   
   AIModelManager._update_model_state
   ai_model_manager
   
   AIModelManager._create_success_result
   ai_model_manager
   
   AIModelManager._create_failure_result
   ai_model_manager
   
   AIModelManager.make_decision
   ai_model_manager
   
   AIModelManager._validate_decision_with_scalable_net
   ai_model_manager
   
   AIModelManager._merge_dual_outputs
   ai_model_manager
   
   AIModelManager.get_model_status
   ai_model_manager
   
   AIModelManager.update_from_experience
   ai_model_manager
   
   AIModelManager._save_model
   ai_model_manager
   
   AIModelManager.load_model
   ai_model_manager
   
   AIModelManager.predict_batch
   ai_model_manager
   
   AIModelManager._create_experience_adapter
   ai_model_manager
   - 模組: 認知核心模組

2. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 205

- **長度**: 2 步
- **起點**: external_loop_connector
- **終點**: model_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

2. **AI組件**
   ModelTrainer
   model_trainer
   
   ModelTrainer.__init__
   model_trainer
   
   ModelTrainer.train
   model_trainer
   
   ModelTrainer.train_supervised
   model_trainer
   
   ModelTrainer.train_reinforcement
   model_trainer
   
   ModelTrainer.train_dqn
   model_trainer
   
   ModelTrainer.train_ppo
   model_trainer
   
   ModelTrainer._prepare_supervised_data
   model_trainer
   
   ModelTrainer._extract_features
   model_trainer
   
   ModelTrainer._prepare_rl_data
   model_trainer
   
   ModelTrainer._build_state_vector
   model_trainer
   
   ModelTrainer._encode_attack_type
   model_trainer
   
   ModelTrainer._encode_action
   model_trainer
   
   ModelTrainer._calculate_step_reward
   model_trainer
   
   ModelTrainer._train_model_supervised
   model_trainer
   
   ModelTrainer._train_model_rl
   model_trainer
   
   ModelTrainer._evaluate_model
   model_trainer
   
   ModelTrainer._save_model
   model_trainer
   
   ModelTrainer.load_model
   model_trainer
   
   ModelTrainer.test_on_scenario
   model_trainer
   
   ModelTrainer._increment_version
   model_trainer
   
   ModelTrainer._persist_training_result
   model_trainer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 229

- **長度**: 2 步
- **起點**: external_loop_connector
- **終點**: ast_trace_comparator
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   ExternalLoopConnector
   external_loop_connector
   
   ExternalLoopConnector.__init__
   external_loop_connector
   
   ExternalLoopConnector.comparator
   external_loop_connector
   
   ExternalLoopConnector.trainer
   external_loop_connector
   
   ExternalLoopConnector.weight_manager
   external_loop_connector
   
   ExternalLoopConnector.process_execution_result
   external_loop_connector
   
   ExternalLoopConnector._analyze_deviations
   external_loop_connector
   
   ExternalLoopConnector._is_significant_deviation
   external_loop_connector
   
   ExternalLoopConnector._train_from_experience
   external_loop_connector
   
   ExternalLoopConnector._register_new_weights
   external_loop_connector
   
   ExternalLoopConnector.get_loop_status
   external_loop_connector
   - 模組: 認知核心模組

2. **程式組件**
   ComparisonMetrics
   ast_trace_comparator
   
   ComparisonMetrics.to_dict
   ast_trace_comparator
   
   StepComparison
   ast_trace_comparator
   
   ASTTraceComparator
   ast_trace_comparator
   
   ASTTraceComparator.__init__
   ast_trace_comparator
   
   ASTTraceComparator.compare
   ast_trace_comparator
   
   ASTTraceComparator._extract_expected_steps
   ast_trace_comparator
   
   ASTTraceComparator._extract_actual_steps
   ast_trace_comparator
   
   ASTTraceComparator._calculate_completion
   ast_trace_comparator
   
   ASTTraceComparator._calculate_sequence_match
   ast_trace_comparator
   
   ASTTraceComparator._longest_common_subsequence
   ast_trace_comparator
   
   ASTTraceComparator._find_extra_steps
   ast_trace_comparator
   
   ASTTraceComparator._count_success_failure
   ast_trace_comparator
   
   ASTTraceComparator._calculate_timing
   ast_trace_comparator
   
   ASTTraceComparator._calculate_overall_score
   ast_trace_comparator
   
   ASTTraceComparator.generate_feedback
   ast_trace_comparator
   - 模組: 認知核心模組(學習子系統)

---

### Flow 234

- **長度**: 2 步
- **起點**: continuous_learning
- **終點**: experience_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ContinuousLearningEngine
   continuous_learning
   
   ContinuousLearningEngine.__init__
   continuous_learning
   
   ContinuousLearningEngine.process_sandbox_experience
   continuous_learning
   
   ContinuousLearningEngine.process_production_experience
   continuous_learning
   
   ContinuousLearningEngine._check_and_trigger_batch_training
   continuous_learning
   
   ContinuousLearningEngine._experience_to_tensors
   continuous_learning
   
   ContinuousLearningEngine.get_statistics
   continuous_learning
   
   create_continuous_learning_engine
   continuous_learning
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   ExperienceTransition
   experience_manager
   
   ExperienceTransition.__init__
   experience_manager
   
   ExperienceTransition.to_dict
   experience_manager
   
   ExperienceManager
   experience_manager
   
   ExperienceManager.__init__
   experience_manager
   
   ExperienceManager.push
   experience_manager
   
   ExperienceManager._persist_to_integration
   experience_manager
   
   ExperienceManager.load_from_integration
   experience_manager
   
   ExperienceManager.get_experiences_by_environment
   experience_manager
   
   ExperienceManager.sample
   experience_manager
   
   ExperienceManager.prioritized_sample
   experience_manager
   
   ExperienceManager.create_dataset
   experience_manager
   
   ExperienceManager.get_statistics
   experience_manager
   
   ExperienceManager.clear
   experience_manager
   
   ExperienceManager.__len__
   experience_manager
   
   ExperienceManager.__repr__
   experience_manager
   
   ExperienceManager.add_sample
   experience_manager
   
   ExperienceManager.get_high_quality_samples
   experience_manager
   
   ExperienceManager.trigger_learning_with_rag
   experience_manager
   
   ExperienceManager._generate_optimization_plan
   experience_manager
   
   ExperienceManager._validate_optimization
   experience_manager
   
   ExperienceManager._save_optimization
   experience_manager
   
   integrate_with_repository_example
   experience_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 235

- **長度**: 2 步
- **起點**: continuous_learning
- **終點**: model_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **程式組件**
   ContinuousLearningEngine
   continuous_learning
   
   ContinuousLearningEngine.__init__
   continuous_learning
   
   ContinuousLearningEngine.process_sandbox_experience
   continuous_learning
   
   ContinuousLearningEngine.process_production_experience
   continuous_learning
   
   ContinuousLearningEngine._check_and_trigger_batch_training
   continuous_learning
   
   ContinuousLearningEngine._experience_to_tensors
   continuous_learning
   
   ContinuousLearningEngine.get_statistics
   continuous_learning
   
   create_continuous_learning_engine
   continuous_learning
   - 模組: 認知核心模組(學習子系統)

2. **AI組件**
   ModelTrainer
   model_trainer
   
   ModelTrainer.__init__
   model_trainer
   
   ModelTrainer.train
   model_trainer
   
   ModelTrainer.train_supervised
   model_trainer
   
   ModelTrainer.train_reinforcement
   model_trainer
   
   ModelTrainer.train_dqn
   model_trainer
   
   ModelTrainer.train_ppo
   model_trainer
   
   ModelTrainer._prepare_supervised_data
   model_trainer
   
   ModelTrainer._extract_features
   model_trainer
   
   ModelTrainer._prepare_rl_data
   model_trainer
   
   ModelTrainer._build_state_vector
   model_trainer
   
   ModelTrainer._encode_attack_type
   model_trainer
   
   ModelTrainer._encode_action
   model_trainer
   
   ModelTrainer._calculate_step_reward
   model_trainer
   
   ModelTrainer._train_model_supervised
   model_trainer
   
   ModelTrainer._train_model_rl
   model_trainer
   
   ModelTrainer._evaluate_model
   model_trainer
   
   ModelTrainer._save_model
   model_trainer
   
   ModelTrainer.load_model
   model_trainer
   
   ModelTrainer.test_on_scenario
   model_trainer
   
   ModelTrainer._increment_version
   model_trainer
   
   ModelTrainer._persist_training_result
   model_trainer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 257

- **長度**: 2 步
- **起點**: execution_status_monitor
- **終點**: module_knowledge_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   EnvironmentType
   execution_status_monitor
   
   ExecutionContext
   execution_status_monitor
   
   ExecutionContext.__post_init__
   execution_status_monitor
   
   ExecutionContext.to_dict
   execution_status_monitor
   
   ExecutionContext.from_dict
   execution_status_monitor
   
   ExecutionMonitor
   execution_status_monitor
   
   ExecutionMonitor.__init__
   execution_status_monitor
   
   ExecutionMonitor.start_task_execution
   execution_status_monitor
   
   ExecutionMonitor.complete_task_execution
   execution_status_monitor
   
   ExecutionMonitor.record_error
   execution_status_monitor
   
   ExecutionMonitor.record_step
   execution_status_monitor
   
   ExecutionMonitor.record_decision_point
   execution_status_monitor
   
   ExecutionMonitor.record_tool_invocation
   execution_status_monitor
   
   ExecutionMonitor.get_task_traces
   execution_status_monitor
   
   ExecutionMonitor.get_task_errors
   execution_status_monitor
   
   ExecutionStatusMonitor
   execution_status_monitor
   
   ExecutionStatusMonitor.__init__
   execution_status_monitor
   
   ExecutionStatusMonitor.record_worker_heartbeat
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_start
   execution_status_monitor
   
   ExecutionStatusMonitor.record_task_completion
   execution_status_monitor
   
   ExecutionStatusMonitor.get_system_health
   execution_status_monitor
   
   ExecutionStatusMonitor.check_sla_violations
   execution_status_monitor
   
   ExecutionStatusMonitor._get_recent_alerts
   execution_status_monitor
   
   ExecutionStatusMonitor.add_alert
   execution_status_monitor
   
   ExecutionStatusMonitor.start_monitoring
   execution_status_monitor
   - 模組: 任務規劃模組

2. **程式組件**
   ExecutionContext
   module_knowledge_manager
   
   ExecutionContext.to_dict
   module_knowledge_manager
   
   KnowledgeMatch
   module_knowledge_manager
   
   LearningRecommendation
   module_knowledge_manager
   
   ModuleKnowledgeManager
   module_knowledge_manager
   
   ModuleKnowledgeManager.__init__
   module_knowledge_manager
   
   ModuleKnowledgeManager.load_all_knowledge
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_module_name
   module_knowledge_manager
   
   ModuleKnowledgeManager._parse_markdown_report
   module_knowledge_manager
   
   ModuleKnowledgeManager._build_scenario_index
   module_knowledge_manager
   
   ModuleKnowledgeManager.match_execution_result
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_features
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_keywords
   module_knowledge_manager
   
   ModuleKnowledgeManager._calculate_similarity
   module_knowledge_manager
   
   ModuleKnowledgeManager.generate_recommendation
   module_knowledge_manager
   
   ModuleKnowledgeManager._build_rag_query
   module_knowledge_manager
   
   ModuleKnowledgeManager._query_rag
   module_knowledge_manager
   
   ModuleKnowledgeManager._generate_recommendation_id
   module_knowledge_manager
   
   ModuleKnowledgeManager.get_statistics
   module_knowledge_manager
   
   ModuleKnowledgeManager.export_knowledge_summary
   module_knowledge_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 292

- **長度**: 2 步
- **起點**: unified_executor
- **終點**: continuous_learning
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   AttackTarget
   unified_executor
   
   AttackPlan
   unified_executor
   
   ExperienceSample
   unified_executor
   
   ExecutionResult
   unified_executor
   
   ModelTrainingConfig
   unified_executor
   
   AttackFeedback
   unified_executor
   
   StrategyOptimization
   unified_executor
   
   UnifiedAttackExecutor
   unified_executor
   
   UnifiedAttackExecutor.__init__
   unified_executor
   
   UnifiedAttackExecutor.rag_engine
   unified_executor
   
   UnifiedAttackExecutor.experience_manager
   unified_executor
   
   UnifiedAttackExecutor.model_trainer
   unified_executor
   
   UnifiedAttackExecutor.continuous_learning_engine
   unified_executor
   
   UnifiedAttackExecutor.message_broker
   unified_executor
   
   UnifiedAttackExecutor.feedback_optimizer
   unified_executor
   
   UnifiedAttackExecutor.execute
   unified_executor
   
   UnifiedAttackExecutor.execute_with_context
   unified_executor
   
   UnifiedAttackExecutor._sandbox_execution
   unified_executor
   
   UnifiedAttackExecutor._production_execution
   unified_executor
   
   UnifiedAttackExecutor._should_learn_from_production
   unified_executor
   
   UnifiedAttackExecutor._generate_enhanced_plan
   unified_executor
   
   UnifiedAttackExecutor._execute_attack_plan
   unified_executor
   
   UnifiedAttackExecutor._learn_from_execution
   unified_executor
   
   UnifiedAttackExecutor._extract_experience_samples
   unified_executor
   
   UnifiedAttackExecutor._calculate_step_reward
   unified_executor
   
   UnifiedAttackExecutor._calculate_quality_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_confidence
   unified_executor
   
   UnifiedAttackExecutor._update_rag_knowledge
   unified_executor
   
   UnifiedAttackExecutor._auto_train
   unified_executor
   
   UnifiedAttackExecutor.get_learning_status
   unified_executor
   
   UnifiedAttackExecutor.collect_feedback
   unified_executor
   
   UnifiedAttackExecutor._calculate_effectiveness_score
   unified_executor
   
   UnifiedAttackExecutor._calculate_error_rate
   unified_executor
   
   UnifiedAttackExecutor._optimize_strategies
   unified_executor
   
   UnifiedAttackExecutor._apply_optimization
   unified_executor
   
   UnifiedAttackExecutor.get_optimization_report
   unified_executor
   
   FeedbackOptimizer
   unified_executor
   
   FeedbackOptimizer.__init__
   unified_executor
   
   FeedbackOptimizer.analyze_and_optimize
   unified_executor
   
   FeedbackOptimizer._identify_poor_strategies
   unified_executor
   
   FeedbackOptimizer._identify_error_patterns
   unified_executor
   
   FeedbackOptimizer._identify_waf_bypass_opportunities
   unified_executor
   
   FeedbackOptimizer._generate_optimization
   unified_executor
   
   FeedbackOptimizer._generate_error_fix
   unified_executor
   
   FeedbackOptimizer._generate_waf_bypass_optimization
   unified_executor
   - 模組: 任務規劃模組

2. **程式組件**
   ContinuousLearningEngine
   continuous_learning
   
   ContinuousLearningEngine.__init__
   continuous_learning
   
   ContinuousLearningEngine.process_sandbox_experience
   continuous_learning
   
   ContinuousLearningEngine.process_production_experience
   continuous_learning
   
   ContinuousLearningEngine._check_and_trigger_batch_training
   continuous_learning
   
   ContinuousLearningEngine._experience_to_tensors
   continuous_learning
   
   ContinuousLearningEngine.get_statistics
   continuous_learning
   
   create_continuous_learning_engine
   continuous_learning
   - 模組: 認知核心模組(學習子系統)

---

### Flow 315

- **長度**: 3 步
- **起點**: unified_tracer
- **終點**: trace_recorder
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   get_global_recorder
   execution_tracer
   
   record_execution_trace
   execution_tracer
   - 模組: 認知核心模組(學習子系統)

3. **程式組件**
   TraceType
   trace_recorder
   
   TraceEntry
   trace_recorder
   
   TraceEntry.to_dict
   trace_recorder
   
   TraceEntry.to_json
   trace_recorder
   
   ExecutionTrace
   trace_recorder
   
   ExecutionTrace.add_entry
   trace_recorder
   
   ExecutionTrace.get_entries_by_task
   trace_recorder
   
   ExecutionTrace.get_entries_by_type
   trace_recorder
   
   ExecutionTrace.finalize
   trace_recorder
   
   ExecutionTrace.to_dict
   trace_recorder
   
   ExecutionTrace.to_json
   trace_recorder
   
   TraceRecorder
   trace_recorder
   
   TraceRecorder.__init__
   trace_recorder
   
   TraceRecorder.start_trace
   trace_recorder
   
   TraceRecorder.record
   trace_recorder
   
   TraceRecorder.record_task_start
   trace_recorder
   
   TraceRecorder.record_task_end
   trace_recorder
   
   TraceRecorder.record_http_request
   trace_recorder
   
   TraceRecorder.record_http_response
   trace_recorder
   
   TraceRecorder.record_log
   trace_recorder
   
   TraceRecorder.record_error
   trace_recorder
   
   TraceRecorder.finalize_trace
   trace_recorder
   
   TraceRecorder.get_trace
   trace_recorder
   - 模組: 認知核心模組(學習子系統)

---

### Flow 332

- **長度**: 2 步
- **起點**: enhanced_decision_agent
- **終點**: unified_tracer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **AI對外能力**
   DecisionContext
   enhanced_decision_agent
   
   DecisionContext.__init__
   enhanced_decision_agent
   
   Decision
   enhanced_decision_agent
   
   Decision.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.__init__
   enhanced_decision_agent
   
   EnhancedDecisionAgent._setup_logger
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide
   enhanced_decision_agent
   
   EnhancedDecisionAgent._sync_make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._convert_decision_to_intent
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_neural_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._ensemble_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.query_internal_capabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.record_execution_feedback
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_target_vulnerabilities
   enhanced_decision_agent
   
   EnhancedDecisionAgent.identify_high_risk_cves
   enhanced_decision_agent
   
   EnhancedDecisionAgent.generate_waf_bypass_payloads
   enhanced_decision_agent
   
   EnhancedDecisionAgent.analyze_web_architecture
   enhanced_decision_agent
   
   EnhancedDecisionAgent.make_enhanced_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._async_wrapper
   enhanced_decision_agent
   
   EnhancedDecisionAgent._assess_risk_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_experience_driven_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._find_similar_experiences
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_similarity
   enhanced_decision_agent
   
   EnhancedDecisionAgent._apply_decision_rules
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_rule_action
   enhanced_decision_agent
   
   EnhancedDecisionAgent._select_best_tool
   enhanced_decision_agent
   
   EnhancedDecisionAgent._suggest_alternative_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.execute_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_tool_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_vulnerability_test
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_mode_switch
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_strategy_change
   enhanced_decision_agent
   
   EnhancedDecisionAgent._execute_stop
   enhanced_decision_agent
   
   EnhancedDecisionAgent._make_default_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent._record_decision
   enhanced_decision_agent
   
   EnhancedDecisionAgent.get_decision_stats
   enhanced_decision_agent
   
   EnhancedDecisionAgent.export_decision_analysis
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_scan_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_scan_target
   enhanced_decision_agent
   
   EnhancedDecisionAgent._build_scan_params
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_scan_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase1_strategy
   enhanced_decision_agent
   
   EnhancedDecisionAgent.decide_phase2_targets
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_tools_for_vuln
   enhanced_decision_agent
   
   EnhancedDecisionAgent._get_default_bounty_table
   enhanced_decision_agent
   
   EnhancedDecisionAgent.evaluate_phase2_results
   enhanced_decision_agent
   
   EnhancedDecisionAgent._analyze_vulnerability_chains
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_report_guidance
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_cvss
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_action_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._decide_waf_bypass
   enhanced_decision_agent
   
   EnhancedDecisionAgent.adaptive_rate_limiting
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_waf_strategies
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_rate_profiles
   enhanced_decision_agent
   
   EnhancedDecisionAgent._initialize_bounty_matrix
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_phase0_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._encode_asset_features
   enhanced_decision_agent
   
   EnhancedDecisionAgent._estimate_phase1_time
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_bounty_value
   enhanced_decision_agent
   
   EnhancedDecisionAgent._calculate_waf_interference
   enhanced_decision_agent
   
   EnhancedDecisionAgent._query_historical_success
   enhanced_decision_agent
   
   EnhancedDecisionAgent._generate_next_steps
   enhanced_decision_agent
   
   demo_enhanced_decision_agent
   enhanced_decision_agent
   - 模組: 認知核心模組

2. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 335

- **長度**: 2 步
- **起點**: event_listener
- **終點**: module_knowledge_manager
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   ExternalLearningListener
   event_listener
   
   ExternalLearningListener.__init__
   event_listener
   
   ExternalLearningListener.broker
   event_listener
   
   ExternalLearningListener.connector
   event_listener
   
   ExternalLearningListener.knowledge_manager
   event_listener
   
   ExternalLearningListener.start_listening
   event_listener
   
   ExternalLearningListener.stop_listening
   event_listener
   
   ExternalLearningListener._on_result_received
   event_listener
   
   ExternalLearningListener._process_finding
   event_listener
   
   ExternalLearningListener.get_statistics
   event_listener
   
   main
   event_listener
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   ExecutionContext
   module_knowledge_manager
   
   ExecutionContext.to_dict
   module_knowledge_manager
   
   KnowledgeMatch
   module_knowledge_manager
   
   LearningRecommendation
   module_knowledge_manager
   
   ModuleKnowledgeManager
   module_knowledge_manager
   
   ModuleKnowledgeManager.__init__
   module_knowledge_manager
   
   ModuleKnowledgeManager.load_all_knowledge
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_module_name
   module_knowledge_manager
   
   ModuleKnowledgeManager._parse_markdown_report
   module_knowledge_manager
   
   ModuleKnowledgeManager._build_scenario_index
   module_knowledge_manager
   
   ModuleKnowledgeManager.match_execution_result
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_features
   module_knowledge_manager
   
   ModuleKnowledgeManager._extract_keywords
   module_knowledge_manager
   
   ModuleKnowledgeManager._calculate_similarity
   module_knowledge_manager
   
   ModuleKnowledgeManager.generate_recommendation
   module_knowledge_manager
   
   ModuleKnowledgeManager._build_rag_query
   module_knowledge_manager
   
   ModuleKnowledgeManager._query_rag
   module_knowledge_manager
   
   ModuleKnowledgeManager._generate_recommendation_id
   module_knowledge_manager
   
   ModuleKnowledgeManager.get_statistics
   module_knowledge_manager
   
   ModuleKnowledgeManager.export_knowledge_summary
   module_knowledge_manager
   - 模組: 認知核心模組(學習子系統)

---

### Flow 345

- **長度**: 2 步
- **起點**: trace_recorder
- **終點**: unified_tracer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   TraceType
   trace_recorder
   
   TraceEntry
   trace_recorder
   
   TraceEntry.to_dict
   trace_recorder
   
   TraceEntry.to_json
   trace_recorder
   
   ExecutionTrace
   trace_recorder
   
   ExecutionTrace.add_entry
   trace_recorder
   
   ExecutionTrace.get_entries_by_task
   trace_recorder
   
   ExecutionTrace.get_entries_by_type
   trace_recorder
   
   ExecutionTrace.finalize
   trace_recorder
   
   ExecutionTrace.to_dict
   trace_recorder
   
   ExecutionTrace.to_json
   trace_recorder
   
   TraceRecorder
   trace_recorder
   
   TraceRecorder.__init__
   trace_recorder
   
   TraceRecorder.start_trace
   trace_recorder
   
   TraceRecorder.record
   trace_recorder
   
   TraceRecorder.record_task_start
   trace_recorder
   
   TraceRecorder.record_task_end
   trace_recorder
   
   TraceRecorder.record_http_request
   trace_recorder
   
   TraceRecorder.record_http_response
   trace_recorder
   
   TraceRecorder.record_log
   trace_recorder
   
   TraceRecorder.record_error
   trace_recorder
   
   TraceRecorder.finalize_trace
   trace_recorder
   
   TraceRecorder.get_trace
   trace_recorder
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   TraceType
   unified_tracer
   
   ExecutionTrace
   unified_tracer
   
   ExecutionTrace.__post_init__
   unified_tracer
   
   UnifiedTracer
   unified_tracer
   
   UnifiedTracer.__init__
   unified_tracer
   
   UnifiedTracer.start_session
   unified_tracer
   
   UnifiedTracer.record_trace
   unified_tracer
   
   UnifiedTracer.log_task_execution
   unified_tracer
   
   UnifiedTracer.get_traces
   unified_tracer
   
   UnifiedTracer.get_trace_records
   unified_tracer
   
   UnifiedTracer.complete_session
   unified_tracer
   
   UnifiedTracer.fail_session
   unified_tracer
   
   UnifiedTracer.create_session
   unified_tracer
   
   UnifiedTracer.get_session
   unified_tracer
   
   UnifiedTracer.abort_session
   unified_tracer
   
   UnifiedTracer.clear_traces
   unified_tracer
   
   UnifiedTracer.get_session_summary
   unified_tracer
   
   UnifiedTracer._persist_trace_record
   unified_tracer
   
   get_global_tracer
   unified_tracer
   
   record_execution_trace
   unified_tracer
   - 模組: 認知核心模組(學習子系統)

---

### Flow 372

- **長度**: 2 步
- **起點**: ai_decision_core
- **終點**: rag_trigger
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI組件

**執行路徑**:

1. **AI組件**
   UserConstraints
   ai_decision_core
   
   ScanStrategy
   ai_decision_core
   
   AIDecisionCore
   ai_decision_core
   
   AIDecisionCore.__init__
   ai_decision_core
   
   AIDecisionCore.initialize
   ai_decision_core
   
   AIDecisionCore.decide_scan_strategy
   ai_decision_core
   
   AIDecisionCore._filter_capabilities
   ai_decision_core
   
   AIDecisionCore._search_rag_suggestions
   ai_decision_core
   
   AIDecisionCore._generate_strategy
   ai_decision_core
   
   AIDecisionCore.get_flow_execution_order
   ai_decision_core
   
   AIDecisionCore.generate_attack_plan
   ai_decision_core
   
   quick_decision
   ai_decision_core
   - 模組: 認知核心模組

2. **AI組件**
   UnknownSituationAlert
   rag_trigger
   
   UnknownSituationAlert.__init__
   rag_trigger
   
   UnknownSituationAlert.to_dict
   rag_trigger
   
   RAGTrigger
   rag_trigger
   
   RAGTrigger.__init__
   rag_trigger
   
   RAGTrigger.calculate_similarity
   rag_trigger
   
   RAGTrigger._extract_features
   rag_trigger
   
   RAGTrigger.check_if_known_situation
   rag_trigger
   
   RAGTrigger.trigger_rag_if_needed
   rag_trigger
   
   RAGTrigger._generate_search_query
   rag_trigger
   
   RAGTrigger._perform_rag_search
   rag_trigger
   
   RAGTrigger._search_internal_vector_store
   rag_trigger
   
   RAGTrigger._search_external_resources
   rag_trigger
   
   RAGTrigger._search_cve_database
   rag_trigger
   
   RAGTrigger._search_exploit_db
   rag_trigger
   
   RAGTrigger._search_google
   rag_trigger
   
   RAGTrigger._search_github_advisory
   rag_trigger
   
   RAGTrigger.get_alert_history
   rag_trigger
   
   RAGTrigger.clear_alert_history
   rag_trigger
   
   SelfOptimizationTrigger
   rag_trigger
   
   SelfOptimizationTrigger.__init__
   rag_trigger
   
   SelfOptimizationTrigger.trigger_internal_analysis
   rag_trigger
   
   SelfOptimizationTrigger.trigger_external_feedback
   rag_trigger
   
   SelfOptimizationTrigger.generate_optimization_decisions
   rag_trigger
   
   SelfOptimizationTrigger._collect_system_health
   rag_trigger
   
   SelfOptimizationTrigger._analyze_capabilities
   rag_trigger
   
   SelfOptimizationTrigger._assess_code_quality
   rag_trigger
   
   SelfOptimizationTrigger._collect_performance_metrics
   rag_trigger
   
   SelfOptimizationTrigger._analyze_attack_effectiveness
   rag_trigger
   
   SelfOptimizationTrigger._identify_target_patterns
   rag_trigger
   
   SelfOptimizationTrigger._detect_defense_mechanisms
   rag_trigger
   
   SelfOptimizationTrigger._extract_internal_priorities
   rag_trigger
   
   SelfOptimizationTrigger._extract_external_priorities
   rag_trigger
   
   SelfOptimizationTrigger._generate_action_recommendations
   rag_trigger
   
   SelfOptimizationTrigger._estimate_optimization_impact
   rag_trigger
   
   trigger_internal_optimization
   rag_trigger
   
   trigger_external_optimization
   rag_trigger
   
   generate_ai_optimization_plan
   rag_trigger
   - 模組: 認知核心模組(學習子系統)

---

### Flow 379

- **長度**: 2 步
- **起點**: trace_recorder
- **終點**: trace_recorder
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: 程式組件

**執行路徑**:

1. **程式組件**
   TraceType
   trace_recorder
   
   TraceEntry
   trace_recorder
   
   TraceEntry.to_dict
   trace_recorder
   
   TraceEntry.to_json
   trace_recorder
   
   ExecutionTrace
   trace_recorder
   
   ExecutionTrace.add_entry
   trace_recorder
   
   ExecutionTrace.get_entries_by_task
   trace_recorder
   
   ExecutionTrace.get_entries_by_type
   trace_recorder
   
   ExecutionTrace.finalize
   trace_recorder
   
   ExecutionTrace.to_dict
   trace_recorder
   
   ExecutionTrace.to_json
   trace_recorder
   
   TraceRecorder
   trace_recorder
   
   TraceRecorder.__init__
   trace_recorder
   
   TraceRecorder.start_trace
   trace_recorder
   
   TraceRecorder.record
   trace_recorder
   
   TraceRecorder.record_task_start
   trace_recorder
   
   TraceRecorder.record_task_end
   trace_recorder
   
   TraceRecorder.record_http_request
   trace_recorder
   
   TraceRecorder.record_http_response
   trace_recorder
   
   TraceRecorder.record_log
   trace_recorder
   
   TraceRecorder.record_error
   trace_recorder
   
   TraceRecorder.finalize_trace
   trace_recorder
   
   TraceRecorder.get_trace
   trace_recorder
   - 模組: 認知核心模組(學習子系統)

2. **程式組件**
   TraceType
   trace_recorder
   
   TraceEntry
   trace_recorder
   
   TraceEntry.to_dict
   trace_recorder
   
   TraceEntry.to_json
   trace_recorder
   
   ExecutionTrace
   trace_recorder
   
   ExecutionTrace.add_entry
   trace_recorder
   
   ExecutionTrace.get_entries_by_task
   trace_recorder
   
   ExecutionTrace.get_entries_by_type
   trace_recorder
   
   ExecutionTrace.finalize
   trace_recorder
   
   ExecutionTrace.to_dict
   trace_recorder
   
   ExecutionTrace.to_json
   trace_recorder
   
   TraceRecorder
   trace_recorder
   
   TraceRecorder.__init__
   trace_recorder
   
   TraceRecorder.start_trace
   trace_recorder
   
   TraceRecorder.record
   trace_recorder
   
   TraceRecorder.record_task_start
   trace_recorder
   
   TraceRecorder.record_task_end
   trace_recorder
   
   TraceRecorder.record_http_request
   trace_recorder
   
   TraceRecorder.record_http_response
   trace_recorder
   
   TraceRecorder.record_log
   trace_recorder
   
   TraceRecorder.record_error
   trace_recorder
   
   TraceRecorder.finalize_trace
   trace_recorder
   
   TraceRecorder.get_trace
   trace_recorder
   - 模組: 認知核心模組(學習子系統)

---

### Flow 385

- **長度**: 2 步
- **起點**: scalable_bio_trainer
- **終點**: scalable_bio_trainer
- **主要模組**: 認知核心模組(學習子系統)
- **主要組件類型**: AI內部能力

**執行路徑**:

1. **AI內部能力**
   ScalableBioTrainingConfig
   scalable_bio_trainer
   
   ScalableBioTrainer
   scalable_bio_trainer
   
   ScalableBioTrainer.__init__
   scalable_bio_trainer
   
   ScalableBioTrainer.train
   scalable_bio_trainer
   
   ScalableBioTrainer._train_epoch
   scalable_bio_trainer
   
   ScalableBioTrainer._validate
   scalable_bio_trainer
   
   ScalableBioTrainer._compute_loss
   scalable_bio_trainer
   
   ScalableBioTrainer._count_correct_predictions
   scalable_bio_trainer
   
   ScalableBioTrainer.get_training_history
   scalable_bio_trainer
   
   ScalableBioTrainer.save_model
   scalable_bio_trainer
   
   ScalableBioTrainer.load_model
   scalable_bio_trainer
   - 模組: 認知核心模組(學習子系統)

2. **AI內部能力**
   ScalableBioTrainingConfig
   scalable_bio_trainer
   
   ScalableBioTrainer
   scalable_bio_trainer
   
   ScalableBioTrainer.__init__
   scalable_bio_trainer
   
   ScalableBioTrainer.train
   scalable_bio_trainer
   
   ScalableBioTrainer._train_epoch
   scalable_bio_trainer
   
   ScalableBioTrainer._validate
   scalable_bio_trainer
   
   ScalableBioTrainer._compute_loss
   scalable_bio_trainer
   
   ScalableBioTrainer._count_correct_predictions
   scalable_bio_trainer
   
   ScalableBioTrainer.get_training_history
   scalable_bio_trainer
   
   ScalableBioTrainer.save_model
   scalable_bio_trainer
   
   ScalableBioTrainer.load_model
   scalable_bio_trainer
   - 模組: 認知核心模組(學習子系統)

---

