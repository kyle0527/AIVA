// Example code that deserializes and serializes the model.
// extern crate serde;
// #[macro_use]
// extern crate serde_derive;
// extern crate serde_json;
//
// use generated_module::AIVASchemas;
//
// fn main() {
//     let json = r#"{"answer": 42}"#;
//     let model: AIVASchemas = serde_json::from_str(&json).unwrap();
// }

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct AivaSchemas {
    #[serde(rename = "$schema")]
    schema: String,

    #[serde(rename = "$id")]
    id: String,

    title: String,

    description: String,

    version: String,

    generated_at: String,

    generator: String,

    #[serde(rename = "$defs")]
    defs: Defs,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Defs {
    module_name: Box<AssetType>,

    message_header: MessageHeader,

    authentication: Authentication,

    rate_limit: AdaptiveBehaviorInfo,

    scan_scope: ScanScope,

    asset: Asset,

    summary: Box<StoppingReason>,

    fingerprints: AdaptiveBehaviorInfo,

    execution_error: ExecutionError,

    risk_factor: RiskFactor,

    task_dependency: TaskDependency,

    topic: Box<AssetType>,

    aiva_message: AivaMessage,

    #[serde(rename = "AIVARequest")]
    aiva_request: AivaRequest,

    #[serde(rename = "AIVAResponse")]
    aiva_response: AivaResponse,

    #[serde(rename = "AIVAEvent")]
    aiva_event: AivaEvent,

    #[serde(rename = "AIVACommand")]
    aiva_command: AivaCommand,

    scan_start_payload: ScanStartPayload,

    scan_completed_payload: ScanCompletedPayload,

    function_task_context: Authentication,

    function_task_target: FunctionTaskTarget,

    function_task_test_config: AdaptiveBehaviorInfo,

    function_task_payload: FunctionTaskPayload,

    feedback_event_payload: FeedbackEventPayload,

    task_update_payload: TaskUpdatePayload,

    config_update_payload: ConfigUpdatePayload,

    #[serde(rename = "IOCType")]
    ioc_type: Box<EnhancedFunctionTelemetry>,

    intel_source: Box<EnhancedFunctionTelemetry>,

    threat_intel_lookup_payload: ThreatIntelLookupPayload,

    threat_level: Box<AssetType>,

    threat_intel_result_payload: ThreatIntelResultPayload,

    auth_z_check_payload: AuthZCheckPayload,

    auth_z_analysis_payload: AuthZAnalysisPayload,

    auth_z_result_payload: AuthZResultPayload,

    remediation_type: RemediationType,

    vulnerability_type: AdaptiveBehaviorInfo,

    remediation_generate_payload: RemediationGeneratePayload,

    remediation_result_payload: RemediationResultPayload,

    post_ex_test_type: Box<AssetType>,

    post_ex_test_payload: PostExTestPayload,

    post_ex_result_payload: PostExResultPayload,

    biz_logic_test_payload: BizLogicTestPayload,

    biz_logic_result_payload: BizLogicResultPayload,

    #[serde(rename = "APISchemaPayload")]
    api_schema_payload: ApiSchemaPayload,

    #[serde(rename = "APITestCase")]
    api_test_case: ApiTestCase,

    #[serde(rename = "APISecurityTestPayload")]
    api_security_test_payload: ApiSecurityTestPayload,

    #[serde(rename = "EASMDiscoveryPayload")]
    easm_discovery_payload: EasmDiscoveryPayload,

    #[serde(rename = "EASMDiscoveryResult")]
    easm_discovery_result: EasmDiscoveryResult,

    standard_scenario: StandardScenario,

    scenario_test_result: ScenarioTestResult,

    exploit_payload: ExploitPayload,

    confidence: Authentication,

    test_status: Authentication,

    test_execution: TestExecution,

    severity: Box<AssetType>,

    exploit_result: ExploitResult,

    test_strategy: TestStrategy,

    vulnerability: Vulnerability,

    target: Target,

    finding_target: Target,

    finding_evidence: AdaptiveBehaviorInfo,

    finding_impact: Box<EnhancedFunctionTelemetry>,

    finding_recommendation: Authentication,

    finding_payload: FindingPayload,

    sensitive_match: SensitiveMatch,

    java_script_analysis_result: JavaScriptAnalysisResult,

    vulnerability_correlation: VulnerabilityCorrelation,

    code_level_root_cause: CodeLevelRootCause,

    #[serde(rename = "SASTDASTCorrelation")]
    sastdast_correlation: SastdastCorrelation,

    #[serde(rename = "AIVerificationRequest")]
    ai_verification_request: AiVerificationRequest,

    #[serde(rename = "AIVerificationResult")]
    ai_verification_result: AiVerificationResult,

    heartbeat_payload: HeartbeatPayload,

    module_status: ModuleStatus,

    function_telemetry: Box<AssetType>,

    adaptive_behavior_info: AdaptiveBehaviorInfo,

    early_stopping_info: EarlyStoppingInfo,

    error_category: Environment,

    error_record: ErrorRecord,

    oast_callback_detail: OastCallbackDetail,

    stopping_reason: Box<StoppingReason>,

    enhanced_function_telemetry: Box<EnhancedFunctionTelemetry>,

    function_execution_result: FunctionExecutionResult,

    oast_event: OastEvent,

    oast_probe: OastProbe,

    #[serde(rename = "SIEMEventPayload")]
    siem_event_payload: SiemEventPayload,

    #[serde(rename = "SIEMEvent")]
    siem_event: SiemEvent,

    notification_payload: NotificationPayload,

    #[serde(rename = "CVSSv3Metrics")]
    cvs_sv3_metrics: CvsSv3Metrics,

    attack_step: AttackStep,

    attack_plan: AttackPlan,

    trace_record: TraceRecord,

    plan_execution_metrics: PlanExecutionMetrics,

    plan_execution_result: PlanExecutionResult,

    model_training_config: ModelTrainingConfig,

    #[serde(rename = "AITrainingStartPayload")]
    ai_training_start_payload: AiTrainingStartPayload,

    #[serde(rename = "AITrainingProgressPayload")]
    ai_training_progress_payload: AiTrainingProgressPayload,

    #[serde(rename = "AITrainingCompletedPayload")]
    ai_training_completed_payload: AiTrainingCompletedPayload,

    #[serde(rename = "AIExperienceCreatedEvent")]
    ai_experience_created_event: AiExperienceCreatedEvent,

    #[serde(rename = "AITraceCompletedEvent")]
    ai_trace_completed_event: AiTraceCompletedEvent,

    #[serde(rename = "AIModelUpdatedEvent")]
    ai_model_updated_event: AiModelUpdatedEvent,

    #[serde(rename = "AIModelDeployCommand")]
    ai_model_deploy_command: AiModelDeployCommand,

    #[serde(rename = "RAGKnowledgeUpdatePayload")]
    rag_knowledge_update_payload: RagKnowledgeUpdatePayload,

    #[serde(rename = "RAGQueryPayload")]
    rag_query_payload: RagQueryPayload,

    #[serde(rename = "RAGResponsePayload")]
    rag_response_payload: RagResponsePayload,

    experience_sample: ExperienceSample,

    enhanced_vulnerability: EnhancedVulnerability,

    #[serde(rename = "SARIFLocation")]
    sarif_location: SarifLocation,

    #[serde(rename = "SARIFResult")]
    sarif_result: SarifResult,

    #[serde(rename = "SARIFRule")]
    sarif_rule: SarifRule,

    #[serde(rename = "SARIFTool")]
    sarif_tool: SarifTool,

    #[serde(rename = "SARIFRun")]
    sarif_run: SarifRun,

    #[serde(rename = "SARIFReport")]
    sarif_report: SarifReport,

    asset_exposure: AdaptiveBehaviorInfo,

    asset_type: Box<AssetType>,

    business_criticality: AdaptiveBehaviorInfo,

    compliance_framework: Authentication,

    data_sensitivity: AdaptiveBehaviorInfo,

    environment: Environment,

    asset_lifecycle_payload: AssetLifecyclePayload,

    exploitability: Environment,

    vulnerability_status: AdaptiveBehaviorInfo,

    vulnerability_lifecycle_payload: VulnerabilityLifecyclePayload,

    vulnerability_update_payload: VulnerabilityUpdatePayload,

    discovered_asset: DiscoveredAsset,

    technical_fingerprint: TechnicalFingerprint,

    asset_inventory_item: AssetInventoryItem,

    #[serde(rename = "EASMAsset")]
    easm_asset: EasmAsset,

    risk_assessment_context: RiskAssessmentContext,

    risk_level: Box<AssetType>,

    risk_assessment_result: RiskAssessmentResult,

    risk_trend_analysis: RiskTrendAnalysis,

    attack_path_node_type: Box<AssetType>,

    attack_path_node: AttackPathNode,

    attack_path_edge_type: Box<AssetType>,

    attack_path_edge: AttackPathEdge,

    attack_path_payload: AttackPathPayload,

    attack_path_recommendation: AttackPathRecommendation,

    enhanced_finding_payload: FindingPayload,

    enhanced_scan_scope: Box<EnhancedFunctionTelemetry>,

    enhanced_scan_request: EnhancedScanRequest,

    enhanced_function_task_target: EnhancedFunctionTaskTarget,

    #[serde(rename = "EnhancedIOCRecord")]
    enhanced_ioc_record: EnhancedIocRecord,

    enhanced_risk_assessment: EnhancedRiskAssessment,

    enhanced_attack_path_node: EnhancedAttackPathNode,

    enhanced_attack_path: EnhancedAttackPath,

    enhanced_task_execution: EnhancedTaskExecution,

    enhanced_vulnerability_correlation: EnhancedVulnerabilityCorrelation,

    session_state: SessionState,

    model_training_result: ModelTrainingResult,

    task_queue: TaskQueue,

    enhanced_module_status: EnhancedModuleStatus,

    system_orchestration: SystemOrchestration,

    webhook_payload: WebhookPayload,

    #[serde(rename = "CVEReference")]
    cve_reference: CveReference,

    #[serde(rename = "CWEReference")]
    cwe_reference: CweReference,

    vulnerability_discovery: VulnerabilityDiscovery,

    language_framework: Environment,

    programming_language: Box<AssetType>,

    language_detection_result: LanguageDetectionResult,

    vulnerability_by_language: RemediationType,

    language_specific_vulnerability: LanguageSpecificVulnerability,

    multi_language_codebase: MultiLanguageCodebase,

    code_quality_metric: Authentication,

    security_pattern: Authentication,

    language_specific_scan_config: LanguageSpecificScanConfig,

    cross_language_analysis: CrossLanguageAnalysis,

    language_specific_payload: LanguageSpecificPayload,

    #[serde(rename = "AILanguageModel")]
    ai_language_model: AiLanguageModel,

    code_quality_report: CodeQualityReport,

    language_interoperability: LanguageInteroperability,
}

#[derive(Serialize, Deserialize)]
pub struct ScanScopeProperties {
    exclusions: TrainingId,

    include_subdomains: Box<EnhancedFunctionTelemetry>,

    allowed_hosts: TrainingId,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanScope {
    #[serde(rename = "default")]
    scan_scope_default: Option<ScanScopeDefault>,

    title: String,

    #[serde(rename = "type")]
    scan_scope_type: Option<Type>,

    description: Option<String>,

    additional_properties: Option<bool>,

    items: Option<AssetTypeItems>,

    properties: Option<ScanScopeProperties>,

    minimum: Option<f64>,

    maximum: Option<f64>,

    any_of: Option<Vec<ItemsElement>>,
}

#[derive(Serialize, Deserialize)]
pub struct AuthenticationProperties {
    method: Option<ScanScope>,

    credentials: Option<Framework>,

    fix: Option<Environment>,

    priority: Option<Environment>,

    remediation_steps: Option<Environment>,

    references: Option<Environment>,

    db_type_hint: Option<Box<EnhancedFunctionTelemetry>>,

    waf_detected: Option<Environment>,

    related_findings: Option<Parameters>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Authentication {
    maximum: Option<f64>,

    minimum: Option<f64>,

    title: Option<String>,

    #[serde(rename = "type")]
    authentication_type: Option<Type>,

    #[serde(rename = "default")]
    authentication_default: Option<AuthenticationDefault>,

    description: Option<String>,

    properties: Option<AuthenticationProperties>,

    #[serde(rename = "enum")]
    authentication_enum: Option<Vec<String>>,

    pattern: Option<String>,

    items: Option<StoppingReasonItems>,

    format: Option<Format>,

    additional_properties: Option<AdditionalProperties>,

    any_of: Option<Vec<AuthenticationAnyOf>>,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedFunctionTelemetryProperties {
    payloads_sent: Option<Box<StoppingReason>>,

    detections: Option<Box<StoppingReason>>,

    attempts: Option<Box<StoppingReason>>,

    errors: Option<Authentication>,

    duration_seconds: Option<Box<StoppingReason>>,

    timestamp: Option<Timestamp>,

    error_records: Option<AdaptiveBehaviorInfo>,

    oast_callbacks: Option<Box<AssetType>>,

    early_stopping: Option<ApiSchema>,

    adaptive_behavior: Option<ApiSchema>,

    included_hosts: Option<AdaptiveBehaviorInfo>,

    excluded_hosts: Option<AdaptiveBehaviorInfo>,

    included_paths: Option<AdaptiveBehaviorInfo>,

    excluded_paths: Option<AdaptiveBehaviorInfo>,

    max_depth: Option<Environment>,

    description: Option<TrainingId>,

    business_impact: Option<TrainingId>,

    technical_impact: Option<TrainingId>,

    affected_users: Option<TrainingId>,

    estimated_cost: Option<TrainingId>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnhancedFunctionTelemetry {
    any_of: Option<Vec<EnhancedFunctionTelemetryAnyOf>>,

    #[serde(rename = "default")]
    enhanced_function_telemetry_default: Option<EnhancedFunctionTelemetryDefault>,

    title: String,

    description: Option<String>,

    #[serde(rename = "type")]
    enhanced_function_telemetry_type: Option<Type>,

    additional_properties: Option<AdditionalProperties>,

    properties: Option<EnhancedFunctionTelemetryProperties>,

    items: Option<ItemsElement>,

    #[serde(rename = "enum")]
    enhanced_function_telemetry_enum: Option<Vec<String>>,

    property_names: Option<Config>,

    minimum: Option<f64>,

    maximum: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct AssetTypeProperties {
    payloads_sent: Box<StoppingReason>,

    detections: Box<StoppingReason>,

    attempts: Box<StoppingReason>,

    errors: Box<EnhancedFunctionTelemetry>,

    duration_seconds: Box<StoppingReason>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetType {
    additional_properties: Option<bool>,

    title: String,

    #[serde(rename = "type")]
    asset_type_type: Option<Type>,

    description: Option<String>,

    items: Option<AssetTypeItems>,

    #[serde(rename = "enum")]
    asset_type_enum: Option<Vec<String>>,

    pattern: Option<String>,

    #[serde(rename = "default")]
    asset_type_default: Option<AssetTypeDefault>,

    minimum: Option<f64>,

    maximum: Option<f64>,

    properties: Option<AssetTypeProperties>,

    any_of: Option<Vec<ItemsElement>>,
}

#[derive(Serialize, Deserialize)]
pub struct AdaptiveBehaviorInfoProperties {
    initial_batch_size: Option<Box<StoppingReason>>,

    final_batch_size: Option<Box<StoppingReason>>,

    rate_adjustments: Option<Box<StoppingReason>>,

    protection_detections: Option<Box<StoppingReason>>,

    bypass_attempts: Option<Box<StoppingReason>>,

    success_rate: Option<Box<StoppingReason>>,

    details: Option<Box<AssetType>>,

    payload: Option<TrainingId>,

    response_time_delta: Option<TrainingId>,

    db_version: Option<TrainingId>,

    request: Option<TrainingId>,

    response: Option<TrainingId>,

    proof: Option<TrainingId>,

    web_server: Option<Framework>,

    framework: Option<Framework>,

    language: Option<Framework>,

    waf_detected: Option<Box<EnhancedFunctionTelemetry>>,

    waf_vendor: Option<Box<EnhancedFunctionTelemetry>>,

    payloads: Option<Box<AssetType>>,

    custom_payloads: Option<Box<AssetType>>,

    blind_xss: Option<Box<AssetType>>,

    dom_testing: Option<Box<AssetType>>,

    timeout: Option<TrainingId>,

    requests_per_second: Option<Box<StoppingReason>>,

    burst: Option<Box<StoppingReason>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdaptiveBehaviorInfo {
    additional_properties: Option<AdditionalProperties>,

    description: Option<String>,

    title: String,

    #[serde(rename = "type")]
    adaptive_behavior_info_type: Option<Type>,

    #[serde(rename = "default")]
    adaptive_behavior_info_default: Option<AdaptiveBehaviorInfoDefault>,

    items: Option<AssetTypeItems>,

    properties: Option<AdaptiveBehaviorInfoProperties>,

    #[serde(rename = "enum")]
    adaptive_behavior_info_enum: Option<Vec<String>>,

    any_of: Option<Vec<EnhancedFunctionTelemetryAnyOf>>,

    property_names: Option<Config>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingId {
    any_of: Option<Vec<TrainingIdAnyOf>>,

    #[serde(rename = "default")]
    training_id_default: Option<String>,

    title: String,

    description: Option<String>,

    #[serde(rename = "type")]
    training_id_type: Option<Type>,

    items: Option<ItemsElement>,

    additional_properties: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingIdAnyOf {
    #[serde(rename = "type")]
    any_of_type: Type,

    minimum: Option<f64>,

    format: Option<Format>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Type {
    Array,

    Boolean,

    Integer,

    Null,

    Number,

    Object,

    String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    #[serde(rename = "date-time")]
    DateTime,
}

#[derive(Serialize, Deserialize)]
pub struct ItemsElement {
    #[serde(rename = "type")]
    additional_properties_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AssetTypeItems {
    #[serde(rename = "type")]
    items_type: Option<Type>,

    #[serde(rename = "$ref")]
    items_ref: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScanScopeDefault {
    AnythingArray(Vec<Option<serde_json::Value>>),

    Integer(i64),

    String(String),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Framework {
    any_of: Vec<FrameworkItems>,

    #[serde(rename = "default")]
    framework_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameworkItems {
    additional_properties: Option<ItemsElement>,

    #[serde(rename = "type")]
    items_type: Type,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    items: Option<EnvironmentItems>,

    title: Option<String>,

    #[serde(rename = "type")]
    environment_type: Option<Type>,

    description: Option<String>,

    additional_properties: Option<AdditionalProperties>,

    minimum: Option<f64>,

    maximum: Option<f64>,

    format: Option<Format>,

    #[serde(rename = "default")]
    environment_default: Option<EnvironmentDefault>,

    #[serde(rename = "enum")]
    environment_enum: Option<Vec<String>>,

    any_of: Option<Vec<EnvironmentAnyOf>>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdditionalProperties {
    Bool(bool),

    ItemsElement(ItemsElement),
}

#[derive(Serialize, Deserialize)]
pub struct EnvironmentAnyOf {
    #[serde(rename = "type")]
    any_of_type: Type,

    pattern: Option<String>,

    format: Option<Format>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum EnvironmentDefault {
    Bool(bool),

    Integer(i64),

    String(String),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentItems {
    additional_properties: Option<AdditionalProperties>,

    #[serde(rename = "type")]
    items_type: Option<Type>,

    #[serde(rename = "$ref")]
    items_ref: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Parameters {
    any_of: Vec<CustomPayloadsAdditionalProperties>,

    #[serde(rename = "default")]
    parameters_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
pub struct CustomPayloadsAdditionalProperties {
    items: Option<ItemsElement>,

    #[serde(rename = "type")]
    additional_properties_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AuthenticationAnyOf {
    minimum: Option<i64>,

    #[serde(rename = "type")]
    any_of_type: Type,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AuthenticationDefault {
    Bool(bool),

    Double(f64),

    String(String),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoppingReasonItems {
    #[serde(rename = "$ref")]
    items_ref: Option<String>,

    additional_properties: Option<bool>,

    #[serde(rename = "type")]
    items_type: Option<Type>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiSchema {
    any_of: Vec<AssetTypeItems>,

    #[serde(rename = "default")]
    api_schema_default: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct StoppingReasonProperties {
    urls_found: Box<StoppingReason>,

    forms_found: Box<StoppingReason>,

    apis_found: Box<StoppingReason>,

    scan_duration_seconds: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoppingReason {
    #[serde(rename = "default")]
    stopping_reason_default: Option<f64>,

    title: String,

    #[serde(rename = "type")]
    stopping_reason_type: Option<Type>,

    description: Option<String>,

    items: Option<StoppingReasonItems>,

    additional_properties: Option<bool>,

    any_of: Option<Vec<StoppingReasonAnyOf>>,

    #[serde(rename = "enum")]
    stopping_reason_enum: Option<Vec<String>>,

    properties: Option<StoppingReasonProperties>,
}

#[derive(Serialize, Deserialize)]
pub struct StoppingReasonAnyOf {
    #[serde(rename = "type")]
    any_of_type: Type,

    maximum: Option<f64>,

    minimum: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct Timestamp {
    format: Format,

    title: String,

    #[serde(rename = "type")]
    timestamp_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedFunctionTelemetryAnyOf {
    maximum: Option<f64>,

    minimum: Option<f64>,

    #[serde(rename = "type")]
    any_of_type: Type,

    format: Option<Format>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum EnhancedFunctionTelemetryDefault {
    Bool(bool),

    String(String),
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "$ref")]
    config_ref: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssetTypeDefault {
    Bool(bool),

    DefaultClass(DefaultClass),

    Integer(i64),

    String(String),
}

#[derive(Serialize, Deserialize)]
pub struct DefaultClass {
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdaptiveBehaviorInfoDefault {
    Bool(bool),

    DefaultClass(DefaultClass),

    String(String),
}

#[derive(Serialize, Deserialize)]
pub struct AiExperienceCreatedEvent {
    description: String,

    properties: AiExperienceCreatedEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_experience_created_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiExperienceCreatedEventProperties {
    experience_id: ExperienceId,

    training_id: TrainingId,

    trace_id: ExperienceId,

    vulnerability_type: ExperienceId,

    quality_score: Authentication,

    success: ExperienceId,

    plan_summary: Box<AssetType>,

    result_summary: Box<AssetType>,

    metadata: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceId {
    title: String,

    #[serde(rename = "type")]
    experience_id_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiLanguageModel {
    description: String,

    properties: AiLanguageModelProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_language_model_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiLanguageModelProperties {
    model_name: AdaptiveBehaviorInfo,

    supported_languages: AdaptiveBehaviorInfo,

    model_type: AdaptiveBehaviorInfo,

    version: AdaptiveBehaviorInfo,

    capabilities: AdaptiveBehaviorInfo,

    training_data_size: TrainingId,

    accuracy_metrics: AdaptiveBehaviorInfo,

    api_endpoint: TrainingId,

    authentication_required: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct AiModelDeployCommand {
    description: String,

    properties: AiModelDeployCommandProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_model_deploy_command_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiModelDeployCommandProperties {
    model_id: ExperienceId,

    model_version: ExperienceId,

    checkpoint_path: ExperienceId,

    deployment_target: ScanScope,

    deployment_config: Box<AssetType>,

    require_validation: AdaptiveBehaviorInfo,

    min_performance_threshold: AdaptiveBehaviorInfo,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AiModelUpdatedEvent {
    description: String,

    properties: AiModelUpdatedEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_model_updated_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiModelUpdatedEventProperties {
    model_id: ExperienceId,

    model_version: ExperienceId,

    training_id: TrainingId,

    update_type: ExperienceId,

    performance_metrics: AdaptiveBehaviorInfo,

    model_path: TrainingId,

    checkpoint_path: TrainingId,

    is_deployed: AdaptiveBehaviorInfo,

    metadata: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AiTraceCompletedEvent {
    description: String,

    properties: AiTraceCompletedEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_trace_completed_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiTraceCompletedEventProperties {
    trace_id: ExperienceId,

    session_id: TrainingId,

    training_id: TrainingId,

    total_steps: ExperienceId,

    successful_steps: ExperienceId,

    failed_steps: ExperienceId,

    duration_seconds: ExperienceId,

    final_success: ExperienceId,

    plan_type: ExperienceId,

    metadata: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingCompletedPayload {
    description: String,

    properties: AiTrainingCompletedPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_training_completed_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingCompletedPayloadProperties {
    training_id: ExperienceId,

    status: ExperienceId,

    total_episodes: ExperienceId,

    successful_episodes: ExperienceId,

    failed_episodes: ExperienceId,

    total_duration_seconds: ExperienceId,

    total_samples: ExperienceId,

    high_quality_samples: ExperienceId,

    medium_quality_samples: ExperienceId,

    low_quality_samples: ExperienceId,

    final_avg_reward: TrainingId,

    final_avg_quality: TrainingId,

    best_episode_reward: TrainingId,

    model_checkpoint_path: TrainingId,

    model_metrics: AdaptiveBehaviorInfo,

    error_message: TrainingId,

    metadata: Box<AssetType>,

    completed_at: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingProgressPayload {
    description: String,

    properties: AiTrainingProgressPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_training_progress_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingProgressPayloadProperties {
    training_id: ExperienceId,

    episode_number: ExperienceId,

    total_episodes: ExperienceId,

    successful_episodes: Box<StoppingReason>,

    failed_episodes: Box<StoppingReason>,

    total_samples: Box<StoppingReason>,

    high_quality_samples: Box<StoppingReason>,

    avg_reward: TrainingId,

    avg_quality: TrainingId,

    best_reward: TrainingId,

    model_metrics: AdaptiveBehaviorInfo,

    status: ScanScope,

    metadata: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingStartPayload {
    description: String,

    properties: AiTrainingStartPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_training_start_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiTrainingStartPayloadProperties {
    training_id: ExperienceId,

    training_type: ExperienceId,

    scenario_id: TrainingId,

    target_vulnerability: TrainingId,

    config: Config,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AiVerificationRequest {
    description: String,

    properties: AiVerificationRequestProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_verification_request_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiVerificationRequestProperties {
    verification_id: ExperienceId,

    finding_id: ExperienceId,

    scan_id: ExperienceId,

    vulnerability_type: Config,

    target: Config,

    evidence: Config,

    verification_mode: ScanScope,

    context: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AiVerificationResult {
    description: String,

    properties: AiVerificationResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    ai_verification_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AiVerificationResultProperties {
    verification_id: ExperienceId,

    finding_id: ExperienceId,

    verification_status: ExperienceId,

    confidence_score: ExperienceId,

    verification_method: ExperienceId,

    test_steps: AdaptiveBehaviorInfo,

    observations: AdaptiveBehaviorInfo,

    recommendations: AdaptiveBehaviorInfo,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AivaCommand {
    description: String,

    properties: AivaCommandProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    aiva_command_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AivaCommandProperties {
    command_id: ExperienceId,

    command_type: ExperienceId,

    source_module: ExperienceId,

    target_module: ExperienceId,

    payload: Box<AssetType>,

    priority: Authentication,

    trace_id: TrainingId,

    metadata: Box<AssetType>,

    timestamp: ExperienceId,
}

#[derive(Serialize, Deserialize)]
pub struct AivaEvent {
    description: String,

    properties: AivaEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    aiva_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AivaEventProperties {
    event_id: ExperienceId,

    event_type: ExperienceId,

    source_module: ExperienceId,

    payload: Box<AssetType>,

    trace_id: TrainingId,

    metadata: Box<AssetType>,

    timestamp: ExperienceId,
}

#[derive(Serialize, Deserialize)]
pub struct AivaMessage {
    description: String,

    properties: AivaMessageProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    aiva_message_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AivaMessageProperties {
    header: Config,

    topic: Config,

    schema_version: ScanScope,

    payload: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AivaRequest {
    description: String,

    properties: AivaRequestProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    aiva_request_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AivaRequestProperties {
    request_id: ExperienceId,

    source_module: ExperienceId,

    target_module: ExperienceId,

    request_type: ExperienceId,

    payload: Box<AssetType>,

    trace_id: TrainingId,

    timeout_seconds: Authentication,

    metadata: Box<AssetType>,

    timestamp: ExperienceId,
}

#[derive(Serialize, Deserialize)]
pub struct AivaResponse {
    description: String,

    properties: AivaResponseProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    aiva_response_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AivaResponseProperties {
    request_id: ExperienceId,

    response_type: ExperienceId,

    success: ExperienceId,

    payload: Payload,

    error_code: TrainingId,

    error_message: TrainingId,

    metadata: Box<AssetType>,

    timestamp: ExperienceId,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Payload {
    any_of: Vec<JsonDataAnyOf>,

    #[serde(rename = "default")]
    payload_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonDataAnyOf {
    additional_properties: Option<bool>,

    #[serde(rename = "type")]
    any_of_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ApiSchemaPayload {
    description: String,

    properties: ApiSchemaPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    api_schema_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ApiSchemaPayloadProperties {
    schema_id: ExperienceId,

    scan_id: ExperienceId,

    schema_type: ExperienceId,

    schema_content: SchemaContent,

    base_url: ExperienceId,

    authentication: Config,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaContent {
    any_of: Vec<JsonDataAnyOf>,

    title: String,
}

#[derive(Serialize, Deserialize)]
pub struct ApiSecurityTestPayload {
    description: String,

    properties: ApiSecurityTestPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    api_security_test_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ApiSecurityTestPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    api_type: ExperienceId,

    api_schema: ApiSchema,

    test_cases: AdaptiveBehaviorInfo,

    authentication: Config,
}

#[derive(Serialize, Deserialize)]
pub struct ApiTestCase {
    description: String,

    properties: ApiTestCaseProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    api_test_case_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ApiTestCaseProperties {
    test_id: ExperienceId,

    test_type: ExperienceId,

    endpoint: ExperienceId,

    method: ExperienceId,

    test_vectors: Environment,

    expected_behavior: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct Asset {
    description: String,

    properties: AssetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    asset_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AssetProperties {
    asset_id: ExperienceId,

    #[serde(rename = "type")]
    properties_type: ExperienceId,

    value: ExperienceId,

    parameters: Parameters,

    has_form: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct AssetInventoryItem {
    description: String,

    properties: AssetInventoryItemProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    asset_inventory_item_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AssetInventoryItemProperties {
    asset_id: AdaptiveBehaviorInfo,

    asset_type: AdaptiveBehaviorInfo,

    name: Box<AssetType>,

    ip_address: TrainingId,

    hostname: TrainingId,

    domain: AdaptiveBehaviorInfo,

    ports: Box<AssetType>,

    fingerprints: AdaptiveBehaviorInfo,

    business_criticality: AdaptiveBehaviorInfo,

    owner: TrainingId,

    environment: AdaptiveBehaviorInfo,

    last_scanned: AdaptiveBehaviorInfo,

    vulnerability_count: Authentication,

    risk_score: Authentication,

    discovered_at: Timestamp,

    updated_at: Timestamp,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AssetLifecyclePayload {
    description: String,

    properties: AssetLifecyclePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    asset_lifecycle_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AssetLifecyclePayloadProperties {
    asset_id: ExperienceId,

    asset_type: Config,

    value: ExperienceId,

    environment: Config,

    business_criticality: Config,

    data_sensitivity: ApiSchema,

    asset_exposure: ApiSchema,

    owner: TrainingId,

    team: TrainingId,

    compliance_tags: AdaptiveBehaviorInfo,

    metadata: Box<AssetType>,

    created_at: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathEdge {
    description: String,

    properties: AttackPathEdgeProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_path_edge_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathEdgeProperties {
    edge_id: ExperienceId,

    source_node_id: ExperienceId,

    target_node_id: ExperienceId,

    edge_type: Config,

    risk_score: Box<StoppingReason>,

    properties: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathNode {
    description: String,

    properties: AttackPathNodeProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_path_node_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathNodeProperties {
    node_id: ExperienceId,

    node_type: Config,

    name: ExperienceId,

    properties: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathPayload {
    description: String,

    properties: AttackPathPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_path_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathPayloadProperties {
    path_id: ExperienceId,

    scan_id: ExperienceId,

    source_node: Config,

    target_node: Config,

    nodes: AdaptiveBehaviorInfo,

    edges: AdaptiveBehaviorInfo,

    total_risk_score: ExperienceId,

    path_length: ExperienceId,

    description: TrainingId,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathRecommendation {
    description: String,

    properties: AttackPathRecommendationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_path_recommendation_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPathRecommendationProperties {
    path_id: ExperienceId,

    risk_level: Config,

    priority_score: ExperienceId,

    executive_summary: ExperienceId,

    technical_explanation: ExperienceId,

    business_impact: ExperienceId,

    remediation_steps: Box<AssetType>,

    quick_wins: AdaptiveBehaviorInfo,

    affected_assets: AdaptiveBehaviorInfo,

    estimated_effort: ExperienceId,

    estimated_risk_reduction: ExperienceId,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPlan {
    description: String,

    properties: AttackPlanProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_plan_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackPlanProperties {
    plan_id: ExperienceId,

    scan_id: ExperienceId,

    attack_type: Config,

    steps: AdaptiveBehaviorInfo,

    dependencies: RemediationType,

    context: Box<AssetType>,

    target_info: Box<AssetType>,

    created_at: Timestamp,

    created_by: ScanScope,

    mitre_techniques: AdaptiveBehaviorInfo,

    mitre_tactics: Box<AssetType>,

    capec_id: CapecId,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapecId {
    any_of: Vec<CapecIdAnyOf>,

    #[serde(rename = "default")]
    capec_id_default: Option<serde_json::Value>,

    title: String,

    description: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct CapecIdAnyOf {
    pattern: Option<String>,

    #[serde(rename = "type")]
    any_of_type: Type,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemediationType {
    additional_properties: Option<CustomPayloadsAdditionalProperties>,

    title: String,

    #[serde(rename = "type")]
    remediation_type_type: Option<Type>,

    description: Option<String>,

    items: Option<Config>,

    #[serde(rename = "enum")]
    remediation_type_enum: Option<Vec<String>>,

    #[serde(rename = "default")]
    remediation_type_default: Option<EnhancedFunctionTelemetryDefault>,

    any_of: Option<Vec<ItemsElement>>,

    format: Option<Format>,
}

#[derive(Serialize, Deserialize)]
pub struct AttackStep {
    description: String,

    properties: AttackStepProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    attack_step_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AttackStepProperties {
    step_id: ExperienceId,

    action: ExperienceId,

    tool_type: ExperienceId,

    target: Box<AssetType>,

    parameters: Box<AssetType>,

    expected_result: TrainingId,

    timeout_seconds: Box<StoppingReason>,

    retry_count: Box<StoppingReason>,

    mitre_technique_id: CapecId,

    mitre_tactic: TrainingId,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZAnalysisPayload {
    description: String,

    properties: AuthZAnalysisPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    auth_z_analysis_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZAnalysisPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    analysis_type: ExperienceId,

    target: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZCheckPayload {
    description: String,

    properties: AuthZCheckPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    auth_z_check_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZCheckPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    user_id: ExperienceId,

    resource: ExperienceId,

    permission: ExperienceId,

    context: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZResultPayload {
    description: String,

    properties: AuthZResultPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    auth_z_result_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct AuthZResultPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    decision: ExperienceId,

    analysis: Box<AssetType>,

    recommendations: AdaptiveBehaviorInfo,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct BizLogicResultPayload {
    description: String,

    properties: BizLogicResultPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    biz_logic_result_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct BizLogicResultPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    test_type: ExperienceId,

    status: ExperienceId,

    findings: Environment,

    statistics: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct BizLogicTestPayload {
    description: String,

    properties: BizLogicTestPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    biz_logic_test_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct BizLogicTestPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    test_type: ExperienceId,

    target_urls: AdaptiveBehaviorInfo,

    test_config: Box<AssetType>,

    product_id: AdaptiveBehaviorInfo,

    workflow_steps: WorkflowSteps,
}

#[derive(Serialize, Deserialize)]
pub struct WorkflowSteps {
    items: FrameworkItems,

    title: String,

    #[serde(rename = "type")]
    workflow_steps_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CodeLevelRootCause {
    description: String,

    properties: CodeLevelRootCauseProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    code_level_root_cause_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CodeLevelRootCauseProperties {
    analysis_id: ExperienceId,

    vulnerable_component: ExperienceId,

    affected_findings: Box<AssetType>,

    code_location: TrainingId,

    vulnerability_pattern: TrainingId,

    fix_recommendation: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct CodeQualityReport {
    description: String,

    properties: CodeQualityReportProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    code_quality_report_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CodeQualityReportProperties {
    language: Language,

    file_path: AdaptiveBehaviorInfo,

    metrics: AdaptiveBehaviorInfo,

    issues: AdaptiveBehaviorInfo,

    suggestions: Authentication,

    overall_score: Authentication,

    timestamp: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct Language {
    #[serde(rename = "$ref")]
    language_ref: String,

    description: String,
}

#[derive(Serialize, Deserialize)]
pub struct ConfigUpdatePayload {
    description: String,

    properties: ConfigUpdatePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    config_update_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ConfigUpdatePayloadProperties {
    update_id: ExperienceId,

    config_items: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct CrossLanguageAnalysis {
    description: String,

    properties: CrossLanguageAnalysisProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    cross_language_analysis_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CrossLanguageAnalysisProperties {
    analysis_id: AdaptiveBehaviorInfo,

    project_name: AdaptiveBehaviorInfo,

    languages_analyzed: AdaptiveBehaviorInfo,

    cross_language_issues: AdaptiveBehaviorInfo,

    integration_points: AdaptiveBehaviorInfo,

    security_boundaries: Authentication,

    data_flow_risks: AdaptiveBehaviorInfo,

    recommendations: AdaptiveBehaviorInfo,

    risk_score: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct CveReference {
    description: String,

    properties: CveReferenceProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    cve_reference_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CveReferenceProperties {
    cve_id: Box<AssetType>,

    description: TrainingId,

    cvss_score: Box<EnhancedFunctionTelemetry>,

    cvss_vector: TrainingId,

    references: AdaptiveBehaviorInfo,

    published_date: AdaptiveBehaviorInfo,

    last_modified_date: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct CvsSv3Metrics {
    description: String,

    properties: CvsSv3MetricsProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    cvs_sv3_metrics_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CvsSv3MetricsProperties {
    attack_vector: Authentication,

    attack_complexity: Authentication,

    privileges_required: Authentication,

    user_interaction: Authentication,

    scope: Authentication,

    confidentiality: Authentication,

    integrity: Authentication,

    availability: Authentication,

    exploit_code_maturity: Authentication,

    remediation_level: Authentication,

    report_confidence: Authentication,

    confidentiality_requirement: Authentication,

    integrity_requirement: Authentication,

    availability_requirement: Authentication,

    base_score: Box<EnhancedFunctionTelemetry>,

    temporal_score: Box<EnhancedFunctionTelemetry>,

    environmental_score: Box<EnhancedFunctionTelemetry>,

    vector_string: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct CweReference {
    description: String,

    properties: CweReferenceProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    cwe_reference_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct CweReferenceProperties {
    cwe_id: Authentication,

    name: TrainingId,

    description: AdaptiveBehaviorInfo,

    weakness_category: TrainingId,

    likelihood_of_exploit: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct DiscoveredAsset {
    description: String,

    properties: DiscoveredAssetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    discovered_asset_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct DiscoveredAssetProperties {
    asset_id: ExperienceId,

    asset_type: Config,

    value: ExperienceId,

    discovery_method: ExperienceId,

    confidence: Config,

    metadata: Box<AssetType>,

    discovered_at: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct EarlyStoppingInfo {
    description: String,

    properties: EarlyStoppingInfoProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    early_stopping_info_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EarlyStoppingInfoProperties {
    reason: Config,

    timestamp: Timestamp,

    total_tests: ExperienceId,

    completed_tests: ExperienceId,

    remaining_tests: ExperienceId,

    details: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct EasmAsset {
    description: String,

    properties: EasmAssetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    easm_asset_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EasmAssetProperties {
    asset_id: Authentication,

    asset_type: Authentication,

    value: Environment,

    discovery_method: Authentication,

    discovery_source: Authentication,

    first_discovered: Authentication,

    last_seen: Authentication,

    status: Authentication,

    confidence: Authentication,

    technologies: Authentication,

    services: Authentication,

    certificates: Environment,

    risk_score: Authentication,

    vulnerability_count: Environment,

    exposure_level: Authentication,

    business_unit: TrainingId,

    owner: TrainingId,

    criticality: Authentication,

    compliance_status: Environment,

    policy_violations: Environment,

    metadata: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct EasmDiscoveryPayload {
    description: String,

    properties: EasmDiscoveryPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    easm_discovery_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EasmDiscoveryPayloadProperties {
    discovery_id: ExperienceId,

    scan_id: ExperienceId,

    discovery_type: ExperienceId,

    targets: AdaptiveBehaviorInfo,

    scope: Config,

    max_depth: Box<StoppingReason>,

    passive_only: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct EasmDiscoveryResult {
    description: String,

    properties: EasmDiscoveryResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    easm_discovery_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EasmDiscoveryResultProperties {
    discovery_id: ExperienceId,

    scan_id: ExperienceId,

    status: ExperienceId,

    discovered_assets: Authentication,

    statistics: Authentication,

    timestamp: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedAttackPath {
    description: String,

    properties: EnhancedAttackPathProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_attack_path_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedAttackPathProperties {
    path_id: Environment,

    target_asset: Environment,

    nodes: Box<AssetType>,

    edges: Environment,

    path_feasibility: Environment,

    estimated_time: Box<AssetType>,

    skill_level_required: Environment,

    success_probability: Environment,

    detection_probability: Environment,

    overall_risk: Environment,

    blocking_controls: AdaptiveBehaviorInfo,

    detection_controls: Box<AssetType>,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedAttackPathNode {
    description: String,

    properties: EnhancedAttackPathNodeProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_attack_path_node_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedAttackPathNodeProperties {
    node_id: Box<AssetType>,

    node_type: Box<AssetType>,

    name: Box<AssetType>,

    description: TrainingId,

    exploitability: Box<AssetType>,

    impact: Box<AssetType>,

    difficulty: Box<AssetType>,

    mitre_technique: TrainingId,

    mitre_tactic: TrainingId,

    prerequisites: Box<AssetType>,

    consequences: Box<AssetType>,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct FindingPayload {
    description: String,

    properties: EnhancedFindingPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    finding_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedFindingPayloadProperties {
    finding_id: ExperienceId,

    task_id: ExperienceId,

    scan_id: ExperienceId,

    status: ExperienceId,

    vulnerability: Config,

    target: Config,

    strategy: TrainingId,

    evidence: ApiSchema,

    impact: ApiSchema,

    recommendation: ApiSchema,

    sarif_result: Option<ApiSchema>,

    metadata: Metadata,

    created_at: Timestamp,

    updated_at: Timestamp,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    additional_properties: bool,

    title: String,

    #[serde(rename = "type")]
    metadata_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedFunctionTaskTarget {
    description: String,

    properties: EnhancedFunctionTaskTargetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_function_task_target_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedFunctionTaskTargetProperties {
    url: PurpleUrl,

    method: Box<EnhancedFunctionTelemetry>,

    headers: Box<EnhancedFunctionTelemetry>,

    cookies: Box<EnhancedFunctionTelemetry>,

    parameters: Box<EnhancedFunctionTelemetry>,

    body: Box<EnhancedFunctionTelemetry>,

    auth_required: Box<EnhancedFunctionTelemetry>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PurpleUrl {
    description: String,

    format: String,

    max_length: i64,

    min_length: i64,

    title: String,

    #[serde(rename = "type")]
    url_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedIocRecord {
    description: String,

    properties: EnhancedIocRecordProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_ioc_record_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedIocRecordProperties {
    ioc_id: AdaptiveBehaviorInfo,

    ioc_type: AdaptiveBehaviorInfo,

    value: AdaptiveBehaviorInfo,

    threat_type: AdaptiveBehaviorInfo,

    malware_family: AdaptiveBehaviorInfo,

    campaign: TrainingId,

    severity: Language,

    confidence: Box<AssetType>,

    reputation_score: Environment,

    first_seen: AdaptiveBehaviorInfo,

    last_seen: AdaptiveBehaviorInfo,

    expires_at: AdaptiveBehaviorInfo,

    tags: Environment,

    mitre_techniques: AdaptiveBehaviorInfo,

    metadata: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedModuleStatus {
    description: String,

    properties: EnhancedModuleStatusProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_module_status_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedModuleStatusProperties {
    module_name: Language,

    version: Environment,

    status: Environment,

    health_score: Environment,

    cpu_usage: Environment,

    memory_usage: Environment,

    active_connections: Environment,

    tasks_processed: Environment,

    tasks_pending: Environment,

    error_count: Environment,

    started_at: Environment,

    last_heartbeat: Environment,

    uptime_seconds: Environment,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedRiskAssessment {
    description: String,

    properties: EnhancedRiskAssessmentProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_risk_assessment_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedRiskAssessmentProperties {
    assessment_id: Authentication,

    target_id: Authentication,

    overall_risk_score: Authentication,

    likelihood_score: Box<AssetType>,

    impact_score: Box<AssetType>,

    risk_level: Language,

    risk_category: Authentication,

    risk_factors: Authentication,

    cvss_metrics: CvssMetrics,

    business_impact: TrainingId,

    affected_assets: Box<AssetType>,

    mitigation_strategies: Box<AssetType>,

    residual_risk: Authentication,

    assessed_at: Authentication,

    valid_until: Box<EnhancedFunctionTelemetry>,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CvssMetrics {
    any_of: Vec<AssetTypeItems>,

    #[serde(rename = "default")]
    cvss_metrics_default: Option<serde_json::Value>,

    description: String,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedScanRequest {
    description: String,

    properties: EnhancedScanRequestProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_scan_request_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedScanRequestProperties {
    scan_id: Authentication,

    targets: PurpleTargets,

    scope: Language,

    strategy: Authentication,

    priority: Authentication,

    max_duration: Authentication,

    metadata: Authentication,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PurpleTargets {
    description: String,

    items: FormUrlItems,

    min_items: i64,

    title: String,

    #[serde(rename = "type")]
    targets_type: Type,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormUrlItems {
    format: Option<String>,

    max_length: Option<i64>,

    min_length: Option<i64>,

    #[serde(rename = "type")]
    items_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedTaskExecution {
    description: String,

    properties: EnhancedTaskExecutionProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_task_execution_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedTaskExecutionProperties {
    task_id: AdaptiveBehaviorInfo,

    task_type: AdaptiveBehaviorInfo,

    module_name: Language,

    priority: Box<AssetType>,

    timeout: Box<AssetType>,

    retry_count: Box<AssetType>,

    dependencies: Box<AssetType>,

    status: Language,

    progress: Box<AssetType>,

    result_data: Box<AssetType>,

    error_message: TrainingId,

    cpu_usage: TrainingId,

    memory_usage: TrainingId,

    created_at: Timestamp,

    started_at: AdaptiveBehaviorInfo,

    completed_at: AdaptiveBehaviorInfo,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedVulnerability {
    description: String,

    properties: EnhancedVulnerabilityProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_vulnerability_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedVulnerabilityProperties {
    vulnerability_id: Authentication,

    title: Authentication,

    description: Authentication,

    vulnerability_type: Authentication,

    severity: Authentication,

    url: Authentication,

    parameter: AdaptiveBehaviorInfo,

    location: AdaptiveBehaviorInfo,

    cvss_metrics: CvssMetrics,

    ai_confidence: Authentication,

    ai_risk_assessment: Authentication,

    exploitability_score: Authentication,

    attack_vector: Authentication,

    attack_complexity: Authentication,

    prerequisites: Authentication,

    business_impact: Authentication,

    technical_impact: Authentication,

    remediation_effort: Authentication,

    remediation_priority: Authentication,

    fix_recommendations: Authentication,

    poc_available: Authentication,

    verified: Authentication,

    false_positive_probability: Authentication,

    discovered_at: Authentication,

    last_verified_at: AdaptiveBehaviorInfo,

    tags: Authentication,

    references: Authentication,

    metadata: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedVulnerabilityCorrelation {
    description: String,

    properties: EnhancedVulnerabilityCorrelationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    enhanced_vulnerability_correlation_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct EnhancedVulnerabilityCorrelationProperties {
    correlation_id: AdaptiveBehaviorInfo,

    primary_vulnerability: Environment,

    related_vulnerabilities: Environment,

    correlation_strength: Environment,

    correlation_type: Environment,

    combined_risk_score: Environment,

    exploitation_complexity: Environment,

    attack_scenarios: AdaptiveBehaviorInfo,

    recommended_order: Environment,

    coordinated_mitigation: Environment,

    priority_ranking: Environment,

    analyzed_at: Timestamp,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct ErrorRecord {
    description: String,

    properties: ErrorRecordProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    error_record_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ErrorRecordProperties {
    category: Config,

    message: ExperienceId,

    timestamp: Timestamp,

    details: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct ExecutionError {
    description: String,

    properties: ExecutionErrorProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    execution_error_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ExecutionErrorProperties {
    error_id: ExperienceId,

    error_type: ExperienceId,

    message: ExperienceId,

    payload: AdaptiveBehaviorInfo,

    vector: AdaptiveBehaviorInfo,

    timestamp: Authentication,

    attempts: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceSample {
    description: String,

    properties: ExperienceSampleProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    experience_sample_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ExperienceSampleProperties {
    sample_id: Authentication,

    session_id: Authentication,

    plan_id: Box<AssetType>,

    state_before: Authentication,

    action_taken: Authentication,

    state_after: Authentication,

    reward: Authentication,

    reward_breakdown: Authentication,

    context: Authentication,

    target_info: Authentication,

    timestamp: Authentication,

    duration_ms: Authentication,

    quality_score: AdaptiveBehaviorInfo,

    is_positive: Box<EnhancedFunctionTelemetry>,

    confidence: Authentication,

    learning_tags: Box<EnhancedFunctionTelemetry>,

    difficulty_level: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct ExploitPayload {
    description: String,

    properties: ExploitPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    exploit_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ExploitPayloadProperties {
    payload_id: Box<AssetType>,

    payload_type: Box<AssetType>,

    payload_content: Box<AssetType>,

    encoding: Box<AssetType>,

    obfuscation: Box<AssetType>,

    bypass_technique: TrainingId,

    target_technology: Box<AssetType>,

    required_context: Box<AssetType>,

    effectiveness_score: Box<AssetType>,

    detection_evasion: Box<AssetType>,

    success_rate: Box<AssetType>,

    usage_count: Box<AssetType>,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct ExploitResult {
    description: String,

    properties: ExploitResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    exploit_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ExploitResultProperties {
    result_id: Environment,

    exploit_id: Environment,

    target_id: Environment,

    success: Environment,

    severity: Language,

    impact_level: Environment,

    exploit_technique: Environment,

    payload_used: Environment,

    execution_time: Environment,

    access_gained: Environment,

    data_extracted: Environment,

    system_impact: Environment,

    detection_bypassed: Environment,

    artifacts_left: Environment,

    remediation_verified: Environment,

    retest_required: Environment,

    executed_at: Environment,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct FeedbackEventPayload {
    description: String,

    properties: FeedbackEventPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    feedback_event_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FeedbackEventPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    event_type: ExperienceId,

    details: AdaptiveBehaviorInfo,

    form_url: FormUrl,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormUrl {
    any_of: Vec<FormUrlItems>,

    #[serde(rename = "default")]
    form_url_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
pub struct Target {
    description: String,

    properties: FindingTargetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    target_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FindingTargetProperties {
    url: FluffyUrl,

    parameter: Body,

    method: Body,

    headers: Headers,

    params: Metadata,

    body: Body,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Body {
    any_of: Vec<ItemsElement>,

    #[serde(rename = "default")]
    body_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Headers {
    additional_properties: ItemsElement,

    title: String,

    #[serde(rename = "type")]
    headers_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FluffyUrl {
    title: String,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionExecutionResult {
    description: String,

    properties: FunctionExecutionResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    function_execution_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionExecutionResultProperties {
    findings: Authentication,

    telemetry: Authentication,

    errors: Authentication,

    duration_seconds: Box<StoppingReason>,

    timestamp: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionTaskPayload {
    description: String,

    properties: FunctionTaskPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    function_task_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionTaskPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    priority: Box<StoppingReason>,

    target: Config,

    context: Config,

    strategy: ScanScope,

    custom_payloads: Parameters,

    test_config: Config,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionTaskTarget {
    description: String,

    properties: FunctionTaskTargetProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    function_task_target_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct FunctionTaskTargetProperties {
    url: FluffyUrl,

    parameter: TrainingId,

    method: ScanScope,

    parameter_location: ScanScope,

    headers: AdaptiveBehaviorInfo,

    cookies: AdaptiveBehaviorInfo,

    form_data: Box<AssetType>,

    json_data: Payload,

    body: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct HeartbeatPayload {
    description: String,

    properties: HeartbeatPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    heartbeat_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct HeartbeatPayloadProperties {
    module: Config,

    worker_id: ExperienceId,

    capacity: ExperienceId,
}

#[derive(Serialize, Deserialize)]
pub struct JavaScriptAnalysisResult {
    description: String,

    properties: JavaScriptAnalysisResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    java_script_analysis_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct JavaScriptAnalysisResultProperties {
    analysis_id: ExperienceId,

    url: ExperienceId,

    source_size_bytes: ExperienceId,

    dangerous_functions: Authentication,

    external_resources: Authentication,

    data_leaks: WorkflowSteps,

    findings: Authentication,

    apis_called: Authentication,

    ajax_endpoints: Authentication,

    suspicious_patterns: Authentication,

    risk_score: Authentication,

    security_score: Authentication,

    timestamp: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    description: String,

    properties: LanguageDetectionResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    language_detection_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageDetectionResultProperties {
    primary_language: Language,

    confidence: Authentication,

    secondary_languages: Environment,

    frameworks: Authentication,

    file_extensions: Authentication,

    lines_of_code: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageInteroperability {
    description: String,

    properties: LanguageInteroperabilityProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    language_interoperability_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageInteroperabilityProperties {
    source_language: Language,

    target_language: Language,

    interop_method: Authentication,

    security_considerations: Authentication,

    performance_impact: Authentication,

    compatibility_issues: Authentication,

    recommendations: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificPayload {
    description: String,

    properties: LanguageSpecificPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    language_specific_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificPayloadProperties {
    language: Language,

    payload_type: TrainingId,

    payload_content: TrainingId,

    encoding: TrainingId,

    expected_behavior: TrainingId,

    bypass_techniques: Authentication,

    target_functions: Authentication,

    success_indicators: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificScanConfig {
    description: String,

    properties: LanguageSpecificScanConfigProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    language_specific_scan_config_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificScanConfigProperties {
    language: Language,

    scan_patterns: Box<StoppingReason>,

    quality_metrics: Box<StoppingReason>,

    exclude_paths: AdaptiveBehaviorInfo,

    include_patterns: AdaptiveBehaviorInfo,

    custom_rules: Authentication,

    max_file_size: Box<StoppingReason>,

    timeout_seconds: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificVulnerability {
    description: String,

    properties: LanguageSpecificVulnerabilityProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    language_specific_vulnerability_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct LanguageSpecificVulnerabilityProperties {
    language: Language,

    vulnerability_type: Language,

    severity: Language,

    description: AdaptiveBehaviorInfo,

    code_snippet: AdaptiveBehaviorInfo,

    line_number: AdaptiveBehaviorInfo,

    file_path: AdaptiveBehaviorInfo,

    function_name: AdaptiveBehaviorInfo,

    remediation: AdaptiveBehaviorInfo,

    cwe_id: AdaptiveBehaviorInfo,

    owasp_category: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct MessageHeader {
    description: String,

    properties: MessageHeaderProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    message_header_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct MessageHeaderProperties {
    message_id: ExperienceId,

    trace_id: ExperienceId,

    correlation_id: AdaptiveBehaviorInfo,

    source_module: Config,

    timestamp: Timestamp,

    version: ScanScope,
}

#[derive(Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    description: String,

    properties: ModelTrainingConfigProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    model_training_config_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ModelTrainingConfigProperties {
    config_id: ExperienceId,

    model_type: ExperienceId,

    training_mode: ExperienceId,

    batch_size: Box<StoppingReason>,

    learning_rate: Box<StoppingReason>,

    epochs: Box<StoppingReason>,

    validation_split: Box<StoppingReason>,

    early_stopping: AdaptiveBehaviorInfo,

    patience: Box<StoppingReason>,

    reward_function: Box<AssetType>,

    discount_factor: Box<StoppingReason>,

    exploration_rate: Box<StoppingReason>,

    hyperparameters: AdaptiveBehaviorInfo,

    metadata: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct ModelTrainingResult {
    description: String,

    properties: ModelTrainingResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    model_training_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ModelTrainingResultProperties {
    training_id: ExperienceId,

    config: Box<AssetType>,

    model_version: ExperienceId,

    training_samples: ExperienceId,

    validation_samples: ExperienceId,

    training_loss: ExperienceId,

    validation_loss: ExperienceId,

    accuracy: AdaptiveBehaviorInfo,

    precision: Box<EnhancedFunctionTelemetry>,

    recall: Box<EnhancedFunctionTelemetry>,

    f1_score: Box<EnhancedFunctionTelemetry>,

    average_reward: Box<EnhancedFunctionTelemetry>,

    training_duration_seconds: Box<StoppingReason>,

    started_at: Timestamp,

    completed_at: Timestamp,

    metrics: Box<AssetType>,

    model_path: Box<EnhancedFunctionTelemetry>,

    metadata: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct ModuleStatus {
    description: String,

    properties: ModuleStatusProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    module_status_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ModuleStatusProperties {
    module: Config,

    status: ExperienceId,

    worker_id: ExperienceId,

    worker_count: Box<StoppingReason>,

    queue_size: Box<StoppingReason>,

    tasks_completed: Box<StoppingReason>,

    tasks_failed: Box<StoppingReason>,

    last_heartbeat: Environment,

    metrics: Environment,

    uptime_seconds: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct MultiLanguageCodebase {
    description: String,

    properties: MultiLanguageCodebaseProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    multi_language_codebase_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct MultiLanguageCodebaseProperties {
    project_name: Box<EnhancedFunctionTelemetry>,

    languages: Box<EnhancedFunctionTelemetry>,

    primary_language: Language,

    frameworks: RemediationType,

    total_files: Box<EnhancedFunctionTelemetry>,

    total_lines: Box<EnhancedFunctionTelemetry>,

    vulnerability_distribution: Box<EnhancedFunctionTelemetry>,

    dependencies: RemediationType,
}

#[derive(Serialize, Deserialize)]
pub struct NotificationPayload {
    description: String,

    properties: NotificationPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    notification_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct NotificationPayloadProperties {
    notification_id: ExperienceId,

    notification_type: ExperienceId,

    priority: ExperienceId,

    title: ExperienceId,

    message: ExperienceId,

    details: Environment,

    recipients: Box<EnhancedFunctionTelemetry>,

    attachments: Environment,

    timestamp: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct OastCallbackDetail {
    description: String,

    properties: OastCallbackDetailProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    oast_callback_detail_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct OastCallbackDetailProperties {
    callback_type: ExperienceId,

    token: ExperienceId,

    source_ip: ExperienceId,

    timestamp: Authentication,

    protocol: Authentication,

    raw_data: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct OastEvent {
    description: String,

    properties: OastEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    oast_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct OastEventProperties {
    event_id: ExperienceId,

    probe_token: ExperienceId,

    event_type: ExperienceId,

    source_ip: ExperienceId,

    timestamp: Authentication,

    protocol: AdaptiveBehaviorInfo,

    raw_request: AdaptiveBehaviorInfo,

    raw_data: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct OastProbe {
    description: String,

    properties: OastProbeProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    oast_probe_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct OastProbeProperties {
    probe_id: ExperienceId,

    token: ExperienceId,

    callback_url: ExperienceId,

    task_id: ExperienceId,

    scan_id: ExperienceId,

    created_at: Timestamp,

    expires_at: AdaptiveBehaviorInfo,

    status: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct PlanExecutionMetrics {
    description: String,

    properties: PlanExecutionMetricsProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    plan_execution_metrics_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct PlanExecutionMetricsProperties {
    plan_id: ExperienceId,

    session_id: ExperienceId,

    expected_steps: ExperienceId,

    executed_steps: ExperienceId,

    completed_steps: ExperienceId,

    failed_steps: ExperienceId,

    skipped_steps: ExperienceId,

    extra_actions: ExperienceId,

    completion_rate: ExperienceId,

    success_rate: ExperienceId,

    sequence_accuracy: ExperienceId,

    goal_achieved: ExperienceId,

    reward_score: ExperienceId,

    total_execution_time: ExperienceId,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct PlanExecutionResult {
    description: String,

    properties: PlanExecutionResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    plan_execution_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct PlanExecutionResultProperties {
    result_id: ExperienceId,

    plan_id: ExperienceId,

    session_id: ExperienceId,

    plan: Config,

    trace: Box<StoppingReason>,

    metrics: Config,

    findings: Environment,

    anomalies: AdaptiveBehaviorInfo,

    recommendations: AdaptiveBehaviorInfo,

    status: ExperienceId,

    completed_at: Timestamp,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct PostExResultPayload {
    description: String,

    properties: PostExResultPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    post_ex_result_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct PostExResultPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    test_type: Config,

    findings: Environment,

    risk_level: Config,

    safe_mode: ExperienceId,

    authorization_verified: AdaptiveBehaviorInfo,

    timestamp: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct PostExTestPayload {
    description: String,

    properties: PostExTestPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    post_ex_test_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct PostExTestPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    test_type: Config,

    target: ExperienceId,

    safe_mode: Box<AssetType>,

    authorization_token: TrainingId,

    context: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct RagKnowledgeUpdatePayload {
    description: String,

    properties: RagKnowledgeUpdatePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    rag_knowledge_update_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RagKnowledgeUpdatePayloadProperties {
    knowledge_type: ExperienceId,

    content: ExperienceId,

    source_id: Box<EnhancedFunctionTelemetry>,

    category: TrainingId,

    tags: Box<EnhancedFunctionTelemetry>,

    related_cve: Box<EnhancedFunctionTelemetry>,

    related_cwe: Box<EnhancedFunctionTelemetry>,

    mitre_techniques: Authentication,

    confidence: Authentication,

    metadata: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct RagQueryPayload {
    description: String,

    properties: RagQueryPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    rag_query_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RagQueryPayloadProperties {
    query_id: ExperienceId,

    query_text: ExperienceId,

    top_k: Authentication,

    min_similarity: Authentication,

    knowledge_types: Parameters,

    categories: Parameters,

    metadata: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct RagResponsePayload {
    description: String,

    properties: RagResponsePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    rag_response_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RagResponsePayloadProperties {
    query_id: ExperienceId,

    results: Environment,

    total_results: ExperienceId,

    avg_similarity: Box<EnhancedFunctionTelemetry>,

    enhanced_context: TrainingId,

    metadata: AdaptiveBehaviorInfo,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct RemediationGeneratePayload {
    description: String,

    properties: RemediationGeneratePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    remediation_generate_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RemediationGeneratePayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    finding_id: ExperienceId,

    vulnerability_type: Config,

    remediation_type: Config,

    context: Environment,

    auto_apply: Box<EnhancedFunctionTelemetry>,
}

#[derive(Serialize, Deserialize)]
pub struct RemediationResultPayload {
    description: String,

    properties: RemediationResultPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    remediation_result_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RemediationResultPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    finding_id: ExperienceId,

    remediation_type: Config,

    status: ExperienceId,

    patch_content: TrainingId,

    instructions: Environment,

    verification_steps: Environment,

    risk_assessment: Environment,

    timestamp: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct RiskAssessmentContext {
    description: String,

    properties: RiskAssessmentContextProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    risk_assessment_context_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RiskAssessmentContextProperties {
    environment: Config,

    business_criticality: Config,

    data_sensitivity: ApiSchema,

    asset_exposure: ApiSchema,

    compliance_tags: Box<AssetType>,

    asset_value: TrainingId,

    user_base: TrainingId,

    sla_hours: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct RiskAssessmentResult {
    description: String,

    properties: RiskAssessmentResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    risk_assessment_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RiskAssessmentResultProperties {
    finding_id: ExperienceId,

    technical_risk_score: ExperienceId,

    business_risk_score: ExperienceId,

    risk_level: Config,

    priority_score: ExperienceId,

    context_multiplier: ExperienceId,

    business_impact: Box<AssetType>,

    recommendations: Box<AssetType>,

    estimated_effort: TrainingId,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct RiskFactor {
    description: String,

    properties: RiskFactorProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    risk_factor_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RiskFactorProperties {
    factor_name: Box<AssetType>,

    weight: Box<AssetType>,

    value: Box<AssetType>,

    description: Box<AssetType>,
}

#[derive(Serialize, Deserialize)]
pub struct RiskTrendAnalysis {
    description: String,

    properties: RiskTrendAnalysisProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    risk_trend_analysis_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct RiskTrendAnalysisProperties {
    period_start: Authentication,

    period_end: Authentication,

    total_vulnerabilities: ExperienceId,

    risk_distribution: Authentication,

    average_risk_score: ExperienceId,

    trend: ExperienceId,

    improvement_percentage: Box<EnhancedFunctionTelemetry>,

    top_risks: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct SarifLocation {
    description: String,

    properties: SarifLocationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_location_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifLocationProperties {
    uri: Box<EnhancedFunctionTelemetry>,

    start_line: Box<EnhancedFunctionTelemetry>,

    start_column: Box<EnhancedFunctionTelemetry>,

    end_line: Box<EnhancedFunctionTelemetry>,

    end_column: Box<EnhancedFunctionTelemetry>,
}

#[derive(Serialize, Deserialize)]
pub struct SarifReport {
    description: String,

    properties: SarifReportProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_report_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifReportProperties {
    version: ScanScope,

    #[serde(rename = "$schema")]
    schema: ScanScope,

    runs: ScanScope,

    properties: ScanScope,
}

#[derive(Serialize, Deserialize)]
pub struct SarifResult {
    description: String,

    properties: SarifResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifResultProperties {
    rule_id: Authentication,

    message: Authentication,

    level: Authentication,

    locations: Authentication,

    partial_fingerprints: Authentication,

    properties: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct SarifRule {
    description: String,

    properties: SarifRuleProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_rule_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifRuleProperties {
    id: AdaptiveBehaviorInfo,

    name: AdaptiveBehaviorInfo,

    short_description: AdaptiveBehaviorInfo,

    full_description: AdaptiveBehaviorInfo,

    help_uri: AdaptiveBehaviorInfo,

    default_level: AdaptiveBehaviorInfo,

    properties: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
pub struct SarifRun {
    description: String,

    properties: SarifRunProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_run_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifRunProperties {
    tool: Language,

    results: Box<StoppingReason>,

    invocations: Box<StoppingReason>,

    artifacts: Box<StoppingReason>,

    properties: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct SarifTool {
    description: String,

    properties: SarifToolProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sarif_tool_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SarifToolProperties {
    name: Box<StoppingReason>,

    version: Box<StoppingReason>,

    information_uri: Box<StoppingReason>,

    rules: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct SastdastCorrelation {
    description: String,

    properties: SastdastCorrelationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sastdast_correlation_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SastdastCorrelationProperties {
    correlation_id: ExperienceId,

    sast_finding_id: ExperienceId,

    dast_finding_id: ExperienceId,

    data_flow_path: Environment,

    verification_status: ExperienceId,

    confidence_score: ExperienceId,

    explanation: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct ScanCompletedPayload {
    description: String,

    properties: ScanCompletedPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    scan_completed_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ScanCompletedPayloadProperties {
    scan_id: ExperienceId,

    status: ExperienceId,

    summary: Config,

    assets: ScanScope,

    fingerprints: ApiSchema,

    error_info: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct ScanStartPayload {
    description: String,

    properties: ScanStartPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    scan_start_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ScanStartPayloadProperties {
    scan_id: ExperienceId,

    targets: FluffyTargets,

    scope: Config,

    authentication: Config,

    strategy: TrainingId,

    rate_limit: Config,

    custom_headers: Authentication,

    x_forwarded_for: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct FluffyTargets {
    items: FormUrlItems,

    title: String,

    #[serde(rename = "type")]
    targets_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ScenarioTestResult {
    description: String,

    properties: ScenarioTestResultProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    scenario_test_result_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ScenarioTestResultProperties {
    test_id: ExperienceId,

    scenario_id: ExperienceId,

    model_version: ExperienceId,

    generated_plan: Authentication,

    execution_result: Authentication,

    score: ExperienceId,

    comparison: Authentication,

    passed: ExperienceId,

    tested_at: Authentication,

    metadata: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct SensitiveMatch {
    description: String,

    properties: SensitiveMatchProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    sensitive_match_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SensitiveMatchProperties {
    match_id: ExperienceId,

    pattern_name: ExperienceId,

    matched_text: ExperienceId,

    context: ExperienceId,

    confidence: Environment,

    line_number: AdaptiveBehaviorInfo,

    file_path: Environment,

    url: AdaptiveBehaviorInfo,

    severity: Severity,
}

#[derive(Serialize, Deserialize)]
pub struct Severity {
    #[serde(rename = "$ref")]
    severity_ref: String,

    #[serde(rename = "default")]
    severity_default: String,
}

#[derive(Serialize, Deserialize)]
pub struct SessionState {
    description: String,

    properties: SessionStateProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    session_state_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SessionStateProperties {
    session_id: ExperienceId,

    plan_id: ExperienceId,

    scan_id: ExperienceId,

    status: ExperienceId,

    current_step_index: Box<StoppingReason>,

    completed_steps: AdaptiveBehaviorInfo,

    pending_steps: Box<AssetType>,

    context: Box<StoppingReason>,

    variables: Box<AssetType>,

    started_at: RemediationType,

    updated_at: RemediationType,

    timeout_at: Box<EnhancedFunctionTelemetry>,

    metadata: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct SiemEvent {
    description: String,

    properties: SiemEventProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    siem_event_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SiemEventProperties {
    event_id: Environment,

    event_type: Environment,

    source_system: RemediationType,

    timestamp: RemediationType,

    received_at: Environment,

    severity: Language,

    category: Environment,

    subcategory: RemediationType,

    source_ip: Environment,

    source_port: TrainingId,

    destination_ip: Environment,

    destination_port: Environment,

    username: RemediationType,

    asset_id: Environment,

    hostname: Environment,

    description: Environment,

    raw_log: Environment,

    correlation_rules: Environment,

    related_events: Environment,

    status: RemediationType,

    assigned_to: Environment,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct SiemEventPayload {
    description: String,

    properties: SiemEventPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    siem_event_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SiemEventPayloadProperties {
    event_id: ExperienceId,

    event_type: ExperienceId,

    severity: ExperienceId,

    source: ExperienceId,

    destination: RemediationType,

    message: ExperienceId,

    details: Box<AssetType>,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct StandardScenario {
    description: String,

    properties: StandardScenarioProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    standard_scenario_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct StandardScenarioProperties {
    scenario_id: ExperienceId,

    name: ExperienceId,

    description: ExperienceId,

    vulnerability_type: Config,

    difficulty_level: ExperienceId,

    target_config: Box<StoppingReason>,

    expected_plan: Box<StoppingReason>,

    success_criteria: Box<StoppingReason>,

    tags: Box<EnhancedFunctionTelemetry>,

    created_at: RemediationType,

    metadata: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct SystemOrchestration {
    description: String,

    properties: SystemOrchestrationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    system_orchestration_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct SystemOrchestrationProperties {
    orchestration_id: ScanScope,

    orchestration_name: ScanScope,

    module_statuses: ScanScope,

    scan_configuration: ScanScope,

    resource_allocation: ScanScope,

    overall_status: ScanScope,

    active_scans: ScanScope,

    queued_tasks: ScanScope,

    system_cpu: ScanScope,

    system_memory: ScanScope,

    network_throughput: ScanScope,

    created_at: Timestamp,

    updated_at: Timestamp,

    metadata: ScanScope,
}

#[derive(Serialize, Deserialize)]
pub struct TaskDependency {
    description: String,

    properties: TaskDependencyProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    task_dependency_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TaskDependencyProperties {
    dependency_type: RemediationType,

    dependent_task_id: RemediationType,

    condition: ScanScope,

    required: RemediationType,
}

#[derive(Serialize, Deserialize)]
pub struct TaskQueue {
    description: String,

    properties: TaskQueueProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    task_queue_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TaskQueueProperties {
    queue_id: ScanScope,

    queue_name: ScanScope,

    max_concurrent_tasks: ScanScope,

    task_timeout: ScanScope,

    pending_tasks: ScanScope,

    running_tasks: ScanScope,

    completed_tasks: ScanScope,

    total_processed: ScanScope,

    success_rate: ScanScope,

    average_execution_time: ScanScope,

    created_at: Timestamp,

    last_activity: Timestamp,

    metadata: ScanScope,
}

#[derive(Serialize, Deserialize)]
pub struct TaskUpdatePayload {
    description: String,

    properties: TaskUpdatePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    task_update_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TaskUpdatePayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    status: ExperienceId,

    worker_id: ExperienceId,

    details: Payload,
}

#[derive(Serialize, Deserialize)]
pub struct TechnicalFingerprint {
    description: String,

    properties: TechnicalFingerprintProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    technical_fingerprint_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TechnicalFingerprintProperties {
    technology: Environment,

    version: Environment,

    confidence: Environment,

    detection_method: Environment,

    evidence: Environment,

    category: TrainingId,

    subcategory: Environment,

    known_vulnerabilities: Environment,

    eol_status: Environment,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct TestExecution {
    description: String,

    properties: TestExecutionProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    test_execution_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TestExecutionProperties {
    execution_id: TrainingId,

    test_case_id: Authentication,

    target_url: Authentication,

    timeout: Authentication,

    retry_attempts: Authentication,

    status: Language,

    start_time: Authentication,

    end_time: TrainingId,

    duration: TrainingId,

    success: Authentication,

    vulnerability_found: Authentication,

    confidence_level: Language,

    request_data: TrainingId,

    response_data: TrainingId,

    evidence: TrainingId,

    error_message: TrainingId,

    cpu_usage: Environment,

    memory_usage: TrainingId,

    network_traffic: TrainingId,

    metadata: TrainingId,
}

#[derive(Serialize, Deserialize)]
pub struct TestStrategy {
    description: String,

    properties: TestStrategyProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    test_strategy_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TestStrategyProperties {
    strategy_id: Box<EnhancedFunctionTelemetry>,

    strategy_name: Box<EnhancedFunctionTelemetry>,

    target_type: Box<EnhancedFunctionTelemetry>,

    test_categories: Box<EnhancedFunctionTelemetry>,

    test_sequence: Box<EnhancedFunctionTelemetry>,

    parallel_execution: Box<EnhancedFunctionTelemetry>,

    trigger_conditions: Box<EnhancedFunctionTelemetry>,

    stop_conditions: Box<EnhancedFunctionTelemetry>,

    priority_weights: Box<EnhancedFunctionTelemetry>,

    resource_limits: Box<EnhancedFunctionTelemetry>,

    learning_enabled: Box<EnhancedFunctionTelemetry>,

    adaptation_threshold: Box<EnhancedFunctionTelemetry>,

    effectiveness_score: Box<EnhancedFunctionTelemetry>,

    usage_count: Box<EnhancedFunctionTelemetry>,

    success_rate: Box<EnhancedFunctionTelemetry>,

    created_at: Timestamp,

    metadata: Box<EnhancedFunctionTelemetry>,
}

#[derive(Serialize, Deserialize)]
pub struct ThreatIntelLookupPayload {
    description: String,

    properties: ThreatIntelLookupPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    threat_intel_lookup_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ThreatIntelLookupPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    indicator: ExperienceId,

    indicator_type: Config,

    sources: Sources,

    enrich: AdaptiveBehaviorInfo,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sources {
    any_of: Vec<SourcesAnyOf>,

    #[serde(rename = "default")]
    sources_default: Option<serde_json::Value>,

    title: String,
}

#[derive(Serialize, Deserialize)]
pub struct SourcesAnyOf {
    items: Option<Config>,

    #[serde(rename = "type")]
    any_of_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ThreatIntelResultPayload {
    description: String,

    properties: ThreatIntelResultPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    threat_intel_result_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct ThreatIntelResultPayloadProperties {
    task_id: ExperienceId,

    scan_id: ExperienceId,

    indicator: ExperienceId,

    indicator_type: Config,

    threat_level: Config,

    sources: Box<AssetType>,

    mitre_techniques: Box<AssetType>,

    enrichment_data: TrainingId,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct TraceRecord {
    description: String,

    properties: TraceRecordProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    trace_record_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct TraceRecordProperties {
    trace_id: ExperienceId,

    plan_id: ExperienceId,

    step_id: ExperienceId,

    session_id: ExperienceId,

    tool_name: ExperienceId,

    input_data: Box<StoppingReason>,

    output_data: Box<StoppingReason>,

    status: ExperienceId,

    error_message: Box<StoppingReason>,

    execution_time_seconds: Box<StoppingReason>,

    timestamp: Timestamp,

    environment_response: Box<StoppingReason>,

    metadata: Box<StoppingReason>,
}

#[derive(Serialize, Deserialize)]
pub struct Vulnerability {
    description: String,

    properties: VulnerabilityProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    vulnerability_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityProperties {
    name: Config,

    cwe: CapecId,

    cve: CapecId,

    severity: Config,

    confidence: Config,

    description: Environment,

    cvss_score: Box<StoppingReason>,

    cvss_vector: Environment,

    owasp_category: RemediationType,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityCorrelation {
    description: String,

    properties: VulnerabilityCorrelationProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    vulnerability_correlation_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityCorrelationProperties {
    correlation_id: ExperienceId,

    correlation_type: ExperienceId,

    related_findings: AdaptiveBehaviorInfo,

    confidence_score: ExperienceId,

    root_cause: AdaptiveBehaviorInfo,

    common_components: ScanScope,

    explanation: ScanScope,

    timestamp: Timestamp,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityDiscovery {
    description: String,

    properties: VulnerabilityDiscoveryProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    vulnerability_discovery_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityDiscoveryProperties {
    discovery_id: Authentication,

    vulnerability_id: Authentication,

    asset_id: AdaptiveBehaviorInfo,

    title: Authentication,

    description: Authentication,

    severity: Language,

    confidence: Language,

    vulnerability_type: Authentication,

    affected_component: AdaptiveBehaviorInfo,

    attack_vector: AdaptiveBehaviorInfo,

    detection_method: Authentication,

    scanner_name: Authentication,

    scan_rule_id: Authentication,

    evidence: Authentication,

    proof_of_concept: Authentication,

    false_positive_likelihood: Authentication,

    impact_assessment: Authentication,

    exploitability: Authentication,

    remediation_advice: Authentication,

    remediation_priority: Authentication,

    cve_ids: AdaptiveBehaviorInfo,

    cwe_ids: Authentication,

    cvss_score: AdaptiveBehaviorInfo,

    discovered_at: Authentication,

    metadata: Authentication,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityLifecyclePayload {
    description: String,

    properties: VulnerabilityLifecyclePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    vulnerability_lifecycle_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityLifecyclePayloadProperties {
    vulnerability_id: ExperienceId,

    finding_id: ExperienceId,

    asset_id: ExperienceId,

    vulnerability_type: Config,

    severity: Config,

    confidence: Config,

    status: Config,

    exploitability: ApiSchema,

    assigned_to: Box<StoppingReason>,

    due_date: AdaptiveBehaviorInfo,

    first_detected: Environment,

    last_seen: Environment,

    resolution_date: AdaptiveBehaviorInfo,

    metadata: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityUpdatePayload {
    description: String,

    properties: VulnerabilityUpdatePayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    vulnerability_update_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct VulnerabilityUpdatePayloadProperties {
    vulnerability_id: ExperienceId,

    status: Config,

    assigned_to: Box<StoppingReason>,

    comment: Box<StoppingReason>,

    metadata: Box<StoppingReason>,

    updated_by: Environment,

    timestamp: Environment,
}

#[derive(Serialize, Deserialize)]
pub struct WebhookPayload {
    description: String,

    properties: WebhookPayloadProperties,

    required: Vec<String>,

    title: String,

    #[serde(rename = "type")]
    webhook_payload_type: Type,
}

#[derive(Serialize, Deserialize)]
pub struct WebhookPayloadProperties {
    webhook_id: Environment,

    event_type: Environment,

    source: Environment,

    timestamp: Environment,

    data: Environment,

    delivery_url: Box<StoppingReason>,

    retry_count: Environment,

    max_retries: Environment,

    status: Environment,

    delivered_at: Environment,

    error_message: Environment,

    metadata: Environment,
}
