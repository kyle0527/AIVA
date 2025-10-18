// Code generated from JSON Schema using quicktype. DO NOT EDIT.
// To parse and unparse this JSON data, add this code to your project and do:
//
//    aIVASchemas, err := UnmarshalAIVASchemas(bytes)
//    bytes, err = aIVASchemas.Marshal()

package main

import "bytes"
import "errors"
import "time"

import "encoding/json"

func UnmarshalAIVASchemas(data []byte) (AIVASchemas, error) {
	var r AIVASchemas
	err := json.Unmarshal(data, &r)
	return r, err
}

func (r *AIVASchemas) Marshal() ([]byte, error) {
	return json.Marshal(r)
}

type AIVASchemas struct {
	Schema      string    `json:"$schema"`
	ID          string    `json:"$id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Version     string    `json:"version"`
	GeneratedAt time.Time `json:"generated_at"`
	Generator   string    `json:"generator"`
	Defs        Defs      `json:"$defs"`
}

type Defs struct {
	ModuleName                       AssetType                        `json:"ModuleName"`
	MessageHeader                    MessageHeader                    `json:"MessageHeader"`
	Authentication                   Authentication                   `json:"Authentication"`
	RateLimit                        AdaptiveBehaviorInfo             `json:"RateLimit"`
	ScanScope                        ScanScope                        `json:"ScanScope"`
	Asset                            Asset                            `json:"Asset"`
	Summary                          StoppingReason                   `json:"Summary"`
	Fingerprints                     AdaptiveBehaviorInfo             `json:"Fingerprints"`
	ExecutionError                   ExecutionError                   `json:"ExecutionError"`
	RiskFactor                       RiskFactor                       `json:"RiskFactor"`
	TaskDependency                   TaskDependency                   `json:"TaskDependency"`
	Topic                            AssetType                        `json:"Topic"`
	AivaMessage                      AivaMessage                      `json:"AivaMessage"`
	AIVARequest                      AIVARequest                      `json:"AIVARequest"`
	AIVAResponse                     AIVAResponse                     `json:"AIVAResponse"`
	AIVAEvent                        AIVAEvent                        `json:"AIVAEvent"`
	AIVACommand                      AIVACommand                      `json:"AIVACommand"`
	ScanStartPayload                 ScanStartPayload                 `json:"ScanStartPayload"`
	ScanCompletedPayload             ScanCompletedPayload             `json:"ScanCompletedPayload"`
	FunctionTaskContext              Authentication                   `json:"FunctionTaskContext"`
	FunctionTaskTarget               FunctionTaskTarget               `json:"FunctionTaskTarget"`
	FunctionTaskTestConfig           AdaptiveBehaviorInfo             `json:"FunctionTaskTestConfig"`
	FunctionTaskPayload              FunctionTaskPayload              `json:"FunctionTaskPayload"`
	FeedbackEventPayload             FeedbackEventPayload             `json:"FeedbackEventPayload"`
	TaskUpdatePayload                TaskUpdatePayload                `json:"TaskUpdatePayload"`
	ConfigUpdatePayload              ConfigUpdatePayload              `json:"ConfigUpdatePayload"`
	IOCType                          EnhancedFunctionTelemetry        `json:"IOCType"`
	IntelSource                      EnhancedFunctionTelemetry        `json:"IntelSource"`
	ThreatIntelLookupPayload         ThreatIntelLookupPayload         `json:"ThreatIntelLookupPayload"`
	ThreatLevel                      AssetType                        `json:"ThreatLevel"`
	ThreatIntelResultPayload         ThreatIntelResultPayload         `json:"ThreatIntelResultPayload"`
	AuthZCheckPayload                AuthZCheckPayload                `json:"AuthZCheckPayload"`
	AuthZAnalysisPayload             AuthZAnalysisPayload             `json:"AuthZAnalysisPayload"`
	AuthZResultPayload               AuthZResultPayload               `json:"AuthZResultPayload"`
	RemediationType                  RemediationType                  `json:"RemediationType"`
	VulnerabilityType                AdaptiveBehaviorInfo             `json:"VulnerabilityType"`
	RemediationGeneratePayload       RemediationGeneratePayload       `json:"RemediationGeneratePayload"`
	RemediationResultPayload         RemediationResultPayload         `json:"RemediationResultPayload"`
	PostExTestType                   AssetType                        `json:"PostExTestType"`
	PostExTestPayload                PostExTestPayload                `json:"PostExTestPayload"`
	PostExResultPayload              PostExResultPayload              `json:"PostExResultPayload"`
	BizLogicTestPayload              BizLogicTestPayload              `json:"BizLogicTestPayload"`
	BizLogicResultPayload            BizLogicResultPayload            `json:"BizLogicResultPayload"`
	APISchemaPayload                 APISchemaPayload                 `json:"APISchemaPayload"`
	APITestCase                      APITestCase                      `json:"APITestCase"`
	APISecurityTestPayload           APISecurityTestPayload           `json:"APISecurityTestPayload"`
	EASMDiscoveryPayload             EASMDiscoveryPayload             `json:"EASMDiscoveryPayload"`
	EASMDiscoveryResult              EASMDiscoveryResult              `json:"EASMDiscoveryResult"`
	StandardScenario                 StandardScenario                 `json:"StandardScenario"`
	ScenarioTestResult               ScenarioTestResult               `json:"ScenarioTestResult"`
	ExploitPayload                   ExploitPayload                   `json:"ExploitPayload"`
	Confidence                       Authentication                   `json:"Confidence"`
	TestStatus                       Authentication                   `json:"TestStatus"`
	TestExecution                    TestExecution                    `json:"TestExecution"`
	Severity                         AssetType                        `json:"Severity"`
	ExploitResult                    ExploitResult                    `json:"ExploitResult"`
	TestStrategy                     TestStrategy                     `json:"TestStrategy"`
	Vulnerability                    Vulnerability                    `json:"Vulnerability"`
	Target                           Target                           `json:"Target"`
	FindingTarget                    Target                           `json:"FindingTarget"`
	FindingEvidence                  AdaptiveBehaviorInfo             `json:"FindingEvidence"`
	FindingImpact                    EnhancedFunctionTelemetry        `json:"FindingImpact"`
	FindingRecommendation            Authentication                   `json:"FindingRecommendation"`
	FindingPayload                   FindingPayload                   `json:"FindingPayload"`
	SensitiveMatch                   SensitiveMatch                   `json:"SensitiveMatch"`
	JavaScriptAnalysisResult         JavaScriptAnalysisResult         `json:"JavaScriptAnalysisResult"`
	VulnerabilityCorrelation         VulnerabilityCorrelation         `json:"VulnerabilityCorrelation"`
	CodeLevelRootCause               CodeLevelRootCause               `json:"CodeLevelRootCause"`
	SASTDASTCorrelation              SASTDASTCorrelation              `json:"SASTDASTCorrelation"`
	AIVerificationRequest            AIVerificationRequest            `json:"AIVerificationRequest"`
	AIVerificationResult             AIVerificationResult             `json:"AIVerificationResult"`
	HeartbeatPayload                 HeartbeatPayload                 `json:"HeartbeatPayload"`
	ModuleStatus                     ModuleStatus                     `json:"ModuleStatus"`
	FunctionTelemetry                AssetType                        `json:"FunctionTelemetry"`
	AdaptiveBehaviorInfo             AdaptiveBehaviorInfo             `json:"AdaptiveBehaviorInfo"`
	EarlyStoppingInfo                EarlyStoppingInfo                `json:"EarlyStoppingInfo"`
	ErrorCategory                    Environment                      `json:"ErrorCategory"`
	ErrorRecord                      ErrorRecord                      `json:"ErrorRecord"`
	OastCallbackDetail               OastCallbackDetail               `json:"OastCallbackDetail"`
	StoppingReason                   StoppingReason                   `json:"StoppingReason"`
	EnhancedFunctionTelemetry        EnhancedFunctionTelemetry        `json:"EnhancedFunctionTelemetry"`
	FunctionExecutionResult          FunctionExecutionResult          `json:"FunctionExecutionResult"`
	OastEvent                        OastEvent                        `json:"OastEvent"`
	OastProbe                        OastProbe                        `json:"OastProbe"`
	SIEMEventPayload                 SIEMEventPayload                 `json:"SIEMEventPayload"`
	SIEMEvent                        SIEMEvent                        `json:"SIEMEvent"`
	NotificationPayload              NotificationPayload              `json:"NotificationPayload"`
	CVSSv3Metrics                    CVSSv3Metrics                    `json:"CVSSv3Metrics"`
	AttackStep                       AttackStep                       `json:"AttackStep"`
	AttackPlan                       AttackPlan                       `json:"AttackPlan"`
	TraceRecord                      TraceRecord                      `json:"TraceRecord"`
	PlanExecutionMetrics             PlanExecutionMetrics             `json:"PlanExecutionMetrics"`
	PlanExecutionResult              PlanExecutionResult              `json:"PlanExecutionResult"`
	ModelTrainingConfig              ModelTrainingConfig              `json:"ModelTrainingConfig"`
	AITrainingStartPayload           AITrainingStartPayload           `json:"AITrainingStartPayload"`
	AITrainingProgressPayload        AITrainingProgressPayload        `json:"AITrainingProgressPayload"`
	AITrainingCompletedPayload       AITrainingCompletedPayload       `json:"AITrainingCompletedPayload"`
	AIExperienceCreatedEvent         AIExperienceCreatedEvent         `json:"AIExperienceCreatedEvent"`
	AITraceCompletedEvent            AITraceCompletedEvent            `json:"AITraceCompletedEvent"`
	AIModelUpdatedEvent              AIModelUpdatedEvent              `json:"AIModelUpdatedEvent"`
	AIModelDeployCommand             AIModelDeployCommand             `json:"AIModelDeployCommand"`
	RAGKnowledgeUpdatePayload        RAGKnowledgeUpdatePayload        `json:"RAGKnowledgeUpdatePayload"`
	RAGQueryPayload                  RAGQueryPayload                  `json:"RAGQueryPayload"`
	RAGResponsePayload               RAGResponsePayload               `json:"RAGResponsePayload"`
	ExperienceSample                 ExperienceSample                 `json:"ExperienceSample"`
	EnhancedVulnerability            EnhancedVulnerability            `json:"EnhancedVulnerability"`
	SARIFLocation                    SARIFLocation                    `json:"SARIFLocation"`
	SARIFResult                      SARIFResult                      `json:"SARIFResult"`
	SARIFRule                        SARIFRule                        `json:"SARIFRule"`
	SARIFTool                        SARIFTool                        `json:"SARIFTool"`
	SARIFRun                         SARIFRun                         `json:"SARIFRun"`
	SARIFReport                      SARIFReport                      `json:"SARIFReport"`
	AssetExposure                    AdaptiveBehaviorInfo             `json:"AssetExposure"`
	AssetType                        AssetType                        `json:"AssetType"`
	BusinessCriticality              AdaptiveBehaviorInfo             `json:"BusinessCriticality"`
	ComplianceFramework              Authentication                   `json:"ComplianceFramework"`
	DataSensitivity                  AdaptiveBehaviorInfo             `json:"DataSensitivity"`
	Environment                      Environment                      `json:"Environment"`
	AssetLifecyclePayload            AssetLifecyclePayload            `json:"AssetLifecyclePayload"`
	Exploitability                   Environment                      `json:"Exploitability"`
	VulnerabilityStatus              AdaptiveBehaviorInfo             `json:"VulnerabilityStatus"`
	VulnerabilityLifecyclePayload    VulnerabilityLifecyclePayload    `json:"VulnerabilityLifecyclePayload"`
	VulnerabilityUpdatePayload       VulnerabilityUpdatePayload       `json:"VulnerabilityUpdatePayload"`
	DiscoveredAsset                  DiscoveredAsset                  `json:"DiscoveredAsset"`
	TechnicalFingerprint             TechnicalFingerprint             `json:"TechnicalFingerprint"`
	AssetInventoryItem               AssetInventoryItem               `json:"AssetInventoryItem"`
	EASMAsset                        EASMAsset                        `json:"EASMAsset"`
	RiskAssessmentContext            RiskAssessmentContext            `json:"RiskAssessmentContext"`
	RiskLevel                        AssetType                        `json:"RiskLevel"`
	RiskAssessmentResult             RiskAssessmentResult             `json:"RiskAssessmentResult"`
	RiskTrendAnalysis                RiskTrendAnalysis                `json:"RiskTrendAnalysis"`
	AttackPathNodeType               AssetType                        `json:"AttackPathNodeType"`
	AttackPathNode                   AttackPathNode                   `json:"AttackPathNode"`
	AttackPathEdgeType               AssetType                        `json:"AttackPathEdgeType"`
	AttackPathEdge                   AttackPathEdge                   `json:"AttackPathEdge"`
	AttackPathPayload                AttackPathPayload                `json:"AttackPathPayload"`
	AttackPathRecommendation         AttackPathRecommendation         `json:"AttackPathRecommendation"`
	EnhancedFindingPayload           FindingPayload                   `json:"EnhancedFindingPayload"`
	EnhancedScanScope                EnhancedFunctionTelemetry        `json:"EnhancedScanScope"`
	EnhancedScanRequest              EnhancedScanRequest              `json:"EnhancedScanRequest"`
	EnhancedFunctionTaskTarget       EnhancedFunctionTaskTarget       `json:"EnhancedFunctionTaskTarget"`
	EnhancedIOCRecord                EnhancedIOCRecord                `json:"EnhancedIOCRecord"`
	EnhancedRiskAssessment           EnhancedRiskAssessment           `json:"EnhancedRiskAssessment"`
	EnhancedAttackPathNode           EnhancedAttackPathNode           `json:"EnhancedAttackPathNode"`
	EnhancedAttackPath               EnhancedAttackPath               `json:"EnhancedAttackPath"`
	EnhancedTaskExecution            EnhancedTaskExecution            `json:"EnhancedTaskExecution"`
	EnhancedVulnerabilityCorrelation EnhancedVulnerabilityCorrelation `json:"EnhancedVulnerabilityCorrelation"`
	SessionState                     SessionState                     `json:"SessionState"`
	ModelTrainingResult              ModelTrainingResult              `json:"ModelTrainingResult"`
	TaskQueue                        TaskQueue                        `json:"TaskQueue"`
	EnhancedModuleStatus             EnhancedModuleStatus             `json:"EnhancedModuleStatus"`
	SystemOrchestration              SystemOrchestration              `json:"SystemOrchestration"`
	WebhookPayload                   WebhookPayload                   `json:"WebhookPayload"`
	CVEReference                     CVEReference                     `json:"CVEReference"`
	CWEReference                     CWEReference                     `json:"CWEReference"`
	VulnerabilityDiscovery           VulnerabilityDiscovery           `json:"VulnerabilityDiscovery"`
	LanguageFramework                Environment                      `json:"LanguageFramework"`
	ProgrammingLanguage              AssetType                        `json:"ProgrammingLanguage"`
	LanguageDetectionResult          LanguageDetectionResult          `json:"LanguageDetectionResult"`
	VulnerabilityByLanguage          RemediationType                  `json:"VulnerabilityByLanguage"`
	LanguageSpecificVulnerability    LanguageSpecificVulnerability    `json:"LanguageSpecificVulnerability"`
	MultiLanguageCodebase            MultiLanguageCodebase            `json:"MultiLanguageCodebase"`
	CodeQualityMetric                Authentication                   `json:"CodeQualityMetric"`
	SecurityPattern                  Authentication                   `json:"SecurityPattern"`
	LanguageSpecificScanConfig       LanguageSpecificScanConfig       `json:"LanguageSpecificScanConfig"`
	CrossLanguageAnalysis            CrossLanguageAnalysis            `json:"CrossLanguageAnalysis"`
	LanguageSpecificPayload          LanguageSpecificPayload          `json:"LanguageSpecificPayload"`
	AILanguageModel                  AILanguageModel                  `json:"AILanguageModel"`
	CodeQualityReport                CodeQualityReport                `json:"CodeQualityReport"`
	LanguageInteroperability         LanguageInteroperability         `json:"LanguageInteroperability"`
}

type AIExperienceCreatedEvent struct {
	Description string                             `json:"description"`
	Properties  AIExperienceCreatedEventProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type AIExperienceCreatedEventProperties struct {
	ExperienceID      ExperienceID   `json:"experience_id"`
	TrainingID        TrainingID     `json:"training_id"`
	TraceID           ExperienceID   `json:"trace_id"`
	VulnerabilityType ExperienceID   `json:"vulnerability_type"`
	QualityScore      Authentication `json:"quality_score"`
	Success           ExperienceID   `json:"success"`
	PlanSummary       AssetType      `json:"plan_summary"`
	ResultSummary     AssetType      `json:"result_summary"`
	Metadata          AssetType      `json:"metadata"`
	Timestamp         Timestamp      `json:"timestamp"`
}

type ExperienceID struct {
	Title string `json:"title"`
	Type  Type   `json:"type"`
}

type ScanScopeProperties struct {
	Exclusions        TrainingID                `json:"exclusions"`
	IncludeSubdomains EnhancedFunctionTelemetry `json:"include_subdomains"`
	AllowedHosts      TrainingID                `json:"allowed_hosts"`
}

type ScanScope struct {
	Default              *ScanScopeDefault             `json:"default"`
	Title                string                        `json:"title"`
	Type                 *Type                         `json:"type,omitempty"`
	Description          *string                       `json:"description,omitempty"`
	AdditionalProperties *bool                         `json:"additionalProperties,omitempty"`
	Items                *AssetTypeItems               `json:"items,omitempty"`
	Properties           *ScanScopeProperties          `json:"properties,omitempty"`
	Minimum              *float64                      `json:"minimum,omitempty"`
	Maximum              *float64                      `json:"maximum,omitempty"`
	AnyOf                []AdditionalPropertiesElement `json:"anyOf,omitempty"`
}

type AuthenticationProperties struct {
	Method           *ScanScope                 `json:"method,omitempty"`
	Credentials      *Framework                 `json:"credentials,omitempty"`
	Fix              *Environment               `json:"fix,omitempty"`
	Priority         *Environment               `json:"priority,omitempty"`
	RemediationSteps *Environment               `json:"remediation_steps,omitempty"`
	References       *Environment               `json:"references,omitempty"`
	DBTypeHint       *EnhancedFunctionTelemetry `json:"db_type_hint,omitempty"`
	WafDetected      *Environment               `json:"waf_detected,omitempty"`
	RelatedFindings  *Parameters                `json:"related_findings,omitempty"`
}

type Authentication struct {
	Maximum              *float64                   `json:"maximum,omitempty"`
	Minimum              *float64                   `json:"minimum,omitempty"`
	Title                *string                    `json:"title,omitempty"`
	Type                 *Type                      `json:"type,omitempty"`
	Default              *AuthenticationDefault     `json:"default"`
	Description          *string                    `json:"description,omitempty"`
	Properties           *AuthenticationProperties  `json:"properties,omitempty"`
	Enum                 []string                   `json:"enum,omitempty"`
	Pattern              *string                    `json:"pattern,omitempty"`
	Items                *StoppingReasonItems       `json:"items,omitempty"`
	Format               *Format                    `json:"format,omitempty"`
	AdditionalProperties *AdditionalPropertiesUnion `json:"additionalProperties"`
	AnyOf                []AuthenticationAnyOf      `json:"anyOf,omitempty"`
}

type AdaptiveBehaviorInfoProperties struct {
	InitialBatchSize     *StoppingReason            `json:"initial_batch_size,omitempty"`
	FinalBatchSize       *StoppingReason            `json:"final_batch_size,omitempty"`
	RateAdjustments      *StoppingReason            `json:"rate_adjustments,omitempty"`
	ProtectionDetections *StoppingReason            `json:"protection_detections,omitempty"`
	BypassAttempts       *StoppingReason            `json:"bypass_attempts,omitempty"`
	SuccessRate          *StoppingReason            `json:"success_rate,omitempty"`
	Details              *AssetType                 `json:"details,omitempty"`
	Payload              *TrainingID                `json:"payload,omitempty"`
	ResponseTimeDelta    *TrainingID                `json:"response_time_delta,omitempty"`
	DBVersion            *TrainingID                `json:"db_version,omitempty"`
	Request              *TrainingID                `json:"request,omitempty"`
	Response             *TrainingID                `json:"response,omitempty"`
	Proof                *TrainingID                `json:"proof,omitempty"`
	WebServer            *Framework                 `json:"web_server,omitempty"`
	Framework            *Framework                 `json:"framework,omitempty"`
	Language             *Framework                 `json:"language,omitempty"`
	WafDetected          *EnhancedFunctionTelemetry `json:"waf_detected,omitempty"`
	WafVendor            *EnhancedFunctionTelemetry `json:"waf_vendor,omitempty"`
	Payloads             *AssetType                 `json:"payloads,omitempty"`
	CustomPayloads       *AssetType                 `json:"custom_payloads,omitempty"`
	BlindXSS             *AssetType                 `json:"blind_xss,omitempty"`
	DOMTesting           *AssetType                 `json:"dom_testing,omitempty"`
	Timeout              *TrainingID                `json:"timeout,omitempty"`
	RequestsPerSecond    *StoppingReason            `json:"requests_per_second,omitempty"`
	Burst                *StoppingReason            `json:"burst,omitempty"`
}

type AdaptiveBehaviorInfo struct {
	AdditionalProperties *AdditionalPropertiesUnion       `json:"additionalProperties"`
	Description          *string                          `json:"description,omitempty"`
	Title                string                           `json:"title"`
	Type                 *Type                            `json:"type,omitempty"`
	Default              *AdaptiveBehaviorInfoDefault     `json:"default"`
	Items                *AssetTypeItems                  `json:"items,omitempty"`
	Properties           *AdaptiveBehaviorInfoProperties  `json:"properties,omitempty"`
	Enum                 []string                         `json:"enum,omitempty"`
	AnyOf                []EnhancedFunctionTelemetryAnyOf `json:"anyOf,omitempty"`
	PropertyNames        *PropertyNames                   `json:"propertyNames,omitempty"`
}

type EnhancedFunctionTelemetryProperties struct {
	PayloadsSent     *StoppingReason       `json:"payloads_sent,omitempty"`
	Detections       *StoppingReason       `json:"detections,omitempty"`
	Attempts         *StoppingReason       `json:"attempts,omitempty"`
	Errors           *Authentication       `json:"errors,omitempty"`
	DurationSeconds  *StoppingReason       `json:"duration_seconds,omitempty"`
	Timestamp        *Timestamp            `json:"timestamp,omitempty"`
	ErrorRecords     *AdaptiveBehaviorInfo `json:"error_records,omitempty"`
	OastCallbacks    *AssetType            `json:"oast_callbacks,omitempty"`
	EarlyStopping    *APISchema            `json:"early_stopping,omitempty"`
	AdaptiveBehavior *APISchema            `json:"adaptive_behavior,omitempty"`
	IncludedHosts    *AdaptiveBehaviorInfo `json:"included_hosts,omitempty"`
	ExcludedHosts    *AdaptiveBehaviorInfo `json:"excluded_hosts,omitempty"`
	IncludedPaths    *AdaptiveBehaviorInfo `json:"included_paths,omitempty"`
	ExcludedPaths    *AdaptiveBehaviorInfo `json:"excluded_paths,omitempty"`
	MaxDepth         *Environment          `json:"max_depth,omitempty"`
	Description      *TrainingID           `json:"description,omitempty"`
	BusinessImpact   *TrainingID           `json:"business_impact,omitempty"`
	TechnicalImpact  *TrainingID           `json:"technical_impact,omitempty"`
	AffectedUsers    *TrainingID           `json:"affected_users,omitempty"`
	EstimatedCost    *TrainingID           `json:"estimated_cost,omitempty"`
}

type EnhancedFunctionTelemetry struct {
	AnyOf                []EnhancedFunctionTelemetryAnyOf     `json:"anyOf,omitempty"`
	Default              *EnhancedFunctionTelemetryDefault    `json:"default"`
	Title                string                               `json:"title"`
	Description          *string                              `json:"description,omitempty"`
	Type                 *Type                                `json:"type,omitempty"`
	AdditionalProperties *AdditionalPropertiesUnion           `json:"additionalProperties"`
	Properties           *EnhancedFunctionTelemetryProperties `json:"properties,omitempty"`
	Items                *AdditionalPropertiesElement         `json:"items,omitempty"`
	Enum                 []string                             `json:"enum,omitempty"`
	PropertyNames        *PropertyNames                       `json:"propertyNames,omitempty"`
	Minimum              *float64                             `json:"minimum,omitempty"`
	Maximum              *float64                             `json:"maximum,omitempty"`
}

type AssetTypeProperties struct {
	PayloadsSent    StoppingReason            `json:"payloads_sent"`
	Detections      StoppingReason            `json:"detections"`
	Attempts        StoppingReason            `json:"attempts"`
	Errors          EnhancedFunctionTelemetry `json:"errors"`
	DurationSeconds StoppingReason            `json:"duration_seconds"`
	Timestamp       Timestamp                 `json:"timestamp"`
}

type AssetType struct {
	AdditionalProperties *bool                         `json:"additionalProperties,omitempty"`
	Title                string                        `json:"title"`
	Type                 *Type                         `json:"type,omitempty"`
	Description          *string                       `json:"description,omitempty"`
	Items                *AssetTypeItems               `json:"items,omitempty"`
	Enum                 []string                      `json:"enum,omitempty"`
	Pattern              *string                       `json:"pattern,omitempty"`
	Default              *AssetTypeDefault             `json:"default"`
	Minimum              *float64                      `json:"minimum,omitempty"`
	Maximum              *float64                      `json:"maximum,omitempty"`
	Properties           *AssetTypeProperties          `json:"properties,omitempty"`
	AnyOf                []AdditionalPropertiesElement `json:"anyOf,omitempty"`
}

type TrainingID struct {
	AnyOf                []TrainingIDAnyOf            `json:"anyOf,omitempty"`
	Default              *string                      `json:"default"`
	Title                string                       `json:"title"`
	Description          *string                      `json:"description,omitempty"`
	Type                 *Type                        `json:"type,omitempty"`
	Items                *AdditionalPropertiesElement `json:"items,omitempty"`
	AdditionalProperties *bool                        `json:"additionalProperties,omitempty"`
}

type TrainingIDAnyOf struct {
	Type    Type     `json:"type"`
	Minimum *float64 `json:"minimum,omitempty"`
	Format  *Format  `json:"format,omitempty"`
}

type AdditionalPropertiesElement struct {
	Type Type `json:"type"`
}

type AssetTypeItems struct {
	Type *Type   `json:"type,omitempty"`
	Ref  *string `json:"$ref,omitempty"`
}

type Framework struct {
	AnyOf   []FrameworkItems `json:"anyOf"`
	Default interface{}      `json:"default"`
	Title   string           `json:"title"`
}

type FrameworkItems struct {
	AdditionalProperties *AdditionalPropertiesElement `json:"additionalProperties,omitempty"`
	Type                 Type                         `json:"type"`
}

type Environment struct {
	Items                *EnvironmentItems          `json:"items,omitempty"`
	Title                *string                    `json:"title,omitempty"`
	Type                 *Type                      `json:"type,omitempty"`
	Description          *string                    `json:"description,omitempty"`
	AdditionalProperties *AdditionalPropertiesUnion `json:"additionalProperties"`
	Minimum              *float64                   `json:"minimum,omitempty"`
	Maximum              *float64                   `json:"maximum,omitempty"`
	Format               *Format                    `json:"format,omitempty"`
	Default              *EnvironmentDefault        `json:"default"`
	Enum                 []string                   `json:"enum,omitempty"`
	AnyOf                []EnvironmentAnyOf         `json:"anyOf,omitempty"`
}

type EnvironmentAnyOf struct {
	Type    Type    `json:"type"`
	Pattern *string `json:"pattern,omitempty"`
	Format  *Format `json:"format,omitempty"`
}

type EnvironmentItems struct {
	AdditionalProperties *AdditionalPropertiesUnion `json:"additionalProperties"`
	Type                 *Type                      `json:"type,omitempty"`
	Ref                  *string                    `json:"$ref,omitempty"`
}

type Parameters struct {
	AnyOf   []AdditionalProperties `json:"anyOf"`
	Default interface{}            `json:"default"`
	Title   string                 `json:"title"`
}

type AdditionalProperties struct {
	Items *AdditionalPropertiesElement `json:"items,omitempty"`
	Type  Type                         `json:"type"`
}

type AuthenticationAnyOf struct {
	Minimum *int64 `json:"minimum,omitempty"`
	Type    Type   `json:"type"`
}

type StoppingReasonItems struct {
	Ref                  *string `json:"$ref,omitempty"`
	AdditionalProperties *bool   `json:"additionalProperties,omitempty"`
	Type                 *Type   `json:"type,omitempty"`
}

type StoppingReasonProperties struct {
	UrlsFound           StoppingReason `json:"urls_found"`
	FormsFound          StoppingReason `json:"forms_found"`
	ApisFound           StoppingReason `json:"apis_found"`
	ScanDurationSeconds StoppingReason `json:"scan_duration_seconds"`
}

type StoppingReason struct {
	Default              *float64                  `json:"default"`
	Title                string                    `json:"title"`
	Type                 *Type                     `json:"type,omitempty"`
	Description          *string                   `json:"description,omitempty"`
	Items                *StoppingReasonItems      `json:"items,omitempty"`
	AdditionalProperties *bool                     `json:"additionalProperties,omitempty"`
	AnyOf                []StoppingReasonAnyOf     `json:"anyOf,omitempty"`
	Enum                 []string                  `json:"enum,omitempty"`
	Properties           *StoppingReasonProperties `json:"properties,omitempty"`
}

type StoppingReasonAnyOf struct {
	Type    Type     `json:"type"`
	Maximum *float64 `json:"maximum,omitempty"`
	Minimum *float64 `json:"minimum,omitempty"`
}

type EnhancedFunctionTelemetryAnyOf struct {
	Maximum *float64 `json:"maximum,omitempty"`
	Minimum *float64 `json:"minimum,omitempty"`
	Type    Type     `json:"type"`
	Format  *Format  `json:"format,omitempty"`
}

type DefaultClass struct {
}

type PropertyNames struct {
	Ref string `json:"$ref"`
}

type APISchema struct {
	AnyOf   []AssetTypeItems `json:"anyOf"`
	Default interface{}      `json:"default"`
}

type Timestamp struct {
	Format Format `json:"format"`
	Title  string `json:"title"`
	Type   Type   `json:"type"`
}

type AILanguageModel struct {
	Description string                    `json:"description"`
	Properties  AILanguageModelProperties `json:"properties"`
	Required    []string                  `json:"required"`
	Title       string                    `json:"title"`
	Type        Type                      `json:"type"`
}

type AILanguageModelProperties struct {
	ModelName              AdaptiveBehaviorInfo `json:"model_name"`
	SupportedLanguages     AdaptiveBehaviorInfo `json:"supported_languages"`
	ModelType              AdaptiveBehaviorInfo `json:"model_type"`
	Version                AdaptiveBehaviorInfo `json:"version"`
	Capabilities           AdaptiveBehaviorInfo `json:"capabilities"`
	TrainingDataSize       TrainingID           `json:"training_data_size"`
	AccuracyMetrics        AdaptiveBehaviorInfo `json:"accuracy_metrics"`
	APIEndpoint            TrainingID           `json:"api_endpoint"`
	AuthenticationRequired AdaptiveBehaviorInfo `json:"authentication_required"`
}

type AIModelDeployCommand struct {
	Description string                         `json:"description"`
	Properties  AIModelDeployCommandProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type AIModelDeployCommandProperties struct {
	ModelID                 ExperienceID         `json:"model_id"`
	ModelVersion            ExperienceID         `json:"model_version"`
	CheckpointPath          ExperienceID         `json:"checkpoint_path"`
	DeploymentTarget        ScanScope            `json:"deployment_target"`
	DeploymentConfig        AssetType            `json:"deployment_config"`
	RequireValidation       AdaptiveBehaviorInfo `json:"require_validation"`
	MinPerformanceThreshold AdaptiveBehaviorInfo `json:"min_performance_threshold"`
	Metadata                AssetType            `json:"metadata"`
}

type AIModelUpdatedEvent struct {
	Description string                        `json:"description"`
	Properties  AIModelUpdatedEventProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type AIModelUpdatedEventProperties struct {
	ModelID            ExperienceID         `json:"model_id"`
	ModelVersion       ExperienceID         `json:"model_version"`
	TrainingID         TrainingID           `json:"training_id"`
	UpdateType         ExperienceID         `json:"update_type"`
	PerformanceMetrics AdaptiveBehaviorInfo `json:"performance_metrics"`
	ModelPath          TrainingID           `json:"model_path"`
	CheckpointPath     TrainingID           `json:"checkpoint_path"`
	IsDeployed         AdaptiveBehaviorInfo `json:"is_deployed"`
	Metadata           AssetType            `json:"metadata"`
	Timestamp          Timestamp            `json:"timestamp"`
}

type AITraceCompletedEvent struct {
	Description string                          `json:"description"`
	Properties  AITraceCompletedEventProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type AITraceCompletedEventProperties struct {
	TraceID         ExperienceID `json:"trace_id"`
	SessionID       TrainingID   `json:"session_id"`
	TrainingID      TrainingID   `json:"training_id"`
	TotalSteps      ExperienceID `json:"total_steps"`
	SuccessfulSteps ExperienceID `json:"successful_steps"`
	FailedSteps     ExperienceID `json:"failed_steps"`
	DurationSeconds ExperienceID `json:"duration_seconds"`
	FinalSuccess    ExperienceID `json:"final_success"`
	PlanType        ExperienceID `json:"plan_type"`
	Metadata        AssetType    `json:"metadata"`
	Timestamp       Timestamp    `json:"timestamp"`
}

type AITrainingCompletedPayload struct {
	Description string                               `json:"description"`
	Properties  AITrainingCompletedPayloadProperties `json:"properties"`
	Required    []string                             `json:"required"`
	Title       string                               `json:"title"`
	Type        Type                                 `json:"type"`
}

type AITrainingCompletedPayloadProperties struct {
	TrainingID           ExperienceID         `json:"training_id"`
	Status               ExperienceID         `json:"status"`
	TotalEpisodes        ExperienceID         `json:"total_episodes"`
	SuccessfulEpisodes   ExperienceID         `json:"successful_episodes"`
	FailedEpisodes       ExperienceID         `json:"failed_episodes"`
	TotalDurationSeconds ExperienceID         `json:"total_duration_seconds"`
	TotalSamples         ExperienceID         `json:"total_samples"`
	HighQualitySamples   ExperienceID         `json:"high_quality_samples"`
	MediumQualitySamples ExperienceID         `json:"medium_quality_samples"`
	LowQualitySamples    ExperienceID         `json:"low_quality_samples"`
	FinalAvgReward       TrainingID           `json:"final_avg_reward"`
	FinalAvgQuality      TrainingID           `json:"final_avg_quality"`
	BestEpisodeReward    TrainingID           `json:"best_episode_reward"`
	ModelCheckpointPath  TrainingID           `json:"model_checkpoint_path"`
	ModelMetrics         AdaptiveBehaviorInfo `json:"model_metrics"`
	ErrorMessage         TrainingID           `json:"error_message"`
	Metadata             AssetType            `json:"metadata"`
	CompletedAt          Timestamp            `json:"completed_at"`
}

type AITrainingProgressPayload struct {
	Description string                              `json:"description"`
	Properties  AITrainingProgressPayloadProperties `json:"properties"`
	Required    []string                            `json:"required"`
	Title       string                              `json:"title"`
	Type        Type                                `json:"type"`
}

type AITrainingProgressPayloadProperties struct {
	TrainingID         ExperienceID         `json:"training_id"`
	EpisodeNumber      ExperienceID         `json:"episode_number"`
	TotalEpisodes      ExperienceID         `json:"total_episodes"`
	SuccessfulEpisodes StoppingReason       `json:"successful_episodes"`
	FailedEpisodes     StoppingReason       `json:"failed_episodes"`
	TotalSamples       StoppingReason       `json:"total_samples"`
	HighQualitySamples StoppingReason       `json:"high_quality_samples"`
	AvgReward          TrainingID           `json:"avg_reward"`
	AvgQuality         TrainingID           `json:"avg_quality"`
	BestReward         TrainingID           `json:"best_reward"`
	ModelMetrics       AdaptiveBehaviorInfo `json:"model_metrics"`
	Status             ScanScope            `json:"status"`
	Metadata           AssetType            `json:"metadata"`
	Timestamp          Timestamp            `json:"timestamp"`
}

type AITrainingStartPayload struct {
	Description string                           `json:"description"`
	Properties  AITrainingStartPayloadProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type AITrainingStartPayloadProperties struct {
	TrainingID          ExperienceID  `json:"training_id"`
	TrainingType        ExperienceID  `json:"training_type"`
	ScenarioID          TrainingID    `json:"scenario_id"`
	TargetVulnerability TrainingID    `json:"target_vulnerability"`
	Config              PropertyNames `json:"config"`
	Metadata            AssetType     `json:"metadata"`
}

type AIVACommand struct {
	Description string                `json:"description"`
	Properties  AIVACommandProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type AIVACommandProperties struct {
	CommandID    ExperienceID   `json:"command_id"`
	CommandType  ExperienceID   `json:"command_type"`
	SourceModule ExperienceID   `json:"source_module"`
	TargetModule ExperienceID   `json:"target_module"`
	Payload      AssetType      `json:"payload"`
	Priority     Authentication `json:"priority"`
	TraceID      TrainingID     `json:"trace_id"`
	Metadata     AssetType      `json:"metadata"`
	Timestamp    ExperienceID   `json:"timestamp"`
}

type AIVAEvent struct {
	Description string              `json:"description"`
	Properties  AIVAEventProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type AIVAEventProperties struct {
	EventID      ExperienceID `json:"event_id"`
	EventType    ExperienceID `json:"event_type"`
	SourceModule ExperienceID `json:"source_module"`
	Payload      AssetType    `json:"payload"`
	TraceID      TrainingID   `json:"trace_id"`
	Metadata     AssetType    `json:"metadata"`
	Timestamp    ExperienceID `json:"timestamp"`
}

type AIVARequest struct {
	Description string                `json:"description"`
	Properties  AIVARequestProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type AIVARequestProperties struct {
	RequestID      ExperienceID   `json:"request_id"`
	SourceModule   ExperienceID   `json:"source_module"`
	TargetModule   ExperienceID   `json:"target_module"`
	RequestType    ExperienceID   `json:"request_type"`
	Payload        AssetType      `json:"payload"`
	TraceID        TrainingID     `json:"trace_id"`
	TimeoutSeconds Authentication `json:"timeout_seconds"`
	Metadata       AssetType      `json:"metadata"`
	Timestamp      ExperienceID   `json:"timestamp"`
}

type AIVAResponse struct {
	Description string                 `json:"description"`
	Properties  AIVAResponseProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type AIVAResponseProperties struct {
	RequestID    ExperienceID `json:"request_id"`
	ResponseType ExperienceID `json:"response_type"`
	Success      ExperienceID `json:"success"`
	Payload      Payload      `json:"payload"`
	ErrorCode    TrainingID   `json:"error_code"`
	ErrorMessage TrainingID   `json:"error_message"`
	Metadata     AssetType    `json:"metadata"`
	Timestamp    ExperienceID `json:"timestamp"`
}

type Payload struct {
	AnyOf   []JSONDataAnyOf `json:"anyOf"`
	Default interface{}     `json:"default"`
	Title   string          `json:"title"`
}

type JSONDataAnyOf struct {
	AdditionalProperties *bool `json:"additionalProperties,omitempty"`
	Type                 Type  `json:"type"`
}

type AIVerificationRequest struct {
	Description string                          `json:"description"`
	Properties  AIVerificationRequestProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type AIVerificationRequestProperties struct {
	VerificationID    ExperienceID  `json:"verification_id"`
	FindingID         ExperienceID  `json:"finding_id"`
	ScanID            ExperienceID  `json:"scan_id"`
	VulnerabilityType PropertyNames `json:"vulnerability_type"`
	Target            PropertyNames `json:"target"`
	Evidence          PropertyNames `json:"evidence"`
	VerificationMode  ScanScope     `json:"verification_mode"`
	Context           AssetType     `json:"context"`
}

type AIVerificationResult struct {
	Description string                         `json:"description"`
	Properties  AIVerificationResultProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type AIVerificationResultProperties struct {
	VerificationID     ExperienceID         `json:"verification_id"`
	FindingID          ExperienceID         `json:"finding_id"`
	VerificationStatus ExperienceID         `json:"verification_status"`
	ConfidenceScore    ExperienceID         `json:"confidence_score"`
	VerificationMethod ExperienceID         `json:"verification_method"`
	TestSteps          AdaptiveBehaviorInfo `json:"test_steps"`
	Observations       AdaptiveBehaviorInfo `json:"observations"`
	Recommendations    AdaptiveBehaviorInfo `json:"recommendations"`
	Timestamp          Timestamp            `json:"timestamp"`
}

type APISchemaPayload struct {
	Description string                     `json:"description"`
	Properties  APISchemaPayloadProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type APISchemaPayloadProperties struct {
	SchemaID       ExperienceID  `json:"schema_id"`
	ScanID         ExperienceID  `json:"scan_id"`
	SchemaType     ExperienceID  `json:"schema_type"`
	SchemaContent  SchemaContent `json:"schema_content"`
	BaseURL        ExperienceID  `json:"base_url"`
	Authentication PropertyNames `json:"authentication"`
}

type SchemaContent struct {
	AnyOf []JSONDataAnyOf `json:"anyOf"`
	Title string          `json:"title"`
}

type APISecurityTestPayload struct {
	Description string                           `json:"description"`
	Properties  APISecurityTestPayloadProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type APISecurityTestPayloadProperties struct {
	TaskID         ExperienceID         `json:"task_id"`
	ScanID         ExperienceID         `json:"scan_id"`
	APIType        ExperienceID         `json:"api_type"`
	APISchema      APISchema            `json:"api_schema"`
	TestCases      AdaptiveBehaviorInfo `json:"test_cases"`
	Authentication PropertyNames        `json:"authentication"`
}

type APITestCase struct {
	Description string                `json:"description"`
	Properties  APITestCaseProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type APITestCaseProperties struct {
	TestID           ExperienceID `json:"test_id"`
	TestType         ExperienceID `json:"test_type"`
	Endpoint         ExperienceID `json:"endpoint"`
	Method           ExperienceID `json:"method"`
	TestVectors      Environment  `json:"test_vectors"`
	ExpectedBehavior TrainingID   `json:"expected_behavior"`
}

type AivaMessage struct {
	Description string                `json:"description"`
	Properties  AivaMessageProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type AivaMessageProperties struct {
	Header        PropertyNames `json:"header"`
	Topic         PropertyNames `json:"topic"`
	SchemaVersion ScanScope     `json:"schema_version"`
	Payload       AssetType     `json:"payload"`
}

type Asset struct {
	Description string          `json:"description"`
	Properties  AssetProperties `json:"properties"`
	Required    []string        `json:"required"`
	Title       string          `json:"title"`
	Type        Type            `json:"type"`
}

type AssetProperties struct {
	AssetID    ExperienceID         `json:"asset_id"`
	Type       ExperienceID         `json:"type"`
	Value      ExperienceID         `json:"value"`
	Parameters Parameters           `json:"parameters"`
	HasForm    AdaptiveBehaviorInfo `json:"has_form"`
}

type AssetInventoryItem struct {
	Description string                       `json:"description"`
	Properties  AssetInventoryItemProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type AssetInventoryItemProperties struct {
	AssetID             AdaptiveBehaviorInfo `json:"asset_id"`
	AssetType           AdaptiveBehaviorInfo `json:"asset_type"`
	Name                AssetType            `json:"name"`
	IPAddress           TrainingID           `json:"ip_address"`
	Hostname            TrainingID           `json:"hostname"`
	Domain              AdaptiveBehaviorInfo `json:"domain"`
	Ports               AssetType            `json:"ports"`
	Fingerprints        AdaptiveBehaviorInfo `json:"fingerprints"`
	BusinessCriticality AdaptiveBehaviorInfo `json:"business_criticality"`
	Owner               TrainingID           `json:"owner"`
	Environment         AdaptiveBehaviorInfo `json:"environment"`
	LastScanned         AdaptiveBehaviorInfo `json:"last_scanned"`
	VulnerabilityCount  Authentication       `json:"vulnerability_count"`
	RiskScore           Authentication       `json:"risk_score"`
	DiscoveredAt        Timestamp            `json:"discovered_at"`
	UpdatedAt           Timestamp            `json:"updated_at"`
	Metadata            AssetType            `json:"metadata"`
}

type AssetLifecyclePayload struct {
	Description string                          `json:"description"`
	Properties  AssetLifecyclePayloadProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type AssetLifecyclePayloadProperties struct {
	AssetID             ExperienceID         `json:"asset_id"`
	AssetType           PropertyNames        `json:"asset_type"`
	Value               ExperienceID         `json:"value"`
	Environment         PropertyNames        `json:"environment"`
	BusinessCriticality PropertyNames        `json:"business_criticality"`
	DataSensitivity     APISchema            `json:"data_sensitivity"`
	AssetExposure       APISchema            `json:"asset_exposure"`
	Owner               TrainingID           `json:"owner"`
	Team                TrainingID           `json:"team"`
	ComplianceTags      AdaptiveBehaviorInfo `json:"compliance_tags"`
	Metadata            AssetType            `json:"metadata"`
	CreatedAt           Timestamp            `json:"created_at"`
}

type AttackPathEdge struct {
	Description string                   `json:"description"`
	Properties  AttackPathEdgeProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type AttackPathEdgeProperties struct {
	EdgeID       ExperienceID   `json:"edge_id"`
	SourceNodeID ExperienceID   `json:"source_node_id"`
	TargetNodeID ExperienceID   `json:"target_node_id"`
	EdgeType     PropertyNames  `json:"edge_type"`
	RiskScore    StoppingReason `json:"risk_score"`
	Properties   AssetType      `json:"properties"`
}

type AttackPathNode struct {
	Description string                   `json:"description"`
	Properties  AttackPathNodeProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type AttackPathNodeProperties struct {
	NodeID     ExperienceID  `json:"node_id"`
	NodeType   PropertyNames `json:"node_type"`
	Name       ExperienceID  `json:"name"`
	Properties AssetType     `json:"properties"`
}

type AttackPathPayload struct {
	Description string                      `json:"description"`
	Properties  AttackPathPayloadProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type AttackPathPayloadProperties struct {
	PathID         ExperienceID         `json:"path_id"`
	ScanID         ExperienceID         `json:"scan_id"`
	SourceNode     PropertyNames        `json:"source_node"`
	TargetNode     PropertyNames        `json:"target_node"`
	Nodes          AdaptiveBehaviorInfo `json:"nodes"`
	Edges          AdaptiveBehaviorInfo `json:"edges"`
	TotalRiskScore ExperienceID         `json:"total_risk_score"`
	PathLength     ExperienceID         `json:"path_length"`
	Description    TrainingID           `json:"description"`
	Timestamp      Timestamp            `json:"timestamp"`
}

type AttackPathRecommendation struct {
	Description string                             `json:"description"`
	Properties  AttackPathRecommendationProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type AttackPathRecommendationProperties struct {
	PathID                 ExperienceID         `json:"path_id"`
	RiskLevel              PropertyNames        `json:"risk_level"`
	PriorityScore          ExperienceID         `json:"priority_score"`
	ExecutiveSummary       ExperienceID         `json:"executive_summary"`
	TechnicalExplanation   ExperienceID         `json:"technical_explanation"`
	BusinessImpact         ExperienceID         `json:"business_impact"`
	RemediationSteps       AssetType            `json:"remediation_steps"`
	QuickWINS              AdaptiveBehaviorInfo `json:"quick_wins"`
	AffectedAssets         AdaptiveBehaviorInfo `json:"affected_assets"`
	EstimatedEffort        ExperienceID         `json:"estimated_effort"`
	EstimatedRiskReduction ExperienceID         `json:"estimated_risk_reduction"`
	Timestamp              Timestamp            `json:"timestamp"`
}

type AttackPlan struct {
	Description string               `json:"description"`
	Properties  AttackPlanProperties `json:"properties"`
	Required    []string             `json:"required"`
	Title       string               `json:"title"`
	Type        Type                 `json:"type"`
}

type AttackPlanProperties struct {
	PlanID          ExperienceID         `json:"plan_id"`
	ScanID          ExperienceID         `json:"scan_id"`
	AttackType      PropertyNames        `json:"attack_type"`
	Steps           AdaptiveBehaviorInfo `json:"steps"`
	Dependencies    RemediationType      `json:"dependencies"`
	Context         AssetType            `json:"context"`
	TargetInfo      AssetType            `json:"target_info"`
	CreatedAt       Timestamp            `json:"created_at"`
	CreatedBy       ScanScope            `json:"created_by"`
	MitreTechniques AdaptiveBehaviorInfo `json:"mitre_techniques"`
	MitreTactics    AssetType            `json:"mitre_tactics"`
	CapecID         CapecID              `json:"capec_id"`
	Metadata        AssetType            `json:"metadata"`
}

type CapecID struct {
	AnyOf       []CapecIDAnyOf `json:"anyOf"`
	Default     interface{}    `json:"default"`
	Title       string         `json:"title"`
	Description *string        `json:"description,omitempty"`
}

type CapecIDAnyOf struct {
	Pattern *string `json:"pattern,omitempty"`
	Type    Type    `json:"type"`
}

type RemediationType struct {
	AdditionalProperties *AdditionalProperties             `json:"additionalProperties,omitempty"`
	Title                string                            `json:"title"`
	Type                 *Type                             `json:"type,omitempty"`
	Description          *string                           `json:"description,omitempty"`
	Items                *PropertyNames                    `json:"items,omitempty"`
	Enum                 []string                          `json:"enum,omitempty"`
	Default              *EnhancedFunctionTelemetryDefault `json:"default"`
	AnyOf                []AdditionalPropertiesElement     `json:"anyOf,omitempty"`
	Format               *Format                           `json:"format,omitempty"`
}

type AttackStep struct {
	Description string               `json:"description"`
	Properties  AttackStepProperties `json:"properties"`
	Required    []string             `json:"required"`
	Title       string               `json:"title"`
	Type        Type                 `json:"type"`
}

type AttackStepProperties struct {
	StepID           ExperienceID   `json:"step_id"`
	Action           ExperienceID   `json:"action"`
	ToolType         ExperienceID   `json:"tool_type"`
	Target           AssetType      `json:"target"`
	Parameters       AssetType      `json:"parameters"`
	ExpectedResult   TrainingID     `json:"expected_result"`
	TimeoutSeconds   StoppingReason `json:"timeout_seconds"`
	RetryCount       StoppingReason `json:"retry_count"`
	MitreTechniqueID CapecID        `json:"mitre_technique_id"`
	MitreTactic      TrainingID     `json:"mitre_tactic"`
	Metadata         AssetType      `json:"metadata"`
}

type AuthZAnalysisPayload struct {
	Description string                         `json:"description"`
	Properties  AuthZAnalysisPayloadProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type AuthZAnalysisPayloadProperties struct {
	TaskID       ExperienceID         `json:"task_id"`
	ScanID       ExperienceID         `json:"scan_id"`
	AnalysisType ExperienceID         `json:"analysis_type"`
	Target       AdaptiveBehaviorInfo `json:"target"`
}

type AuthZCheckPayload struct {
	Description string                      `json:"description"`
	Properties  AuthZCheckPayloadProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type AuthZCheckPayloadProperties struct {
	TaskID     ExperienceID `json:"task_id"`
	ScanID     ExperienceID `json:"scan_id"`
	UserID     ExperienceID `json:"user_id"`
	Resource   ExperienceID `json:"resource"`
	Permission ExperienceID `json:"permission"`
	Context    AssetType    `json:"context"`
}

type AuthZResultPayload struct {
	Description string                       `json:"description"`
	Properties  AuthZResultPayloadProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type AuthZResultPayloadProperties struct {
	TaskID          ExperienceID         `json:"task_id"`
	ScanID          ExperienceID         `json:"scan_id"`
	Decision        ExperienceID         `json:"decision"`
	Analysis        AssetType            `json:"analysis"`
	Recommendations AdaptiveBehaviorInfo `json:"recommendations"`
	Timestamp       Timestamp            `json:"timestamp"`
}

type BizLogicResultPayload struct {
	Description string                          `json:"description"`
	Properties  BizLogicResultPayloadProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type BizLogicResultPayloadProperties struct {
	TaskID     ExperienceID `json:"task_id"`
	ScanID     ExperienceID `json:"scan_id"`
	TestType   ExperienceID `json:"test_type"`
	Status     ExperienceID `json:"status"`
	Findings   Environment  `json:"findings"`
	Statistics AssetType    `json:"statistics"`
	Timestamp  Timestamp    `json:"timestamp"`
}

type BizLogicTestPayload struct {
	Description string                        `json:"description"`
	Properties  BizLogicTestPayloadProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type BizLogicTestPayloadProperties struct {
	TaskID        ExperienceID         `json:"task_id"`
	ScanID        ExperienceID         `json:"scan_id"`
	TestType      ExperienceID         `json:"test_type"`
	TargetUrls    AdaptiveBehaviorInfo `json:"target_urls"`
	TestConfig    AssetType            `json:"test_config"`
	ProductID     AdaptiveBehaviorInfo `json:"product_id"`
	WorkflowSteps WorkflowSteps        `json:"workflow_steps"`
}

type WorkflowSteps struct {
	Items FrameworkItems `json:"items"`
	Title string         `json:"title"`
	Type  Type           `json:"type"`
}

type CVEReference struct {
	Description string                 `json:"description"`
	Properties  CVEReferenceProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type CVEReferenceProperties struct {
	CveID            AssetType                 `json:"cve_id"`
	Description      TrainingID                `json:"description"`
	CvssScore        EnhancedFunctionTelemetry `json:"cvss_score"`
	CvssVector       TrainingID                `json:"cvss_vector"`
	References       AdaptiveBehaviorInfo      `json:"references"`
	PublishedDate    AdaptiveBehaviorInfo      `json:"published_date"`
	LastModifiedDate AdaptiveBehaviorInfo      `json:"last_modified_date"`
}

type CVSSv3Metrics struct {
	Description string                  `json:"description"`
	Properties  CVSSv3MetricsProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type CVSSv3MetricsProperties struct {
	AttackVector               Authentication            `json:"attack_vector"`
	AttackComplexity           Authentication            `json:"attack_complexity"`
	PrivilegesRequired         Authentication            `json:"privileges_required"`
	UserInteraction            Authentication            `json:"user_interaction"`
	Scope                      Authentication            `json:"scope"`
	Confidentiality            Authentication            `json:"confidentiality"`
	Integrity                  Authentication            `json:"integrity"`
	Availability               Authentication            `json:"availability"`
	ExploitCodeMaturity        Authentication            `json:"exploit_code_maturity"`
	RemediationLevel           Authentication            `json:"remediation_level"`
	ReportConfidence           Authentication            `json:"report_confidence"`
	ConfidentialityRequirement Authentication            `json:"confidentiality_requirement"`
	IntegrityRequirement       Authentication            `json:"integrity_requirement"`
	AvailabilityRequirement    Authentication            `json:"availability_requirement"`
	BaseScore                  EnhancedFunctionTelemetry `json:"base_score"`
	TemporalScore              EnhancedFunctionTelemetry `json:"temporal_score"`
	EnvironmentalScore         EnhancedFunctionTelemetry `json:"environmental_score"`
	VectorString               TrainingID                `json:"vector_string"`
}

type CWEReference struct {
	Description string                 `json:"description"`
	Properties  CWEReferenceProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type CWEReferenceProperties struct {
	CweID               Authentication       `json:"cwe_id"`
	Name                TrainingID           `json:"name"`
	Description         AdaptiveBehaviorInfo `json:"description"`
	WeaknessCategory    TrainingID           `json:"weakness_category"`
	LikelihoodOfExploit TrainingID           `json:"likelihood_of_exploit"`
}

type CodeLevelRootCause struct {
	Description string                       `json:"description"`
	Properties  CodeLevelRootCauseProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type CodeLevelRootCauseProperties struct {
	AnalysisID           ExperienceID `json:"analysis_id"`
	VulnerableComponent  ExperienceID `json:"vulnerable_component"`
	AffectedFindings     AssetType    `json:"affected_findings"`
	CodeLocation         TrainingID   `json:"code_location"`
	VulnerabilityPattern TrainingID   `json:"vulnerability_pattern"`
	FixRecommendation    TrainingID   `json:"fix_recommendation"`
}

type CodeQualityReport struct {
	Description string                      `json:"description"`
	Properties  CodeQualityReportProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type CodeQualityReportProperties struct {
	Language     Language             `json:"language"`
	FilePath     AdaptiveBehaviorInfo `json:"file_path"`
	Metrics      AdaptiveBehaviorInfo `json:"metrics"`
	Issues       AdaptiveBehaviorInfo `json:"issues"`
	Suggestions  Authentication       `json:"suggestions"`
	OverallScore Authentication       `json:"overall_score"`
	Timestamp    Authentication       `json:"timestamp"`
}

type Language struct {
	Ref         string `json:"$ref"`
	Description string `json:"description"`
}

type ConfigUpdatePayload struct {
	Description string                        `json:"description"`
	Properties  ConfigUpdatePayloadProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type ConfigUpdatePayloadProperties struct {
	UpdateID    ExperienceID `json:"update_id"`
	ConfigItems AssetType    `json:"config_items"`
}

type CrossLanguageAnalysis struct {
	Description string                          `json:"description"`
	Properties  CrossLanguageAnalysisProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type CrossLanguageAnalysisProperties struct {
	AnalysisID          AdaptiveBehaviorInfo `json:"analysis_id"`
	ProjectName         AdaptiveBehaviorInfo `json:"project_name"`
	LanguagesAnalyzed   AdaptiveBehaviorInfo `json:"languages_analyzed"`
	CrossLanguageIssues AdaptiveBehaviorInfo `json:"cross_language_issues"`
	IntegrationPoints   AdaptiveBehaviorInfo `json:"integration_points"`
	SecurityBoundaries  Authentication       `json:"security_boundaries"`
	DataFlowRisks       AdaptiveBehaviorInfo `json:"data_flow_risks"`
	Recommendations     AdaptiveBehaviorInfo `json:"recommendations"`
	RiskScore           Authentication       `json:"risk_score"`
}

type DiscoveredAsset struct {
	Description string                    `json:"description"`
	Properties  DiscoveredAssetProperties `json:"properties"`
	Required    []string                  `json:"required"`
	Title       string                    `json:"title"`
	Type        Type                      `json:"type"`
}

type DiscoveredAssetProperties struct {
	AssetID         ExperienceID  `json:"asset_id"`
	AssetType       PropertyNames `json:"asset_type"`
	Value           ExperienceID  `json:"value"`
	DiscoveryMethod ExperienceID  `json:"discovery_method"`
	Confidence      PropertyNames `json:"confidence"`
	Metadata        AssetType     `json:"metadata"`
	DiscoveredAt    Timestamp     `json:"discovered_at"`
}

type EASMAsset struct {
	Description string              `json:"description"`
	Properties  EASMAssetProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type EASMAssetProperties struct {
	AssetID            Authentication `json:"asset_id"`
	AssetType          Authentication `json:"asset_type"`
	Value              Environment    `json:"value"`
	DiscoveryMethod    Authentication `json:"discovery_method"`
	DiscoverySource    Authentication `json:"discovery_source"`
	FirstDiscovered    Authentication `json:"first_discovered"`
	LastSeen           Authentication `json:"last_seen"`
	Status             Authentication `json:"status"`
	Confidence         Authentication `json:"confidence"`
	Technologies       Authentication `json:"technologies"`
	Services           Authentication `json:"services"`
	Certificates       Environment    `json:"certificates"`
	RiskScore          Authentication `json:"risk_score"`
	VulnerabilityCount Environment    `json:"vulnerability_count"`
	ExposureLevel      Authentication `json:"exposure_level"`
	BusinessUnit       TrainingID     `json:"business_unit"`
	Owner              TrainingID     `json:"owner"`
	Criticality        Authentication `json:"criticality"`
	ComplianceStatus   Environment    `json:"compliance_status"`
	PolicyViolations   Environment    `json:"policy_violations"`
	Metadata           Authentication `json:"metadata"`
}

type EASMDiscoveryPayload struct {
	Description string                         `json:"description"`
	Properties  EASMDiscoveryPayloadProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type EASMDiscoveryPayloadProperties struct {
	DiscoveryID   ExperienceID         `json:"discovery_id"`
	ScanID        ExperienceID         `json:"scan_id"`
	DiscoveryType ExperienceID         `json:"discovery_type"`
	Targets       AdaptiveBehaviorInfo `json:"targets"`
	Scope         PropertyNames        `json:"scope"`
	MaxDepth      StoppingReason       `json:"max_depth"`
	PassiveOnly   AdaptiveBehaviorInfo `json:"passive_only"`
}

type EASMDiscoveryResult struct {
	Description string                        `json:"description"`
	Properties  EASMDiscoveryResultProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type EASMDiscoveryResultProperties struct {
	DiscoveryID      ExperienceID   `json:"discovery_id"`
	ScanID           ExperienceID   `json:"scan_id"`
	Status           ExperienceID   `json:"status"`
	DiscoveredAssets Authentication `json:"discovered_assets"`
	Statistics       Authentication `json:"statistics"`
	Timestamp        Authentication `json:"timestamp"`
}

type EarlyStoppingInfo struct {
	Description string                      `json:"description"`
	Properties  EarlyStoppingInfoProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type EarlyStoppingInfoProperties struct {
	Reason         PropertyNames `json:"reason"`
	Timestamp      Timestamp     `json:"timestamp"`
	TotalTests     ExperienceID  `json:"total_tests"`
	CompletedTests ExperienceID  `json:"completed_tests"`
	RemainingTests ExperienceID  `json:"remaining_tests"`
	Details        AssetType     `json:"details"`
}

type EnhancedAttackPath struct {
	Description string                       `json:"description"`
	Properties  EnhancedAttackPathProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type EnhancedAttackPathProperties struct {
	PathID               Environment          `json:"path_id"`
	TargetAsset          Environment          `json:"target_asset"`
	Nodes                AssetType            `json:"nodes"`
	Edges                Environment          `json:"edges"`
	PathFeasibility      Environment          `json:"path_feasibility"`
	EstimatedTime        AssetType            `json:"estimated_time"`
	SkillLevelRequired   Environment          `json:"skill_level_required"`
	SuccessProbability   Environment          `json:"success_probability"`
	DetectionProbability Environment          `json:"detection_probability"`
	OverallRisk          Environment          `json:"overall_risk"`
	BlockingControls     AdaptiveBehaviorInfo `json:"blocking_controls"`
	DetectionControls    AssetType            `json:"detection_controls"`
	Metadata             AssetType            `json:"metadata"`
}

type EnhancedAttackPathNode struct {
	Description string                           `json:"description"`
	Properties  EnhancedAttackPathNodeProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type EnhancedAttackPathNodeProperties struct {
	NodeID         AssetType  `json:"node_id"`
	NodeType       AssetType  `json:"node_type"`
	Name           AssetType  `json:"name"`
	Description    TrainingID `json:"description"`
	Exploitability AssetType  `json:"exploitability"`
	Impact         AssetType  `json:"impact"`
	Difficulty     AssetType  `json:"difficulty"`
	MitreTechnique TrainingID `json:"mitre_technique"`
	MitreTactic    TrainingID `json:"mitre_tactic"`
	Prerequisites  AssetType  `json:"prerequisites"`
	Consequences   AssetType  `json:"consequences"`
	Metadata       AssetType  `json:"metadata"`
}

type FindingPayload struct {
	Description string                           `json:"description"`
	Properties  EnhancedFindingPayloadProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type EnhancedFindingPayloadProperties struct {
	FindingID      ExperienceID  `json:"finding_id"`
	TaskID         ExperienceID  `json:"task_id"`
	ScanID         ExperienceID  `json:"scan_id"`
	Status         ExperienceID  `json:"status"`
	Vulnerability  PropertyNames `json:"vulnerability"`
	Target         PropertyNames `json:"target"`
	Strategy       TrainingID    `json:"strategy"`
	Evidence       APISchema     `json:"evidence"`
	Impact         APISchema     `json:"impact"`
	Recommendation APISchema     `json:"recommendation"`
	SarifResult    *APISchema    `json:"sarif_result,omitempty"`
	Metadata       Metadata      `json:"metadata"`
	CreatedAt      Timestamp     `json:"created_at"`
	UpdatedAt      Timestamp     `json:"updated_at"`
}

type Metadata struct {
	AdditionalProperties bool   `json:"additionalProperties"`
	Title                string `json:"title"`
	Type                 Type   `json:"type"`
}

type EnhancedFunctionTaskTarget struct {
	Description string                               `json:"description"`
	Properties  EnhancedFunctionTaskTargetProperties `json:"properties"`
	Required    []string                             `json:"required"`
	Title       string                               `json:"title"`
	Type        Type                                 `json:"type"`
}

type EnhancedFunctionTaskTargetProperties struct {
	URL          PurpleURL                 `json:"url"`
	Method       EnhancedFunctionTelemetry `json:"method"`
	Headers      EnhancedFunctionTelemetry `json:"headers"`
	Cookies      EnhancedFunctionTelemetry `json:"cookies"`
	Parameters   EnhancedFunctionTelemetry `json:"parameters"`
	Body         EnhancedFunctionTelemetry `json:"body"`
	AuthRequired EnhancedFunctionTelemetry `json:"auth_required"`
}

type PurpleURL struct {
	Description string `json:"description"`
	Format      string `json:"format"`
	MaxLength   int64  `json:"maxLength"`
	MinLength   int64  `json:"minLength"`
	Title       string `json:"title"`
	Type        Type   `json:"type"`
}

type EnhancedIOCRecord struct {
	Description string                      `json:"description"`
	Properties  EnhancedIOCRecordProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type EnhancedIOCRecordProperties struct {
	IocID           AdaptiveBehaviorInfo `json:"ioc_id"`
	IocType         AdaptiveBehaviorInfo `json:"ioc_type"`
	Value           AdaptiveBehaviorInfo `json:"value"`
	ThreatType      AdaptiveBehaviorInfo `json:"threat_type"`
	MalwareFamily   AdaptiveBehaviorInfo `json:"malware_family"`
	Campaign        TrainingID           `json:"campaign"`
	Severity        Language             `json:"severity"`
	Confidence      AssetType            `json:"confidence"`
	ReputationScore Environment          `json:"reputation_score"`
	FirstSeen       AdaptiveBehaviorInfo `json:"first_seen"`
	LastSeen        AdaptiveBehaviorInfo `json:"last_seen"`
	ExpiresAt       AdaptiveBehaviorInfo `json:"expires_at"`
	Tags            Environment          `json:"tags"`
	MitreTechniques AdaptiveBehaviorInfo `json:"mitre_techniques"`
	Metadata        AdaptiveBehaviorInfo `json:"metadata"`
}

type EnhancedModuleStatus struct {
	Description string                         `json:"description"`
	Properties  EnhancedModuleStatusProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type EnhancedModuleStatusProperties struct {
	ModuleName        Language    `json:"module_name"`
	Version           Environment `json:"version"`
	Status            Environment `json:"status"`
	HealthScore       Environment `json:"health_score"`
	CPUUsage          Environment `json:"cpu_usage"`
	MemoryUsage       Environment `json:"memory_usage"`
	ActiveConnections Environment `json:"active_connections"`
	TasksProcessed    Environment `json:"tasks_processed"`
	TasksPending      Environment `json:"tasks_pending"`
	ErrorCount        Environment `json:"error_count"`
	StartedAt         Environment `json:"started_at"`
	LastHeartbeat     Environment `json:"last_heartbeat"`
	UptimeSeconds     Environment `json:"uptime_seconds"`
	Metadata          Environment `json:"metadata"`
}

type EnhancedRiskAssessment struct {
	Description string                           `json:"description"`
	Properties  EnhancedRiskAssessmentProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type EnhancedRiskAssessmentProperties struct {
	AssessmentID         Authentication            `json:"assessment_id"`
	TargetID             Authentication            `json:"target_id"`
	OverallRiskScore     Authentication            `json:"overall_risk_score"`
	LikelihoodScore      AssetType                 `json:"likelihood_score"`
	ImpactScore          AssetType                 `json:"impact_score"`
	RiskLevel            Language                  `json:"risk_level"`
	RiskCategory         Authentication            `json:"risk_category"`
	RiskFactors          Authentication            `json:"risk_factors"`
	CvssMetrics          CvssMetrics               `json:"cvss_metrics"`
	BusinessImpact       TrainingID                `json:"business_impact"`
	AffectedAssets       AssetType                 `json:"affected_assets"`
	MitigationStrategies AssetType                 `json:"mitigation_strategies"`
	ResidualRisk         Authentication            `json:"residual_risk"`
	AssessedAt           Authentication            `json:"assessed_at"`
	ValidUntil           EnhancedFunctionTelemetry `json:"valid_until"`
	Metadata             AssetType                 `json:"metadata"`
}

type CvssMetrics struct {
	AnyOf       []AssetTypeItems `json:"anyOf"`
	Default     interface{}      `json:"default"`
	Description string           `json:"description"`
}

type EnhancedScanRequest struct {
	Description string                        `json:"description"`
	Properties  EnhancedScanRequestProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type EnhancedScanRequestProperties struct {
	ScanID      Authentication `json:"scan_id"`
	Targets     PurpleTargets  `json:"targets"`
	Scope       Language       `json:"scope"`
	Strategy    Authentication `json:"strategy"`
	Priority    Authentication `json:"priority"`
	MaxDuration Authentication `json:"max_duration"`
	Metadata    Authentication `json:"metadata"`
}

type PurpleTargets struct {
	Description string       `json:"description"`
	Items       FormURLItems `json:"items"`
	MinItems    int64        `json:"minItems"`
	Title       string       `json:"title"`
	Type        Type         `json:"type"`
}

type FormURLItems struct {
	Format    *string `json:"format,omitempty"`
	MaxLength *int64  `json:"maxLength,omitempty"`
	MinLength *int64  `json:"minLength,omitempty"`
	Type      Type    `json:"type"`
}

type EnhancedTaskExecution struct {
	Description string                          `json:"description"`
	Properties  EnhancedTaskExecutionProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type EnhancedTaskExecutionProperties struct {
	TaskID       AdaptiveBehaviorInfo `json:"task_id"`
	TaskType     AdaptiveBehaviorInfo `json:"task_type"`
	ModuleName   Language             `json:"module_name"`
	Priority     AssetType            `json:"priority"`
	Timeout      AssetType            `json:"timeout"`
	RetryCount   AssetType            `json:"retry_count"`
	Dependencies AssetType            `json:"dependencies"`
	Status       Language             `json:"status"`
	Progress     AssetType            `json:"progress"`
	ResultData   AssetType            `json:"result_data"`
	ErrorMessage TrainingID           `json:"error_message"`
	CPUUsage     TrainingID           `json:"cpu_usage"`
	MemoryUsage  TrainingID           `json:"memory_usage"`
	CreatedAt    Timestamp            `json:"created_at"`
	StartedAt    AdaptiveBehaviorInfo `json:"started_at"`
	CompletedAt  AdaptiveBehaviorInfo `json:"completed_at"`
	Metadata     AssetType            `json:"metadata"`
}

type EnhancedVulnerability struct {
	Description string                          `json:"description"`
	Properties  EnhancedVulnerabilityProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type EnhancedVulnerabilityProperties struct {
	VulnerabilityID          Authentication       `json:"vulnerability_id"`
	Title                    Authentication       `json:"title"`
	Description              Authentication       `json:"description"`
	VulnerabilityType        Authentication       `json:"vulnerability_type"`
	Severity                 Authentication       `json:"severity"`
	URL                      Authentication       `json:"url"`
	Parameter                AdaptiveBehaviorInfo `json:"parameter"`
	Location                 AdaptiveBehaviorInfo `json:"location"`
	CvssMetrics              CvssMetrics          `json:"cvss_metrics"`
	AIConfidence             Authentication       `json:"ai_confidence"`
	AIRiskAssessment         Authentication       `json:"ai_risk_assessment"`
	ExploitabilityScore      Authentication       `json:"exploitability_score"`
	AttackVector             Authentication       `json:"attack_vector"`
	AttackComplexity         Authentication       `json:"attack_complexity"`
	Prerequisites            Authentication       `json:"prerequisites"`
	BusinessImpact           Authentication       `json:"business_impact"`
	TechnicalImpact          Authentication       `json:"technical_impact"`
	RemediationEffort        Authentication       `json:"remediation_effort"`
	RemediationPriority      Authentication       `json:"remediation_priority"`
	FixRecommendations       Authentication       `json:"fix_recommendations"`
	PocAvailable             Authentication       `json:"poc_available"`
	Verified                 Authentication       `json:"verified"`
	FalsePositiveProbability Authentication       `json:"false_positive_probability"`
	DiscoveredAt             Authentication       `json:"discovered_at"`
	LastVerifiedAt           AdaptiveBehaviorInfo `json:"last_verified_at"`
	Tags                     Authentication       `json:"tags"`
	References               Authentication       `json:"references"`
	Metadata                 AdaptiveBehaviorInfo `json:"metadata"`
}

type EnhancedVulnerabilityCorrelation struct {
	Description string                                     `json:"description"`
	Properties  EnhancedVulnerabilityCorrelationProperties `json:"properties"`
	Required    []string                                   `json:"required"`
	Title       string                                     `json:"title"`
	Type        Type                                       `json:"type"`
}

type EnhancedVulnerabilityCorrelationProperties struct {
	CorrelationID          AdaptiveBehaviorInfo `json:"correlation_id"`
	PrimaryVulnerability   Environment          `json:"primary_vulnerability"`
	RelatedVulnerabilities Environment          `json:"related_vulnerabilities"`
	CorrelationStrength    Environment          `json:"correlation_strength"`
	CorrelationType        Environment          `json:"correlation_type"`
	CombinedRiskScore      Environment          `json:"combined_risk_score"`
	ExploitationComplexity Environment          `json:"exploitation_complexity"`
	AttackScenarios        AdaptiveBehaviorInfo `json:"attack_scenarios"`
	RecommendedOrder       Environment          `json:"recommended_order"`
	CoordinatedMitigation  Environment          `json:"coordinated_mitigation"`
	PriorityRanking        Environment          `json:"priority_ranking"`
	AnalyzedAt             Timestamp            `json:"analyzed_at"`
	Metadata               Environment          `json:"metadata"`
}

type ErrorRecord struct {
	Description string                `json:"description"`
	Properties  ErrorRecordProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type ErrorRecordProperties struct {
	Category  PropertyNames        `json:"category"`
	Message   ExperienceID         `json:"message"`
	Timestamp Timestamp            `json:"timestamp"`
	Details   AdaptiveBehaviorInfo `json:"details"`
}

type ExecutionError struct {
	Description string                   `json:"description"`
	Properties  ExecutionErrorProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type ExecutionErrorProperties struct {
	ErrorID   ExperienceID         `json:"error_id"`
	ErrorType ExperienceID         `json:"error_type"`
	Message   ExperienceID         `json:"message"`
	Payload   AdaptiveBehaviorInfo `json:"payload"`
	Vector    AdaptiveBehaviorInfo `json:"vector"`
	Timestamp Authentication       `json:"timestamp"`
	Attempts  StoppingReason       `json:"attempts"`
}

type ExperienceSample struct {
	Description string                     `json:"description"`
	Properties  ExperienceSampleProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type ExperienceSampleProperties struct {
	SampleID        Authentication            `json:"sample_id"`
	SessionID       Authentication            `json:"session_id"`
	PlanID          AssetType                 `json:"plan_id"`
	StateBefore     Authentication            `json:"state_before"`
	ActionTaken     Authentication            `json:"action_taken"`
	StateAfter      Authentication            `json:"state_after"`
	Reward          Authentication            `json:"reward"`
	RewardBreakdown Authentication            `json:"reward_breakdown"`
	Context         Authentication            `json:"context"`
	TargetInfo      Authentication            `json:"target_info"`
	Timestamp       Authentication            `json:"timestamp"`
	DurationMS      Authentication            `json:"duration_ms"`
	QualityScore    AdaptiveBehaviorInfo      `json:"quality_score"`
	IsPositive      EnhancedFunctionTelemetry `json:"is_positive"`
	Confidence      Authentication            `json:"confidence"`
	LearningTags    EnhancedFunctionTelemetry `json:"learning_tags"`
	DifficultyLevel Authentication            `json:"difficulty_level"`
}

type ExploitPayload struct {
	Description string                   `json:"description"`
	Properties  ExploitPayloadProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type ExploitPayloadProperties struct {
	PayloadID          AssetType  `json:"payload_id"`
	PayloadType        AssetType  `json:"payload_type"`
	PayloadContent     AssetType  `json:"payload_content"`
	Encoding           AssetType  `json:"encoding"`
	Obfuscation        AssetType  `json:"obfuscation"`
	BypassTechnique    TrainingID `json:"bypass_technique"`
	TargetTechnology   AssetType  `json:"target_technology"`
	RequiredContext    AssetType  `json:"required_context"`
	EffectivenessScore AssetType  `json:"effectiveness_score"`
	DetectionEvasion   AssetType  `json:"detection_evasion"`
	SuccessRate        AssetType  `json:"success_rate"`
	UsageCount         AssetType  `json:"usage_count"`
	Metadata           AssetType  `json:"metadata"`
}

type ExploitResult struct {
	Description string                  `json:"description"`
	Properties  ExploitResultProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type ExploitResultProperties struct {
	ResultID            Environment `json:"result_id"`
	ExploitID           Environment `json:"exploit_id"`
	TargetID            Environment `json:"target_id"`
	Success             Environment `json:"success"`
	Severity            Language    `json:"severity"`
	ImpactLevel         Environment `json:"impact_level"`
	ExploitTechnique    Environment `json:"exploit_technique"`
	PayloadUsed         Environment `json:"payload_used"`
	ExecutionTime       Environment `json:"execution_time"`
	AccessGained        Environment `json:"access_gained"`
	DataExtracted       Environment `json:"data_extracted"`
	SystemImpact        Environment `json:"system_impact"`
	DetectionBypassed   Environment `json:"detection_bypassed"`
	ArtifactsLeft       Environment `json:"artifacts_left"`
	RemediationVerified Environment `json:"remediation_verified"`
	RetestRequired      Environment `json:"retest_required"`
	ExecutedAt          Environment `json:"executed_at"`
	Metadata            Environment `json:"metadata"`
}

type FeedbackEventPayload struct {
	Description string                         `json:"description"`
	Properties  FeedbackEventPayloadProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type FeedbackEventPayloadProperties struct {
	TaskID    ExperienceID         `json:"task_id"`
	ScanID    ExperienceID         `json:"scan_id"`
	EventType ExperienceID         `json:"event_type"`
	Details   AdaptiveBehaviorInfo `json:"details"`
	FormURL   FormURL              `json:"form_url"`
}

type FormURL struct {
	AnyOf   []FormURLItems `json:"anyOf"`
	Default interface{}    `json:"default"`
	Title   string         `json:"title"`
}

type Target struct {
	Description string                  `json:"description"`
	Properties  FindingTargetProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type FindingTargetProperties struct {
	URL       FluffyURL `json:"url"`
	Parameter Body      `json:"parameter"`
	Method    Body      `json:"method"`
	Headers   Headers   `json:"headers"`
	Params    Metadata  `json:"params"`
	Body      Body      `json:"body"`
}

type Body struct {
	AnyOf   []AdditionalPropertiesElement `json:"anyOf"`
	Default interface{}                   `json:"default"`
	Title   string                        `json:"title"`
}

type Headers struct {
	AdditionalProperties AdditionalPropertiesElement `json:"additionalProperties"`
	Title                string                      `json:"title"`
	Type                 Type                        `json:"type"`
}

type FluffyURL struct {
	Title string `json:"title"`
}

type FunctionExecutionResult struct {
	Description string                            `json:"description"`
	Properties  FunctionExecutionResultProperties `json:"properties"`
	Required    []string                          `json:"required"`
	Title       string                            `json:"title"`
	Type        Type                              `json:"type"`
}

type FunctionExecutionResultProperties struct {
	Findings        Authentication `json:"findings"`
	Telemetry       Authentication `json:"telemetry"`
	Errors          Authentication `json:"errors"`
	DurationSeconds StoppingReason `json:"duration_seconds"`
	Timestamp       Authentication `json:"timestamp"`
}

type FunctionTaskPayload struct {
	Description string                        `json:"description"`
	Properties  FunctionTaskPayloadProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type FunctionTaskPayloadProperties struct {
	TaskID         ExperienceID   `json:"task_id"`
	ScanID         ExperienceID   `json:"scan_id"`
	Priority       StoppingReason `json:"priority"`
	Target         PropertyNames  `json:"target"`
	Context        PropertyNames  `json:"context"`
	Strategy       ScanScope      `json:"strategy"`
	CustomPayloads Parameters     `json:"custom_payloads"`
	TestConfig     PropertyNames  `json:"test_config"`
}

type FunctionTaskTarget struct {
	Description string                       `json:"description"`
	Properties  FunctionTaskTargetProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type FunctionTaskTargetProperties struct {
	URL               FluffyURL            `json:"url"`
	Parameter         TrainingID           `json:"parameter"`
	Method            ScanScope            `json:"method"`
	ParameterLocation ScanScope            `json:"parameter_location"`
	Headers           AdaptiveBehaviorInfo `json:"headers"`
	Cookies           AdaptiveBehaviorInfo `json:"cookies"`
	FormData          AssetType            `json:"form_data"`
	JSONData          Payload              `json:"json_data"`
	Body              TrainingID           `json:"body"`
}

type HeartbeatPayload struct {
	Description string                     `json:"description"`
	Properties  HeartbeatPayloadProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type HeartbeatPayloadProperties struct {
	Module   PropertyNames `json:"module"`
	WorkerID ExperienceID  `json:"worker_id"`
	Capacity ExperienceID  `json:"capacity"`
}

type JavaScriptAnalysisResult struct {
	Description string                             `json:"description"`
	Properties  JavaScriptAnalysisResultProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type JavaScriptAnalysisResultProperties struct {
	AnalysisID         ExperienceID   `json:"analysis_id"`
	URL                ExperienceID   `json:"url"`
	SourceSizeBytes    ExperienceID   `json:"source_size_bytes"`
	DangerousFunctions Authentication `json:"dangerous_functions"`
	ExternalResources  Authentication `json:"external_resources"`
	DataLeaks          WorkflowSteps  `json:"data_leaks"`
	Findings           Authentication `json:"findings"`
	ApisCalled         Authentication `json:"apis_called"`
	AjaxEndpoints      Authentication `json:"ajax_endpoints"`
	SuspiciousPatterns Authentication `json:"suspicious_patterns"`
	RiskScore          Authentication `json:"risk_score"`
	SecurityScore      Authentication `json:"security_score"`
	Timestamp          Authentication `json:"timestamp"`
}

type LanguageDetectionResult struct {
	Description string                            `json:"description"`
	Properties  LanguageDetectionResultProperties `json:"properties"`
	Required    []string                          `json:"required"`
	Title       string                            `json:"title"`
	Type        Type                              `json:"type"`
}

type LanguageDetectionResultProperties struct {
	PrimaryLanguage    Language       `json:"primary_language"`
	Confidence         Authentication `json:"confidence"`
	SecondaryLanguages Environment    `json:"secondary_languages"`
	Frameworks         Authentication `json:"frameworks"`
	FileExtensions     Authentication `json:"file_extensions"`
	LinesOfCode        Environment    `json:"lines_of_code"`
}

type LanguageInteroperability struct {
	Description string                             `json:"description"`
	Properties  LanguageInteroperabilityProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type LanguageInteroperabilityProperties struct {
	SourceLanguage         Language       `json:"source_language"`
	TargetLanguage         Language       `json:"target_language"`
	InteropMethod          Authentication `json:"interop_method"`
	SecurityConsiderations Authentication `json:"security_considerations"`
	PerformanceImpact      Authentication `json:"performance_impact"`
	CompatibilityIssues    Authentication `json:"compatibility_issues"`
	Recommendations        Authentication `json:"recommendations"`
}

type LanguageSpecificPayload struct {
	Description string                            `json:"description"`
	Properties  LanguageSpecificPayloadProperties `json:"properties"`
	Required    []string                          `json:"required"`
	Title       string                            `json:"title"`
	Type        Type                              `json:"type"`
}

type LanguageSpecificPayloadProperties struct {
	Language          Language       `json:"language"`
	PayloadType       TrainingID     `json:"payload_type"`
	PayloadContent    TrainingID     `json:"payload_content"`
	Encoding          TrainingID     `json:"encoding"`
	ExpectedBehavior  TrainingID     `json:"expected_behavior"`
	BypassTechniques  Authentication `json:"bypass_techniques"`
	TargetFunctions   Authentication `json:"target_functions"`
	SuccessIndicators TrainingID     `json:"success_indicators"`
}

type LanguageSpecificScanConfig struct {
	Description string                               `json:"description"`
	Properties  LanguageSpecificScanConfigProperties `json:"properties"`
	Required    []string                             `json:"required"`
	Title       string                               `json:"title"`
	Type        Type                                 `json:"type"`
}

type LanguageSpecificScanConfigProperties struct {
	Language        Language             `json:"language"`
	ScanPatterns    StoppingReason       `json:"scan_patterns"`
	QualityMetrics  StoppingReason       `json:"quality_metrics"`
	ExcludePaths    AdaptiveBehaviorInfo `json:"exclude_paths"`
	IncludePatterns AdaptiveBehaviorInfo `json:"include_patterns"`
	CustomRules     Authentication       `json:"custom_rules"`
	MaxFileSize     StoppingReason       `json:"max_file_size"`
	TimeoutSeconds  StoppingReason       `json:"timeout_seconds"`
}

type LanguageSpecificVulnerability struct {
	Description string                                  `json:"description"`
	Properties  LanguageSpecificVulnerabilityProperties `json:"properties"`
	Required    []string                                `json:"required"`
	Title       string                                  `json:"title"`
	Type        Type                                    `json:"type"`
}

type LanguageSpecificVulnerabilityProperties struct {
	Language          Language             `json:"language"`
	VulnerabilityType Language             `json:"vulnerability_type"`
	Severity          Language             `json:"severity"`
	Description       AdaptiveBehaviorInfo `json:"description"`
	CodeSnippet       AdaptiveBehaviorInfo `json:"code_snippet"`
	LineNumber        AdaptiveBehaviorInfo `json:"line_number"`
	FilePath          AdaptiveBehaviorInfo `json:"file_path"`
	FunctionName      AdaptiveBehaviorInfo `json:"function_name"`
	Remediation       AdaptiveBehaviorInfo `json:"remediation"`
	CweID             AdaptiveBehaviorInfo `json:"cwe_id"`
	OwaspCategory     AdaptiveBehaviorInfo `json:"owasp_category"`
}

type MessageHeader struct {
	Description string                  `json:"description"`
	Properties  MessageHeaderProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type MessageHeaderProperties struct {
	MessageID     ExperienceID         `json:"message_id"`
	TraceID       ExperienceID         `json:"trace_id"`
	CorrelationID AdaptiveBehaviorInfo `json:"correlation_id"`
	SourceModule  PropertyNames        `json:"source_module"`
	Timestamp     Timestamp            `json:"timestamp"`
	Version       ScanScope            `json:"version"`
}

type ModelTrainingConfig struct {
	Description string                        `json:"description"`
	Properties  ModelTrainingConfigProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type ModelTrainingConfigProperties struct {
	ConfigID        ExperienceID         `json:"config_id"`
	ModelType       ExperienceID         `json:"model_type"`
	TrainingMode    ExperienceID         `json:"training_mode"`
	BatchSize       StoppingReason       `json:"batch_size"`
	LearningRate    StoppingReason       `json:"learning_rate"`
	Epochs          StoppingReason       `json:"epochs"`
	ValidationSplit StoppingReason       `json:"validation_split"`
	EarlyStopping   AdaptiveBehaviorInfo `json:"early_stopping"`
	Patience        StoppingReason       `json:"patience"`
	RewardFunction  AssetType            `json:"reward_function"`
	DiscountFactor  StoppingReason       `json:"discount_factor"`
	ExplorationRate StoppingReason       `json:"exploration_rate"`
	Hyperparameters AdaptiveBehaviorInfo `json:"hyperparameters"`
	Metadata        AdaptiveBehaviorInfo `json:"metadata"`
}

type ModelTrainingResult struct {
	Description string                        `json:"description"`
	Properties  ModelTrainingResultProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type ModelTrainingResultProperties struct {
	TrainingID              ExperienceID              `json:"training_id"`
	Config                  AssetType                 `json:"config"`
	ModelVersion            ExperienceID              `json:"model_version"`
	TrainingSamples         ExperienceID              `json:"training_samples"`
	ValidationSamples       ExperienceID              `json:"validation_samples"`
	TrainingLoss            ExperienceID              `json:"training_loss"`
	ValidationLoss          ExperienceID              `json:"validation_loss"`
	Accuracy                AdaptiveBehaviorInfo      `json:"accuracy"`
	Precision               EnhancedFunctionTelemetry `json:"precision"`
	Recall                  EnhancedFunctionTelemetry `json:"recall"`
	F1Score                 EnhancedFunctionTelemetry `json:"f1_score"`
	AverageReward           EnhancedFunctionTelemetry `json:"average_reward"`
	TrainingDurationSeconds StoppingReason            `json:"training_duration_seconds"`
	StartedAt               Timestamp                 `json:"started_at"`
	CompletedAt             Timestamp                 `json:"completed_at"`
	Metrics                 AssetType                 `json:"metrics"`
	ModelPath               EnhancedFunctionTelemetry `json:"model_path"`
	Metadata                AssetType                 `json:"metadata"`
}

type ModuleStatus struct {
	Description string                 `json:"description"`
	Properties  ModuleStatusProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type ModuleStatusProperties struct {
	Module         PropertyNames  `json:"module"`
	Status         ExperienceID   `json:"status"`
	WorkerID       ExperienceID   `json:"worker_id"`
	WorkerCount    StoppingReason `json:"worker_count"`
	QueueSize      StoppingReason `json:"queue_size"`
	TasksCompleted StoppingReason `json:"tasks_completed"`
	TasksFailed    StoppingReason `json:"tasks_failed"`
	LastHeartbeat  Environment    `json:"last_heartbeat"`
	Metrics        Environment    `json:"metrics"`
	UptimeSeconds  StoppingReason `json:"uptime_seconds"`
}

type MultiLanguageCodebase struct {
	Description string                          `json:"description"`
	Properties  MultiLanguageCodebaseProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type MultiLanguageCodebaseProperties struct {
	ProjectName               EnhancedFunctionTelemetry `json:"project_name"`
	Languages                 EnhancedFunctionTelemetry `json:"languages"`
	PrimaryLanguage           Language                  `json:"primary_language"`
	Frameworks                RemediationType           `json:"frameworks"`
	TotalFiles                EnhancedFunctionTelemetry `json:"total_files"`
	TotalLines                EnhancedFunctionTelemetry `json:"total_lines"`
	VulnerabilityDistribution EnhancedFunctionTelemetry `json:"vulnerability_distribution"`
	Dependencies              RemediationType           `json:"dependencies"`
}

type NotificationPayload struct {
	Description string                        `json:"description"`
	Properties  NotificationPayloadProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type NotificationPayloadProperties struct {
	NotificationID   ExperienceID              `json:"notification_id"`
	NotificationType ExperienceID              `json:"notification_type"`
	Priority         ExperienceID              `json:"priority"`
	Title            ExperienceID              `json:"title"`
	Message          ExperienceID              `json:"message"`
	Details          Environment               `json:"details"`
	Recipients       EnhancedFunctionTelemetry `json:"recipients"`
	Attachments      Environment               `json:"attachments"`
	Timestamp        Authentication            `json:"timestamp"`
}

type OastCallbackDetail struct {
	Description string                       `json:"description"`
	Properties  OastCallbackDetailProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type OastCallbackDetailProperties struct {
	CallbackType ExperienceID         `json:"callback_type"`
	Token        ExperienceID         `json:"token"`
	SourceIP     ExperienceID         `json:"source_ip"`
	Timestamp    Authentication       `json:"timestamp"`
	Protocol     Authentication       `json:"protocol"`
	RawData      AdaptiveBehaviorInfo `json:"raw_data"`
}

type OastEvent struct {
	Description string              `json:"description"`
	Properties  OastEventProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type OastEventProperties struct {
	EventID    ExperienceID         `json:"event_id"`
	ProbeToken ExperienceID         `json:"probe_token"`
	EventType  ExperienceID         `json:"event_type"`
	SourceIP   ExperienceID         `json:"source_ip"`
	Timestamp  Authentication       `json:"timestamp"`
	Protocol   AdaptiveBehaviorInfo `json:"protocol"`
	RawRequest AdaptiveBehaviorInfo `json:"raw_request"`
	RawData    AdaptiveBehaviorInfo `json:"raw_data"`
}

type OastProbe struct {
	Description string              `json:"description"`
	Properties  OastProbeProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type OastProbeProperties struct {
	ProbeID     ExperienceID         `json:"probe_id"`
	Token       ExperienceID         `json:"token"`
	CallbackURL ExperienceID         `json:"callback_url"`
	TaskID      ExperienceID         `json:"task_id"`
	ScanID      ExperienceID         `json:"scan_id"`
	CreatedAt   Timestamp            `json:"created_at"`
	ExpiresAt   AdaptiveBehaviorInfo `json:"expires_at"`
	Status      TrainingID           `json:"status"`
}

type PlanExecutionMetrics struct {
	Description string                         `json:"description"`
	Properties  PlanExecutionMetricsProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type PlanExecutionMetricsProperties struct {
	PlanID             ExperienceID `json:"plan_id"`
	SessionID          ExperienceID `json:"session_id"`
	ExpectedSteps      ExperienceID `json:"expected_steps"`
	ExecutedSteps      ExperienceID `json:"executed_steps"`
	CompletedSteps     ExperienceID `json:"completed_steps"`
	FailedSteps        ExperienceID `json:"failed_steps"`
	SkippedSteps       ExperienceID `json:"skipped_steps"`
	ExtraActions       ExperienceID `json:"extra_actions"`
	CompletionRate     ExperienceID `json:"completion_rate"`
	SuccessRate        ExperienceID `json:"success_rate"`
	SequenceAccuracy   ExperienceID `json:"sequence_accuracy"`
	GoalAchieved       ExperienceID `json:"goal_achieved"`
	RewardScore        ExperienceID `json:"reward_score"`
	TotalExecutionTime ExperienceID `json:"total_execution_time"`
	Timestamp          Timestamp    `json:"timestamp"`
}

type PlanExecutionResult struct {
	Description string                        `json:"description"`
	Properties  PlanExecutionResultProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type PlanExecutionResultProperties struct {
	ResultID        ExperienceID         `json:"result_id"`
	PlanID          ExperienceID         `json:"plan_id"`
	SessionID       ExperienceID         `json:"session_id"`
	Plan            PropertyNames        `json:"plan"`
	Trace           StoppingReason       `json:"trace"`
	Metrics         PropertyNames        `json:"metrics"`
	Findings        Environment          `json:"findings"`
	Anomalies       AdaptiveBehaviorInfo `json:"anomalies"`
	Recommendations AdaptiveBehaviorInfo `json:"recommendations"`
	Status          ExperienceID         `json:"status"`
	CompletedAt     Timestamp            `json:"completed_at"`
	Metadata        Environment          `json:"metadata"`
}

type PostExResultPayload struct {
	Description string                        `json:"description"`
	Properties  PostExResultPayloadProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type PostExResultPayloadProperties struct {
	TaskID                ExperienceID         `json:"task_id"`
	ScanID                ExperienceID         `json:"scan_id"`
	TestType              PropertyNames        `json:"test_type"`
	Findings              Environment          `json:"findings"`
	RiskLevel             PropertyNames        `json:"risk_level"`
	SafeMode              ExperienceID         `json:"safe_mode"`
	AuthorizationVerified AdaptiveBehaviorInfo `json:"authorization_verified"`
	Timestamp             Environment          `json:"timestamp"`
}

type PostExTestPayload struct {
	Description string                      `json:"description"`
	Properties  PostExTestPayloadProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type PostExTestPayloadProperties struct {
	TaskID             ExperienceID  `json:"task_id"`
	ScanID             ExperienceID  `json:"scan_id"`
	TestType           PropertyNames `json:"test_type"`
	Target             ExperienceID  `json:"target"`
	SafeMode           AssetType     `json:"safe_mode"`
	AuthorizationToken TrainingID    `json:"authorization_token"`
	Context            AssetType     `json:"context"`
}

type RAGKnowledgeUpdatePayload struct {
	Description string                              `json:"description"`
	Properties  RAGKnowledgeUpdatePayloadProperties `json:"properties"`
	Required    []string                            `json:"required"`
	Title       string                              `json:"title"`
	Type        Type                                `json:"type"`
}

type RAGKnowledgeUpdatePayloadProperties struct {
	KnowledgeType   ExperienceID              `json:"knowledge_type"`
	Content         ExperienceID              `json:"content"`
	SourceID        EnhancedFunctionTelemetry `json:"source_id"`
	Category        TrainingID                `json:"category"`
	Tags            EnhancedFunctionTelemetry `json:"tags"`
	RelatedCve      EnhancedFunctionTelemetry `json:"related_cve"`
	RelatedCwe      EnhancedFunctionTelemetry `json:"related_cwe"`
	MitreTechniques Authentication            `json:"mitre_techniques"`
	Confidence      Authentication            `json:"confidence"`
	Metadata        Authentication            `json:"metadata"`
}

type RAGQueryPayload struct {
	Description string                    `json:"description"`
	Properties  RAGQueryPayloadProperties `json:"properties"`
	Required    []string                  `json:"required"`
	Title       string                    `json:"title"`
	Type        Type                      `json:"type"`
}

type RAGQueryPayloadProperties struct {
	QueryID        ExperienceID         `json:"query_id"`
	QueryText      ExperienceID         `json:"query_text"`
	TopK           Authentication       `json:"top_k"`
	MinSimilarity  Authentication       `json:"min_similarity"`
	KnowledgeTypes Parameters           `json:"knowledge_types"`
	Categories     Parameters           `json:"categories"`
	Metadata       AdaptiveBehaviorInfo `json:"metadata"`
}

type RAGResponsePayload struct {
	Description string                       `json:"description"`
	Properties  RAGResponsePayloadProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type RAGResponsePayloadProperties struct {
	QueryID         ExperienceID              `json:"query_id"`
	Results         Environment               `json:"results"`
	TotalResults    ExperienceID              `json:"total_results"`
	AvgSimilarity   EnhancedFunctionTelemetry `json:"avg_similarity"`
	EnhancedContext TrainingID                `json:"enhanced_context"`
	Metadata        AdaptiveBehaviorInfo      `json:"metadata"`
	Timestamp       Timestamp                 `json:"timestamp"`
}

type RemediationGeneratePayload struct {
	Description string                               `json:"description"`
	Properties  RemediationGeneratePayloadProperties `json:"properties"`
	Required    []string                             `json:"required"`
	Title       string                               `json:"title"`
	Type        Type                                 `json:"type"`
}

type RemediationGeneratePayloadProperties struct {
	TaskID            ExperienceID              `json:"task_id"`
	ScanID            ExperienceID              `json:"scan_id"`
	FindingID         ExperienceID              `json:"finding_id"`
	VulnerabilityType PropertyNames             `json:"vulnerability_type"`
	RemediationType   PropertyNames             `json:"remediation_type"`
	Context           Environment               `json:"context"`
	AutoApply         EnhancedFunctionTelemetry `json:"auto_apply"`
}

type RemediationResultPayload struct {
	Description string                             `json:"description"`
	Properties  RemediationResultPayloadProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type RemediationResultPayloadProperties struct {
	TaskID            ExperienceID  `json:"task_id"`
	ScanID            ExperienceID  `json:"scan_id"`
	FindingID         ExperienceID  `json:"finding_id"`
	RemediationType   PropertyNames `json:"remediation_type"`
	Status            ExperienceID  `json:"status"`
	PatchContent      TrainingID    `json:"patch_content"`
	Instructions      Environment   `json:"instructions"`
	VerificationSteps Environment   `json:"verification_steps"`
	RiskAssessment    Environment   `json:"risk_assessment"`
	Timestamp         Environment   `json:"timestamp"`
}

type RiskAssessmentContext struct {
	Description string                          `json:"description"`
	Properties  RiskAssessmentContextProperties `json:"properties"`
	Required    []string                        `json:"required"`
	Title       string                          `json:"title"`
	Type        Type                            `json:"type"`
}

type RiskAssessmentContextProperties struct {
	Environment         PropertyNames `json:"environment"`
	BusinessCriticality PropertyNames `json:"business_criticality"`
	DataSensitivity     APISchema     `json:"data_sensitivity"`
	AssetExposure       APISchema     `json:"asset_exposure"`
	ComplianceTags      AssetType     `json:"compliance_tags"`
	AssetValue          TrainingID    `json:"asset_value"`
	UserBase            TrainingID    `json:"user_base"`
	SlaHours            TrainingID    `json:"sla_hours"`
}

type RiskAssessmentResult struct {
	Description string                         `json:"description"`
	Properties  RiskAssessmentResultProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type RiskAssessmentResultProperties struct {
	FindingID          ExperienceID  `json:"finding_id"`
	TechnicalRiskScore ExperienceID  `json:"technical_risk_score"`
	BusinessRiskScore  ExperienceID  `json:"business_risk_score"`
	RiskLevel          PropertyNames `json:"risk_level"`
	PriorityScore      ExperienceID  `json:"priority_score"`
	ContextMultiplier  ExperienceID  `json:"context_multiplier"`
	BusinessImpact     AssetType     `json:"business_impact"`
	Recommendations    AssetType     `json:"recommendations"`
	EstimatedEffort    TrainingID    `json:"estimated_effort"`
	Timestamp          Timestamp     `json:"timestamp"`
}

type RiskFactor struct {
	Description string               `json:"description"`
	Properties  RiskFactorProperties `json:"properties"`
	Required    []string             `json:"required"`
	Title       string               `json:"title"`
	Type        Type                 `json:"type"`
}

type RiskFactorProperties struct {
	FactorName  AssetType `json:"factor_name"`
	Weight      AssetType `json:"weight"`
	Value       AssetType `json:"value"`
	Description AssetType `json:"description"`
}

type RiskTrendAnalysis struct {
	Description string                      `json:"description"`
	Properties  RiskTrendAnalysisProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type RiskTrendAnalysisProperties struct {
	PeriodStart           Authentication            `json:"period_start"`
	PeriodEnd             Authentication            `json:"period_end"`
	TotalVulnerabilities  ExperienceID              `json:"total_vulnerabilities"`
	RiskDistribution      Authentication            `json:"risk_distribution"`
	AverageRiskScore      ExperienceID              `json:"average_risk_score"`
	Trend                 ExperienceID              `json:"trend"`
	ImprovementPercentage EnhancedFunctionTelemetry `json:"improvement_percentage"`
	TopRisks              Authentication            `json:"top_risks"`
}

type SARIFLocation struct {
	Description string                  `json:"description"`
	Properties  SARIFLocationProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type SARIFLocationProperties struct {
	URI         EnhancedFunctionTelemetry `json:"uri"`
	StartLine   EnhancedFunctionTelemetry `json:"start_line"`
	StartColumn EnhancedFunctionTelemetry `json:"start_column"`
	EndLine     EnhancedFunctionTelemetry `json:"end_line"`
	EndColumn   EnhancedFunctionTelemetry `json:"end_column"`
}

type SARIFReport struct {
	Description string                `json:"description"`
	Properties  SARIFReportProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type SARIFReportProperties struct {
	Version    ScanScope `json:"version"`
	Schema     ScanScope `json:"$schema"`
	Runs       ScanScope `json:"runs"`
	Properties ScanScope `json:"properties"`
}

type SARIFResult struct {
	Description string                `json:"description"`
	Properties  SARIFResultProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type SARIFResultProperties struct {
	RuleID              Authentication `json:"rule_id"`
	Message             Authentication `json:"message"`
	Level               Authentication `json:"level"`
	Locations           Authentication `json:"locations"`
	PartialFingerprints Authentication `json:"partial_fingerprints"`
	Properties          Authentication `json:"properties"`
}

type SARIFRule struct {
	Description string              `json:"description"`
	Properties  SARIFRuleProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type SARIFRuleProperties struct {
	ID               AdaptiveBehaviorInfo `json:"id"`
	Name             AdaptiveBehaviorInfo `json:"name"`
	ShortDescription AdaptiveBehaviorInfo `json:"short_description"`
	FullDescription  AdaptiveBehaviorInfo `json:"full_description"`
	HelpURI          AdaptiveBehaviorInfo `json:"help_uri"`
	DefaultLevel     AdaptiveBehaviorInfo `json:"default_level"`
	Properties       AdaptiveBehaviorInfo `json:"properties"`
}

type SARIFRun struct {
	Description string             `json:"description"`
	Properties  SARIFRunProperties `json:"properties"`
	Required    []string           `json:"required"`
	Title       string             `json:"title"`
	Type        Type               `json:"type"`
}

type SARIFRunProperties struct {
	Tool        Language       `json:"tool"`
	Results     StoppingReason `json:"results"`
	Invocations StoppingReason `json:"invocations"`
	Artifacts   StoppingReason `json:"artifacts"`
	Properties  StoppingReason `json:"properties"`
}

type SARIFTool struct {
	Description string              `json:"description"`
	Properties  SARIFToolProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type SARIFToolProperties struct {
	Name           StoppingReason `json:"name"`
	Version        StoppingReason `json:"version"`
	InformationURI StoppingReason `json:"information_uri"`
	Rules          StoppingReason `json:"rules"`
}

type SASTDASTCorrelation struct {
	Description string                        `json:"description"`
	Properties  SASTDASTCorrelationProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type SASTDASTCorrelationProperties struct {
	CorrelationID      ExperienceID `json:"correlation_id"`
	SastFindingID      ExperienceID `json:"sast_finding_id"`
	DastFindingID      ExperienceID `json:"dast_finding_id"`
	DataFlowPath       Environment  `json:"data_flow_path"`
	VerificationStatus ExperienceID `json:"verification_status"`
	ConfidenceScore    ExperienceID `json:"confidence_score"`
	Explanation        Environment  `json:"explanation"`
}

type SIEMEvent struct {
	Description string              `json:"description"`
	Properties  SIEMEventProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type SIEMEventProperties struct {
	EventID          Environment     `json:"event_id"`
	EventType        Environment     `json:"event_type"`
	SourceSystem     RemediationType `json:"source_system"`
	Timestamp        RemediationType `json:"timestamp"`
	ReceivedAt       Environment     `json:"received_at"`
	Severity         Language        `json:"severity"`
	Category         Environment     `json:"category"`
	Subcategory      RemediationType `json:"subcategory"`
	SourceIP         Environment     `json:"source_ip"`
	SourcePort       TrainingID      `json:"source_port"`
	DestinationIP    Environment     `json:"destination_ip"`
	DestinationPort  Environment     `json:"destination_port"`
	Username         RemediationType `json:"username"`
	AssetID          Environment     `json:"asset_id"`
	Hostname         Environment     `json:"hostname"`
	Description      Environment     `json:"description"`
	RawLog           Environment     `json:"raw_log"`
	CorrelationRules Environment     `json:"correlation_rules"`
	RelatedEvents    Environment     `json:"related_events"`
	Status           RemediationType `json:"status"`
	AssignedTo       Environment     `json:"assigned_to"`
	Metadata         Environment     `json:"metadata"`
}

type SIEMEventPayload struct {
	Description string                     `json:"description"`
	Properties  SIEMEventPayloadProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type SIEMEventPayloadProperties struct {
	EventID     ExperienceID    `json:"event_id"`
	EventType   ExperienceID    `json:"event_type"`
	Severity    ExperienceID    `json:"severity"`
	Source      ExperienceID    `json:"source"`
	Destination RemediationType `json:"destination"`
	Message     ExperienceID    `json:"message"`
	Details     AssetType       `json:"details"`
	Timestamp   Timestamp       `json:"timestamp"`
}

type ScanCompletedPayload struct {
	Description string                         `json:"description"`
	Properties  ScanCompletedPayloadProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type ScanCompletedPayloadProperties struct {
	ScanID       ExperienceID  `json:"scan_id"`
	Status       ExperienceID  `json:"status"`
	Summary      PropertyNames `json:"summary"`
	Assets       ScanScope     `json:"assets"`
	Fingerprints APISchema     `json:"fingerprints"`
	ErrorInfo    TrainingID    `json:"error_info"`
}

type ScanStartPayload struct {
	Description string                     `json:"description"`
	Properties  ScanStartPayloadProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type ScanStartPayloadProperties struct {
	ScanID         ExperienceID   `json:"scan_id"`
	Targets        FluffyTargets  `json:"targets"`
	Scope          PropertyNames  `json:"scope"`
	Authentication PropertyNames  `json:"authentication"`
	Strategy       TrainingID     `json:"strategy"`
	RateLimit      PropertyNames  `json:"rate_limit"`
	CustomHeaders  Authentication `json:"custom_headers"`
	XForwardedFor  Authentication `json:"x_forwarded_for"`
}

type FluffyTargets struct {
	Items FormURLItems `json:"items"`
	Title string       `json:"title"`
	Type  Type         `json:"type"`
}

type ScenarioTestResult struct {
	Description string                       `json:"description"`
	Properties  ScenarioTestResultProperties `json:"properties"`
	Required    []string                     `json:"required"`
	Title       string                       `json:"title"`
	Type        Type                         `json:"type"`
}

type ScenarioTestResultProperties struct {
	TestID          ExperienceID   `json:"test_id"`
	ScenarioID      ExperienceID   `json:"scenario_id"`
	ModelVersion    ExperienceID   `json:"model_version"`
	GeneratedPlan   Authentication `json:"generated_plan"`
	ExecutionResult Authentication `json:"execution_result"`
	Score           ExperienceID   `json:"score"`
	Comparison      Authentication `json:"comparison"`
	Passed          ExperienceID   `json:"passed"`
	TestedAt        Authentication `json:"tested_at"`
	Metadata        Authentication `json:"metadata"`
}

type SensitiveMatch struct {
	Description string                   `json:"description"`
	Properties  SensitiveMatchProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type SensitiveMatchProperties struct {
	MatchID     ExperienceID         `json:"match_id"`
	PatternName ExperienceID         `json:"pattern_name"`
	MatchedText ExperienceID         `json:"matched_text"`
	Context     ExperienceID         `json:"context"`
	Confidence  Environment          `json:"confidence"`
	LineNumber  AdaptiveBehaviorInfo `json:"line_number"`
	FilePath    Environment          `json:"file_path"`
	URL         AdaptiveBehaviorInfo `json:"url"`
	Severity    Severity             `json:"severity"`
}

type Severity struct {
	Ref     string `json:"$ref"`
	Default string `json:"default"`
}

type SessionState struct {
	Description string                 `json:"description"`
	Properties  SessionStateProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type SessionStateProperties struct {
	SessionID        ExperienceID              `json:"session_id"`
	PlanID           ExperienceID              `json:"plan_id"`
	ScanID           ExperienceID              `json:"scan_id"`
	Status           ExperienceID              `json:"status"`
	CurrentStepIndex StoppingReason            `json:"current_step_index"`
	CompletedSteps   AdaptiveBehaviorInfo      `json:"completed_steps"`
	PendingSteps     AssetType                 `json:"pending_steps"`
	Context          StoppingReason            `json:"context"`
	Variables        AssetType                 `json:"variables"`
	StartedAt        RemediationType           `json:"started_at"`
	UpdatedAt        RemediationType           `json:"updated_at"`
	TimeoutAt        EnhancedFunctionTelemetry `json:"timeout_at"`
	Metadata         StoppingReason            `json:"metadata"`
}

type StandardScenario struct {
	Description string                     `json:"description"`
	Properties  StandardScenarioProperties `json:"properties"`
	Required    []string                   `json:"required"`
	Title       string                     `json:"title"`
	Type        Type                       `json:"type"`
}

type StandardScenarioProperties struct {
	ScenarioID        ExperienceID              `json:"scenario_id"`
	Name              ExperienceID              `json:"name"`
	Description       ExperienceID              `json:"description"`
	VulnerabilityType PropertyNames             `json:"vulnerability_type"`
	DifficultyLevel   ExperienceID              `json:"difficulty_level"`
	TargetConfig      StoppingReason            `json:"target_config"`
	ExpectedPlan      StoppingReason            `json:"expected_plan"`
	SuccessCriteria   StoppingReason            `json:"success_criteria"`
	Tags              EnhancedFunctionTelemetry `json:"tags"`
	CreatedAt         RemediationType           `json:"created_at"`
	Metadata          StoppingReason            `json:"metadata"`
}

type SystemOrchestration struct {
	Description string                        `json:"description"`
	Properties  SystemOrchestrationProperties `json:"properties"`
	Required    []string                      `json:"required"`
	Title       string                        `json:"title"`
	Type        Type                          `json:"type"`
}

type SystemOrchestrationProperties struct {
	OrchestrationID    ScanScope `json:"orchestration_id"`
	OrchestrationName  ScanScope `json:"orchestration_name"`
	ModuleStatuses     ScanScope `json:"module_statuses"`
	ScanConfiguration  ScanScope `json:"scan_configuration"`
	ResourceAllocation ScanScope `json:"resource_allocation"`
	OverallStatus      ScanScope `json:"overall_status"`
	ActiveScans        ScanScope `json:"active_scans"`
	QueuedTasks        ScanScope `json:"queued_tasks"`
	SystemCPU          ScanScope `json:"system_cpu"`
	SystemMemory       ScanScope `json:"system_memory"`
	NetworkThroughput  ScanScope `json:"network_throughput"`
	CreatedAt          Timestamp `json:"created_at"`
	UpdatedAt          Timestamp `json:"updated_at"`
	Metadata           ScanScope `json:"metadata"`
}

type TaskDependency struct {
	Description string                   `json:"description"`
	Properties  TaskDependencyProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type TaskDependencyProperties struct {
	DependencyType  RemediationType `json:"dependency_type"`
	DependentTaskID RemediationType `json:"dependent_task_id"`
	Condition       ScanScope       `json:"condition"`
	Required        RemediationType `json:"required"`
}

type TaskQueue struct {
	Description string              `json:"description"`
	Properties  TaskQueueProperties `json:"properties"`
	Required    []string            `json:"required"`
	Title       string              `json:"title"`
	Type        Type                `json:"type"`
}

type TaskQueueProperties struct {
	QueueID              ScanScope `json:"queue_id"`
	QueueName            ScanScope `json:"queue_name"`
	MaxConcurrentTasks   ScanScope `json:"max_concurrent_tasks"`
	TaskTimeout          ScanScope `json:"task_timeout"`
	PendingTasks         ScanScope `json:"pending_tasks"`
	RunningTasks         ScanScope `json:"running_tasks"`
	CompletedTasks       ScanScope `json:"completed_tasks"`
	TotalProcessed       ScanScope `json:"total_processed"`
	SuccessRate          ScanScope `json:"success_rate"`
	AverageExecutionTime ScanScope `json:"average_execution_time"`
	CreatedAt            Timestamp `json:"created_at"`
	LastActivity         Timestamp `json:"last_activity"`
	Metadata             ScanScope `json:"metadata"`
}

type TaskUpdatePayload struct {
	Description string                      `json:"description"`
	Properties  TaskUpdatePayloadProperties `json:"properties"`
	Required    []string                    `json:"required"`
	Title       string                      `json:"title"`
	Type        Type                        `json:"type"`
}

type TaskUpdatePayloadProperties struct {
	TaskID   ExperienceID `json:"task_id"`
	ScanID   ExperienceID `json:"scan_id"`
	Status   ExperienceID `json:"status"`
	WorkerID ExperienceID `json:"worker_id"`
	Details  Payload      `json:"details"`
}

type TechnicalFingerprint struct {
	Description string                         `json:"description"`
	Properties  TechnicalFingerprintProperties `json:"properties"`
	Required    []string                       `json:"required"`
	Title       string                         `json:"title"`
	Type        Type                           `json:"type"`
}

type TechnicalFingerprintProperties struct {
	Technology           Environment `json:"technology"`
	Version              Environment `json:"version"`
	Confidence           Environment `json:"confidence"`
	DetectionMethod      Environment `json:"detection_method"`
	Evidence             Environment `json:"evidence"`
	Category             TrainingID  `json:"category"`
	Subcategory          Environment `json:"subcategory"`
	KnownVulnerabilities Environment `json:"known_vulnerabilities"`
	EOLStatus            Environment `json:"eol_status"`
	Metadata             Environment `json:"metadata"`
}

type TestExecution struct {
	Description string                  `json:"description"`
	Properties  TestExecutionProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type TestExecutionProperties struct {
	ExecutionID        TrainingID     `json:"execution_id"`
	TestCaseID         Authentication `json:"test_case_id"`
	TargetURL          Authentication `json:"target_url"`
	Timeout            Authentication `json:"timeout"`
	RetryAttempts      Authentication `json:"retry_attempts"`
	Status             Language       `json:"status"`
	StartTime          Authentication `json:"start_time"`
	EndTime            TrainingID     `json:"end_time"`
	Duration           TrainingID     `json:"duration"`
	Success            Authentication `json:"success"`
	VulnerabilityFound Authentication `json:"vulnerability_found"`
	ConfidenceLevel    Language       `json:"confidence_level"`
	RequestData        TrainingID     `json:"request_data"`
	ResponseData       TrainingID     `json:"response_data"`
	Evidence           TrainingID     `json:"evidence"`
	ErrorMessage       TrainingID     `json:"error_message"`
	CPUUsage           Environment    `json:"cpu_usage"`
	MemoryUsage        TrainingID     `json:"memory_usage"`
	NetworkTraffic     TrainingID     `json:"network_traffic"`
	Metadata           TrainingID     `json:"metadata"`
}

type TestStrategy struct {
	Description string                 `json:"description"`
	Properties  TestStrategyProperties `json:"properties"`
	Required    []string               `json:"required"`
	Title       string                 `json:"title"`
	Type        Type                   `json:"type"`
}

type TestStrategyProperties struct {
	StrategyID          EnhancedFunctionTelemetry `json:"strategy_id"`
	StrategyName        EnhancedFunctionTelemetry `json:"strategy_name"`
	TargetType          EnhancedFunctionTelemetry `json:"target_type"`
	TestCategories      EnhancedFunctionTelemetry `json:"test_categories"`
	TestSequence        EnhancedFunctionTelemetry `json:"test_sequence"`
	ParallelExecution   EnhancedFunctionTelemetry `json:"parallel_execution"`
	TriggerConditions   EnhancedFunctionTelemetry `json:"trigger_conditions"`
	StopConditions      EnhancedFunctionTelemetry `json:"stop_conditions"`
	PriorityWeights     EnhancedFunctionTelemetry `json:"priority_weights"`
	ResourceLimits      EnhancedFunctionTelemetry `json:"resource_limits"`
	LearningEnabled     EnhancedFunctionTelemetry `json:"learning_enabled"`
	AdaptationThreshold EnhancedFunctionTelemetry `json:"adaptation_threshold"`
	EffectivenessScore  EnhancedFunctionTelemetry `json:"effectiveness_score"`
	UsageCount          EnhancedFunctionTelemetry `json:"usage_count"`
	SuccessRate         EnhancedFunctionTelemetry `json:"success_rate"`
	CreatedAt           Timestamp                 `json:"created_at"`
	Metadata            EnhancedFunctionTelemetry `json:"metadata"`
}

type ThreatIntelLookupPayload struct {
	Description string                             `json:"description"`
	Properties  ThreatIntelLookupPayloadProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type ThreatIntelLookupPayloadProperties struct {
	TaskID        ExperienceID         `json:"task_id"`
	ScanID        ExperienceID         `json:"scan_id"`
	Indicator     ExperienceID         `json:"indicator"`
	IndicatorType PropertyNames        `json:"indicator_type"`
	Sources       Sources              `json:"sources"`
	Enrich        AdaptiveBehaviorInfo `json:"enrich"`
}

type Sources struct {
	AnyOf   []SourcesAnyOf `json:"anyOf"`
	Default interface{}    `json:"default"`
	Title   string         `json:"title"`
}

type SourcesAnyOf struct {
	Items *PropertyNames `json:"items,omitempty"`
	Type  Type           `json:"type"`
}

type ThreatIntelResultPayload struct {
	Description string                             `json:"description"`
	Properties  ThreatIntelResultPayloadProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type ThreatIntelResultPayloadProperties struct {
	TaskID          ExperienceID  `json:"task_id"`
	ScanID          ExperienceID  `json:"scan_id"`
	Indicator       ExperienceID  `json:"indicator"`
	IndicatorType   PropertyNames `json:"indicator_type"`
	ThreatLevel     PropertyNames `json:"threat_level"`
	Sources         AssetType     `json:"sources"`
	MitreTechniques AssetType     `json:"mitre_techniques"`
	EnrichmentData  TrainingID    `json:"enrichment_data"`
	Timestamp       Timestamp     `json:"timestamp"`
}

type TraceRecord struct {
	Description string                `json:"description"`
	Properties  TraceRecordProperties `json:"properties"`
	Required    []string              `json:"required"`
	Title       string                `json:"title"`
	Type        Type                  `json:"type"`
}

type TraceRecordProperties struct {
	TraceID              ExperienceID   `json:"trace_id"`
	PlanID               ExperienceID   `json:"plan_id"`
	StepID               ExperienceID   `json:"step_id"`
	SessionID            ExperienceID   `json:"session_id"`
	ToolName             ExperienceID   `json:"tool_name"`
	InputData            StoppingReason `json:"input_data"`
	OutputData           StoppingReason `json:"output_data"`
	Status               ExperienceID   `json:"status"`
	ErrorMessage         StoppingReason `json:"error_message"`
	ExecutionTimeSeconds StoppingReason `json:"execution_time_seconds"`
	Timestamp            Timestamp      `json:"timestamp"`
	EnvironmentResponse  StoppingReason `json:"environment_response"`
	Metadata             StoppingReason `json:"metadata"`
}

type Vulnerability struct {
	Description string                  `json:"description"`
	Properties  VulnerabilityProperties `json:"properties"`
	Required    []string                `json:"required"`
	Title       string                  `json:"title"`
	Type        Type                    `json:"type"`
}

type VulnerabilityProperties struct {
	Name          PropertyNames   `json:"name"`
	Cwe           CapecID         `json:"cwe"`
	Cve           CapecID         `json:"cve"`
	Severity      PropertyNames   `json:"severity"`
	Confidence    PropertyNames   `json:"confidence"`
	Description   Environment     `json:"description"`
	CvssScore     StoppingReason  `json:"cvss_score"`
	CvssVector    Environment     `json:"cvss_vector"`
	OwaspCategory RemediationType `json:"owasp_category"`
}

type VulnerabilityCorrelation struct {
	Description string                             `json:"description"`
	Properties  VulnerabilityCorrelationProperties `json:"properties"`
	Required    []string                           `json:"required"`
	Title       string                             `json:"title"`
	Type        Type                               `json:"type"`
}

type VulnerabilityCorrelationProperties struct {
	CorrelationID    ExperienceID         `json:"correlation_id"`
	CorrelationType  ExperienceID         `json:"correlation_type"`
	RelatedFindings  AdaptiveBehaviorInfo `json:"related_findings"`
	ConfidenceScore  ExperienceID         `json:"confidence_score"`
	RootCause        AdaptiveBehaviorInfo `json:"root_cause"`
	CommonComponents ScanScope            `json:"common_components"`
	Explanation      ScanScope            `json:"explanation"`
	Timestamp        Timestamp            `json:"timestamp"`
}

type VulnerabilityDiscovery struct {
	Description string                           `json:"description"`
	Properties  VulnerabilityDiscoveryProperties `json:"properties"`
	Required    []string                         `json:"required"`
	Title       string                           `json:"title"`
	Type        Type                             `json:"type"`
}

type VulnerabilityDiscoveryProperties struct {
	DiscoveryID             Authentication       `json:"discovery_id"`
	VulnerabilityID         Authentication       `json:"vulnerability_id"`
	AssetID                 AdaptiveBehaviorInfo `json:"asset_id"`
	Title                   Authentication       `json:"title"`
	Description             Authentication       `json:"description"`
	Severity                Language             `json:"severity"`
	Confidence              Language             `json:"confidence"`
	VulnerabilityType       Authentication       `json:"vulnerability_type"`
	AffectedComponent       AdaptiveBehaviorInfo `json:"affected_component"`
	AttackVector            AdaptiveBehaviorInfo `json:"attack_vector"`
	DetectionMethod         Authentication       `json:"detection_method"`
	ScannerName             Authentication       `json:"scanner_name"`
	ScanRuleID              Authentication       `json:"scan_rule_id"`
	Evidence                Authentication       `json:"evidence"`
	ProofOfConcept          Authentication       `json:"proof_of_concept"`
	FalsePositiveLikelihood Authentication       `json:"false_positive_likelihood"`
	ImpactAssessment        Authentication       `json:"impact_assessment"`
	Exploitability          Authentication       `json:"exploitability"`
	RemediationAdvice       Authentication       `json:"remediation_advice"`
	RemediationPriority     Authentication       `json:"remediation_priority"`
	CveIDS                  AdaptiveBehaviorInfo `json:"cve_ids"`
	CweIDS                  Authentication       `json:"cwe_ids"`
	CvssScore               AdaptiveBehaviorInfo `json:"cvss_score"`
	DiscoveredAt            Authentication       `json:"discovered_at"`
	Metadata                Authentication       `json:"metadata"`
}

type VulnerabilityLifecyclePayload struct {
	Description string                                  `json:"description"`
	Properties  VulnerabilityLifecyclePayloadProperties `json:"properties"`
	Required    []string                                `json:"required"`
	Title       string                                  `json:"title"`
	Type        Type                                    `json:"type"`
}

type VulnerabilityLifecyclePayloadProperties struct {
	VulnerabilityID   ExperienceID         `json:"vulnerability_id"`
	FindingID         ExperienceID         `json:"finding_id"`
	AssetID           ExperienceID         `json:"asset_id"`
	VulnerabilityType PropertyNames        `json:"vulnerability_type"`
	Severity          PropertyNames        `json:"severity"`
	Confidence        PropertyNames        `json:"confidence"`
	Status            PropertyNames        `json:"status"`
	Exploitability    APISchema            `json:"exploitability"`
	AssignedTo        StoppingReason       `json:"assigned_to"`
	DueDate           AdaptiveBehaviorInfo `json:"due_date"`
	FirstDetected     Environment          `json:"first_detected"`
	LastSeen          Environment          `json:"last_seen"`
	ResolutionDate    AdaptiveBehaviorInfo `json:"resolution_date"`
	Metadata          Environment          `json:"metadata"`
}

type VulnerabilityUpdatePayload struct {
	Description string                               `json:"description"`
	Properties  VulnerabilityUpdatePayloadProperties `json:"properties"`
	Required    []string                             `json:"required"`
	Title       string                               `json:"title"`
	Type        Type                                 `json:"type"`
}

type VulnerabilityUpdatePayloadProperties struct {
	VulnerabilityID ExperienceID   `json:"vulnerability_id"`
	Status          PropertyNames  `json:"status"`
	AssignedTo      StoppingReason `json:"assigned_to"`
	Comment         StoppingReason `json:"comment"`
	Metadata        StoppingReason `json:"metadata"`
	UpdatedBy       Environment    `json:"updated_by"`
	Timestamp       Environment    `json:"timestamp"`
}

type WebhookPayload struct {
	Description string                   `json:"description"`
	Properties  WebhookPayloadProperties `json:"properties"`
	Required    []string                 `json:"required"`
	Title       string                   `json:"title"`
	Type        Type                     `json:"type"`
}

type WebhookPayloadProperties struct {
	WebhookID    Environment    `json:"webhook_id"`
	EventType    Environment    `json:"event_type"`
	Source       Environment    `json:"source"`
	Timestamp    Environment    `json:"timestamp"`
	Data         Environment    `json:"data"`
	DeliveryURL  StoppingReason `json:"delivery_url"`
	RetryCount   Environment    `json:"retry_count"`
	MaxRetries   Environment    `json:"max_retries"`
	Status       Environment    `json:"status"`
	DeliveredAt  Environment    `json:"delivered_at"`
	ErrorMessage Environment    `json:"error_message"`
	Metadata     Environment    `json:"metadata"`
}

type Type string

const (
	Array   Type = "array"
	Boolean Type = "boolean"
	Integer Type = "integer"
	Null    Type = "null"
	Number  Type = "number"
	Object  Type = "object"
	String  Type = "string"
)

type Format string

const (
	DateTime Format = "date-time"
)

type ScanScopeDefault struct {
	AnythingArray []interface{}
	Integer       *int64
	String        *string
}

func (x *ScanScopeDefault) UnmarshalJSON(data []byte) error {
	x.AnythingArray = nil
	object, err := unmarshalUnion(data, &x.Integer, nil, nil, &x.String, true, &x.AnythingArray, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *ScanScopeDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(x.Integer, nil, nil, x.String, x.AnythingArray != nil, x.AnythingArray, false, nil, false, nil, false, nil, true)
}

type AdditionalPropertiesUnion struct {
	AdditionalPropertiesElement *AdditionalPropertiesElement
	Bool                        *bool
}

func (x *AdditionalPropertiesUnion) UnmarshalJSON(data []byte) error {
	x.AdditionalPropertiesElement = nil
	var c AdditionalPropertiesElement
	object, err := unmarshalUnion(data, nil, nil, &x.Bool, nil, false, nil, true, &c, false, nil, false, nil, false)
	if err != nil {
		return err
	}
	if object {
		x.AdditionalPropertiesElement = &c
	}
	return nil
}

func (x *AdditionalPropertiesUnion) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, x.Bool, nil, false, nil, x.AdditionalPropertiesElement != nil, x.AdditionalPropertiesElement, false, nil, false, nil, false)
}

type EnvironmentDefault struct {
	Bool    *bool
	Integer *int64
	String  *string
}

func (x *EnvironmentDefault) UnmarshalJSON(data []byte) error {
	object, err := unmarshalUnion(data, &x.Integer, nil, &x.Bool, &x.String, false, nil, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *EnvironmentDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(x.Integer, nil, x.Bool, x.String, false, nil, false, nil, false, nil, false, nil, true)
}

type AuthenticationDefault struct {
	Bool   *bool
	Double *float64
	String *string
}

func (x *AuthenticationDefault) UnmarshalJSON(data []byte) error {
	object, err := unmarshalUnion(data, nil, &x.Double, &x.Bool, &x.String, false, nil, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *AuthenticationDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, x.Double, x.Bool, x.String, false, nil, false, nil, false, nil, false, nil, true)
}

type AdaptiveBehaviorInfoDefault struct {
	Bool         *bool
	DefaultClass *DefaultClass
	String       *string
}

func (x *AdaptiveBehaviorInfoDefault) UnmarshalJSON(data []byte) error {
	x.DefaultClass = nil
	var c DefaultClass
	object, err := unmarshalUnion(data, nil, nil, &x.Bool, &x.String, false, nil, true, &c, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
		x.DefaultClass = &c
	}
	return nil
}

func (x *AdaptiveBehaviorInfoDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, x.Bool, x.String, false, nil, x.DefaultClass != nil, x.DefaultClass, false, nil, false, nil, true)
}

type EnhancedFunctionTelemetryDefault struct {
	Bool   *bool
	String *string
}

func (x *EnhancedFunctionTelemetryDefault) UnmarshalJSON(data []byte) error {
	object, err := unmarshalUnion(data, nil, nil, &x.Bool, &x.String, false, nil, false, nil, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
	}
	return nil
}

func (x *EnhancedFunctionTelemetryDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(nil, nil, x.Bool, x.String, false, nil, false, nil, false, nil, false, nil, true)
}

type AssetTypeDefault struct {
	Bool         *bool
	DefaultClass *DefaultClass
	Integer      *int64
	String       *string
}

func (x *AssetTypeDefault) UnmarshalJSON(data []byte) error {
	x.DefaultClass = nil
	var c DefaultClass
	object, err := unmarshalUnion(data, &x.Integer, nil, &x.Bool, &x.String, false, nil, true, &c, false, nil, false, nil, true)
	if err != nil {
		return err
	}
	if object {
		x.DefaultClass = &c
	}
	return nil
}

func (x *AssetTypeDefault) MarshalJSON() ([]byte, error) {
	return marshalUnion(x.Integer, nil, x.Bool, x.String, false, nil, x.DefaultClass != nil, x.DefaultClass, false, nil, false, nil, true)
}

func unmarshalUnion(data []byte, pi **int64, pf **float64, pb **bool, ps **string, haveArray bool, pa interface{}, haveObject bool, pc interface{}, haveMap bool, pm interface{}, haveEnum bool, pe interface{}, nullable bool) (bool, error) {
	if pi != nil {
			*pi = nil
	}
	if pf != nil {
			*pf = nil
	}
	if pb != nil {
			*pb = nil
	}
	if ps != nil {
			*ps = nil
	}

	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
			return false, err
	}

	switch v := tok.(type) {
	case json.Number:
			if pi != nil {
					i, err := v.Int64()
					if err == nil {
							*pi = &i
							return false, nil
					}
			}
			if pf != nil {
					f, err := v.Float64()
					if err == nil {
							*pf = &f
							return false, nil
					}
					return false, errors.New("Unparsable number")
			}
			return false, errors.New("Union does not contain number")
	case float64:
			return false, errors.New("Decoder should not return float64")
	case bool:
			if pb != nil {
					*pb = &v
					return false, nil
			}
			return false, errors.New("Union does not contain bool")
	case string:
			if haveEnum {
					return false, json.Unmarshal(data, pe)
			}
			if ps != nil {
					*ps = &v
					return false, nil
			}
			return false, errors.New("Union does not contain string")
	case nil:
			if nullable {
					return false, nil
			}
			return false, errors.New("Union does not contain null")
	case json.Delim:
			if v == '{' {
					if haveObject {
							return true, json.Unmarshal(data, pc)
					}
					if haveMap {
							return false, json.Unmarshal(data, pm)
					}
					return false, errors.New("Union does not contain object")
			}
			if v == '[' {
					if haveArray {
							return false, json.Unmarshal(data, pa)
					}
					return false, errors.New("Union does not contain array")
			}
			return false, errors.New("Cannot handle delimiter")
	}
	return false, errors.New("Cannot unmarshal union")
}

func marshalUnion(pi *int64, pf *float64, pb *bool, ps *string, haveArray bool, pa interface{}, haveObject bool, pc interface{}, haveMap bool, pm interface{}, haveEnum bool, pe interface{}, nullable bool) ([]byte, error) {
	if pi != nil {
			return json.Marshal(*pi)
	}
	if pf != nil {
			return json.Marshal(*pf)
	}
	if pb != nil {
			return json.Marshal(*pb)
	}
	if ps != nil {
			return json.Marshal(*ps)
	}
	if haveArray {
			return json.Marshal(pa)
	}
	if haveObject {
			return json.Marshal(pc)
	}
	if haveMap {
			return json.Marshal(pm)
	}
	if haveEnum {
			return json.Marshal(pe)
	}
	if nullable {
			return json.Marshal(nil)
	}
	return nil, errors.New("Union must not be null")
}
