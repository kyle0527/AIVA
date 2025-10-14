package schemas

import "time"

// ==================== 核心訊息結構 ====================

// MessageHeader 對應 Python aiva_common.schemas.MessageHeader
type MessageHeader struct {
	MessageID     string    `json:"message_id"`
	TraceID       string    `json:"trace_id"`
	CorrelationID *string   `json:"correlation_id,omitempty"`
	SourceModule  string    `json:"source_module"`
	Timestamp     time.Time `json:"timestamp"`
	Version       string    `json:"version"`
}

// AivaMessage 對應 Python aiva_common.schemas.AivaMessage
type AivaMessage struct {
	Header        MessageHeader          `json:"header"`
	Topic         string                 `json:"topic"`
	SchemaVersion string                 `json:"schema_version"`
	Payload       map[string]interface{} `json:"payload"`
}

// ==================== 功能任務相關 ====================

// FunctionTaskPayload 對應 Python aiva_common.schemas.FunctionTaskPayload
type FunctionTaskPayload struct {
	TaskID         string                  `json:"task_id"`
	ScanID         string                  `json:"scan_id"`
	Priority       int                     `json:"priority"`
	Target         FunctionTaskTarget      `json:"target"`
	Context        FunctionTaskContext     `json:"context"`
	Strategy       string                  `json:"strategy"`
	CustomPayloads []string                `json:"custom_payloads,omitempty"`
	TestConfig     FunctionTaskTestConfig  `json:"test_config"`
}

// FunctionTaskTarget 對應 Python aiva_common.schemas.FunctionTaskTarget
type FunctionTaskTarget struct {
	URL                string                 `json:"url"`
	Parameter          *string                `json:"parameter,omitempty"`
	Method             string                 `json:"method"`
	ParameterLocation  string                 `json:"parameter_location"`
	Headers            map[string]string      `json:"headers"`
	Cookies            map[string]string      `json:"cookies"`
	FormData           map[string]interface{} `json:"form_data"`
	JSONData           map[string]interface{} `json:"json_data,omitempty"`
	Body               *string                `json:"body,omitempty"`
}

// FunctionTaskContext 對應 Python aiva_common.schemas.FunctionTaskContext
type FunctionTaskContext struct {
	DBTypeHint      *string  `json:"db_type_hint,omitempty"`
	WAFDetected     bool     `json:"waf_detected"`
	RelatedFindings []string `json:"related_findings,omitempty"`
}

// FunctionTaskTestConfig 對應 Python aiva_common.schemas.FunctionTaskTestConfig
type FunctionTaskTestConfig struct {
	Payloads       []string `json:"payloads"`
	CustomPayloads []string `json:"custom_payloads"`
	BlindXSS       bool     `json:"blind_xss"`
	DOMTesting     bool     `json:"dom_testing"`
	Timeout        *float64 `json:"timeout,omitempty"`
}

// ==================== 漏洞發現相關 ====================

// FindingPayload 對應 Python aiva_common.schemas.FindingPayload
type FindingPayload struct {
	FindingID      string                  `json:"finding_id"`
	TaskID         string                  `json:"task_id"`
	ScanID         string                  `json:"scan_id"`
	Status         string                  `json:"status"`
	Vulnerability  Vulnerability           `json:"vulnerability"`
	Target         Target                  `json:"target"`
	Strategy       *string                 `json:"strategy,omitempty"`
	Evidence       *FindingEvidence        `json:"evidence,omitempty"`
	Impact         *FindingImpact          `json:"impact,omitempty"`
	Recommendation *FindingRecommendation  `json:"recommendation,omitempty"`
	Metadata       map[string]interface{}  `json:"metadata,omitempty"`
	CreatedAt      time.Time               `json:"created_at"`
	UpdatedAt      time.Time               `json:"updated_at"`
}

// Vulnerability 對應 Python aiva_common.schemas.Vulnerability
type Vulnerability struct {
	Name        string  `json:"name"`
	CWE         *string `json:"cwe,omitempty"`
	Severity    string  `json:"severity"`
	Confidence  string  `json:"confidence"`
	Description *string `json:"description,omitempty"`
}

// Target 對應 Python aiva_common.schemas.Target (FindingTarget 的別名)
type Target struct {
	URL       string                 `json:"url"`
	Parameter *string                `json:"parameter,omitempty"`
	Method    *string                `json:"method,omitempty"`
	Headers   map[string]string      `json:"headers,omitempty"`
	Params    map[string]interface{} `json:"params,omitempty"`
	Body      *string                `json:"body,omitempty"`
}

// FindingTarget 是 Target 的別名,保持向後相容
type FindingTarget = Target

// FindingEvidence 對應 Python aiva_common.schemas.FindingEvidence
type FindingEvidence struct {
	Payload           *string  `json:"payload,omitempty"`
	ResponseTimeDelta *float64 `json:"response_time_delta,omitempty"`
	DBVersion         *string  `json:"db_version,omitempty"`
	Request           *string  `json:"request,omitempty"`
	Response          *string  `json:"response,omitempty"`
	Proof             *string  `json:"proof,omitempty"`
}

// FindingImpact 對應 Python aiva_common.schemas.FindingImpact
type FindingImpact struct {
	Description     *string  `json:"description,omitempty"`
	BusinessImpact  *string  `json:"business_impact,omitempty"`
	TechnicalImpact *string  `json:"technical_impact,omitempty"`
	AffectedUsers   *int     `json:"affected_users,omitempty"`
	EstimatedCost   *float64 `json:"estimated_cost,omitempty"`
}

// FindingRecommendation 對應 Python aiva_common.schemas.FindingRecommendation
type FindingRecommendation struct {
	Fix              *string  `json:"fix,omitempty"`
	Priority         *string  `json:"priority,omitempty"`
	RemediationSteps []string `json:"remediation_steps,omitempty"`
	References       []string `json:"references,omitempty"`
}

// ==================== 通用輔助結構 ====================

// LocationInfo 位置資訊 (保留用於舊版相容)
type LocationInfo struct {
	URL       string  `json:"url,omitempty"`
	FilePath  string  `json:"file_path,omitempty"`
	Line      *int    `json:"line,omitempty"`
	Column    *int    `json:"column,omitempty"`
	Function  string  `json:"function,omitempty"`
	Method    string  `json:"method,omitempty"`
	Parameter string  `json:"parameter,omitempty"`
}
