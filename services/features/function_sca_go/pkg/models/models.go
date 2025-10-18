package models

// FunctionTaskPayload 功能任務載荷
type FunctionTaskPayload struct {
	TaskID       string                 `json:"task_id"`
	FunctionType string                 `json:"function_type"`
	Target       FunctionTaskTarget     `json:"target"`
	Context      FunctionTaskContext    `json:"context,omitempty"`
	TestConfig   FunctionTaskTestConfig `json:"test_config,omitempty"`
}

// FunctionTaskTarget 任務目標
type FunctionTaskTarget struct {
	URL        string            `json:"url"`
	Method     string            `json:"method,omitempty"`
	Headers    map[string]string `json:"headers,omitempty"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

// FunctionTaskContext 任務上下文
type FunctionTaskContext struct {
	TraceID        string                 `json:"trace_id,omitempty"`
	Assets         []interface{}          `json:"assets,omitempty"`
	Authentication map[string]interface{} `json:"authentication,omitempty"`
}

// FunctionTaskTestConfig 測試配置
type FunctionTaskTestConfig struct {
	Timeout        int      `json:"timeout,omitempty"`
	MaxConcurrency int      `json:"max_concurrency,omitempty"`
	FollowRedirect bool     `json:"follow_redirect,omitempty"`
	Payloads       []string `json:"payloads,omitempty"`
}

// FindingPayload 發現載荷
type FindingPayload struct {
	FindingID      string                `json:"finding_id"`
	TaskID         string                `json:"task_id"`
	Vulnerability  Vulnerability         `json:"vulnerability"`
	Severity       string                `json:"severity"`
	Confidence     string                `json:"confidence"`
	Target         FindingTarget         `json:"target"`
	Evidence       FindingEvidence       `json:"evidence"`
	Impact         FindingImpact         `json:"impact"`
	Recommendation FindingRecommendation `json:"recommendation"`
	Tags           []string              `json:"tags"`
}

// Vulnerability 漏洞資訊
type Vulnerability struct {
	Type        string   `json:"type"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	CVEID       string   `json:"cve_id,omitempty"`
	GHSAID      string   `json:"ghsa_id,omitempty"`
	CWEIDs      []string `json:"cwe_ids,omitempty"`
}

// FindingTarget 發現目標
type FindingTarget struct {
	URL       string `json:"url"`
	Method    string `json:"method,omitempty"`
	Parameter string `json:"parameter,omitempty"`
}

// FindingEvidence 證據
type FindingEvidence struct {
	Request        string `json:"request"`
	Response       string `json:"response"`
	Payload        string `json:"payload,omitempty"`
	ProofOfConcept string `json:"proof_of_concept,omitempty"`
}

// FindingImpact 影響
type FindingImpact struct {
	Confidentiality string `json:"confidentiality"`
	Integrity       string `json:"integrity"`
	Availability    string `json:"availability"`
	BusinessImpact  string `json:"business_impact"`
}

// FindingRecommendation 修復建議
type FindingRecommendation struct {
	Remediation string   `json:"remediation"`
	References  []string `json:"references"`
}
