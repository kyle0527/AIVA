package models

// FunctionTaskPayload CSPM 任務載荷
type FunctionTaskPayload struct {
	TaskID       string      `json:"task_id"`
	FunctionType string      `json:"function_type"`
	Target       TaskTarget  `json:"target"`
	Options      TaskOptions `json:"options,omitempty"`
}

// TaskTarget 掃描目標
type TaskTarget struct {
	URL        string `json:"url,omitempty"`
	ConfigPath string `json:"config_path,omitempty"`
	Provider   string `json:"provider,omitempty"` // aws, azure, gcp, kubernetes
}

// TaskOptions 掃描選項
type TaskOptions struct {
	SeverityThreshold string   `json:"severity_threshold,omitempty"`
	PolicyFiles       []string `json:"policy_files,omitempty"`
	ExcludeRules      []string `json:"exclude_rules,omitempty"`
}

// FindingPayload CSPM 發現載荷
type FindingPayload struct {
	FindingID      string          `json:"finding_id"`
	TaskID         string          `json:"task_id"`
	ScanID         string          `json:"scan_id"`
	Status         string          `json:"status"`
	Vulnerability  Vulnerability   `json:"vulnerability"`
	Target         FindingTarget   `json:"target"`
	Evidence       FindingEvidence `json:"evidence"`
	Impact         FindingImpact   `json:"impact"`
	Recommendation string          `json:"recommendation"`
}

// Vulnerability 漏洞資訊
type Vulnerability struct {
	Name       string `json:"name"`
	CWE        string `json:"cwe"`
	Severity   string `json:"severity"`
	Confidence string `json:"confidence"`
}

// FindingTarget 發現目標
type FindingTarget struct {
	ResourceType string            `json:"resource_type"`
	ResourceID   string            `json:"resource_id"`
	Provider     string            `json:"provider"`
	Region       string            `json:"region,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// FindingEvidence 證據
type FindingEvidence struct {
	RuleID          string `json:"rule_id"`
	ConfigPath      string `json:"config_path"`
	MisconfigDetail string `json:"misconfig_detail"`
	ActualValue     string `json:"actual_value"`
	ExpectedValue   string `json:"expected_value"`
}

// FindingImpact 影響
type FindingImpact struct {
	Description    string `json:"description"`
	BusinessImpact string `json:"business_impact"`
	Exploitability string `json:"exploitability"`
}

// CSPMMisconfig Trivy CSPM 錯誤配置
type CSPMMisconfig struct {
	ID          string
	Title       string
	Description string
	Severity    string
	Resolution  string
	Provider    string
	Service     string
	ResourceID  string
	FilePath    string
	LineNumber  int
}
