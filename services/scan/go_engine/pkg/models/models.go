package models

import "time"

type Target struct {
	URL string `json:"url"`
}

type ScanTaskPayload struct {
	TaskID   string `json:"task_id"`
	ScanID   string `json:"scan_id"`
	Target   Target `json:"target"`
}

type FindingPayload struct {
	FindingID     string                `json:"finding_id"`
	TaskID        string                `json:"task_id"`
	ScanID        string                `json:"scan_id"`
	Type          string                `json:"type"`
	Severity      string                `json:"severity"`
	Title         string                `json:"title"`
	Description   string                `json:"description"`
	Status        string                `json:"status"`
	Target        *Target               `json:"target,omitempty"`
	Evidence      *FindingEvidence      `json:"evidence,omitempty"`
	Impact        *FindingImpact        `json:"impact,omitempty"`
	Recommendation *FindingRecommendation `json:"recommendation,omitempty"`
	Vulnerability *Vulnerability        `json:"vulnerability,omitempty"`
	Vulnerabilities []Vulnerability       `json:"vulnerabilities,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time             `json:"created_at"`
	UpdatedAt     time.Time             `json:"updated_at"`
}

type Vulnerability struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Package     string  `json:"package"`
	Version     string  `json:"version"`
	FixedVersion string  `json:"fixed_version"`
	Description *string `json:"description"`
	CWE         []string `json:"cwe"`
	Severity    string  `json:"severity"`
	Confidence  string  `json:"confidence"`
}

type FindingEvidence struct {
	Data     *string `json:"data"`
	Request  *string `json:"request,omitempty"`
	Response *string `json:"response,omitempty"`
	Payload  *string `json:"payload,omitempty"`
	Proof    *string `json:"proof,omitempty"`
}

type FindingImpact struct {
	Description     *string `json:"description"`
	BusinessImpact  *string `json:"business_impact,omitempty"`
	TechnicalImpact *string `json:"technical_impact,omitempty"`
}

type FindingRecommendation struct {
	Description      *string  `json:"description"`
	Fix              *string  `json:"fix,omitempty"`
	Priority         *string  `json:"priority,omitempty"`
	RemediationSteps []string `json:"remediation_steps,omitempty"`
	References       []string `json:"references,omitempty"`
}
