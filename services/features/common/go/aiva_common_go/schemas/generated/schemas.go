// AIVA Go Schema - 自動生成
// 版本: 1.0.0
// 基於 core_schema_sot.yaml 作為單一事實來源
//
// ⚠️  此檔案自動生成，請勿手動修改
// 📅 最後更新: 2025-10-27T08:15:28.157056

package schemas

import (
	"time"
)

// ==================== 枚舉類型 ====================

type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

type Confidence string

const (
	ConfidenceConfirmed Confidence = "confirmed"
	ConfidenceFirm      Confidence = "firm"
	ConfidenceTentative Confidence = "tentative"
)

type FindingStatus string

const (
	FindingStatusNew           FindingStatus = "new"
	FindingStatusConfirmed     FindingStatus = "confirmed"
	FindingStatusFalsePositive FindingStatus = "false_positive"
	FindingStatusFixed         FindingStatus = "fixed"
	FindingStatusIgnored       FindingStatus = "ignored"
)

// ==================== 核心結構定義 ====================

type MessageHeader struct {
	MessageID     string    `json:"message_id"`
	TraceID       string    `json:"trace_id"`
	CorrelationID *string   `json:"correlation_id,omitempty"`
	SourceModule  string    `json:"source_module"`
	Timestamp     time.Time `json:"timestamp"`
	Version       string    `json:"version"`
}

type Target struct {
	URL       string                 `json:"url"`
	Parameter *string                `json:"parameter,omitempty"`
	Method    string                 `json:"method"`
	Headers   map[string]string      `json:"headers"`
	Params    map[string]interface{} `json:"params"`
	Body      *string                `json:"body,omitempty"`
}

type Vulnerability struct {
	Name        string     `json:"name"`
	CWE         *string    `json:"cwe,omitempty"`
	Severity    Severity   `json:"severity"`
	Confidence  Confidence `json:"confidence"`
	Description *string    `json:"description,omitempty"`
}

type FindingEvidence struct {
	Payload           *string  `json:"payload,omitempty"`
	ResponseTimeDelta *float64 `json:"response_time_delta,omitempty"`
	DBVersion         *string  `json:"db_version,omitempty"`
	Request           *string  `json:"request,omitempty"`
	Response          *string  `json:"response,omitempty"`
	Proof             *string  `json:"proof,omitempty"`
}

type FindingImpact struct {
	Description     *string  `json:"description,omitempty"`
	BusinessImpact  *string  `json:"business_impact,omitempty"`
	TechnicalImpact *string  `json:"technical_impact,omitempty"`
	AffectedUsers   *int     `json:"affected_users,omitempty"`
	EstimatedCost   *float64 `json:"estimated_cost,omitempty"`
}

type FindingRecommendation struct {
	Fix              *string  `json:"fix,omitempty"`
	Priority         *string  `json:"priority,omitempty"`
	RemediationSteps []string `json:"remediation_steps"`
	References       []string `json:"references"`
}

// ==================== 主要 Payload 結構 ====================

type FindingPayload struct {
	FindingID      string                  `json:"finding_id"`
	TaskID         string                  `json:"task_id"`
	ScanID         string                  `json:"scan_id"`
	Status         FindingStatus           `json:"status"`
	Vulnerability  Vulnerability           `json:"vulnerability"`
	Target         Target                  `json:"target"`
	Strategy       *string                 `json:"strategy,omitempty"`
	Evidence       *FindingEvidence        `json:"evidence,omitempty"`
	Impact         *FindingImpact          `json:"impact,omitempty"`
	Recommendation *FindingRecommendation  `json:"recommendation,omitempty"`
	Metadata       map[string]interface{} `json:"metadata"`
	CreatedAt      time.Time               `json:"created_at"`
	UpdatedAt      time.Time               `json:"updated_at"`
}

// NewFindingPayload 創建新的 FindingPayload 實例
func NewFindingPayload(
	findingID, taskID, scanID string,
	status FindingStatus,
	vulnerability Vulnerability,
	target Target,
) *FindingPayload {
	now := time.Now()
	return &FindingPayload{
		FindingID:      findingID,
		TaskID:         taskID,
		ScanID:         scanID,
		Status:         status,
		Vulnerability:  vulnerability,
		Target:         target,
		Metadata:       make(map[string]interface{}),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
}

// Touch 更新時間戳
func (fp *FindingPayload) Touch() {
	fp.UpdatedAt = time.Now()
}
