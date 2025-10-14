package schemas

import "time"

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

// TaskPayload 通用任務 Payload
type TaskPayload struct {
	TaskID  string                 `json:"task_id"`
	ScanID  string                 `json:"scan_id"`
	Input   map[string]interface{} `json:"input"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// FindingPayload 對應 Python 的 Finding Schema
type FindingPayload struct {
	FindingID         string                 `json:"finding_id"`
	ScanID            string                 `json:"scan_id"`
	VulnerabilityType string                 `json:"vulnerability_type"`
	Severity          string                 `json:"severity"`
	Confidence        string                 `json:"confidence"`
	Title             string                 `json:"title"`
	Description       string                 `json:"description"`
	Location          LocationInfo           `json:"location"`
	Evidence          map[string]interface{} `json:"evidence,omitempty"`
	Recommendation    string                 `json:"recommendation,omitempty"`
	References        []string               `json:"references,omitempty"`
	Tags              []string               `json:"tags,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// LocationInfo 位置資訊
type LocationInfo struct {
	URL       string `json:"url,omitempty"`
	FilePath  string `json:"file_path,omitempty"`
	Line      *int   `json:"line,omitempty"`
	Column    *int   `json:"column,omitempty"`
	Function  string `json:"function,omitempty"`
	Method    string `json:"method,omitempty"`
	Parameter string `json:"parameter,omitempty"`
}
