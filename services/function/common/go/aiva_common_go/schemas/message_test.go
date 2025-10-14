package schemas

import (
	"encoding/json"
	"testing"
	"time"
)

func TestMessageHeaderSerialization(t *testing.T) {
	header := MessageHeader{
		MessageID:     "msg_123",
		TraceID:       "trace_456",
		CorrelationID: stringPtr("corr_789"),
		SourceModule:  "test_module",
		Timestamp:     time.Now(),
		Version:       "1.0",
	}

	// 序列化
	data, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// 反序列化
	var decoded MessageHeader
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	// 驗證
	if decoded.MessageID != header.MessageID {
		t.Errorf("MessageID = %v, want %v", decoded.MessageID, header.MessageID)
	}
	if decoded.TraceID != header.TraceID {
		t.Errorf("TraceID = %v, want %v", decoded.TraceID, header.TraceID)
	}
}

func TestFindingPayloadSerialization(t *testing.T) {
	finding := FindingPayload{
		FindingID:         "finding_001",
		ScanID:            "scan_123",
		VulnerabilityType: "sql_injection",
		Severity:          "HIGH",
		Confidence:        "HIGH",
		Title:             "SQL Injection Vulnerability",
		Description:       "Potential SQL injection found",
		Location: LocationInfo{
			URL:      "https://example.com/api/users",
			FilePath: "api/users.py",
			Line:     intPtr(42),
			Function: "get_user_by_id",
		},
	}

	// 序列化
	data, err := json.Marshal(finding)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// 反序列化
	var decoded FindingPayload
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	// 驗證
	if decoded.FindingID != finding.FindingID {
		t.Errorf("FindingID = %v, want %v", decoded.FindingID, finding.FindingID)
	}
	if decoded.Severity != finding.Severity {
		t.Errorf("Severity = %v, want %v", decoded.Severity, finding.Severity)
	}
	if *decoded.Location.Line != *finding.Location.Line {
		t.Errorf("Location.Line = %v, want %v", *decoded.Location.Line, *finding.Location.Line)
	}
}

// 輔助函數
func stringPtr(s string) *string {
	return &s
}

func intPtr(i int) *int {
	return &i
}
