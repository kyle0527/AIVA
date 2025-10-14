package schemas

import (
	"encoding/json"
	"testing"
	"time"
)

// 輔助函數
func stringPtr(s string) *string {
	return &s
}

func intPtr(i int) *int {
	return &i
}

func float64Ptr(f float64) *float64 {
	return &f
}

// TestMessageHeaderSerialization 測試 MessageHeader 的序列化
func TestMessageHeaderSerialization(t *testing.T) {
	now := time.Now()
	header := MessageHeader{
		MessageID:     "msg_123",
		TraceID:       "trace_456",
		CorrelationID: stringPtr("corr_789"),
		SourceModule:  "test_module",
		Timestamp:     now,
		Version:       "1.0",
	}

	data, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded MessageHeader
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if decoded.MessageID != header.MessageID {
		t.Errorf("MessageID mismatch: got %s, want %s", decoded.MessageID, header.MessageID)
	}
	if decoded.TraceID != header.TraceID {
		t.Errorf("TraceID mismatch: got %s, want %s", decoded.TraceID, header.TraceID)
	}
}

// TestFunctionTaskPayloadSerialization 測試 FunctionTaskPayload 的序列化
func TestFunctionTaskPayloadSerialization(t *testing.T) {
	customPayloads := []string{"payload1", "payload2"}

	task := FunctionTaskPayload{
		TaskID:   "task_123",
		ScanID:   "scan_456",
		Priority: 5,
		Target: FunctionTaskTarget{
			URL:               "https://example.com",
			Method:            "GET",
			ParameterLocation: "query",
			Headers:           map[string]string{"Authorization": "Bearer token"},
			Cookies:           map[string]string{},
			FormData:          map[string]interface{}{},
		},
		Context: FunctionTaskContext{
			WAFDetected: false,
		},
		Strategy:       "full",
		CustomPayloads: customPayloads,
		TestConfig: FunctionTaskTestConfig{
			Payloads:       []string{"basic"},
			CustomPayloads: []string{},
			BlindXSS:       false,
			DOMTesting:     false,
			Timeout:        float64Ptr(30.0),
		},
	}

	data, err := json.Marshal(task)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded FunctionTaskPayload
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if decoded.TaskID != task.TaskID {
		t.Errorf("TaskID mismatch: got %s, want %s", decoded.TaskID, task.TaskID)
	}
	if decoded.ScanID != task.ScanID {
		t.Errorf("ScanID mismatch: got %s, want %s", decoded.ScanID, task.ScanID)
	}
	if decoded.Priority != task.Priority {
		t.Errorf("Priority mismatch: got %d, want %d", decoded.Priority, task.Priority)
	}
}

// TestFindingPayloadSerialization 測試 FindingPayload 的序列化
func TestFindingPayloadSerialization(t *testing.T) {
	now := time.Now()
	cwe := "CWE-89"
	desc := "SQL Injection vulnerability found"

	finding := FindingPayload{
		FindingID: "finding_123",
		TaskID:    "task_456",
		ScanID:    "scan_789",
		Status:    "confirmed",
		Vulnerability: Vulnerability{
			Name:        "SQL Injection",
			CWE:         &cwe,
			Severity:    "high",
			Confidence:  "high",
			Description: &desc,
		},
		Target: Target{
			URL:     "https://example.com/api",
			Headers: map[string]string{},
			Params:  map[string]interface{}{},
		},
		Impact: &FindingImpact{
			AffectedUsers: intPtr(1000),
			EstimatedCost: float64Ptr(50000.0),
		},
		CreatedAt: now,
		UpdatedAt: now,
	}

	data, err := json.Marshal(finding)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded FindingPayload
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if decoded.FindingID != finding.FindingID {
		t.Errorf("FindingID mismatch: got %s, want %s", decoded.FindingID, finding.FindingID)
	}
	if decoded.Vulnerability.Name != finding.Vulnerability.Name {
		t.Errorf("Vulnerability.Name mismatch: got %s, want %s", decoded.Vulnerability.Name, finding.Vulnerability.Name)
	}
}

// TestOptionalFieldsOmitempty 測試 omitempty 的行為
func TestOptionalFieldsOmitempty(t *testing.T) {
	// 測試有可選欄位的結構
	target := Target{
		URL:     "https://example.com",
		Headers: map[string]string{},
		Params:  map[string]interface{}{},
	}

	data, err := json.Marshal(target)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// 驗證 nil 的指標欄位不會出現在 JSON 中
	var jsonMap map[string]interface{}
	err = json.Unmarshal(data, &jsonMap)
	if err != nil {
		t.Fatalf("Unmarshal to map failed: %v", err)
	}

	// parameter, method, body 應該不存在 (因為是 nil 指標 + omitempty)
	if _, exists := jsonMap["parameter"]; exists {
		t.Errorf("Expected 'parameter' to be omitted, but it exists")
	}
	if _, exists := jsonMap["method"]; exists {
		t.Errorf("Expected 'method' to be omitted, but it exists")
	}
	if _, exists := jsonMap["body"]; exists {
		t.Errorf("Expected 'body' to be omitted, but it exists")
	}
}

// TestOptionalFieldsWithValue 測試設定可選欄位的值
func TestOptionalFieldsWithValue(t *testing.T) {
	param := "id"
	method := "POST"
	body := `{"test": "data"}`

	target := Target{
		URL:       "https://example.com",
		Parameter: &param,
		Method:    &method,
		Body:      &body,
		Headers:   map[string]string{},
		Params:    map[string]interface{}{},
	}

	data, err := json.Marshal(target)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var decoded Target
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// 驗證值正確解析
	if decoded.Parameter == nil || *decoded.Parameter != param {
		t.Errorf("Parameter mismatch: got %v, want %s", decoded.Parameter, param)
	}
	if decoded.Method == nil || *decoded.Method != method {
		t.Errorf("Method mismatch: got %v, want %s", decoded.Method, method)
	}
	if decoded.Body == nil || *decoded.Body != body {
		t.Errorf("Body mismatch: got %v, want %s", decoded.Body, body)
	}
}

// TestPythonGoInteroperability 測試 Python-Go 互操作性
func TestPythonGoInteroperability(t *testing.T) {
	// 模擬 Python 生成的 JSON
	pythonJSON := `{
		"finding_id": "finding_abc123",
		"task_id": "task_xyz789",
		"scan_id": "scan_def456",
		"status": "confirmed",
		"vulnerability": {
			"name": "XSS",
			"cwe": "CWE-79",
			"severity": "medium",
			"confidence": "high",
			"description": "Cross-Site Scripting vulnerability"
		},
		"target": {
			"url": "https://test.com",
			"parameter": "q",
			"method": "GET"
		},
		"evidence": {
			"payload": "<script>alert(1)</script>",
			"request": "GET /?q=<script>alert(1)</script> HTTP/1.1",
			"response": "HTTP/1.1 200 OK\n\n<script>alert(1)</script>"
		},
		"created_at": "2025-10-14T10:00:00Z",
		"updated_at": "2025-10-14T10:00:00Z"
	}`

	var finding FindingPayload
	err := json.Unmarshal([]byte(pythonJSON), &finding)
	if err != nil {
		t.Fatalf("Failed to unmarshal Python JSON: %v", err)
	}

	// 驗證基本欄位
	if finding.FindingID != "finding_abc123" {
		t.Errorf("FindingID mismatch: got %s, want finding_abc123", finding.FindingID)
	}
	if finding.Vulnerability.Name != "XSS" {
		t.Errorf("Vulnerability.Name mismatch: got %s, want XSS", finding.Vulnerability.Name)
	}

	// 驗證可選欄位
	if finding.Target.Parameter == nil || *finding.Target.Parameter != "q" {
		t.Errorf("Target.Parameter mismatch: got %v, want q", finding.Target.Parameter)
	}
	if finding.Evidence == nil {
		t.Error("Expected Evidence to be populated")
	} else if finding.Evidence.Payload == nil || *finding.Evidence.Payload != "<script>alert(1)</script>" {
		t.Errorf("Evidence.Payload mismatch: got %v", finding.Evidence.Payload)
	}

	// 反序列化回 JSON 並驗證格式
	data, err := json.Marshal(finding)
	if err != nil {
		t.Fatalf("Failed to marshal back to JSON: %v", err)
	}

	var jsonMap map[string]interface{}
	err = json.Unmarshal(data, &jsonMap)
	if err != nil {
		t.Fatalf("Failed to unmarshal to map: %v", err)
	}

	// 驗證必填欄位存在
	requiredFields := []string{"finding_id", "task_id", "scan_id", "status", "vulnerability", "target", "created_at", "updated_at"}
	for _, field := range requiredFields {
		if _, exists := jsonMap[field]; !exists {
			t.Errorf("Required field %s is missing", field)
		}
	}
}

// TestZeroValueVsNilPointer 測試零值與 nil 指標的區別
func TestZeroValueVsNilPointer(t *testing.T) {
	// 測試 1: nil 指標 + omitempty = 欄位省略
	evidence1 := FindingEvidence{
		Request:  nil, // nil 指標
		Response: nil, // nil 指標
	}

	data1, _ := json.Marshal(evidence1)
	var map1 map[string]interface{}
	json.Unmarshal(data1, &map1)

	if _, exists := map1["request"]; exists {
		t.Error("Expected nil pointer field 'request' to be omitted")
	}

	// 測試 2: 指向空字串的指標 + omitempty = 欄位保留
	emptyStr := ""
	evidence2 := FindingEvidence{
		Request:  &emptyStr, // 指向空字串
		Response: &emptyStr, // 指向空字串
	}

	data2, _ := json.Marshal(evidence2)
	var map2 map[string]interface{}
	json.Unmarshal(data2, &map2)

	if _, exists := map2["request"]; !exists {
		t.Error("Expected non-nil pointer field 'request' to be included")
	}
	if val, _ := map2["request"].(string); val != "" {
		t.Errorf("Expected empty string, got %s", val)
	}
}
