package common

import "encoding/json"

type SARIFReport struct {
    Schema  string     `json:"$schema"`
    Version string     `json:"version"`
    Runs    []SARIFRun `json:"runs"`
}

type SARIFRun struct {
    Tool    SARIFTool     `json:"tool"`
    Results []SARIFResult `json:"results"`
}

type SARIFTool struct {
    Driver SARIFDriver `json:"driver"`
}

type SARIFDriver struct {
    Name    string `json:"name"`
    Version string `json:"version"`
}

type SARIFResult struct {
    RuleID    string          `json:"ruleId"`
    Level     string          `json:"level"`
    Message   SARIFMessage    `json:"message"`
    Locations []SARIFLocation `json:"locations"`
}

type SARIFMessage struct {
    Text string `json:"text"`
}

type SARIFLocation struct {
    PhysicalLocation SARIFPhysicalLocation `json:"physicalLocation"`
}

type SARIFPhysicalLocation struct {
    ArtifactLocation SARIFArtifactLocation `json:"artifactLocation"`
}

type SARIFArtifactLocation struct {
    URI string `json:"uri"`
}

type CVSSv3Metrics struct {
    BaseScore           float32 `json:"base_score"`
    AttackVector        string  `json:"attack_vector"`
    AttackComplexity    string  `json:"attack_complexity"`
    PrivilegesRequired  string  `json:"privileges_required"`
    UserInteraction     string  `json:"user_interaction"`
    Scope               string  `json:"scope"`
    ConfidentialityImpact string `json:"confidentiality_impact"`
    IntegrityImpact     string  `json:"integrity_impact"`
    AvailabilityImpact  string  `json:"availability_impact"`
}

type ScanTask struct {
    TaskID    string                 `json:"task_id"`
    ScanID    string                 `json:"scan_id"`
    SessionID string                 `json:"session_id"`
    Target    ScanTarget             `json:"target"`
    Config    map[string]interface{} `json:"config"`
}

type ScanTarget struct {
    URL  string   `json:"url"`
    URLs []string `json:"urls,omitempty"`
}

type Finding struct {
    ID          string         `json:"id"`
    RuleID      string         `json:"rule_id"`
    Title       string         `json:"title"`
    Description string         `json:"description"`
    Severity    string         `json:"severity"`
    Confidence  string         `json:"confidence"`
    URL         string         `json:"url"`
    CWEIDs      []string       `json:"cwe_ids,omitempty"`
    Evidence    []string       `json:"evidence,omitempty"`
    CVSSMetrics *CVSSv3Metrics `json:"cvss_metrics,omitempty"`
}

type ScanResult struct {
    TaskID   string    `json:"task_id"`
    ScanID   string    `json:"scan_id"`
    Success  bool      `json:"success"`
    Findings []Finding `json:"findings"`
    Error    string    `json:"error,omitempty"`
    Metadata map[string]interface{} `json:"metadata"`
}

func ConvertToSARIF(scannerName string, findings []Finding) *SARIFReport {
    results := make([]SARIFResult, len(findings))
    for i, f := range findings {
        results[i] = SARIFResult{
            RuleID:  f.RuleID,
            Level:   mapSeverityToLevel(f.Severity),
            Message: SARIFMessage{Text: f.Description},
            Locations: []SARIFLocation{{
                PhysicalLocation: SARIFPhysicalLocation{
                    ArtifactLocation: SARIFArtifactLocation{URI: f.URL},
                },
            }},
        }
    }
    return &SARIFReport{
        Schema:  "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        Version: "2.1.0",
        Runs: []SARIFRun{{
            Tool: SARIFTool{Driver: SARIFDriver{Name: scannerName, Version: "1.0.0"}},
            Results: results,
        }},
    }
}

func mapSeverityToLevel(sev string) string {
    switch sev {
    case "CRITICAL", "HIGH":
        return "error"
    case "MEDIUM":
        return "warning"
    default:
        return "note"
    }
}

func ToJSON(v any) []byte {
    b, _ := json.Marshal(v)
    return b
}
