package main

import (
    "context"
    "encoding/json"
    "fmt"
    "strings"

    "aiva/scan/go_scanners/common"
)

type SCAFindingRule struct {
    Package string
    VulnerablePrefix string // versions starting with this prefix
    CVE     string
    Severity string
    Description string
}

type SCAScanner struct {
    rules []SCAFindingRule
}

func NewSCAScanner() *SCAScanner {
    return &SCAScanner{
        rules: []SCAFindingRule{
            {"lodash", "4.17.", "CVE-2021-23337", "HIGH", "Prototype pollution in lodash"},
            {"log4j", "2.14.", "CVE-2021-44228", "CRITICAL", "Log4Shell vulnerability in log4j-core"},
            {"express", "4.16.", "CVE-2019-xxxxx", "MEDIUM", "Known prototype pollution risk"},
        },
    }
}

func (s *SCAScanner) GetName() string { return "AIVA-SCA-GO" }
func (s *SCAScanner) GetVersion() string { return "1.0.0" }
func (s *SCAScanner) GetCapabilities() []string { return []string{"SCA"} }
func (s *SCAScanner) HealthCheck() error { return nil }

func (s *SCAScanner) Scan(ctx context.Context, task common.ScanTask) common.ScanResult {
    // Expect task.Config["manifest"] as a JSON string of { "dependencies": { "name": "version" } }
    findings := []common.Finding{}
    manifestRaw, ok := task.Config["manifest"]
    if !ok {
        return common.ScanResult{ TaskID: task.TaskID, ScanID: task.ScanID, Success: true, Findings: findings, Metadata: map[string]interface{}{"note": "no manifest"} }
    }
    mjson := fmt.Sprintf("%v", manifestRaw)
    var manifest struct { Dependencies map[string]string `json:"dependencies"` }
    _ = json.Unmarshal([]byte(mjson), &manifest)

    for name, ver := range manifest.Dependencies {
        for _, rule := range s.rules {
            if strings.EqualFold(name, rule.Package) && strings.HasPrefix(ver, rule.VulnerablePrefix) {
                findings = append(findings, common.Finding{
                    ID: rule.CVE,
                    RuleID: rule.CVE,
                    Title: rule.Package + " vulnerable version",
                    Description: rule.Description,
                    Severity: rule.Severity,
                    Confidence: "HIGH",
                    URL: task.Target.URL,
                    Evidence: []string{name + "@" + ver},
                })
            }
        }
    }
    return common.ScanResult{ TaskID: task.TaskID, ScanID: task.ScanID, Success: true, Findings: findings, Metadata: map[string]interface{}{"rules": len(s.rules)} }
}
