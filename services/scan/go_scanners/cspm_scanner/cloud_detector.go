package main

import (
    "context"
    "fmt"

    "aiva/scan/go_scanners/common"
)

type CSPMScanner struct {
    rules []CloudRule
}

type CloudRule struct {
    ID string
    Title string
    Description string
    Severity string
    MatchKey string
    MatchValue string
}

func NewCSPMScanner() *CSPMScanner {
    return &CSPMScanner{
        rules: []CloudRule{
            {"CSPM-001", "Public S3 Bucket", "S3 bucket allows public read", "HIGH", "s3_public", "true"},
            {"CSPM-002", "DB Security Group 0.0.0.0/0", "Database open to world", "HIGH", "db_cidr_all", "true"},
            {"CSPM-003", "IAM Root Used", "Root user used for daily ops", "MEDIUM", "iam_root_used", "true"},
        },
    }
}

func (s *CSPMScanner) GetName() string { return "AIVA-CSPM-GO" }
func (s *CSPMScanner) GetVersion() string { return "1.0.0" }
func (s *CSPMScanner) GetCapabilities() []string { return []string{"CSPM"} }
func (s *CSPMScanner) HealthCheck() error { return nil }

func (s *CSPMScanner) Scan(ctx context.Context, task common.ScanTask) common.ScanResult {
    findings := []common.Finding{}
    cfg := task.Config
    for _, r := range s.rules {
        v, ok := cfg[r.MatchKey]
        if ok && fmt.Sprintf("%v", v) == r.MatchValue {
            findings = append(findings, common.Finding{
                ID: r.ID,
                RuleID: r.ID,
                Title: r.Title,
                Description: r.Description,
                Severity: r.Severity,
                Confidence: "MEDIUM",
                URL: task.Target.URL,
                CWEIDs: []string{},
                Evidence: []string{fmt.Sprintf("%s=%v", r.MatchKey, v)},
            })
        }
    }
    return common.ScanResult{ TaskID: task.TaskID, ScanID: task.ScanID, Success: true, Findings: findings, Metadata: map[string]interface{}{"rules": len(s.rules)} }
}
