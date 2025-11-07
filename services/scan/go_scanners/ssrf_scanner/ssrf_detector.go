package main

import (
    "context"
    "fmt"
    "net/http"
    "net/url"
    "strings"
    "sync"
    "time"

    "aiva/scan/go_scanners/common"
)

type SSRFScanner struct {
    httpClient *http.Client
    payloads   []string
}

func NewSSRFScanner() *SSRFScanner {
    return &SSRFScanner{
        httpClient: &http.Client{ Timeout: 8 * time.Second },
        payloads: []string{
            "http://127.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "gopher://127.0.0.1:25/",
        },
    }
}

func (s *SSRFScanner) GetName() string { return "AIVA-SSRF-GO" }
func (s *SSRFScanner) GetVersion() string { return "1.0.0" }
func (s *SSRFScanner) GetCapabilities() []string { return []string{"SSRF", "metadata", "file-scheme"} }
func (s *SSRFScanner) HealthCheck() error { return nil }

func (s *SSRFScanner) Scan(ctx context.Context, task common.ScanTask) common.ScanResult {
    target := task.Target.URL
    findings := []common.Finding{}
    params := extractParams(target)

    var wg sync.WaitGroup
    mu := &sync.Mutex{}
    for pname := range params {
        pname := pname
        wg.Add(1)
        go func() {
            defer wg.Done()
            for _, p := range s.payloads {
                testURL := injectParam(target, pname, p)
                req, _ := http.NewRequestWithContext(ctx, "GET", testURL, nil)
                resp, err := s.httpClient.Do(req)
                if err != nil {
                    continue
                }
                _ = resp.Body.Close()
                if looksLikeSSRF(resp, p) {
                    mu.Lock()
                    findings = append(findings, common.Finding{
                        ID: fmt.Sprintf("ssrf_%s_%d", pname, time.Now().UnixNano()),
                        RuleID: "CWE-918",
                        Title: "Server-Side Request Forgery (SSRF)",
                        Description: fmt.Sprintf("Parameter '%s' appears vulnerable to SSRF.", pname),
                        Severity: "HIGH",
                        Confidence: "MEDIUM",
                        URL: testURL,
                        CWEIDs: []string{"CWE-918"},
                        Evidence: []string{fmt.Sprintf("payload=%s status=%d", p, resp.StatusCode)},
                        CVSSMetrics: &common.CVSSv3Metrics{
                            BaseScore: 8.5, AttackVector: "NETWORK", AttackComplexity: "LOW", PrivilegesRequired: "NONE",
                            UserInteraction: "NONE", Scope: "CHANGED", ConfidentialityImpact: "HIGH", IntegrityImpact: "LOW", AvailabilityImpact: "LOW",
                        },
                    })
                    mu.Unlock()
                }
            }
        }()
    }
    wg.Wait()
    return common.ScanResult{
        TaskID: task.TaskID,
        ScanID: task.ScanID,
        Success: true,
        Findings: findings,
        Metadata: map[string]interface{}{"tested_params": len(params)},
    }
}

func extractParams(u string) map[string]string {
    out := map[string]string{}
    ur, err := url.Parse(u)
    if err != nil { return out }
    for k, v := range ur.Query() {
        if len(v) > 0 {
            out[k] = v[0]
        }
    }
    return out
}
func injectParam(u, name, payload string) string {
    ur, err := url.Parse(u)
    if err != nil { return u }
    q := ur.Query()
    q.Set(name, payload)
    ur.RawQuery = q.Encode()
    return ur.String()
}
func looksLikeSSRF(resp *http.Response, payload string) bool {
    if resp.StatusCode >= 200 && resp.StatusCode < 300 {
        return true
    }
    if strings.HasPrefix(payload, "http://169.254.169.254") && (resp.StatusCode == 200 || resp.StatusCode == 403) {
        return true
    }
    return false
}
