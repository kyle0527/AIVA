// SSRF Detector - 核心檢測邏輯
package detector

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	// 使用標準 schema 定義
	schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"
)

type ScanTask struct {
	TaskID   string            `json:"task_id"`
	Module   string            `json:"module"`
	Target   string            `json:"target"`
	Metadata map[string]string `json:"metadata"`
}

// 使用標準 FindingPayload 替代自定義 Finding 結構
// 這確保與 Python aiva_common 的一致性
type Finding = schemas.FindingPayload

type SSRFDetector struct {
	logger        *zap.Logger
	client        *http.Client
	blockedRanges []*net.IPNet
}

func NewSSRFDetector(logger *zap.Logger) *SSRFDetector {
	// 阻擋的內網 IP 範圍
	blockedCIDRs := []string{
		"10.0.0.0/8",
		"172.16.0.0/12",
		"192.168.0.0/16",
		"127.0.0.0/8",
		"169.254.169.254/32", // AWS IMDS
		"fd00::/8",           // IPv6 ULA
	}

	var ranges []*net.IPNet
	for _, cidr := range blockedCIDRs {
		_, ipNet, _ := net.ParseCIDR(cidr)
		ranges = append(ranges, ipNet)
	}

	// 自定義 HTTP 客戶端 (阻擋內網請求)
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, _, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}

			ip := net.ParseIP(host)
			if ip != nil {
				for _, blocked := range ranges {
					if blocked.Contains(ip) {
						return nil, fmt.Errorf("blocked IP: %s", host)
					}
				}
			}

			return (&net.Dialer{
				Timeout:   5 * time.Second,
				KeepAlive: 5 * time.Second,
			}).DialContext(ctx, network, addr)
		},
	}

	client := &http.Client{
		Timeout:   10 * time.Second,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	return &SSRFDetector{
		logger:        logger,
		client:        client,
		blockedRanges: ranges,
	}
}

func (d *SSRFDetector) Scan(ctx context.Context, task *ScanTask) ([]*Finding, error) {
	d.logger.Info("開始 SSRF 掃描", zap.String("task_id", task.TaskID))

	findings := []*Finding{}

	// SSRF Payloads
	payloads := []string{
		"http://169.254.169.254/latest/meta-data/",            // AWS IMDS
		"http://metadata.google.internal/computeMetadata/v1/", // GCP Metadata
		"http://127.0.0.1:80/admin",                           // Localhost
		"http://localhost:8080/",                              // Localhost alt port
		"http://[::1]/",                                       // IPv6 localhost
		"http://0.0.0.0/",                                     // Wildcard
		"http://192.168.1.1/",                                 // Private IP
	}

	for _, payload := range payloads {
		select {
		case <-ctx.Done():
			return findings, ctx.Err()
		default:
		}

		// 構造測試 URL (假設目標有 url 參數)
		testURL := fmt.Sprintf("%s?url=%s", task.Target, payload)

		d.logger.Debug("測試 SSRF payload",
			zap.String("payload", payload),
			zap.String("test_url", testURL),
		)

		req, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
		if err != nil {
			d.logger.Debug("建立請求失敗", zap.Error(err))
			continue
		}

		resp, err := d.client.Do(req)
		if err != nil {
			// 請求失敗是正常的 (可能被阻擋)
			d.logger.Debug("請求失敗", zap.String("payload", payload), zap.Error(err))
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		// 檢查是否成功訪問內網
		if resp.StatusCode == 200 {
			// 檢查回應內容是否包含敏感資訊
			bodyStr := string(body)
			if d.containsSensitiveInfo(bodyStr, payload) {
				// 創建符合標準 FindingPayload 格式的發現
				evidence := &schemas.FindingEvidence{
					Request:  &testURL,
					Response: &bodyStr,
					Proof: func() *string {
						s := fmt.Sprintf("Status: %d, Body (前100字): %s", resp.StatusCode, truncate(bodyStr, 100))
						return &s
					}(),
				}

				vulnerability := schemas.Vulnerability{
					Name:       "SSRF",
					Cwe:        func() *string { s := "CWE-918"; return &s }(),
					Severity:   "HIGH",
					Confidence: "FIRM",
					Description: func() *string {
						s := "Server-Side Request Forgery vulnerability detected"
						return &s
					}(),
				}

				target := schemas.Target{
					Url: task.Target,
				}

				finding := &Finding{
					FindingId:     fmt.Sprintf("finding_%s_%d", task.TaskID, time.Now().Unix()),
					TaskId:        task.TaskID,
					ScanId:        "ssrf_scan",
					Status:        "confirmed",
					Vulnerability: vulnerability,
					Target:        target,
					Evidence:      evidence,
					CreatedAt:     time.Now(),
					UpdatedAt:     time.Now(),
				}
				findings = append(findings, finding)

				d.logger.Warn("🚨 發現 SSRF 漏洞",
					zap.String("task_id", task.TaskID),
					zap.String("payload", payload),
				)
			}
		}
	}

	return findings, nil
}

func (d *SSRFDetector) containsSensitiveInfo(body, payload string) bool {
	// 檢查是否包含敏感資訊的關鍵字
	keywords := []string{
		"ami-id",                   // AWS
		"instance-id",              // AWS
		"iam/security-credentials", // AWS IAM
		"computeMetadata",          // GCP
		"config",
		"password",
		"secret",
		"token",
		"api_key",
	}

	bodyLower := strings.ToLower(body)
	for _, kw := range keywords {
		if strings.Contains(bodyLower, strings.ToLower(kw)) {
			return true
		}
	}

	// AWS IMDS 特殊檢查
	if strings.Contains(payload, "169.254.169.254") && len(body) > 10 {
		return true
	}

	return false
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
