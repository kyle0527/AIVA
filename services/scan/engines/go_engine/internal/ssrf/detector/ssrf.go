// SSRF Detector - Server-Side Request Forgery 檢測引擎
package detector

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/common"
)

type SSRFDetector struct {
	logger        *zap.Logger
	client        *http.Client
	blockedRanges []*net.IPNet

	// 診斷計數器 (用於追蹤請求生命週期)
	requestsAttempted atomic.Uint64 // testSSRF 被調用次數
	requestsSent      atomic.Uint64 // HTTP 請求實際發送次數
	requestsFailed    atomic.Uint64 // HTTP 請求失敗次數
	responsesRead     atomic.Uint64 // 響應成功讀取次數
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

	// SSRF 掃描器的 HTTP Client（不阻擋內網請求，因為我們需要測試靶場是否會訪問內網）
	// 注意：這個 client 是用來向「靶場應用」發送測試請求，而非直接訪問內網
	client := &http.Client{
		Timeout: 10 * time.Second,
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

// Scan - 實現 Scanner 接口
func (d *SSRFDetector) Scan(ctx context.Context, targets []string, config *common.ScannerConfig) ([]common.Asset, error) {
	d.logger.Info("開始 SSRF 掃描",
		zap.Int("targets", len(targets)),
		zap.Int("concurrency", config.Concurrency),
	)

	var allAssets []common.Asset

	for _, target := range targets {
		select {
		case <-ctx.Done():
			d.logger.Warn("掃描被取消")
			return allAssets, ctx.Err()
		default:
		}

		assets, err := d.scanSingleTarget(ctx, target)
		if err != nil {
			d.logger.Warn("目標掃描失敗", zap.String("target", target), zap.Error(err))
			continue
		}

		allAssets = append(allAssets, assets...)
	}

	// 🔍 診斷: 輸出請求統計信息
	d.logger.Info("SSRF 掃描完成",
		zap.Int("assets_found", len(allAssets)),
		zap.Uint64("requests_attempted", d.requestsAttempted.Load()),
		zap.Uint64("requests_sent", d.requestsSent.Load()),
		zap.Uint64("requests_failed", d.requestsFailed.Load()),
		zap.Uint64("responses_read", d.responsesRead.Load()),
	)
	return allAssets, nil
}

// scanSingleTarget - 掃描單個目標，返回發現的資產
func (d *SSRFDetector) scanSingleTarget(ctx context.Context, target string) ([]common.Asset, error) {
	d.logger.Debug("開始掃描目標", zap.String("target", target))

	var assets []common.Asset

	// 解析目標 URL，確保格式正確
	parsedURL, err := url.Parse(target)
	if err != nil {
		return nil, fmt.Errorf("無效的 URL: %w", err)
	}

	// SSRF 測試 Payloads（使用英文避免編碼問題）
	testPayloads := []struct {
		name        string
		url         string
		description string
	}{
		{
			name:        "AWS IMDS v1",
			url:         "http://169.254.169.254/latest/meta-data/",
			description: "Attempt to access AWS Instance Metadata Service",
		},
		{
			name:        "AWS IMDS v2 Token",
			url:         "http://169.254.169.254/latest/api/token",
			description: "Attempt to obtain AWS IMDS v2 Token",
		},
		{
			name:        "GCP Metadata",
			url:         "http://metadata.google.internal/computeMetadata/v1/",
			description: "Attempt to access GCP Metadata Service",
		},
		{
			name:        "Localhost Admin",
			url:         "http://127.0.0.1/admin",
			description: "Attempt to access local admin interface",
		},
		{
			name:        "Localhost Alternative",
			url:         "http://localhost:8080/",
			description: "Attempt to access local service",
		},
		{
			name:        "IPv6 Localhost",
			url:         "http://[::1]/",
			description: "Attempt to access localhost via IPv6",
		},
		{
			name:        "Private Network",
			url:         "http://192.168.1.1/",
			description: "Attempt to access private network address",
		},
		{
			name:        "File Protocol",
			url:         "file:///etc/passwd",
			description: "Attempt to read local files via file protocol",
		},
	}

	// 常見的 SSRF 參數名稱
	ssrfParams := []string{
		"url", "uri", "path", "redirect", "link", "next", "target",
		"goto", "destination", "source", "callback", "return_to",
		"file", "document", "load", "fetch", "proxy", "forward",
	}

	for _, param := range ssrfParams {
		for _, payload := range testPayloads {
			select {
			case <-ctx.Done():
				return assets, ctx.Err()
			default:
			}

			// 構造測試 URL
			testURL := d.buildTestURL(parsedURL, param, payload.url)

			d.logger.Debug("測試 SSRF",
				zap.String("target", target),
				zap.String("test_url", testURL),
				zap.String("param", param),
				zap.String("payload", payload.name),
			)

			// 執行請求並分析響應
			if asset := d.testSSRF(ctx, testURL, target, param, payload); asset != nil {
				assets = append(assets, *asset)

				d.logger.Warn("🚨 檢測到 SSRF 漏洞",
					zap.String("target", target),
					zap.String("param", param),
					zap.String("payload_type", payload.name),
				)
			}
		}
	}

	return assets, nil
}

// buildTestURL - 構造測試 URL
func (d *SSRFDetector) buildTestURL(baseURL *url.URL, param, payload string) string {
	testURL := *baseURL
	query := testURL.Query()
	query.Set(param, payload)
	testURL.RawQuery = query.Encode()
	return testURL.String()
}

// testSSRF - 執行單次 SSRF 測試
func (d *SSRFDetector) testSSRF(
	ctx context.Context,
	testURL, originalTarget, param string,
	payload struct {
		name        string
		url         string
		description string
	},
) *common.Asset {
	// 🔍 診斷: 記錄 testSSRF 被調用
	d.requestsAttempted.Add(1)

	// 創建請求
	req, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
	if err != nil {
		d.logger.Debug("創建請求失敗", zap.Error(err))
		return nil
	}

	// 設置 User-Agent
	req.Header.Set("User-Agent", "AIVA-SSRF-Scanner/1.0")

	// 🔍 診斷:檢查請求前的狀態
	d.logger.Debug("🔍 即將發送 HTTP 請求",
		zap.String("method", req.Method),
		zap.String("url", req.URL.String()),
		zap.String("host", req.Host),
		zap.Any("headers", req.Header),
	)

	// 執行請求
	startTime := time.Now()
	resp, err := d.client.Do(req)
	duration := time.Since(startTime)

	// 🔍 診斷:記錄請求結果
	d.logger.Debug("🔍 HTTP 請求完成",
		zap.Duration("duration", duration),
		zap.Bool("has_error", err != nil),
		zap.String("error", fmt.Sprintf("%v", err)),
		zap.Bool("has_response", resp != nil),
	)

	// ✅ 修正: 錯誤檢查後立即註冊 defer，避免連接洩漏
	if err != nil {
		// 🔍 診斷: 記錄請求失敗
		d.requestsFailed.Add(1)

		d.logger.Error("🚨 HTTP 請求失敗",
			zap.String("url", testURL),
			zap.Duration("duration", duration),
			zap.Error(err),
		)
		
		if d.isSSRFIndicatorError(err) {
			d.logger.Debug("檢測到 SSRF 指標錯誤",
				zap.String("error", err.Error()),
				zap.String("payload", payload.name),
			)
		}
		// resp 為 nil，無需關閉
		return nil
	}

	// 🔍 診斷: 記錄請求成功發送
	d.requestsSent.Add(1)

	// ✅ 確保響應 Body 一定會被關閉和讀取完整(這是重用連接的關鍵!)
	// 在錯誤檢查之後立即註冊，確保一定會執行
	defer func() {
		if resp != nil && resp.Body != nil {
			// 先丟棄剩餘的 Body 內容(如果有的話)
			io.Copy(io.Discard, resp.Body)
			// 然後關閉
			resp.Body.Close()
		}
	}()

	// 讀取響應內容
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024)) // 限制 1MB
	if err != nil {
		d.logger.Debug("讀取響應失敗",
			zap.String("url", testURL),
			zap.Error(err),
		)
		return nil
	}
	bodyStr := string(body)

	// 🔍 診斷: 記錄響應成功讀取
	d.responsesRead.Add(1)

	// 分析響應，判斷是否存在 SSRF
	if d.isSSRFVulnerable(resp.StatusCode, bodyStr, payload.url) {
			// 生成唯一 Finding ID
			findingID := d.generateFindingID(originalTarget, param, payload.name)

			// 創建 Asset
			return &common.Asset{
				Type:         "web_vulnerability",
				Name:         fmt.Sprintf("SSRF - %s", payload.name),
				Severity:     d.calculateSeverity(payload.name, resp.StatusCode),
				Confidence:   d.calculateConfidence(bodyStr, payload.url),
				SourceEngine: "go",
				Details: map[string]interface{}{
					"finding_id":        findingID,
					"vulnerability_type": "SSRF",
					"cwe":               "CWE-918",
					"description":       payload.description,
					"affected_url":      testURL,
					"vulnerable_param":  param,
					"payload_used":      payload.url,
					"payload_type":      payload.name,
					"http_method":       "GET",
					"response_status":   resp.StatusCode,
					"response_time_ms":  duration.Milliseconds(),
					"response_length":   len(bodyStr),
					"response_preview":  truncate(bodyStr, 200),
					"evidence": map[string]interface{}{
						"request_url":      testURL,
						"response_status":  resp.StatusCode,
						"response_headers": resp.Header,
						"indicators_found": d.extractIndicators(bodyStr, payload.url),
					},
				},
				Metadata: map[string]interface{}{
					"detected_at": time.Now().Format(time.RFC3339),
					"scanner":     "SSRF Detector",
					"target":      originalTarget,
				},
			}
		}

	return nil
}

// GetStats 獲取掃描統計信息
func (d *SSRFDetector) GetStats() map[string]int64 {
	return map[string]int64{
		"attempted": int64(d.requestsAttempted.Load()),
		"sent":      int64(d.requestsSent.Load()),
		"failed":    int64(d.requestsFailed.Load()),
		"read":      int64(d.responsesRead.Load()),
	}
}

// isSSRFVulnerable - 判斷響應是否表明存在 SSRF 漏洞（基於 PortSwigger/OWASP 最佳實踐）
func (d *SSRFDetector) isSSRFVulnerable(statusCode int, body, payload string) bool {
	// 1. 先檢查是否為常見的應用錯誤/登錄頁面（避免誤判）
	if d.isCommonErrorPage(body) {
		return false
	}

	// 2. 針對不同 payload 類型進行精確驗證
	if strings.Contains(payload, "169.254.169.254") {
		// AWS IMDS：檢查元數據格式，而非僅檢查關鍵字
		return d.isAWSMetadataResponse(body, statusCode)
	}

	if strings.Contains(payload, "metadata.google.internal") {
		// GCP Metadata：檢查 JSON 格式響應
		return d.isGCPMetadataResponse(body, statusCode)
	}

	if strings.Contains(payload, "file://") {
		// File Protocol：檢查文件內容格式
		return d.isFileProtocolResponse(body, statusCode)
	}

	// 3. 內網地址：檢查是否與公網響應明顯不同
	if strings.Contains(payload, "127.0.0.1") || strings.Contains(payload, "localhost") || strings.Contains(payload, "192.168.") {
		// 401/403 表示成功到達內部服務（但需要認證）
		if statusCode == 401 || statusCode == 403 {
			return true
		}
		// 200 且內容不是登錄頁面
		if statusCode == 200 && !d.isLoginPage(body) {
			return d.containsInternalServiceIndicators(body)
		}
	}

	return false
}

// isCommonErrorPage - 檢查是否為常見的錯誤/登錄頁面（避免誤判）
func (d *SSRFDetector) isCommonErrorPage(body string) bool {
	bodyLower := strings.ToLower(body)
	// 登錄頁面特徵
	if strings.Contains(bodyLower, "<title>login") || strings.Contains(bodyLower, "<title>sign in") {
		return true
	}
	// 404/403 錯誤頁面
	if strings.Contains(bodyLower, "404 not found") || strings.Contains(bodyLower, "403 forbidden") {
		return true
	}
	return false
}

// isLoginPage - 檢查是否為登錄頁面
func (d *SSRFDetector) isLoginPage(body string) bool {
	bodyLower := strings.ToLower(body)
	return strings.Contains(bodyLower, "<title>login") ||
		strings.Contains(bodyLower, "name=\"username\"") ||
		strings.Contains(bodyLower, "name=\"password\"")
}

// isAWSMetadataResponse - 驗證是否為真實的 AWS 元數據響應
func (d *SSRFDetector) isAWSMetadataResponse(body string, statusCode int) bool {
	if statusCode != 200 {
		return false
	}
	// AWS IMDS 響應特徵：簡短的純文本列表，無 HTML 標籤
	if strings.Contains(body, "<html") || strings.Contains(body, "<body") {
		return false // 不應該是 HTML 頁面
	}
	// 檢查典型的 IMDS 響應內容（目錄列表或特定值）
	if strings.Contains(body, "ami-id") || strings.Contains(body, "instance-id") ||
		strings.Contains(body, "security-credentials") || strings.Contains(body, "iam/") {
		return true
	}
	// 或者是簡短的純文本（如 ami-id 值）
	if len(body) < 500 && !strings.Contains(body, "<") && len(strings.TrimSpace(body)) > 5 {
		return true
	}
	return false
}

// isGCPMetadataResponse - 驗證是否為真實的 GCP 元數據響應
func (d *SSRFDetector) isGCPMetadataResponse(body string, statusCode int) bool {
	if statusCode != 200 {
		return false
	}
	// GCP 元數據通常是 JSON 或純文本
	return (strings.HasPrefix(strings.TrimSpace(body), "{") || !strings.Contains(body, "<html"))
}

// isFileProtocolResponse - 驗證是否為真實的文件讀取響應
func (d *SSRFDetector) isFileProtocolResponse(body string, statusCode int) bool {
	if statusCode != 200 {
		return false
	}
	// 檢查 /etc/passwd 格式：root:x:0:0
	if strings.Contains(body, "root:x:0:0") || strings.Contains(body, "root:x:0:") {
		return true
	}
	// 或者包含典型的 Unix 用戶列表格式
	lines := strings.Split(body, "\n")
	for _, line := range lines {
		if strings.Count(line, ":") >= 6 && !strings.Contains(line, "<") {
			return true // 類似 username:x:uid:gid:...
		}
	}
	return false
}

// containsInternalServiceIndicators - 檢查是否包含內部服務指標
func (d *SSRFDetector) containsInternalServiceIndicators(body string) bool {
	bodyLower := strings.ToLower(body)
	// 內部服務通常有這些特徵
	internalIndicators := []string{
		"internal api", "admin panel", "dashboard",
		"prometheus", "grafana", "kubernetes",
		"consul", "etcd", "redis",
	}
	for _, indicator := range internalIndicators {
		if strings.Contains(bodyLower, indicator) {
			return true
		}
	}
	return false
}

// generateFindingID - 生成唯一的 Finding ID
func (d *SSRFDetector) generateFindingID(target, param, payloadName string) string {
	data := fmt.Sprintf("%s:%s:%s:%d", target, param, payloadName, time.Now().Unix())
	hash := md5.Sum([]byte(data))
	return fmt.Sprintf("ssrf_%s", hex.EncodeToString(hash[:])[:16])
}

// calculateSeverity - 計算漏洞嚴重性
func (d *SSRFDetector) calculateSeverity(payloadName string, statusCode int) string {
	// AWS/GCP Metadata 服務是高危
	if strings.Contains(payloadName, "IMDS") ||
		strings.Contains(payloadName, "Metadata") {
		return "high"
	}

	// 本地文件讀取是高危
	if strings.Contains(payloadName, "File") {
		return "high"
	}

	// 內網訪問是中危
	if strings.Contains(payloadName, "Localhost") ||
		strings.Contains(payloadName, "Private") {
		return "medium"
	}

	return "low"
}

// calculateConfidence - 計算檢測置信度（基於精確證據）
func (d *SSRFDetector) calculateConfidence(body, payload string) string {
	indicators := d.extractIndicators(body, payload)
	
	// 有精確的格式驗證證據 = 高置信度
	if len(indicators) >= 2 {
		return "high"
	}
	if len(indicators) >= 1 {
		return "medium"
	}
	
	// 僅基於狀態碼 = 低置信度
	return "low"
}

// extractIndicators - 提取 SSRF 證據指標（精確驗證，避免誤判）
func (d *SSRFDetector) extractIndicators(body, payload string) []string {
	var indicators []string

	// AWS IMDS 精確檢查
	if strings.Contains(payload, "169.254.169.254") {
		if strings.Contains(body, "ami-id") {
			indicators = append(indicators, "Found AWS AMI ID metadata")
		}
		if strings.Contains(body, "instance-id") {
			indicators = append(indicators, "Found AWS Instance ID metadata")
		}
		if strings.Contains(body, "security-credentials") {
			indicators = append(indicators, "Found AWS IAM credentials path")
		}
		if len(body) < 500 && !strings.Contains(body, "<html") {
			indicators = append(indicators, "Plain text metadata response (non-HTML)")
		}
	}

	// GCP Metadata 精確檢查
	if strings.Contains(payload, "metadata.google.internal") {
		if strings.HasPrefix(strings.TrimSpace(body), "{") {
			indicators = append(indicators, "JSON metadata response from GCP")
		}
	}

	// File Protocol 精確檢查
	if strings.Contains(payload, "file://") {
		if strings.Contains(body, "root:x:0:0") {
			indicators = append(indicators, "Found /etc/passwd root entry")
		}
		if strings.Count(body, ":") > 20 && !strings.Contains(body, "<html") {
			indicators = append(indicators, "Unix password file format detected")
		}
	}

	// 內部服務檢查（排除登錄頁面）
	if !d.isLoginPage(body) {
		bodyLower := strings.ToLower(body)
		if strings.Contains(bodyLower, "internal api") {
			indicators = append(indicators, "Internal API endpoint detected")
		}
		if strings.Contains(bodyLower, "admin panel") || strings.Contains(bodyLower, "dashboard") {
			indicators = append(indicators, "Admin interface detected")
		}
	}

	return indicators
}

// isSSRFIndicatorError - 判斷錯誤是否為 SSRF 指標
func (d *SSRFDetector) isSSRFIndicatorError(err error) bool {
	errStr := strings.ToLower(err.Error())

	// 連接被拒絕可能表示成功訪問內網（但服務未運行）
	if strings.Contains(errStr, "connection refused") {
		return true
	}

	// 超時可能表示正在嘗試連接
	if strings.Contains(errStr, "timeout") {
		return true
	}

	return false
}

// ==================== Scanner 接口實現 ====================

// Name - 返回掃描器名稱
func (d *SSRFDetector) Name() string {
	return "SSRF Scanner"
}

// Type - 返回掃描器類型
func (d *SSRFDetector) Type() string {
	return "ssrf"
}

// ==================== 輔助函數 ====================

// containsSensitiveInfo - 檢查響應是否包含敏感信息
func (d *SSRFDetector) containsSensitiveInfo(body, payload string) bool {
	// 檢查是否包含敏感關鍵字
	keywords := []string{
		"ami-id", "instance-id", "iam/security-credentials",
		"computeMetadata", "metadata",
		"config", "password", "secret", "token", "api_key",
		"root:", "admin", "private",
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

	// GCP Metadata 特殊檢查
	if strings.Contains(payload, "metadata.google.internal") {
		if strings.Contains(body, "project") || strings.Contains(body, "instance") {
			return true
		}
	}

	return false
}

// truncate - 截斷字符串
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
