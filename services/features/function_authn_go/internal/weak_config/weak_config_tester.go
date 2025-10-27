package weak_config

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
	"go.uber.org/zap"
)

// WeakConfigResult 弱配置測試結果
type WeakConfigResult struct {
	ConfigType     string            `json:"config_type"`
	Vulnerable     bool              `json:"vulnerable"`
	Issue          string            `json:"issue"`
	Details        string            `json:"details"`
	Risk           string            `json:"risk"`
	Evidence       map[string]string `json:"evidence"`
	Recommendation string            `json:"recommendation"`
}

// WeakConfigTester 弱配置檢測器
type WeakConfigTester struct {
	logger *zap.Logger
	client *http.Client
}

// NewWeakConfigTester 建立弱配置檢測器
func NewWeakConfigTester(logger *zap.Logger) *WeakConfigTester {
	// 配置HTTP客戶端，允許不安全的TLS (用於檢測弱配置)
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,             // 允許自簽證書
			MinVersion:         tls.VersionTLS10, // 允許舊版TLS
		},
		IdleConnTimeout:    30 * time.Second,
		DisableCompression: true,
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   15 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse // 不自動跟隨重定向
		},
	}

	return &WeakConfigTester{
		logger: logger,
		client: client,
	}
}

// Test 執行弱配置檢測
func (w *WeakConfigTester) Test(ctx context.Context, task *schemas.FunctionTaskPayload) ([]*schemas.FindingPayload, error) {
	w.logger.Info("Starting weak configuration tests", zap.String("task_id", task.TaskID))

	var findings []*schemas.FindingPayload
	var results []WeakConfigResult

	// 解析目標URL
	targetURL, err := url.Parse(task.Target.URL)
	if err != nil {
		return nil, fmt.Errorf("invalid target URL: %w", err)
	}

	// 執行各種弱配置檢測
	configTests := []func(context.Context, *url.URL) []WeakConfigResult{
		w.testWeakTLSConfig,
		w.testDefaultCredentials,
		w.testWeakHeaders,
		w.testDirectoryListing,
		w.testDebugEndpoints,
		w.testMisconfiguredCORS,
		w.testWeakSessionConfig,
		w.testInformationDisclosure,
	}

	for _, testFunc := range configTests {
		select {
		case <-ctx.Done():
			return findings, ctx.Err()
		default:
			testResults := testFunc(ctx, targetURL)
			results = append(results, testResults...)
		}
	}

	// 轉換結果為Finding格式
	for _, result := range results {
		if result.Vulnerable {
			finding := w.createFinding(task.TaskID, targetURL.String(), result)
			findings = append(findings, &finding)
		}
	}

	w.logger.Info("Weak configuration tests completed",
		zap.String("task_id", task.TaskID),
		zap.Int("total_tests", len(results)),
		zap.Int("vulnerabilities_found", len(findings)))

	return findings, nil
}

// testWeakTLSConfig 檢測弱TLS配置
func (w *WeakConfigTester) testWeakTLSConfig(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 測試HTTPS目標
	if targetURL.Scheme != "https" {
		results = append(results, WeakConfigResult{
			ConfigType:     "tls",
			Vulnerable:     true,
			Issue:          "HTTP instead of HTTPS",
			Details:        "服務使用HTTP協議，傳輸數據未加密",
			Risk:           "HIGH",
			Evidence:       map[string]string{"protocol": targetURL.Scheme},
			Recommendation: "啟用HTTPS並強制重定向HTTP到HTTPS",
		})
		return results
	}

	// 檢測TLS版本支援
	tlsVersions := []uint16{
		tls.VersionSSL30, // 已棄用
		tls.VersionTLS10, // 弱版本
		tls.VersionTLS11, // 弱版本
		tls.VersionTLS12, // 可接受
		tls.VersionTLS13, // 最佳
	}

	supportedVersions := []string{}
	for _, version := range tlsVersions {
		if w.testTLSVersion(targetURL, version) {
			versionName := w.getTLSVersionName(version)
			supportedVersions = append(supportedVersions, versionName)

			// 檢測弱版本
			if version <= tls.VersionTLS11 {
				results = append(results, WeakConfigResult{
					ConfigType:     "tls_version",
					Vulnerable:     true,
					Issue:          fmt.Sprintf("Weak TLS version supported: %s", versionName),
					Details:        fmt.Sprintf("服務支援弱TLS版本 %s，存在安全風險", versionName),
					Risk:           "HIGH",
					Evidence:       map[string]string{"tls_version": versionName},
					Recommendation: "禁用TLS 1.1及以下版本，僅支援TLS 1.2和1.3",
				})
			}
		}
	}

	// 檢測證書問題
	if certResult := w.testCertificateIssues(targetURL); certResult.Vulnerable {
		results = append(results, certResult)
	}

	return results
}

// testDefaultCredentials 檢測預設憑證
func (w *WeakConfigTester) testDefaultCredentials(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 常見的預設憑證列表
	defaultCreds := []struct {
		username string
		password string
		service  string
	}{
		{"admin", "admin", "generic"},
		{"admin", "password", "generic"},
		{"admin", "", "generic"},
		{"administrator", "administrator", "generic"},
		{"root", "root", "generic"},
		{"guest", "guest", "generic"},
		{"admin", "123456", "generic"},
		{"admin", "admin123", "generic"},
		{"user", "user", "generic"},
		{"test", "test", "generic"},
		// 特定服務的預設憑證
		{"tomcat", "tomcat", "tomcat"},
		{"jenkins", "jenkins", "jenkins"},
		{"elastic", "changeme", "elasticsearch"},
		{"kibana", "kibana", "kibana"},
	}

	// 常見登入端點
	loginPaths := []string{
		"/login",
		"/admin/login",
		"/administrator/login",
		"/wp-admin/",
		"/admin/",
		"/manager/html",
		"/console/",
		"/dashboard/",
	}

	for _, path := range loginPaths {
		loginURL := fmt.Sprintf("%s://%s%s", targetURL.Scheme, targetURL.Host, path)

		// 檢查端點是否存在
		if !w.checkEndpointExists(loginURL) {
			continue
		}

		for _, cred := range defaultCreds {
			if w.testCredential(loginURL, cred.username, cred.password) {
				results = append(results, WeakConfigResult{
					ConfigType: "default_credentials",
					Vulnerable: true,
					Issue:      "Default credentials accepted",
					Details:    fmt.Sprintf("預設憑證 %s:%s 在 %s 被接受", cred.username, cred.password, path),
					Risk:       "CRITICAL",
					Evidence: map[string]string{
						"username": cred.username,
						"password": cred.password,
						"endpoint": path,
						"service":  cred.service,
					},
					Recommendation: "立即更改所有預設憑證，實施強密碼政策",
				})
			}
		}
	}

	return results
}

// testWeakHeaders 檢測安全標頭配置
func (w *WeakConfigTester) testWeakHeaders(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	resp, err := w.client.Get(targetURL.String())
	if err != nil {
		w.logger.Debug("Failed to fetch headers", zap.Error(err))
		return results
	}
	defer resp.Body.Close()

	// 檢查缺失的安全標頭
	securityHeaders := map[string]string{
		"X-Frame-Options":           "防止點擊劫持攻擊",
		"X-Content-Type-Options":    "防止MIME類型混淆攻擊",
		"X-XSS-Protection":          "啟用瀏覽器XSS防護",
		"Strict-Transport-Security": "強制HTTPS傳輸",
		"Content-Security-Policy":   "防止XSS和代碼注入",
		"Referrer-Policy":           "控制Referrer資訊洩露",
		"Permissions-Policy":        "控制瀏覽器功能權限",
	}

	for header, description := range securityHeaders {
		if resp.Header.Get(header) == "" {
			results = append(results, WeakConfigResult{
				ConfigType:     "security_headers",
				Vulnerable:     true,
				Issue:          fmt.Sprintf("Missing security header: %s", header),
				Details:        fmt.Sprintf("缺少安全標頭 %s - %s", header, description),
				Risk:           "MEDIUM",
				Evidence:       map[string]string{"missing_header": header},
				Recommendation: fmt.Sprintf("添加 %s 標頭以提升安全性", header),
			})
		}
	}

	// 檢查危險的Server標頭
	serverHeader := resp.Header.Get("Server")
	if serverHeader != "" && w.isServerHeaderVulnerable(serverHeader) {
		results = append(results, WeakConfigResult{
			ConfigType:     "information_disclosure",
			Vulnerable:     true,
			Issue:          "Server version disclosure",
			Details:        fmt.Sprintf("Server標頭洩露版本資訊: %s", serverHeader),
			Risk:           "LOW",
			Evidence:       map[string]string{"server_header": serverHeader},
			Recommendation: "隱藏或模糊化Server標頭中的版本資訊",
		})
	}

	return results
}

// testDirectoryListing 檢測目錄清單
func (w *WeakConfigTester) testDirectoryListing(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 常見可能啟用目錄清單的路徑
	testPaths := []string{
		"/",
		"/admin/",
		"/backup/",
		"/logs/",
		"/uploads/",
		"/files/",
		"/tmp/",
		"/test/",
		"/dev/",
		"/config/",
	}

	for _, path := range testPaths {
		testURL := fmt.Sprintf("%s://%s%s", targetURL.Scheme, targetURL.Host, path)

		if w.hasDirectoryListing(testURL) {
			results = append(results, WeakConfigResult{
				ConfigType:     "directory_listing",
				Vulnerable:     true,
				Issue:          "Directory listing enabled",
				Details:        fmt.Sprintf("目錄 %s 啟用了目錄清單功能", path),
				Risk:           "MEDIUM",
				Evidence:       map[string]string{"path": path},
				Recommendation: "禁用Web服務器的目錄清單功能",
			})
		}
	}

	return results
}

// testDebugEndpoints 檢測調試端點
func (w *WeakConfigTester) testDebugEndpoints(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 常見的調試和管理端點
	debugEndpoints := []string{
		"/debug",
		"/debug/pprof/",
		"/actuator/",
		"/actuator/health",
		"/actuator/env",
		"/actuator/configprops",
		"/health",
		"/status",
		"/info",
		"/metrics",
		"/trace",
		"/.env",
		"/phpinfo.php",
		"/server-status",
		"/server-info",
	}

	for _, endpoint := range debugEndpoints {
		testURL := fmt.Sprintf("%s://%s%s", targetURL.Scheme, targetURL.Host, endpoint)

		if w.isDebugEndpointExposed(testURL) {
			results = append(results, WeakConfigResult{
				ConfigType:     "debug_endpoints",
				Vulnerable:     true,
				Issue:          "Debug endpoint exposed",
				Details:        fmt.Sprintf("調試端點 %s 對外暴露", endpoint),
				Risk:           "HIGH",
				Evidence:       map[string]string{"endpoint": endpoint},
				Recommendation: "在生產環境中禁用或保護調試端點",
			})
		}
	}

	return results
}

// testMisconfiguredCORS 檢測CORS配置錯誤
func (w *WeakConfigTester) testMisconfiguredCORS(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 創建帶有Origin標頭的請求
	req, err := http.NewRequestWithContext(ctx, "GET", targetURL.String(), nil)
	if err != nil {
		return results
	}

	// 測試通配符Origin
	req.Header.Set("Origin", "https://evil.example.com")
	resp, err := w.client.Do(req)
	if err != nil {
		return results
	}
	defer resp.Body.Close()

	// 檢查CORS響應
	accessControlOrigin := resp.Header.Get("Access-Control-Allow-Origin")
	if accessControlOrigin == "*" {
		results = append(results, WeakConfigResult{
			ConfigType:     "cors_misconfiguration",
			Vulnerable:     true,
			Issue:          "CORS wildcard origin allowed",
			Details:        "CORS配置允許所有來源 (*) 存取資源",
			Risk:           "MEDIUM",
			Evidence:       map[string]string{"access_control_origin": accessControlOrigin},
			Recommendation: "限制CORS允許的來源為特定域名",
		})
	}

	// 檢查是否反射任意Origin
	if accessControlOrigin == "https://evil.example.com" {
		results = append(results, WeakConfigResult{
			ConfigType:     "cors_misconfiguration",
			Vulnerable:     true,
			Issue:          "CORS reflects arbitrary origins",
			Details:        "CORS配置反射任意來源，可能導致跨域資料洩露",
			Risk:           "HIGH",
			Evidence:       map[string]string{"reflected_origin": accessControlOrigin},
			Recommendation: "實施白名單機制限制允許的CORS來源",
		})
	}

	return results
}

// testWeakSessionConfig 檢測會話配置問題
func (w *WeakConfigTester) testWeakSessionConfig(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	resp, err := w.client.Get(targetURL.String())
	if err != nil {
		return results
	}
	defer resp.Body.Close()

	// 檢查Cookie設定
	for _, cookie := range resp.Cookies() {
		// 檢查缺少Secure標記
		if !cookie.Secure && targetURL.Scheme == "https" {
			results = append(results, WeakConfigResult{
				ConfigType:     "session_config",
				Vulnerable:     true,
				Issue:          "Cookie missing Secure flag",
				Details:        fmt.Sprintf("Cookie %s 缺少Secure標記", cookie.Name),
				Risk:           "MEDIUM",
				Evidence:       map[string]string{"cookie_name": cookie.Name},
				Recommendation: "為所有Cookie設定Secure標記",
			})
		}

		// 檢查缺少HttpOnly標記
		if !cookie.HttpOnly {
			results = append(results, WeakConfigResult{
				ConfigType:     "session_config",
				Vulnerable:     true,
				Issue:          "Cookie missing HttpOnly flag",
				Details:        fmt.Sprintf("Cookie %s 缺少HttpOnly標記", cookie.Name),
				Risk:           "MEDIUM",
				Evidence:       map[string]string{"cookie_name": cookie.Name},
				Recommendation: "為會話Cookie設定HttpOnly標記防止XSS",
			})
		}

		// 檢查SameSite設定
		if cookie.SameSite == http.SameSiteDefaultMode {
			results = append(results, WeakConfigResult{
				ConfigType:     "session_config",
				Vulnerable:     true,
				Issue:          "Cookie missing SameSite attribute",
				Details:        fmt.Sprintf("Cookie %s 缺少SameSite屬性", cookie.Name),
				Risk:           "LOW",
				Evidence:       map[string]string{"cookie_name": cookie.Name},
				Recommendation: "設定SameSite屬性防止CSRF攻擊",
			})
		}
	}

	return results
}

// testInformationDisclosure 檢測資訊洩露
func (w *WeakConfigTester) testInformationDisclosure(ctx context.Context, targetURL *url.URL) []WeakConfigResult {
	var results []WeakConfigResult

	// 測試錯誤頁面資訊洩露
	errorURL := fmt.Sprintf("%s://%s/nonexistentpage12345", targetURL.Scheme, targetURL.Host)
	resp, err := w.client.Get(errorURL)
	if err == nil {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)

		// 檢查是否洩露敏感資訊
		sensitivePatterns := map[string]string{
			`(?i)(apache|nginx|iis)/[\d\.]+`:      "Web服務器版本",
			`(?i)php/[\d\.]+`:                     "PHP版本",
			`(?i)(mysql|postgresql|oracle) error`: "資料庫錯誤訊息",
			`(?i)stack trace|backtrace`:           "程式堆疊追蹤",
			`(?i)/[a-z]:/.*`:                      "Windows檔案路徑",
			`(?i)/home/|/var/|/usr/`:              "Linux檔案路徑",
		}

		for pattern, description := range sensitivePatterns {
			if matched, _ := regexp.MatchString(pattern, bodyStr); matched {
				results = append(results, WeakConfigResult{
					ConfigType:     "information_disclosure",
					Vulnerable:     true,
					Issue:          "Sensitive information in error pages",
					Details:        fmt.Sprintf("錯誤頁面洩露敏感資訊: %s", description),
					Risk:           "LOW",
					Evidence:       map[string]string{"pattern": pattern},
					Recommendation: "配置自定義錯誤頁面，避免洩露技術細節",
				})
			}
		}
	}

	return results
}

// 輔助方法

func (w *WeakConfigTester) testTLSVersion(targetURL *url.URL, version uint16) bool {
	config := &tls.Config{
		InsecureSkipVerify: true,
		MinVersion:         version,
		MaxVersion:         version,
	}

	transport := &http.Transport{TLSClientConfig: config}
	client := &http.Client{Transport: transport, Timeout: 5 * time.Second}

	_, err := client.Get(targetURL.String())
	return err == nil
}

func (w *WeakConfigTester) getTLSVersionName(version uint16) string {
	switch version {
	case tls.VersionSSL30:
		return "SSL 3.0"
	case tls.VersionTLS10:
		return "TLS 1.0"
	case tls.VersionTLS11:
		return "TLS 1.1"
	case tls.VersionTLS12:
		return "TLS 1.2"
	case tls.VersionTLS13:
		return "TLS 1.3"
	default:
		return "Unknown"
	}
}

func (w *WeakConfigTester) testCertificateIssues(targetURL *url.URL) WeakConfigResult {
	// 使用嚴格的TLS配置檢測證書問題
	config := &tls.Config{InsecureSkipVerify: false}
	transport := &http.Transport{TLSClientConfig: config}
	client := &http.Client{Transport: transport, Timeout: 5 * time.Second}

	_, err := client.Get(targetURL.String())
	if err != nil {
		if strings.Contains(err.Error(), "certificate") {
			return WeakConfigResult{
				ConfigType:     "certificate",
				Vulnerable:     true,
				Issue:          "Certificate validation failed",
				Details:        fmt.Sprintf("SSL/TLS證書驗證失敗: %s", err.Error()),
				Risk:           "HIGH",
				Evidence:       map[string]string{"error": err.Error()},
				Recommendation: "使用有效的SSL/TLS證書，確保證書鏈完整",
			}
		}
	}

	return WeakConfigResult{Vulnerable: false}
}

func (w *WeakConfigTester) checkEndpointExists(url string) bool {
	resp, err := w.client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode != 404
}

func (w *WeakConfigTester) testCredential(loginURL, username, password string) bool {
	// 嘗試表單認證
	if w.testFormAuth(loginURL, username, password) {
		return true
	}

	// 嘗試基本認證
	if w.testBasicAuth(loginURL, username, password) {
		return true
	}

	return false
}

func (w *WeakConfigTester) testFormAuth(loginURL, username, password string) bool {
	// 簡化的表單認證測試 - 實際應用中需要更複雜的邏輯
	data := url.Values{}
	data.Set("username", username)
	data.Set("password", password)
	data.Set("user", username)
	data.Set("pass", password)
	data.Set("email", username)

	resp, err := w.client.PostForm(loginURL, data)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	// 檢查響應是否表示成功登入
	body, _ := io.ReadAll(resp.Body)
	bodyStr := strings.ToLower(string(body))

	successIndicators := []string{"dashboard", "welcome", "success", "admin panel"}
	failureIndicators := []string{"invalid", "incorrect", "failed", "error"}

	hasSuccess := false
	hasFailure := false

	for _, indicator := range successIndicators {
		if strings.Contains(bodyStr, indicator) {
			hasSuccess = true
			break
		}
	}

	for _, indicator := range failureIndicators {
		if strings.Contains(bodyStr, indicator) {
			hasFailure = true
			break
		}
	}

	return hasSuccess && !hasFailure
}

func (w *WeakConfigTester) testBasicAuth(loginURL, username, password string) bool {
	req, err := http.NewRequest("GET", loginURL, nil)
	if err != nil {
		return false
	}

	req.SetBasicAuth(username, password)
	resp, err := w.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == 200
}

func (w *WeakConfigTester) isServerHeaderVulnerable(serverHeader string) bool {
	// 檢查是否包含版本資訊
	versionPatterns := []string{
		`\d+\.\d+`,      // 數字版本 如 2.4
		`\d+\.\d+\.\d+`, // 詳細版本 如 2.4.41
		`/\d+`,          // 斜線後數字 如 /7
	}

	for _, pattern := range versionPatterns {
		if matched, _ := regexp.MatchString(pattern, serverHeader); matched {
			return true
		}
	}

	return false
}

func (w *WeakConfigTester) hasDirectoryListing(url string) bool {
	resp, err := w.client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return false
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false
	}

	bodyStr := strings.ToLower(string(body))

	// 檢查目錄清單的特徵
	indicators := []string{
		"index of",
		"directory listing",
		"parent directory",
		"<pre>",
		"[dir]",
		"folder.gif",
	}

	for _, indicator := range indicators {
		if strings.Contains(bodyStr, indicator) {
			return true
		}
	}

	return false
}

func (w *WeakConfigTester) isDebugEndpointExposed(url string) bool {
	resp, err := w.client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return false
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false
	}

	bodyStr := strings.ToLower(string(body))

	// 檢查調試端點的特徵
	debugIndicators := []string{
		"debug",
		"actuator",
		"health",
		"metrics",
		"phpinfo",
		"server-status",
		"configuration",
		"environment",
	}

	for _, indicator := range debugIndicators {
		if strings.Contains(bodyStr, indicator) {
			return true
		}
	}

	return false
}

func (w *WeakConfigTester) createFinding(taskID, url string, result WeakConfigResult) schemas.FindingPayload {
	// 獲取當前時間
	now := time.Now()

	// 映射風險等級到嚴重程度
	severity := "MEDIUM"
	switch result.Risk {
	case "CRITICAL":
		severity = "CRITICAL"
	case "HIGH":
		severity = "HIGH"
	case "MEDIUM":
		severity = "MEDIUM"
	case "LOW":
		severity = "LOW"
	}

	// 建構證據JSON
	evidenceJSON, _ := json.Marshal(result.Evidence)
	evidenceStr := string(evidenceJSON)

	// HTTP方法
	method := "GET"

	// 構建描述
	description := fmt.Sprintf("弱配置檢測發現問題: %s. %s", result.Issue, result.Details)

	// 構建建議
	priority := "HIGH"
	steps := []string{
		result.Recommendation,
		"定期進行安全配置審查",
		"實施安全基線配置",
	}
	refs := []string{
		"https://owasp.org/www-project-top-ten/",
		"https://cwe.mitre.org/data/definitions/16.html",
	}

	return schemas.FindingPayload{
		FindingID: fmt.Sprintf("weak_config_%s_%d", taskID, now.Unix()),
		TaskID:    taskID,
		ScanID:    taskID,
		Vulnerability: schemas.Vulnerability{
			Name:        result.Issue,
			Severity:    severity,
			Confidence:  "HIGH",
			Description: &description,
		},
		Target: schemas.Target{
			URL:    url,
			Method: &method,
		},
		Evidence: &schemas.FindingEvidence{
			Proof:   &evidenceStr,
			Payload: &result.Details,
		},
		Impact: &schemas.FindingImpact{
			Description:     &description,
			BusinessImpact:  &result.Details,
			TechnicalImpact: &result.Details,
		},
		Recommendation: &schemas.FindingRecommendation{
			Fix:              &result.Recommendation,
			Priority:         &priority,
			RemediationSteps: steps,
			References:       refs,
		},
		CreatedAt: now,
		UpdatedAt: now,
	}
}
