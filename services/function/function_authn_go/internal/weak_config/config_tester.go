package weak_config

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/kyle0527/aiva/services/function/function_authn_go/pkg/models"
	"go.uber.org/zap"
)

// WeakConfigTester 弱配置測試器
type WeakConfigTester struct {
	logger *zap.Logger
	client *http.Client
}

// NewWeakConfigTester 建立弱配置測試器
func NewWeakConfigTester(logger *zap.Logger) *WeakConfigTester {
	return &WeakConfigTester{
		logger: logger,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Test 執行弱配置測試
func (w *WeakConfigTester) Test(ctx context.Context, task models.FunctionTaskPayload) ([]interface{}, error) {
	var findings []interface{}

	w.logger.Info("Starting weak config test", zap.String("url", task.Target.URL))

	// 測試項目
	tests := []func(context.Context, models.FunctionTaskPayload) (*models.WeakConfigResult, error){
		w.testPasswordPolicy,
		w.testSessionTimeout,
		w.testAccountLockout,
		w.testSecurityHeaders,
		w.testHTTPS,
	}

	for _, test := range tests {
		result, err := test(ctx, task)
		if err != nil {
			w.logger.Error("Test failed", zap.Error(err))
			continue
		}

		if result != nil && result.Vulnerable {
			finding := w.createFinding(task.TaskID, task.Target.URL, result)
			findings = append(findings, finding)
		}
	}

	return findings, nil
}

// testPasswordPolicy 測試密碼政策
func (w *WeakConfigTester) testPasswordPolicy(ctx context.Context, task models.FunctionTaskPayload) (*models.WeakConfigResult, error) {
	// 嘗試註冊弱密碼帳號
	weakPasswords := []string{"123", "12345", "password"}

	for _, pwd := range weakPasswords {
		payload := fmt.Sprintf(`{"username":"testuser_%d","password":"%s"}`, time.Now().Unix(), pwd)
		req, err := http.NewRequestWithContext(ctx, "POST", task.Target.URL+"/register", strings.NewReader(payload))
		if err != nil {
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		resp, err := w.client.Do(req)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)

		// 如果成功註冊，表示密碼政策過弱
		if resp.StatusCode == 200 || resp.StatusCode == 201 {
			if strings.Contains(bodyStr, "success") || strings.Contains(bodyStr, "created") {
				return &models.WeakConfigResult{
					Vulnerable:       true,
					ConfigType:       "password_policy",
					Details:          fmt.Sprintf("系統接受弱密碼: %s", pwd),
					ActualValue:      fmt.Sprintf("允許長度 %d 的簡單密碼", len(pwd)),
					RecommendedValue: "最少 8 個字元，包含大小寫字母、數字和特殊字符",
				}, nil
			}
		}
	}

	return nil, nil
}

// testSessionTimeout 測試 Session 超時
func (w *WeakConfigTester) testSessionTimeout(ctx context.Context, task models.FunctionTaskPayload) (*models.WeakConfigResult, error) {
	// 檢查 Set-Cookie 標頭
	req, err := http.NewRequestWithContext(ctx, "GET", task.Target.URL, nil)
	if err != nil {
		return nil, err
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	cookies := resp.Cookies()
	for _, cookie := range cookies {
		// 檢查 Session cookie 是否設置超時
		if strings.Contains(strings.ToLower(cookie.Name), "session") ||
			strings.Contains(strings.ToLower(cookie.Name), "token") {

			// 如果沒有設置 MaxAge 或 Expires，表示是 session cookie
			if cookie.MaxAge == 0 && cookie.Expires.IsZero() {
				continue // Session cookie 在瀏覽器關閉時過期，這是正常的
			}

			// 檢查超時時間是否過長（超過 24 小時）
			maxAge := cookie.MaxAge
			if maxAge > 86400 { // 24 hours
				return &models.WeakConfigResult{
					Vulnerable:       true,
					ConfigType:       "session_timeout",
					Details:          fmt.Sprintf("Session cookie '%s' 超時時間過長", cookie.Name),
					ActualValue:      fmt.Sprintf("%d 秒 (%d 小時)", maxAge, maxAge/3600),
					RecommendedValue: "建議設置為 1-2 小時",
				}, nil
			}
		}
	}

	return nil, nil
}

// testAccountLockout 測試帳號鎖定機制
func (w *WeakConfigTester) testAccountLockout(ctx context.Context, task models.FunctionTaskPayload) (*models.WeakConfigResult, error) {
	// 多次失敗登入嘗試
	username := "nonexistentuser"
	password := "wrongpassword"

	for i := 0; i < 10; i++ {
		payload := fmt.Sprintf(`{"username":"%s","password":"%s"}`, username, password)
		req, err := http.NewRequestWithContext(ctx, "POST", task.Target.URL, strings.NewReader(payload))
		if err != nil {
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		resp, err := w.client.Do(req)
		if err != nil {
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		bodyStr := string(body)

		// 檢查是否有帳號鎖定
		if resp.StatusCode == 423 ||
			strings.Contains(bodyStr, "locked") ||
			strings.Contains(bodyStr, "blocked") {
			// 有帳號鎖定機制，這是好的
			return nil, nil
		}

		time.Sleep(100 * time.Millisecond)
	}

	// 10 次失敗後仍未鎖定，表示缺乏帳號鎖定機制
	return &models.WeakConfigResult{
		Vulnerable:       true,
		ConfigType:       "account_lockout",
		Details:          "經過 10 次失敗登入後，帳號仍未被鎖定",
		ActualValue:      "無帳號鎖定機制",
		RecommendedValue: "在 5 次失敗登入後鎖定帳號 15-30 分鐘",
	}, nil
}

// testSecurityHeaders 測試安全標頭
func (w *WeakConfigTester) testSecurityHeaders(ctx context.Context, task models.FunctionTaskPayload) (*models.WeakConfigResult, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", task.Target.URL, nil)
	if err != nil {
		return nil, err
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	missingHeaders := []string{}

	securityHeaders := map[string]string{
		"X-Content-Type-Options":    "nosniff",
		"X-Frame-Options":           "DENY or SAMEORIGIN",
		"Strict-Transport-Security": "max-age=31536000",
		"Content-Security-Policy":   "configured",
	}

	for header := range securityHeaders {
		if resp.Header.Get(header) == "" {
			missingHeaders = append(missingHeaders, header)
		}
	}

	if len(missingHeaders) > 0 {
		return &models.WeakConfigResult{
			Vulnerable:       true,
			ConfigType:       "security_headers",
			Details:          fmt.Sprintf("缺少安全標頭: %s", strings.Join(missingHeaders, ", ")),
			ActualValue:      fmt.Sprintf("缺少 %d 個安全標頭", len(missingHeaders)),
			RecommendedValue: "配置所有推薦的安全標頭",
		}, nil
	}

	return nil, nil
}

// testHTTPS 測試 HTTPS 強制執行
func (w *WeakConfigTester) testHTTPS(ctx context.Context, task models.FunctionTaskPayload) (*models.WeakConfigResult, error) {
	if strings.HasPrefix(task.Target.URL, "http://") {
		return &models.WeakConfigResult{
			Vulnerable:       true,
			ConfigType:       "https_enforcement",
			Details:          "未強制使用 HTTPS 連線",
			ActualValue:      "允許 HTTP 連線",
			RecommendedValue: "強制使用 HTTPS 並配置 HSTS",
		}, nil
	}

	return nil, nil
}

// createFinding 建立 Finding
func (w *WeakConfigTester) createFinding(taskID, url string, result *models.WeakConfigResult) models.FindingPayload {
	findingID := fmt.Sprintf("finding_authn_wc_%d", time.Now().UnixNano())
	scanID := fmt.Sprintf("scan_authn_%d", time.Now().UnixNano())

	severityMap := map[string]string{
		"password_policy":   "HIGH",
		"session_timeout":   "MEDIUM",
		"account_lockout":   "HIGH",
		"security_headers":  "MEDIUM",
		"https_enforcement": "CRITICAL",
	}

	severity := severityMap[result.ConfigType]
	if severity == "" {
		severity = "MEDIUM"
	}

	return models.FindingPayload{
		FindingID: findingID,
		TaskID:    taskID,
		ScanID:    scanID,
		Status:    "CONFIRMED",
		Vulnerability: models.Vulnerability{
			Name:       fmt.Sprintf("Weak Configuration - %s", result.ConfigType),
			CWE:        "CWE-16",
			Severity:   severity,
			Confidence: "HIGH",
		},
		Target: models.FindingTarget{
			URL:      url,
			Endpoint: url,
			Method:   "GET",
		},
		Evidence: models.FindingEvidence{
			Request:        fmt.Sprintf("GET %s", url),
			Response:       result.Details,
			Payload:        "",
			ProofOfConcept: fmt.Sprintf("%s\n當前值: %s\n建議值: %s", result.Details, result.ActualValue, result.RecommendedValue),
			Details: map[string]string{
				"config_type":       result.ConfigType,
				"actual_value":      result.ActualValue,
				"recommended_value": result.RecommendedValue,
			},
		},
		Impact: models.FindingImpact{
			Description:    result.Details,
			BusinessImpact: "弱配置可能導致帳戶被入侵或資料洩漏",
			Exploitability: "高",
		},
		Recommendation: fmt.Sprintf("將 %s 設置為: %s", result.ConfigType, result.RecommendedValue),
	}
}
