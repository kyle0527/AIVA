package brute_force

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
	"go.uber.org/zap"
)

// BruteForceResult 暴力破解測試結果
type BruteForceResult struct {
	Success       bool   `json:"success"`
	Username      string `json:"username"`
	Password      string `json:"password"`
	ResponseTime  int64  `json:"response_time"`
	StatusCode    int    `json:"status_code"`
	ResponseBody  string `json:"response_body"`
	ContentLength int64  `json:"content_length"`
	IsSuccessful  bool   `json:"is_successful"`
	Vulnerable    bool   `json:"vulnerable"`
	AccountLocked bool   `json:"account_locked"`
	RateLimited   bool   `json:"rate_limited"`
	AttemptsCount int    `json:"attempts_count"`
}

// BruteForcer 暴力破解檢測器
type BruteForcer struct {
	logger *zap.Logger
	client *http.Client
}

// NewBruteForcer 建立暴力破解檢測器
func NewBruteForcer(logger *zap.Logger) *BruteForcer {
	client := &http.Client{
		Timeout: 15 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	return &BruteForcer{
		logger: logger,
		client: client,
	}
}

// Test 執行暴力破解測試
func (b *BruteForcer) Test(ctx context.Context, task *schemas.FunctionTaskPayload) ([]*schemas.FindingPayload, error) {
	var findings []*schemas.FindingPayload

	// 使用預設用戶名列表
	usernames := b.getDefaultUsernames()
	passwords := b.getDefaultPasswords()

	maxAttempts := 50
	rateDelay := time.Millisecond * 500

	b.logger.Info("Starting brute force test",
		zap.Int("usernames", len(usernames)),
		zap.Int("passwords", len(passwords)),
		zap.Int("max_attempts", maxAttempts))

	attempts := 0
	for _, username := range usernames {
		for _, password := range passwords {
			if attempts >= maxAttempts {
				b.logger.Warn("Reached max attempts limit")
				break
			}

			// 從任務目標獲取URL
			loginURL := task.Target.URL
			result, err := b.tryCredentials(ctx, loginURL, username, password)
			if err != nil {
				b.logger.Error("Login attempt failed", zap.Error(err))
				continue
			}

			attempts++
			time.Sleep(rateDelay)

			// 如果成功登入，記錄為漏洞
			if result.Vulnerable {
				finding := b.createFinding(task.TaskID, loginURL, result)
				findings = append(findings, &finding)
				b.logger.Warn("Weak credentials found",
					zap.String("username", username),
					zap.String("password", password))
			}
		}
	}

	b.logger.Info("Brute force test completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings", len(findings)))

	return findings, nil
}

// tryCredentials 嘗試憑證
func (b *BruteForcer) tryCredentials(ctx context.Context, loginURL, username, password string) (*BruteForceResult, error) {
	result := &BruteForceResult{
		Username:      username,
		Password:      password,
		AttemptsCount: 1,
	}

	start := time.Now()

	// 嘗試表單認證
	if b.testFormAuth(loginURL, username, password) {
		result.Success = true
		result.Vulnerable = true
		result.IsSuccessful = true
	} else if b.testBasicAuth(loginURL, username, password) {
		// 嘗試基本認證
		result.Success = true
		result.Vulnerable = true
		result.IsSuccessful = true
	}

	result.ResponseTime = time.Since(start).Milliseconds()
	return result, nil
}

// testFormAuth 測試表單認證
func (b *BruteForcer) testFormAuth(loginURL, username, password string) bool {
	data := fmt.Sprintf("username=%s&password=%s", username, password)
	resp, err := b.client.Post(loginURL, "application/x-www-form-urlencoded", strings.NewReader(data))
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	bodyStr := strings.ToLower(string(body))

	// 檢查成功指標
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

// testBasicAuth 測試基本認證
func (b *BruteForcer) testBasicAuth(loginURL, username, password string) bool {
	req, err := http.NewRequest("GET", loginURL, nil)
	if err != nil {
		return false
	}

	req.SetBasicAuth(username, password)
	resp, err := b.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == 200
}

// createFinding 創建發現結果 - 遵循aiva_common規範
func (b *BruteForcer) createFinding(taskID, loginURL string, result *BruteForceResult) schemas.FindingPayload {
	findingID := fmt.Sprintf("brute_force_%s_%d", taskID, time.Now().UnixMilli())

	// 根據OWASP A07:2021 - Identification and Authentication Failures
	cwePtr := "CWE-307" // Improper Restriction of Excessive Authentication Attempts
	methodStr := "POST"
	requestPtr := fmt.Sprintf("POST %s\nContent-Type: application/x-www-form-urlencoded\n\nusername=%s&password=%s",
		loginURL, result.Username, result.Password)
	responsePtr := "HTTP/1.1 200 OK - Authentication successful"
	payloadPtr := fmt.Sprintf("username=%s&password=%s", result.Username, result.Password)
	proofPtr := fmt.Sprintf("成功使用弱憑證 %s:%s 登入系統", result.Username, result.Password)
	descPtr := "系統存在弱認證憑證，攻擊者可透過暴力破解獲得未授權存取權限，違反OWASP A07:2021安全標準"
	businessPtr := "未授權存取可能導致敏感資料洩漏、系統入侵或業務中斷"
	techPtr := "身份驗證繞過，可導致完整系統控制權限"

	now := time.Now()

	return schemas.FindingPayload{
		FindingID: findingID,
		TaskID:    taskID,
		ScanID:    taskID, // 使用TaskID作為ScanID
		Status:    "confirmed",
		Vulnerability: schemas.Vulnerability{
			Name:        "Weak Authentication - Brute Force Vulnerability",
			CWE:         &cwePtr,
			Severity:    "CRITICAL",
			Confidence:  "FIRM",
			Description: &descPtr,
		},
		Target: schemas.Target{
			URL:    loginURL,
			Method: &methodStr,
			Headers: map[string]string{
				"Content-Type": "application/x-www-form-urlencoded",
			},
			Params: map[string]interface{}{
				"username": result.Username,
				"password": result.Password,
			},
		},
		Evidence: &schemas.FindingEvidence{
			Request:  &requestPtr,
			Response: &responsePtr,
			Payload:  &payloadPtr,
			Proof:    &proofPtr,
		},
		Impact: &schemas.FindingImpact{
			Description:     &descPtr,
			BusinessImpact:  &businessPtr,
			TechnicalImpact: &techPtr,
		},
		Recommendation: &schemas.FindingRecommendation{
			Fix: &[]string{
				"實施強密碼政策，要求複雜密碼",
				"啟用帳戶鎖定機制，防止暴力破解",
				"實施多因子認證(MFA)",
				"使用CAPTCHA防止自動化攻擊",
				"監控和記錄登入嘗試",
				"定期審核和更新預設憑證",
			}[0],
			Priority: &[]string{"HIGH"}[0],
			RemediationSteps: []string{
				"立即更改所有預設和弱密碼",
				"配置帳戶鎖定策略（如5次失敗嘗試後鎖定30分鐘）",
				"部署多因子認證系統",
				"實施登入監控和警報機制",
				"進行定期密碼審核",
			},
			References: []string{
				"https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
				"https://cwe.mitre.org/data/definitions/307.html",
				"https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
			},
		},
		Metadata: map[string]interface{}{
			"username":          result.Username,
			"password":          result.Password,
			"attempts_count":    result.AttemptsCount,
			"response_time_ms":  result.ResponseTime,
			"owasp_category":    "A07:2021-Identification_and_Authentication_Failures",
			"cwe_category":      "CWE-307",
			"attack_vector":     "Network",
			"attack_complexity": "Low",
		},
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// getDefaultUsernames 預設用戶名列表 - 基於OWASP測試指南
func (b *BruteForcer) getDefaultUsernames() []string {
	return []string{
		"admin", "administrator", "root", "user",
		"test", "guest", "demo", "manager",
		"sa", "support", "service", "operator",
		"tomcat", "jenkins", "postgres", "mysql",
	}
}

// getDefaultPasswords 預設密碼列表 - 基於常見弱密碼
func (b *BruteForcer) getDefaultPasswords() []string {
	return []string{
		"admin", "password", "123456", "12345678",
		"admin123", "password123", "root", "guest",
		"test", "demo", "", "changeme",
		"default", "qwerty", "letmein", "welcome",
	}
}
