package brute_force

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

// BruteForcer 暴力破解測試器
type BruteForcer struct {
	logger *zap.Logger
	client *http.Client
}

// NewBruteForcer 建立暴力破解測試器
func NewBruteForcer(logger *zap.Logger) *BruteForcer {
	return &BruteForcer{
		logger: logger,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Test 執行暴力破解測試
func (b *BruteForcer) Test(ctx context.Context, task models.FunctionTaskPayload) ([]interface{}, error) {
	var findings []interface{}

	usernames := task.Options.Usernames
	if len(usernames) == 0 {
		usernames = b.getDefaultUsernames()
	}

	passwords := task.Options.Passwords
	if len(passwords) == 0 {
		passwords = b.getDefaultPasswords()
	}

	maxAttempts := task.Options.MaxAttempts
	if maxAttempts == 0 {
		maxAttempts = 50 // 預設最多 50 次嘗試
	}

	rateDelay := task.Options.RateLimitDelay
	if rateDelay == 0 {
		rateDelay = 100 // 預設 100ms 延遲
	}

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

			result, err := b.tryCredentials(ctx, task.Target.URL, username, password)
			if err != nil {
				b.logger.Error("Login attempt failed", zap.Error(err))
				continue
			}

			attempts++

			// 延遲以避免被封鎖
			time.Sleep(time.Duration(rateDelay) * time.Millisecond)

			// 如果成功登入，記錄為漏洞
			if result.Vulnerable {
				finding := b.createFinding(task.TaskID, task.Target.URL, result)
				findings = append(findings, finding)
				b.logger.Warn("Weak credentials found",
					zap.String("username", username),
					zap.String("password", password))
				break // 找到一組就停止該用戶名的測試
			}

			// 檢測帳號鎖定
			if result.AccountLocked {
				b.logger.Warn("Account locked detected", zap.String("username", username))
				break
			}

			// 檢測限速
			if result.RateLimited {
				b.logger.Warn("Rate limiting detected, slowing down")
				time.Sleep(5 * time.Second)
			}
		}
	}

	return findings, nil
}

// tryCredentials 嘗試登入憑證
func (b *BruteForcer) tryCredentials(ctx context.Context, loginURL, username, password string) (models.BruteForceResult, error) {
	// 構建登入請求
	payload := fmt.Sprintf(`{"username":"%s","password":"%s"}`, username, password)
	req, err := http.NewRequestWithContext(ctx, "POST", loginURL, strings.NewReader(payload))
	if err != nil {
		return models.BruteForceResult{}, err
	}

	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := b.client.Do(req)
	if err != nil {
		return models.BruteForceResult{}, err
	}
	defer resp.Body.Close()

	responseTime := time.Since(start).Milliseconds()
	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)

	result := models.BruteForceResult{
		Username:      username,
		Password:      password,
		AttemptsCount: 1,
		ResponseTime:  responseTime,
	}

	// 判斷是否成功登入
	if resp.StatusCode == 200 || resp.StatusCode == 302 {
		if strings.Contains(bodyStr, "token") ||
			strings.Contains(bodyStr, "success") ||
			strings.Contains(bodyStr, "dashboard") {
			result.Vulnerable = true
		}
	}

	// 檢測帳號鎖定
	if resp.StatusCode == 423 ||
		strings.Contains(bodyStr, "locked") ||
		strings.Contains(bodyStr, "blocked") {
		result.AccountLocked = true
	}

	// 檢測限速
	if resp.StatusCode == 429 ||
		strings.Contains(bodyStr, "too many") ||
		strings.Contains(bodyStr, "rate limit") {
		result.RateLimited = true
	}

	return result, nil
}

// createFinding 建立 Finding
func (b *BruteForcer) createFinding(taskID, loginURL string, result models.BruteForceResult) models.FindingPayload {
	findingID := fmt.Sprintf("finding_authn_bf_%d", time.Now().UnixNano())
	scanID := fmt.Sprintf("scan_authn_%d", time.Now().UnixNano())

	return models.FindingPayload{
		FindingID: findingID,
		TaskID:    taskID,
		ScanID:    scanID,
		Status:    "CONFIRMED",
		Vulnerability: models.Vulnerability{
			Name:       "Weak Authentication - Brute Force",
			CWE:        "CWE-307",
			Severity:   "CRITICAL",
			Confidence: "HIGH",
		},
		Target: models.FindingTarget{
			URL:      loginURL,
			Endpoint: loginURL,
			Method:   "POST",
		},
		Evidence: models.FindingEvidence{
			Request:        fmt.Sprintf("POST %s\n{\"username\":\"%s\",\"password\":\"%s\"}", loginURL, result.Username, result.Password),
			Response:       "200 OK - Authentication successful",
			Payload:        fmt.Sprintf("{\"username\":\"%s\",\"password\":\"%s\"}", result.Username, result.Password),
			ProofOfConcept: fmt.Sprintf("使用憑證 %s:%s 成功登入", result.Username, result.Password),
			Details: map[string]string{
				"username":      result.Username,
				"password":      result.Password,
				"attempts":      fmt.Sprintf("%d", result.AttemptsCount),
				"response_time": fmt.Sprintf("%dms", result.ResponseTime),
			},
		},
		Impact: models.FindingImpact{
			Description:    "系統存在弱憑證，攻擊者可透過暴力破解獲得存取權限",
			BusinessImpact: "攻擊者可完全控制帳戶，竊取敏感資料",
			Exploitability: "極高",
		},
		Recommendation: "1. 強制執行強密碼政策\n2. 實施帳戶鎖定機制\n3. 啟用多因素認證(MFA)\n4. 限制登入嘗試次數\n5. 實施 IP 限速",
	}
}

// getDefaultUsernames 預設用戶名列表
func (b *BruteForcer) getDefaultUsernames() []string {
	return []string{
		"admin", "administrator", "root", "user",
		"test", "guest", "demo", "manager",
	}
}

// getDefaultPasswords 預設密碼列表
func (b *BruteForcer) getDefaultPasswords() []string {
	return []string{
		"admin", "password", "123456", "admin123",
		"password123", "root", "test", "demo",
		"12345678", "qwerty", "letmein", "welcome",
	}
}
