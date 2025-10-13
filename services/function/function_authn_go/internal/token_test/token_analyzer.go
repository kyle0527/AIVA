package token_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/kyle0527/aiva/services/function/function_authn_go/pkg/models"
	"go.uber.org/zap"
)

// TokenAnalyzer Token 分析器
type TokenAnalyzer struct {
	logger *zap.Logger
}

// NewTokenAnalyzer 建立 Token 分析器
func NewTokenAnalyzer(logger *zap.Logger) *TokenAnalyzer {
	return &TokenAnalyzer{
		logger: logger,
	}
}

// Test 執行 Token 測試
func (t *TokenAnalyzer) Test(ctx context.Context, task models.FunctionTaskPayload) ([]interface{}, error) {
	var findings []interface{}

	jwtToken := task.Options.JWTToken
	if jwtToken != "" {
		t.logger.Info("Analyzing JWT token")
		results := t.analyzeJWT(jwtToken)
		for _, result := range results {
			if result.Vulnerable {
				finding := t.createFinding(task.TaskID, task.Target.URL, result)
				findings = append(findings, finding)
			}
		}
	}

	sessionToken := task.Options.SessionToken
	if sessionToken != "" {
		t.logger.Info("Analyzing session token")
		result := t.analyzeSessionToken(sessionToken)
		if result.Vulnerable {
			finding := t.createFinding(task.TaskID, task.Target.URL, result)
			findings = append(findings, finding)
		}
	}

	return findings, nil
}

// analyzeJWT 分析 JWT
func (t *TokenAnalyzer) analyzeJWT(tokenString string) []models.TokenTestResult {
	var results []models.TokenTestResult

	// 解析 JWT（不驗證簽名）
	token, _, err := new(jwt.Parser).ParseUnverified(tokenString, jwt.MapClaims{})
	if err != nil {
		t.logger.Error("Failed to parse JWT", zap.Error(err))
		return results
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return results
	}

	// 測試 1: 檢查簽名演算法
	if token.Method.Alg() == "none" {
		results = append(results, models.TokenTestResult{
			Vulnerable:     true,
			TokenType:      "jwt",
			Issue:          "None Algorithm",
			Details:        "JWT 使用 'none' 演算法，允許未簽名的 token",
			DecodedPayload: claims,
		})
	}

	// 測試 2: 檢查弱演算法 (HS256 with weak secret)
	if token.Method.Alg() == "HS256" {
		// 嘗試常見的弱密鑰
		weakSecrets := []string{"secret", "password", "123456", "admin", "test"}
		for _, secret := range weakSecrets {
			_, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
				return []byte(secret), nil
			})
			if err == nil {
				results = append(results, models.TokenTestResult{
					Vulnerable:     true,
					TokenType:      "jwt",
					Issue:          "Weak Secret",
					Details:        fmt.Sprintf("JWT 使用弱密鑰: '%s'", secret),
					DecodedPayload: claims,
				})
				break
			}
		}
	}

	// 測試 3: 檢查過期時間
	if exp, ok := claims["exp"].(float64); ok {
		expTime := time.Unix(int64(exp), 0)
		if time.Since(expTime) > 0 {
			results = append(results, models.TokenTestResult{
				Vulnerable:     true,
				TokenType:      "jwt",
				Issue:          "Expired Token",
				Details:        fmt.Sprintf("Token 已過期於 %s", expTime.Format(time.RFC3339)),
				DecodedPayload: claims,
			})
		} else if expTime.Sub(time.Now()) > 24*time.Hour {
			results = append(results, models.TokenTestResult{
				Vulnerable:     true,
				TokenType:      "jwt",
				Issue:          "Long Expiration",
				Details:        fmt.Sprintf("Token 過期時間過長: %s", expTime.Format(time.RFC3339)),
				DecodedPayload: claims,
			})
		}
	} else {
		// 沒有過期時間
		results = append(results, models.TokenTestResult{
			Vulnerable:     true,
			TokenType:      "jwt",
			Issue:          "No Expiration",
			Details:        "JWT 沒有設置過期時間 (exp claim)",
			DecodedPayload: claims,
		})
	}

	// 測試 4: 檢查敏感資訊洩漏
	sensitiveFields := []string{"password", "secret", "api_key", "private_key"}
	for _, field := range sensitiveFields {
		if _, ok := claims[field]; ok {
			results = append(results, models.TokenTestResult{
				Vulnerable:     true,
				TokenType:      "jwt",
				Issue:          "Sensitive Data Leak",
				Details:        fmt.Sprintf("JWT payload 包含敏感欄位: '%s'", field),
				DecodedPayload: claims,
			})
		}
	}

	// 測試 5: 檢查用戶權限提升
	if role, ok := claims["role"].(string); ok {
		if strings.ToLower(role) == "admin" || strings.ToLower(role) == "root" {
			results = append(results, models.TokenTestResult{
				Vulnerable:     true,
				TokenType:      "jwt",
				Issue:          "Privilege Escalation Risk",
				Details:        fmt.Sprintf("JWT 包含高權限角色: '%s'，可能存在權限提升風險", role),
				DecodedPayload: claims,
			})
		}
	}

	return results
}

// analyzeSessionToken 分析 Session Token
func (t *TokenAnalyzer) analyzeSessionToken(token string) models.TokenTestResult {
	// 檢查 token 長度
	if len(token) < 16 {
		return models.TokenTestResult{
			Vulnerable: true,
			TokenType:  "session",
			Issue:      "Weak Session Token",
			Details:    fmt.Sprintf("Session token 過短 (%d 字元)，容易被暴力破解", len(token)),
		}
	}

	// 檢查是否為純數字
	isNumeric := true
	for _, c := range token {
		if c < '0' || c > '9' {
			isNumeric = false
			break
		}
	}

	if isNumeric {
		return models.TokenTestResult{
			Vulnerable: true,
			TokenType:  "session",
			Issue:      "Predictable Session Token",
			Details:    "Session token 僅包含數字，容易被預測",
		}
	}

	// 嘗試 Base64 解碼
	if decoded, err := base64.StdEncoding.DecodeString(token); err == nil {
		var jsonData map[string]interface{}
		if json.Unmarshal(decoded, &jsonData) == nil {
			// Token 是 Base64 編碼的 JSON
			return models.TokenTestResult{
				Vulnerable:     true,
				TokenType:      "session",
				Issue:          "Unencrypted Session Data",
				Details:        "Session token 包含未加密的 JSON 資料",
				DecodedPayload: jsonData,
			}
		}
	}

	return models.TokenTestResult{
		Vulnerable: false,
		TokenType:  "session",
	}
}

// createFinding 建立 Finding
func (t *TokenAnalyzer) createFinding(taskID, url string, result models.TokenTestResult) models.FindingPayload {
	findingID := fmt.Sprintf("finding_authn_token_%d", time.Now().UnixNano())
	scanID := fmt.Sprintf("scan_authn_%d", time.Now().UnixNano())

	severityMap := map[string]string{
		"None Algorithm":            "CRITICAL",
		"Weak Secret":               "CRITICAL",
		"Expired Token":             "MEDIUM",
		"Long Expiration":           "MEDIUM",
		"No Expiration":             "HIGH",
		"Sensitive Data Leak":       "HIGH",
		"Privilege Escalation Risk": "HIGH",
		"Weak Session Token":        "HIGH",
		"Predictable Session Token": "HIGH",
		"Unencrypted Session Data":  "HIGH",
	}

	severity := severityMap[result.Issue]
	if severity == "" {
		severity = "MEDIUM"
	}

	payloadJSON, _ := json.Marshal(result.DecodedPayload)

	return models.FindingPayload{
		FindingID: findingID,
		TaskID:    taskID,
		ScanID:    scanID,
		Status:    "CONFIRMED",
		Vulnerability: models.Vulnerability{
			Name:       fmt.Sprintf("Token Vulnerability - %s", result.Issue),
			CWE:        "CWE-287",
			Severity:   severity,
			Confidence: "HIGH",
		},
		Target: models.FindingTarget{
			URL:      url,
			Endpoint: url,
			Method:   "GET",
		},
		Evidence: models.FindingEvidence{
			Request:        fmt.Sprintf("Analyzed %s token", result.TokenType),
			Response:       result.Details,
			Payload:        string(payloadJSON),
			ProofOfConcept: fmt.Sprintf("Token 類型: %s\n問題: %s\n詳情: %s", result.TokenType, result.Issue, result.Details),
		},
		Impact: models.FindingImpact{
			Description:    result.Details,
			BusinessImpact: "Token 漏洞可能導致身份竊取、未授權存取或權限提升",
			Exploitability: "高",
		},
		Recommendation: t.getRecommendation(result.Issue),
	}
}

// getRecommendation 獲取修復建議
func (t *TokenAnalyzer) getRecommendation(issue string) string {
	recommendations := map[string]string{
		"None Algorithm":            "禁用 'none' 演算法，強制使用 RS256 或 HS256",
		"Weak Secret":               "使用強密鑰 (至少 256 位)，並定期輪換",
		"Expired Token":             "實施 token 刷新機制，移除過期 token",
		"Long Expiration":           "縮短 token 有效期至 1-2 小時，實施刷新 token 機制",
		"No Expiration":             "為所有 JWT 設置合理的過期時間 (exp claim)",
		"Sensitive Data Leak":       "不要在 JWT payload 中存儲敏感資訊，使用 token ID 關聯伺服器端資料",
		"Privilege Escalation Risk": "驗證 JWT 中的角色聲明，不要僅依賴 token 內容",
		"Weak Session Token":        "使用至少 128 位的隨機 session token",
		"Predictable Session Token": "使用密碼學安全的隨機數生成器 (crypto/rand)",
		"Unencrypted Session Data":  "加密 session 資料或使用隨機 session ID",
	}

	if rec, ok := recommendations[issue]; ok {
		return rec
	}
	return "請參考 OWASP Authentication Cheat Sheet"
}
