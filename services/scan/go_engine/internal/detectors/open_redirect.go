// Package detectors - Open Redirection Detector
// OWASP A01:2021 - Broken Access Control
//
// 檢測開放重定向漏洞
package detectors

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// Finding 漏洞發現結果
type Finding struct {
	Type      string `json:"type"`
	Parameter string `json:"parameter"`
	Payload   string `json:"payload"`
	Evidence  string `json:"evidence"`
	Severity  string `json:"severity"`
	OWASP     string `json:"owasp"`
	URL       string `json:"url"`
}

// OpenRedirectDetector 開放重定向檢測器
type OpenRedirectDetector struct {
	Client *http.Client
}

// NewOpenRedirectDetector 創建新的檢測器
func NewOpenRedirectDetector() *OpenRedirectDetector {
	return &OpenRedirectDetector{
		Client: &http.Client{
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				// 不自動跟隨重定向，我們需要檢查 Location 頭
				return http.ErrUseLastResponse
			},
			Timeout: 10 * 1000000000, // 10 seconds
		},
	}
}

// Test 測試開放重定向漏洞
func (d *OpenRedirectDetector) Test(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	// 測試 payload 列表
	testURLs := []string{
		"https://evil.com",
		"//evil.com",
		"//evil.com/",
		"javascript:alert(1)",
		"data:text/html,<script>alert(1)</script>",
		"http://evil.com",
		"//google.com",              // 常見繞過測試
		"https://www.google.com",    // 白名單測試
		"/\\evil.com",               // 混合斜線繞過
		"https:evil.com",            // 缺少斜線
		"//\\evil.com",              // 多重繞過
	}
	
	for paramName := range params {
		for _, testURL := range testURLs {
			testParams := copyParams(params)
			testParams[paramName] = testURL
			
			resp, err := d.makeRequest(targetURL, testParams)
			if err != nil {
				continue
			}
			
			// 檢查是否發生重定向
			if d.isRedirect(resp.StatusCode) {
				location := resp.Header.Get("Location")
				
				// 檢查是否重定向到惡意域
				if d.isMaliciousRedirect(location, testURL) {
					findings = append(findings, Finding{
						Type:      "Open Redirection",
						Parameter: paramName,
						Payload:   testURL,
						Evidence:  fmt.Sprintf("Redirected to: %s (Status: %d)", location, resp.StatusCode),
						Severity:  "Medium",
						OWASP:     "A01:2021",
						URL:       targetURL,
					})
					
					// 找到一個就夠了
					break
				}
			}
			
			resp.Body.Close()
		}
	}
	
	return findings
}

// TestWithReferrer 使用 Referrer 頭測試
func (d *OpenRedirectDetector) TestWithReferrer(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	// 某些應用程序使用 Referer 進行重定向
	refererPayloads := []string{
		"https://evil.com",
		"https://attacker.com/phishing",
	}
	
	for _, referer := range refererPayloads {
		req, err := http.NewRequest("GET", targetURL, nil)
		if err != nil {
			continue
		}
		
		req.Header.Set("Referer", referer)
		
		resp, err := d.Client.Do(req)
		if err != nil {
			continue
		}
		
		if d.isRedirect(resp.StatusCode) {
			location := resp.Header.Get("Location")
			if strings.Contains(location, "evil.com") || strings.Contains(location, "attacker.com") {
				findings = append(findings, Finding{
					Type:      "Open Redirection (Referer-based)",
					Parameter: "Referer",
					Payload:   referer,
					Evidence:  fmt.Sprintf("Referer-based redirect to: %s", location),
					Severity:  "Medium",
					OWASP:     "A01:2021",
					URL:       targetURL,
				})
			}
		}
		
		resp.Body.Close()
	}
	
	return findings
}

// TestCommonParams 測試常見的重定向參數
func (d *OpenRedirectDetector) TestCommonParams(targetURL string) []Finding {
	findings := []Finding{}
	
	// 常見的重定向參數名
	commonRedirectParams := []string{
		"url",
		"redirect",
		"return",
		"next",
		"continue",
		"callback",
		"returnUrl",
		"redirect_uri",
		"goto",
		"destination",
		"redir",
		"out",
		"view",
		"to",
	}
	
	for _, paramName := range commonRedirectParams {
		params := map[string]string{
			paramName: "https://evil.com",
		}
		
		results := d.Test(targetURL, params)
		findings = append(findings, results...)
	}
	
	return findings
}

// isRedirect 檢查狀態碼是否為重定向
func (d *OpenRedirectDetector) isRedirect(statusCode int) bool {
	return statusCode >= 300 && statusCode < 400
}

// isMaliciousRedirect 檢查是否重定向到惡意域
func (d *OpenRedirectDetector) isMaliciousRedirect(location, payload string) bool {
	if location == "" {
		return false
	}
	
	// 檢查是否包含測試域名
	maliciousDomains := []string{
		"evil.com",
		"attacker.com",
		"google.com", // 用於繞過檢測測試
	}
	
	locationLower := strings.ToLower(location)
	
	for _, domain := range maliciousDomains {
		if strings.Contains(locationLower, domain) {
			return true
		}
	}
	
	// 檢查 javascript: 和 data: URI
	if strings.HasPrefix(locationLower, "javascript:") || 
	   strings.HasPrefix(locationLower, "data:") {
		return true
	}
	
	// 檢查是否重定向到絕對 URL（協議相對）
	if strings.HasPrefix(location, "//") && 
	   !strings.Contains(location, "localhost") &&
	   !strings.Contains(location, "127.0.0.1") {
		// 解析並檢查域名
		parsed, err := url.Parse("http:" + location)
		if err == nil && parsed.Host != "" {
			// 如果主機名不是原始域，則可能是開放重定向
			return true
		}
	}
	
	return false
}

// makeRequest 發送 HTTP 請求
func (d *OpenRedirectDetector) makeRequest(targetURL string, params map[string]string) (*http.Response, error) {
	u, err := url.Parse(targetURL)
	if err != nil {
		return nil, err
	}
	
	// 添加查詢參數
	q := u.Query()
	for key, value := range params {
		q.Set(key, value)
	}
	u.RawQuery = q.Encode()
	
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	
	req.Header.Set("User-Agent", "AIVA-Go-Detector/1.0")
	
	return d.Client.Do(req)
}

// 輔助函數
func copyParams(params map[string]string) map[string]string {
	copied := make(map[string]string)
	for k, v := range params {
		copied[k] = v
	}
	return copied
}
