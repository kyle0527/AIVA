// Package fuzzer - Parameter Fuzzing Engine
// OWASP A03:2021 - Injection
// OWASP A06:2021 - Vulnerable and Outdated Components
//
// 實現 SQL Injection, Command Injection, SSTI 的 time-based 檢測
package fuzzer

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Finding 表示發現的漏洞
type Finding struct {
	Type       string  `json:"type"`
	Parameter  string  `json:"parameter"`
	Payload    string  `json:"payload"`
	Evidence   string  `json:"evidence"`
	Severity   string  `json:"severity"`
	Confidence string  `json:"confidence"`
	OWASP      string  `json:"owasp"`
	URL        string  `json:"url"`
}

// ParameterFuzzer 參數模糊測試引擎
type ParameterFuzzer struct {
	Client      *http.Client
	Payloads    map[string][]string
	Concurrency int
	Timeout     time.Duration
}

// NewParameterFuzzer 創建新的 Fuzzer 實例
func NewParameterFuzzer(concurrency int, timeout time.Duration) *ParameterFuzzer {
	return &ParameterFuzzer{
		Client: &http.Client{
			Timeout: timeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				// 不跟隨重定向
				return http.ErrUseLastResponse
			},
		},
		Payloads:    loadPayloads(),
		Concurrency: concurrency,
		Timeout:     timeout,
	}
}

// loadPayloads 加載各種類型的 payloads
func loadPayloads() map[string][]string {
	return map[string][]string{
		"sql": {
			"' AND SLEEP(5)--",
			"' OR SLEEP(5)--",
			"1' WAITFOR DELAY '00:00:05'--",
			"'; SELECT PG_SLEEP(5)--",
			"1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
			"\" AND SLEEP(5)--",
		},
		"cmd": {
			"; sleep 5",
			"| sleep 5",
			"` sleep 5 `",
			"$( sleep 5 )",
			"& ping -c 5 127.0.0.1",
			"|| sleep 5",
		},
		"ssti": {
			"{{7*7}}",
			"${7*7}",
			"<%= 7*7 %>",
			"#{7*7}",
			"{{config}}",
			"${__import__('os').system('sleep 5')}",
		},
	}
}

// DetectSQLInjection 使用 time-based 檢測 SQL 注入
func (f *ParameterFuzzer) DetectSQLInjection(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	sqlPayloads := f.Payloads["sql"]
	
	for paramName, paramValue := range params {
		// 測試基準時間
		baseline := f.measureResponseTime(targetURL, params)
		
		for _, payload := range sqlPayloads {
			// 創建測試參數
			testParams := copyParams(params)
			testParams[paramName] = paramValue + payload
			
			// 測試響應時間
			start := time.Now()
			resp, err := f.makeRequest(targetURL, testParams)
			duration := time.Since(start)
			
			if err != nil {
				continue
			}
			defer resp.Body.Close()
			
			// 如果響應時間顯著增加（>4.5秒）
			if duration.Seconds() > baseline.Seconds()+4.5 {
				evidence := fmt.Sprintf("Baseline: %.2fs, Test: %.2fs, Delta: +%.2fs", 
					baseline.Seconds(), duration.Seconds(), duration.Seconds()-baseline.Seconds())
				
				findings = append(findings, Finding{
					Type:       "SQL Injection (Time-based)",
					Parameter:  paramName,
					Payload:    payload,
					Evidence:   evidence,
					Severity:   "High",
					Confidence: "Firm",
					OWASP:      "A03:2021",
					URL:        targetURL,
				})
				
				// 找到一個就夠了，避免過度測試
				break
			}
		}
	}
	
	return findings
}

// DetectCommandInjection 檢測命令注入 (OWASP A03)
func (f *ParameterFuzzer) DetectCommandInjection(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	cmdPayloads := f.Payloads["cmd"]
	baseline := f.measureResponseTime(targetURL, params)
	
	for paramName, paramValue := range params {
		for _, payload := range cmdPayloads {
			testParams := copyParams(params)
			testParams[paramName] = paramValue + payload
			
			start := time.Now()
			resp, err := f.makeRequest(targetURL, testParams)
			duration := time.Since(start)
			
			if err != nil {
				continue
			}
			resp.Body.Close()
			
			// Time-based 檢測
			if duration.Seconds() > baseline.Seconds()+4.5 {
				evidence := fmt.Sprintf("Command execution delay detected: %.2fs (baseline: %.2fs)", 
					duration.Seconds(), baseline.Seconds())
				
				findings = append(findings, Finding{
					Type:       "Command Injection (Time-based)",
					Parameter:  paramName,
					Payload:    payload,
					Evidence:   evidence,
					Severity:   "Critical",
					Confidence: "Firm",
					OWASP:      "A03:2021",
					URL:        targetURL,
				})
				
				break
			}
		}
	}
	
	return findings
}

// DetectSSTI 檢測服務端模板注入 (OWASP A03)
func (f *ParameterFuzzer) DetectSSTI(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	// SSTI Payloads 和期望輸出
	sstiTests := map[string]string{
		"{{7*7}}":      "49",      // Jinja2, Twig
		"${7*7}":       "49",      // FreeMarker, Velocity
		"<%= 7*7 %>":   "49",      // ERB
		"#{7*7}":       "49",      // Ruby
		"{{config}}":   "config",  // Jinja2 config leak
	}
	
	for paramName, paramValue := range params {
		for payload, expected := range sstiTests {
			testParams := copyParams(params)
			testParams[paramName] = paramValue + payload
			
			resp, err := f.makeRequest(targetURL, testParams)
			if err != nil {
				continue
			}
			
			body := readBody(resp)
			resp.Body.Close()
			
			// 檢測預期輸出
			if strings.Contains(body, expected) {
				findings = append(findings, Finding{
					Type:       "Server-Side Template Injection",
					Parameter:  paramName,
					Payload:    payload,
					Evidence:   fmt.Sprintf("Found expected output: %s", expected),
					Severity:   "Critical",
					Confidence: "Confirmed",
					OWASP:      "A03:2021",
					URL:        targetURL,
				})
				
				break
			}
		}
	}
	
	return findings
}

// DetectAll 執行所有類型的檢測
func (f *ParameterFuzzer) DetectAll(targetURL string, params map[string]string) []Finding {
	findings := []Finding{}
	
	// SQL Injection
	sqlFindings := f.DetectSQLInjection(targetURL, params)
	findings = append(findings, sqlFindings...)
	
	// Command Injection
	cmdFindings := f.DetectCommandInjection(targetURL, params)
	findings = append(findings, cmdFindings...)
	
	// SSTI
	sstiFindings := f.DetectSSTI(targetURL, params)
	findings = append(findings, sstiFindings...)
	
	return findings
}

// measureResponseTime 測量正常請求的響應時間（基準）
func (f *ParameterFuzzer) measureResponseTime(targetURL string, params map[string]string) time.Duration {
	start := time.Now()
	resp, err := f.makeRequest(targetURL, params)
	duration := time.Since(start)
	
	if err != nil {
		return duration
	}
	defer resp.Body.Close()
	
	return duration
}

// makeRequest 發送 HTTP 請求
func (f *ParameterFuzzer) makeRequest(targetURL string, params map[string]string) (*http.Response, error) {
	// 構建 URL
	u, err := url.Parse(targetURL)
	if err != nil {
		return nil, err
	}
	
	// 添加參數
	q := u.Query()
	for key, value := range params {
		q.Set(key, value)
	}
	u.RawQuery = q.Encode()
	
	// 發送請求
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	
	req.Header.Set("User-Agent", "AIVA-Go-Fuzzer/1.0")
	
	return f.Client.Do(req)
}

// 輔助函數

func copyParams(params map[string]string) map[string]string {
	copied := make(map[string]string)
	for k, v := range params {
		copied[k] = v
	}
	return copied
}

func readBody(resp *http.Response) string {
	if resp == nil || resp.Body == nil {
		return ""
	}
	
	buf := make([]byte, 8192) // 只讀取前 8KB
	n, _ := resp.Body.Read(buf)
	return string(buf[:n])
}
