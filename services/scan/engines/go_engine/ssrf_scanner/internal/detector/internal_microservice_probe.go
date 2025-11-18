package detector

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// InternalServiceProbe 用於探測內部微服務或已知端口
type InternalServiceProbe struct {
	httpClient *http.Client
}

// NewInternalServiceProbe 創建探測器實例
func NewInternalServiceProbe(timeout time.Duration) *InternalServiceProbe {
	// 創建一個自定義 transport，限制連接時間
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   timeout / 2, // 連接超時設置為總超時的一半
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   timeout / 2, // TLS 握手超時
		ResponseHeaderTimeout: timeout / 2, // 響應頭超時
		// 禁用 HTTP Keep-Alive，每次請求都是新的連接，避免受之前連接影響
		DisableKeepAlives: true,
		// 強制使用 HTTP/1.1，避免 HTTP/2 的複雜性影響探測
		ForceAttemptHTTP2: false,
	}

	return &InternalServiceProbe{
		httpClient: &http.Client{
			Timeout:   timeout, // 請求總超時
			Transport: transport,
			// 防止自動重定向，因為我們關心的是直接響應
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

// ProbeTarget 探測指定的內部目標 (IP:Port 或 Hostname:Port)
// target 通常是像 "10.0.0.5:8080", "localhost:9200", "kubernetes.default.svc:443" 這樣的格式
// payloadURL 是觸發 SSRF 的原始 URL，用於日誌記錄
func (p *InternalServiceProbe) ProbeTarget(ctx context.Context, target string, payloadURL string) (bool, string, error) {
	// 構造探測 URL，優先使用 http
	probeURL := fmt.Sprintf("http://%s", target)
	if strings.HasSuffix(target, ":443") || strings.Contains(target, "https") { // 簡單判斷 https
		probeURL = fmt.Sprintf("https://%s", target)
		// 注意：對於自簽名證書，可能需要配置 InsecureSkipVerify
		// p.httpClient.Transport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}

	fmt.Printf("Probing internal target: %s via SSRF payload: %s\n", probeURL, payloadURL)

	req, err := http.NewRequestWithContext(ctx, "GET", probeURL, nil)
	if err != nil {
		fmt.Printf("Failed to create request for %s: %v\n", probeURL, err)
		return false, "request_creation_failed", err
	}

	// 添加一些通用的 Header，模擬瀏覽器行為
	req.Header.Set("User-Agent", "AIVA-SSRF-Internal-Probe/1.0")
	req.Header.Set("Accept", "*/*")
	// 可以考慮添加 Accept-Language 等

	resp, err := p.httpClient.Do(req)
	if err != nil {
		// 檢查錯誤類型
		if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
			fmt.Printf("Timeout probing %s\n", probeURL)
			return false, "timeout", nil // 超時通常意味著端口不開放或無法訪問
		}
		// 處理連接被拒絕
		if strings.Contains(err.Error(), "connection refused") {
			fmt.Printf("Connection refused for %s\n", probeURL)
			return false, "connection_refused", nil // 連接被拒絕也表示端口不開放
		}
		// 其他網絡錯誤
		fmt.Printf("Network error probing %s: %v\n", probeURL, err)
		return false, "network_error", err // 其他錯誤可能需要記錄
	}
	defer resp.Body.Close()

	// 只要收到響應 (即使是 4xx 或 5xx)，就認為目標是可達的
	// 因為 SSRF 關心的是能否觸達內部服務，而不一定是成功的業務響應
	statusCode := resp.StatusCode
	statusText := http.StatusText(statusCode)
	fmt.Printf("Successfully reached internal target %s (via %s), Status: %d %s\n", target, payloadURL, statusCode, statusText)

	// 可以根據需要記錄更多響應信息，例如 Server header
	serverHeader := resp.Header.Get("Server")
	contentType := resp.Header.Get("Content-Type")
	details := fmt.Sprintf("Status: %d, Server: %s, Content-Type: %s", statusCode, serverHeader, contentType)

	return true, details, nil
}

// ProbeCommonPorts 探測常見的內部服務端口列表
func (p *InternalServiceProbe) ProbeCommonPorts(ctx context.Context, ip string, payloadURL string) map[int]string {
	commonPorts := []int{
		80, 443, // Web
		8080, 8443, 9090, // Alt Web / APIs
		22,    // SSH
		23,    // Telnet
		21,    // FTP
		25, 587, 465, // SMTP
		110, 995, // POP3
		143, 993, // IMAP
		3306,  // MySQL
		5432,  // PostgreSQL
		1433,  // MSSQL
		6379,  // Redis
		27017, // MongoDB
		9200, 9300, // Elasticsearch
		1521,  // Oracle DB
		5000,  // Docker Registry? Python apps?
		// 添加更多可能的端口...
	}

	results := make(map[int]string)
	var wg sync.WaitGroup
	var mu sync.Mutex // 保護 results map

	for _, port := range commonPorts {
		wg.Add(1)
		go func(pNum int) {
			defer wg.Done()
			target := fmt.Sprintf("%s:%d", ip, pNum)
			// 為每個 goroutine 創建獨立的子 context，以便更好地控制超時
			probeCtx, cancel := context.WithTimeout(ctx, p.httpClient.Timeout)
			defer cancel()

			reachable, details, err := p.ProbeTarget(probeCtx, target, payloadURL)
			if err != nil {
				// 只記錄非超時和非連接拒絕的錯誤
				if details != "timeout" && details != "connection_refused" {
					fmt.Printf("Error probing %s:%d: %v\n", ip, pNum, err)
				}
			}
			if reachable {
				mu.Lock()
				results[pNum] = details
				mu.Unlock()
			}
		}(port)
	}

	wg.Wait()
	return results
}