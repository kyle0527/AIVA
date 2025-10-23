package detector

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// CloudMetadataScanner 用於檢測是否能通過 SSRF 訪問雲提供商的元數據服務
type CloudMetadataScanner struct {
	httpClient *http.Client
}

// MetadataEndpointInfo 存儲不同雲提供商的元數據端點信息
type MetadataEndpointInfo struct {
	URL          string            // 主要元數據 URL
	Headers      map[string]string // 特定的請求頭 (例如 AWS V2 需要)
	ExpectedHint string            // 響應中預期的關鍵字或模式
	Provider     string            // 雲提供商名稱
}

// KnownMetadataEndpoints 列出已知的元數據端點
var KnownMetadataEndpoints = []MetadataEndpointInfo{
	{
		URL:          "http://169.254.169.254/latest/meta-data/",
		Headers:      nil, // AWS IMDSv1
		ExpectedHint: "instance-id",
		Provider:     "AWS (IMDSv1)",
	},
	{
		URL:          "http://169.254.169.254/latest/api/token", // 需要先獲取 Token
		Headers:      map[string]string{"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, // AWS IMDSv2 Token 請求頭
		ExpectedHint: "", // Token 本身就是證據
		Provider:     "AWS (IMDSv2 Token)",
	},
	{
		URL:          "http://metadata.google.internal/computeMetadata/v1/?recursive=true", // GCP
		Headers:      map[string]string{"Metadata-Flavor": "Google"},
		ExpectedHint: `"instance"`, // GCP 響應是 JSON
		Provider:     "GCP",
	},
	{
		URL:          "http://169.254.169.254/metadata/instance?api-version=2021-02-01", // Azure
		Headers:      map[string]string{"Metadata": "true"},
		ExpectedHint: `"compute"`, // Azure 響應是 JSON
		Provider:     "Azure",
	},
	{
		URL:          "http://169.254.169.254/metadata/v1/", // DigitalOcean
		Headers:      nil,
		ExpectedHint: "droplet_id",
		Provider:     "DigitalOcean",
	},
	{
		URL:          "http://100.100.100.200/latest/meta-data/", // Alibaba Cloud
		Headers:      nil,
		ExpectedHint: "instance-id",
		Provider:     "Alibaba Cloud",
	},
	// 可以添加更多... Oracle Cloud, OpenStack 等
}

// NewCloudMetadataScanner 創建掃描器實例
func NewCloudMetadataScanner(timeout time.Duration) *CloudMetadataScanner {
	// 使用與 InternalServiceProbe 類似的超時設置，但可能需要更短，因為元數據服務響應通常很快
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: timeout / 3}).DialContext,
		TLSHandshakeTimeout:   timeout / 3,
		ResponseHeaderTimeout: timeout / 3,
		DisableKeepAlives:     true,
		ForceAttemptHTTP2:     false,
	}
	return &CloudMetadataScanner{
		httpClient: &http.Client{
			Timeout:   timeout,
			Transport: transport,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse // 不跟隨重定向
			},
		},
	}
}

// ScanForMetadata 嘗試訪問所有已知的元數據端點
// payloadURL 是觸發 SSRF 的原始 URL
// Returns: 發現的提供商列表, 錯誤
func (s *CloudMetadataScanner) ScanForMetadata(ctx context.Context, payloadURL string) ([]string, error) {
	var foundProviders []string
	var wg sync.WaitGroup
	var mu sync.Mutex // 保護 foundProviders

	fmt.Printf("Scanning for cloud metadata endpoints via SSRF payload: %s\n", payloadURL)

	// AWS IMDSv2 特殊處理：先嘗試獲取 Token
	awsV2Token := ""
	for _, endpoint := range KnownMetadataEndpoints {
		if endpoint.Provider == "AWS (IMDSv2 Token)" {
			token, err := s.fetchAwsV2Token(ctx, endpoint, payloadURL)
			if err != nil {
				fmt.Printf("Failed to fetch AWS IMDSv2 token via %s: %v\n", payloadURL, err)
			} else if token != "" {
				awsV2Token = token
				mu.Lock()
				foundProviders = append(foundProviders, endpoint.Provider) // 記錄 Token 獲取成功
				mu.Unlock()
				fmt.Printf("Successfully fetched AWS IMDSv2 token via %s\n", payloadURL)
			}
			break // 只嘗試一次獲取 Token
		}
	}

	for _, endpoint := range KnownMetadataEndpoints {
		// 跳過 AWS V2 Token 端點，因為已處理
		if endpoint.Provider == "AWS (IMDSv2 Token)" {
			continue
		}

		wg.Add(1)
		go func(ep MetadataEndpointInfo) {
			defer wg.Done()
			// 為每個 goroutine 創建獨立的子 context
			probeCtx, cancel := context.WithTimeout(ctx, s.httpClient.Timeout)
			defer cancel()

			headers := ep.Headers

			success, bodyHint, err := s.probeEndpoint(probeCtx, ep.URL, headers, payloadURL)
			if err != nil {
				// 通常只記錄非網絡錯誤 (超時、連接拒絕等是預期行為)
				if !strings.Contains(err.Error(), "Timeout") && !strings.Contains(err.Error(), "refused") {
					fmt.Printf("Error probing metadata endpoint %s (%s): %v\n", ep.Provider, ep.URL, err)
				}
			} else if success {
				// 檢查響應體是否包含預期線索
				if ep.ExpectedHint == "" || strings.Contains(bodyHint, ep.ExpectedHint) {
					fmt.Printf("!!! Potential Cloud Metadata Exposure Found: %s via %s\n", ep.Provider, payloadURL)
					mu.Lock()
					// 避免重複添加
					providerFound := false
					for _, p := range foundProviders {
						if p == ep.Provider {
							providerFound = true
							break
						}
					}
					if !providerFound {
						foundProviders = append(foundProviders, ep.Provider)
					}
					mu.Unlock()
				} else {
					fmt.Printf("Reached metadata endpoint %s (%s) but hint '%s' not found in response.\n", ep.Provider, ep.URL, ep.ExpectedHint)
				}
			}
		}(endpoint)
	}

	wg.Wait()

	if len(foundProviders) > 0 {
		fmt.Printf("Found potential metadata exposure for providers: %v via %s\n", foundProviders, payloadURL)
	} else {
		fmt.Printf("No cloud metadata endpoints seem reachable via %s\n", payloadURL)
	}

	return foundProviders, nil
}

// probeEndpoint 嘗試訪問單個元數據端點
func (s *CloudMetadataScanner) probeEndpoint(ctx context.Context, url string, headers map[string]string, triggerURL string) (bool, string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false, "", fmt.Errorf("creating request failed: %w", err)
	}

	req.Header.Set("User-Agent", "AIVA-SSRF-Metadata-Probe/1.0")
	if headers != nil {
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	}

	fmt.Printf("Probing metadata URL: %s with headers: %v (triggered by %s)\n", url, req.Header, triggerURL)

	resp, err := s.httpClient.Do(req)
	if err != nil {
		// 超時或連接拒絕是常見的，不需要作為錯誤返回，返回 false 即可
		if netErr, ok := err.(net.Error); ok && (netErr.Timeout() || strings.Contains(err.Error(), "connection refused")) {
			fmt.Printf("Probe failed for %s: %v\n", url, err)
			return false, "", nil // Not reachable
		}
		return false, "", fmt.Errorf("http client error: %w", err) // Other errors
	}
	defer resp.Body.Close()

	// 收到任何 2xx 或 3xx 響應都認為是成功的觸達
	// 4xx 可能表示需要特定頭或路徑，但也表明服務存在
	// 5xx 通常是服務端錯誤
	if resp.StatusCode >= 200 && resp.StatusCode < 500 {
		// 讀取部分響應體用於線索檢查
		bodyBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1024)) // Read max 1KB
		bodyHint := ""
		if err == nil {
			bodyHint = string(bodyBytes)
		}
		fmt.Printf("Successful probe for %s (Status: %d). Body hint: %.100s...\n", url, resp.StatusCode, bodyHint)
		return true, bodyHint, nil
	}

	fmt.Printf("Probe for %s returned status %d\n", url, resp.StatusCode)
	return false, "", nil // Not considered a successful reach for metadata
}

// fetchAwsV2Token 專門用於獲取 AWS IMDSv2 的 Token
func (s *CloudMetadataScanner) fetchAwsV2Token(ctx context.Context, endpoint MetadataEndpointInfo, triggerURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "PUT", endpoint.URL, nil) // IMDSv2 Token 使用 PUT
	if err != nil {
		return "", fmt.Errorf("creating token request failed: %w", err)
	}

	req.Header.Set("User-Agent", "AIVA-SSRF-Metadata-Probe/1.0")
	if endpoint.Headers != nil {
		for key, value := range endpoint.Headers {
			req.Header.Set(key, value)
		}
	}

	fmt.Printf("Attempting to fetch AWS IMDSv2 token from %s (triggered by %s)\n", endpoint.URL, triggerURL)

	resp, err := s.httpClient.Do(req)
	if err != nil {
		// 超時或連接拒絕表明 IMDSv2 可能不存在或不可達
		if netErr, ok := err.(net.Error); ok && (netErr.Timeout() || strings.Contains(err.Error(), "connection refused")) {
			return "", nil // Not an error, just didn't get a token
		}
		return "", fmt.Errorf("token http client error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("reading token response body failed: %w", err)
		}
		token := string(bodyBytes)
		if len(token) > 10 { // 簡單檢查 Token 看起來是否有效
			return token, nil
		}
		return "", fmt.Errorf("received invalid token (too short)")
	}

	return "", fmt.Errorf("token request failed with status %d", resp.StatusCode)
}