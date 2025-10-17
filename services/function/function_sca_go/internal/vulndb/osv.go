package vulndb

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"
)

// OSVDatabase OSV (Open Source Vulnerabilities) 資料庫實現
type OSVDatabase struct {
	logger     *zap.Logger
	httpClient *http.Client
	apiURL     string
}

// NewOSVDatabase 建立 OSV 資料庫實例
func NewOSVDatabase(logger *zap.Logger) *OSVDatabase {
	return &OSVDatabase{
		logger: logger,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		apiURL: "https://api.osv.dev/v1",
	}
}

// QueryVulnerabilities 查詢漏洞
func (db *OSVDatabase) QueryVulnerabilities(
	ctx context.Context,
	name, version, ecosystem string,
) ([]Vulnerability, error) {
	// 轉換生態系統名稱為 OSV 格式
	osvEcosystem := mapEcosystemToOSV(ecosystem)

	// 構建查詢請求
	query := map[string]interface{}{
		"package": map[string]string{
			"name":      name,
			"ecosystem": osvEcosystem,
		},
		"version": version,
	}

	queryData, err := json.Marshal(query)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal query: %w", err)
	}

	// 發送 HTTP 請求
	req, err := http.NewRequestWithContext(ctx, "POST", db.apiURL+"/query", strings.NewReader(string(queryData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := db.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, body)
	}

	// 解析回應
	var osvResp struct {
		Vulns []struct {
			ID       string `json:"id"`
			Summary  string `json:"summary"`
			Details  string `json:"details"`
			Modified string `json:"modified"`
			Severity []struct {
				Type  string `json:"type"`
				Score string `json:"score"`
			} `json:"severity"`
			References []struct {
				Type string `json:"type"`
				URL  string `json:"url"`
			} `json:"references"`
		} `json:"vulns"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&osvResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// 轉換為內部格式
	vulns := make([]Vulnerability, 0, len(osvResp.Vulns))
	for _, v := range osvResp.Vulns {
		vuln := Vulnerability{
			ID:          v.ID,
			Description: v.Summary,
			Modified:    v.Modified,
		}

		// 提取 CVSS 和嚴重性
		for _, sev := range v.Severity {
			if sev.Type == "CVSS_V3" || sev.Type == "CVSS_V2" {
				vuln.Severity = parseCVSSSeverity(sev.Score)
				vuln.CVSS = parseCVSSScore(sev.Score)
				break
			}
		}

		if vuln.Severity == "" {
			vuln.Severity = "MEDIUM"
		}

		// 提取參考連結
		refs := make([]string, 0, len(v.References))
		for _, ref := range v.References {
			refs = append(refs, ref.URL)
		}
		vuln.References = refs

		vulns = append(vulns, vuln)
	}

	return vulns, nil
}

// Close 關閉資料庫
func (db *OSVDatabase) Close() error {
	// HTTP 客戶端不需要特別關閉
	return nil
}

// mapEcosystemToOSV 映射生態系統名稱到 OSV 格式
func mapEcosystemToOSV(ecosystem string) string {
	mapping := map[string]string{
		"nodejs": "npm",
		"python": "PyPI",
		"go":     "Go",
		"rust":   "crates.io",
		"php":    "Packagist",
		"ruby":   "RubyGems",
		"java":   "Maven",
		"dotnet": "NuGet",
	}

	if osv, ok := mapping[ecosystem]; ok {
		return osv
	}
	return ecosystem
}

// parseCVSSSeverity 解析 CVSS 嚴重性
func parseCVSSSeverity(cvssString string) string {
	// 從 CVSS 向量字串中提取分數
	// 例如: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
	// 或直接的分數: "9.8"

	// 簡化處理：嘗試提取數字
	score := parseCVSSScore(cvssString)

	if score >= 9.0 {
		return "CRITICAL"
	} else if score >= 7.0 {
		return "HIGH"
	} else if score >= 4.0 {
		return "MEDIUM"
	} else if score > 0 {
		return "LOW"
	}

	return "MEDIUM"
}

// parseCVSSScore 解析 CVSS 分數
func parseCVSSScore(cvssString string) float64 {
	// 簡化實現：嘗試從字串中提取數字
	// 實際應該完整解析 CVSS 向量

	parts := strings.Split(cvssString, "/")
	for _, part := range parts {
		// 嘗試解析為浮點數
		var score float64
		if _, err := fmt.Sscanf(part, "%f", &score); err == nil {
			if score >= 0 && score <= 10 {
				return score
			}
		}
	}

	return 0.0
}
