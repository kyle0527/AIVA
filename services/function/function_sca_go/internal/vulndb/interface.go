package vulndb

import "context"

// VulnDatabase 漏洞資料庫介面
type VulnDatabase interface {
	// QueryVulnerabilities 查詢套件的漏洞
	QueryVulnerabilities(ctx context.Context, name, version, ecosystem string) ([]Vulnerability, error)

	// Close 關閉資料庫連線
	Close() error
}

// Vulnerability 漏洞資訊
type Vulnerability struct {
	ID          string   `json:"id"`
	Severity    string   `json:"severity"`
	Description string   `json:"description"`
	CVSS        float64  `json:"cvss,omitempty"`
	References  []string `json:"references,omitempty"`
	Published   string   `json:"published,omitempty"`
	Modified    string   `json:"modified,omitempty"`
}
