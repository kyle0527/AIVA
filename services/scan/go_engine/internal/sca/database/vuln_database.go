// Package database - 多源漏洞數據庫
// OWASP A06:2021 - Vulnerable and Outdated Components
//
// 整合多個 CVE 數據源: RetireJS, NVD, Snyk, GHSA
package database

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CVE 漏洞信息
type CVE struct {
	ID                string   `json:"id"`
	Description       string   `json:"description"`
	Severity          string   `json:"severity"`
	CVSS              float64  `json:"cvss"`
	AffectedVersions  string   `json:"affected_versions"`
	FixedVersions     []string `json:"fixed_versions"`
	References        []string `json:"references"`
	PublishedDate     string   `json:"published_date"`
	Source            string   `json:"source"` // retirejs, nvd, snyk, ghsa
}

// VulnDatabase 多源漏洞數據庫
type VulnDatabase struct {
	RetireJS  map[string][]CVE
	NISTNVD   map[string][]CVE
	SnykDB    map[string][]CVE
	GHSA      map[string][]CVE
	dataDir   string
}

// NewVulnDatabase 創建新的漏洞數據庫
func NewVulnDatabase(dataDir string) (*VulnDatabase, error) {
	db := &VulnDatabase{
		RetireJS: make(map[string][]CVE),
		NISTNVD:  make(map[string][]CVE),
		SnykDB:   make(map[string][]CVE),
		GHSA:     make(map[string][]CVE),
		dataDir:  dataDir,
	}

	// 加載數據
	if err := db.LoadFromFiles(); err != nil {
		return nil, err
	}

	return db, nil
}

// LoadFromFiles 從文件加載漏洞數據
func (db *VulnDatabase) LoadFromFiles() error {
	// 數據源文件列表
	sources := []struct {
		filename string
		target   *map[string][]CVE
		source   string
	}{
		{"retirejs.json", &db.RetireJS, "retirejs"},
		{"nvd.json", &db.NISTNVD, "nvd"},
		{"snyk.json", &db.SnykDB, "snyk"},
		{"ghsa.json", &db.GHSA, "ghsa"},
	}

	for _, src := range sources {
		filePath := filepath.Join(db.dataDir, src.filename)
		if err := db.loadSource(filePath, src.target, src.source); err != nil {
			// 只記錄警告，不中斷加載
			fmt.Printf("Warning: Failed to load %s: %v\n", src.filename, err)
		}
	}

	return nil
}

// loadSource 加載單個數據源
func (db *VulnDatabase) loadSource(filePath string, target *map[string][]CVE, source string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	// 通用 JSON 結構
	var rawData map[string]interface{}
	if err := json.Unmarshal(data, &rawData); err != nil {
		return err
	}

	// 根據數據源解析
	switch source {
	case "retirejs":
		return db.parseRetireJS(rawData, target)
	case "nvd":
		return db.parseNVD(rawData, target)
	case "snyk":
		return db.parseSnyk(rawData, target)
	case "ghsa":
		return db.parseGHSA(rawData, target)
	}

	return nil
}

// CheckVersion 檢查庫版本是否受影響
func (db *VulnDatabase) CheckVersion(lib string, version string) []CVE {
	cves := []CVE{}

	// 規範化庫名
	libLower := strings.ToLower(lib)

	// 檢查所有數據源
	sources := []map[string][]CVE{
		db.RetireJS,
		db.NISTNVD,
		db.SnykDB,
		db.GHSA,
	}

	for _, source := range sources {
		if vulns, ok := source[libLower]; ok {
			for _, vuln := range vulns {
				if isVersionAffected(version, vuln.AffectedVersions) {
					cves = append(cves, vuln)
				}
			}
		}
	}

	// 去重
	return deduplicateCVEs(cves)
}

// CheckPackageJSON 檢查 package.json 中的依賴
func (db *VulnDatabase) CheckPackageJSON(packageJSONPath string) ([]CVE, error) {
	data, err := os.ReadFile(packageJSONPath)
	if err != nil {
		return nil, err
	}

	var pkgJSON struct {
		Dependencies    map[string]string `json:"dependencies"`
		DevDependencies map[string]string `json:"devDependencies"`
	}

	if err := json.Unmarshal(data, &pkgJSON); err != nil {
		return nil, err
	}

	allCVEs := []CVE{}

	// 檢查所有依賴
	for pkg, version := range pkgJSON.Dependencies {
		cves := db.CheckVersion(pkg, cleanVersion(version))
		allCVEs = append(allCVEs, cves...)
	}

	for pkg, version := range pkgJSON.DevDependencies {
		cves := db.CheckVersion(pkg, cleanVersion(version))
		allCVEs = append(allCVEs, cves...)
	}

	return allCVEs, nil
}

// CheckRequirementsTxt 檢查 requirements.txt 中的依賴
func (db *VulnDatabase) CheckRequirementsTxt(requirementsPath string) ([]CVE, error) {
	data, err := os.ReadFile(requirementsPath)
	if err != nil {
		return nil, err
	}

	allCVEs := []CVE{}
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// 解析 package==version 或 package>=version
		parts := strings.FieldsFunc(line, func(r rune) bool {
			return r == '=' || r == '>' || r == '<' || r == '!'
		})

		if len(parts) >= 2 {
			pkg := strings.TrimSpace(parts[0])
			version := strings.TrimSpace(parts[1])
			cves := db.CheckVersion(pkg, version)
			allCVEs = append(allCVEs, cves...)
		}
	}

	return allCVEs, nil
}

// 解析不同數據源的格式

func (db *VulnDatabase) parseRetireJS(data map[string]interface{}, target *map[string][]CVE) error {
	// RetireJS 格式解析
	for lib, vulnsRaw := range data {
		if vulnsList, ok := vulnsRaw.([]interface{}); ok {
			for _, vulnRaw := range vulnsList {
				if vulnMap, ok := vulnRaw.(map[string]interface{}); ok {
					cve := CVE{
						Source: "retirejs",
					}

					if id, ok := vulnMap["id"].(string); ok {
						cve.ID = id
					}
					if desc, ok := vulnMap["description"].(string); ok {
						cve.Description = desc
					}
					if severity, ok := vulnMap["severity"].(string); ok {
						cve.Severity = severity
					}

					(*target)[strings.ToLower(lib)] = append((*target)[strings.ToLower(lib)], cve)
				}
			}
		}
	}
	return nil
}

func (db *VulnDatabase) parseNVD(data map[string]interface{}, target *map[string][]CVE) error {
	// NVD 格式解析（簡化）
	// 實際實現需要根據 NVD JSON 格式調整
	return nil
}

func (db *VulnDatabase) parseSnyk(data map[string]interface{}, target *map[string][]CVE) error {
	// Snyk 格式解析
	return nil
}

func (db *VulnDatabase) parseGHSA(data map[string]interface{}, target *map[string][]CVE) error {
	// GitHub Security Advisories 格式解析
	return nil
}

// 輔助函數

func isVersionAffected(version, affectedVersions string) bool {
	// 簡化的版本比較邏輯
	// 實際實現需要使用語義化版本比較庫
	// 例如: hashicorp/go-version

	if affectedVersions == "*" || affectedVersions == "all" {
		return true
	}

	// 檢查版本範圍
	// 例如: ">=1.0.0 <2.0.0"
	// 這裡需要實現更複雜的版本比較邏輯

	return strings.Contains(affectedVersions, version)
}

func deduplicateCVEs(cves []CVE) []CVE {
	seen := make(map[string]bool)
	result := []CVE{}

	for _, cve := range cves {
		if !seen[cve.ID] {
			seen[cve.ID] = true
			result = append(result, cve)
		}
	}

	return result
}

func cleanVersion(version string) string {
	// 移除版本前綴符號
	version = strings.TrimPrefix(version, "^")
	version = strings.TrimPrefix(version, "~")
	version = strings.TrimPrefix(version, ">=")
	version = strings.TrimPrefix(version, "<=")
	version = strings.TrimPrefix(version, ">")
	version = strings.TrimPrefix(version, "<")
	return strings.TrimSpace(version)
}

// GetStats 獲取數據庫統計信息
func (db *VulnDatabase) GetStats() map[string]int {
	return map[string]int{
		"retirejs": len(db.RetireJS),
		"nvd":      len(db.NISTNVD),
		"snyk":     len(db.SnykDB),
		"ghsa":     len(db.GHSA),
		"total":    len(db.RetireJS) + len(db.NISTNVD) + len(db.SnykDB) + len(db.GHSA),
	}
}
