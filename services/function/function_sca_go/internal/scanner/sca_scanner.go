package scanner

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/function/function_sca_go/pkg/models"
)

// SCAScanner 軟體組成分析掃描器
type SCAScanner struct {
	logger *zap.Logger
}

// NewSCAScanner 建立 SCA 掃描器
func NewSCAScanner(logger *zap.Logger) *SCAScanner {
	return &SCAScanner{
		logger: logger,
	}
}

// Scan 執行 SCA 掃描
func (s *SCAScanner) Scan(ctx context.Context, task models.FunctionTaskPayload) ([]models.FindingPayload, error) {
	s.logger.Info("Starting SCA scan", zap.String("task_id", task.TaskID))

	// 1. 下載或克隆目標專案（如果是 Git URL）
	projectPath, cleanup, err := s.prepareProject(ctx, task.Target.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare project: %w", err)
	}
	defer cleanup()

	// 2. 偵測套件管理檔案
	packageFiles, err := s.detectPackageFiles(projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to detect package files: %w", err)
	}

	s.logger.Info("Detected package files",
		zap.String("task_id", task.TaskID),
		zap.Int("count", len(packageFiles)),
	)

	// 3. 使用 OSV-Scanner 掃描
	vulnerabilities, err := s.scanWithOSV(ctx, projectPath)
	if err != nil {
		return nil, fmt.Errorf("OSV scan failed: %w", err)
	}

	// 4. 轉換為 FindingPayload
	findings := s.convertToFindings(vulnerabilities, task.TaskID, packageFiles)

	return findings, nil
}

// prepareProject 準備專案檔案
func (s *SCAScanner) prepareProject(ctx context.Context, targetURL string) (string, func(), error) {
	// 如果是本地路徑
	if _, err := os.Stat(targetURL); err == nil {
		return targetURL, func() {}, nil
	}

	// 如果是 Git URL，克隆到臨時目錄
	if strings.HasPrefix(targetURL, "http") || strings.HasPrefix(targetURL, "git@") {
		tmpDir, err := os.MkdirTemp("", "sca-scan-*")
		if err != nil {
			return "", nil, err
		}

		cleanup := func() {
			os.RemoveAll(tmpDir)
		}

		// 執行 git clone
		cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", targetURL, tmpDir)
		output, err := cmd.CombinedOutput()
		if err != nil {
			cleanup()
			return "", nil, fmt.Errorf("git clone failed: %w, output: %s", err, output)
		}

		return tmpDir, cleanup, nil
	}

	return "", nil, fmt.Errorf("unsupported target URL: %s", targetURL)
}

// detectPackageFiles 偵測套件管理檔案
func (s *SCAScanner) detectPackageFiles(projectPath string) ([]string, error) {
	packageFiles := []string{}

	// 支援的套件管理檔案
	patterns := []string{
		"package.json",      // Node.js
		"package-lock.json", // Node.js
		"yarn.lock",         // Node.js
		"pnpm-lock.yaml",    // Node.js
		"pyproject.toml",    // Python
		"requirements.txt",  // Python
		"Pipfile.lock",      // Python
		"poetry.lock",       // Python
		"go.mod",            // Go
		"go.sum",            // Go
		"Cargo.toml",        // Rust
		"Cargo.lock",        // Rust
		"pom.xml",           // Java Maven
		"build.gradle",      // Java Gradle
		"composer.json",     // PHP
		"composer.lock",     // PHP
		"Gemfile.lock",      // Ruby
	}

	// 遍歷專案目錄
	err := filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 跳過隱藏目錄和 node_modules
		if info.IsDir() {
			name := info.Name()
			if name == "node_modules" || name == ".git" || name == "vendor" || name == "target" {
				return filepath.SkipDir
			}
			return nil
		}

		// 檢查檔案名稱
		fileName := filepath.Base(path)
		for _, pattern := range patterns {
			if fileName == pattern {
				packageFiles = append(packageFiles, path)
				break
			}
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return packageFiles, nil
}

// OSVSeverity OSV 嚴重性結構
type OSVSeverity struct {
	Type  string `json:"type"`
	Score string `json:"score"`
}

// OSVReference OSV 參考連結結構
type OSVReference struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// OSVVulnerability OSV 漏洞結構
type OSVVulnerability struct {
	ID         string         `json:"id"`
	Summary    string         `json:"summary"`
	Details    string         `json:"details"`
	Aliases    []string       `json:"aliases"`
	Modified   string         `json:"modified"`
	Severity   []OSVSeverity  `json:"severity"`
	References []OSVReference `json:"references"`
}

// OSVResult OSV-Scanner 輸出結構
type OSVResult struct {
	Results []struct {
		Source   string `json:"source"`
		Packages []struct {
			Package struct {
				Name      string `json:"name"`
				Version   string `json:"version"`
				Ecosystem string `json:"ecosystem"`
			} `json:"package"`
			Vulnerabilities []OSVVulnerability `json:"vulnerabilities"`
		} `json:"packages"`
	} `json:"results"`
}

// scanWithOSV 使用 OSV-Scanner 執行掃描
func (s *SCAScanner) scanWithOSV(ctx context.Context, projectPath string) (*OSVResult, error) {
	// 執行 osv-scanner
	cmd := exec.CommandContext(ctx, "osv-scanner", "--format", "json", "--recursive", projectPath)

	output, _ := cmd.CombinedOutput()

	// OSV-Scanner 在發現漏洞時會返回非零退出碼，這是正常的
	// 只有在無法解析輸出時才視為錯誤
	var result OSVResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse OSV output: %w, output: %s", err, output)
	}

	return &result, nil
}

// convertToFindings 轉換 OSV 結果為 Finding
func (s *SCAScanner) convertToFindings(
	osvResult *OSVResult,
	taskID string,
	packageFiles []string,
) []models.FindingPayload {
	findings := []models.FindingPayload{}

	for _, result := range osvResult.Results {
		for _, pkg := range result.Packages {
			for _, vuln := range pkg.Vulnerabilities {
				finding := s.createFinding(
					taskID,
					pkg.Package.Name,
					pkg.Package.Version,
					pkg.Package.Ecosystem,
					vuln,
					result.Source,
				)
				findings = append(findings, finding)
			}
		}
	}

	return findings
}

// createFinding 建立單一 Finding
func (s *SCAScanner) createFinding(
	taskID string,
	packageName string,
	packageVersion string,
	ecosystem string,
	vuln OSVVulnerability,
	sourceFile string,
) models.FindingPayload {
	// 生成 Finding ID
	findingID := fmt.Sprintf("finding_sca_%d", time.Now().UnixNano())

	// 提取 CVE ID
	cveID := ""
	ghsaID := ""
	for _, alias := range vuln.Aliases {
		if strings.HasPrefix(alias, "CVE-") {
			cveID = alias
		} else if strings.HasPrefix(alias, "GHSA-") {
			ghsaID = alias
		}
	}
	if cveID == "" && strings.HasPrefix(vuln.ID, "CVE-") {
		cveID = vuln.ID
	}
	if ghsaID == "" && strings.HasPrefix(vuln.ID, "GHSA-") {
		ghsaID = vuln.ID
	}

	// 判斷嚴重性
	severity := determineSeverity(vuln.Severity)

	// 建立漏洞物件
	vulnerability := models.Vulnerability{
		Type:        "SCA",
		Name:        fmt.Sprintf("%s in %s@%s", vuln.ID, packageName, packageVersion),
		Description: vuln.Summary,
		CVEID:       cveID,
		GHSAID:      ghsaID,
		CWEIDs:      []string{}, // OSV 通常不提供 CWE
	}

	// 建立目標
	target := models.FindingTarget{
		URL:       sourceFile,
		Parameter: fmt.Sprintf("%s@%s", packageName, packageVersion),
	}

	// 提取修復建議（從 references 中尋找）
	fixVersion := ""
	references := []string{}
	for _, ref := range vuln.References {
		references = append(references, ref.URL)
	}

	// 建立證據
	evidence := models.FindingEvidence{
		Request:        sourceFile,
		Response:       fmt.Sprintf("Package: %s@%s\nEcosystem: %s", packageName, packageVersion, ecosystem),
		Payload:        vuln.ID,
		ProofOfConcept: vuln.Details,
	}

	// 建立影響
	impact := models.FindingImpact{
		Confidentiality: "HIGH",
		Integrity:       "HIGH",
		Availability:    "MEDIUM",
		BusinessImpact: fmt.Sprintf(
			"應用程式使用了存在已知漏洞的第三方套件 %s (版本 %s)，可能被攻擊者利用",
			packageName, packageVersion,
		),
	}

	// 建立修復建議
	remediation := fmt.Sprintf(
		"1. 更新套件 %s 到安全版本%s\n2. 檢查相依性樹，確認是否有其他套件依賴此漏洞版本\n3. 執行安全測試驗證修復",
		packageName,
		func() string {
			if fixVersion != "" {
				return fmt.Sprintf(" (%s 或更高版本)", fixVersion)
			}
			return ""
		}(),
	)

	recommendation := models.FindingRecommendation{
		Remediation: remediation,
		References:  references,
	}

	return models.FindingPayload{
		FindingID:      findingID,
		TaskID:         taskID,
		Vulnerability:  vulnerability,
		Severity:       severity,
		Confidence:     "FIRM",
		Target:         target,
		Evidence:       evidence,
		Impact:         impact,
		Recommendation: recommendation,
		Tags:           []string{"SCA", "Dependency", ecosystem, vuln.ID},
	}
}

// determineSeverity 判斷嚴重性
func determineSeverity(severities []OSVSeverity) string {
	// 預設為 MEDIUM
	if len(severities) == 0 {
		return "MEDIUM"
	}

	// 解析 CVSS 分數
	for _, sev := range severities {
		if sev.Type == "CVSS_V3" || sev.Type == "CVSS_V2" {
			score := sev.Score
			// 簡單解析（實際應該解析完整的 CVSS 向量）
			if strings.Contains(score, "CRITICAL") || strings.Contains(score, "9.") || strings.Contains(score, "10.") {
				return "CRITICAL"
			}
			if strings.Contains(score, "HIGH") || strings.Contains(score, "7.") || strings.Contains(score, "8.") {
				return "HIGH"
			}
			if strings.Contains(score, "MEDIUM") || strings.Contains(score, "4.") || strings.Contains(score, "5.") || strings.Contains(score, "6.") {
				return "MEDIUM"
			}
			return "LOW"
		}
	}

	return "MEDIUM"
}
