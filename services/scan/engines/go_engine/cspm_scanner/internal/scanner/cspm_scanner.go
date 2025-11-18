package scanner

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"

	schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"
	"go.uber.org/zap"
)

// CSPMMisconfig 本地 CSPM 錯誤配置類型
type CSPMMisconfig struct {
	ID          string
	Title       string
	Description string
	Severity    string
	Resolution  string
	FilePath    string
	ResourceID  string
}

// CSPMScanner 雲端安全態勢管理掃描器
type CSPMScanner struct {
	logger *zap.Logger
}

// NewCSPMScanner 建立 CSPM 掃描器
func NewCSPMScanner(logger *zap.Logger) *CSPMScanner {
	return &CSPMScanner{
		logger: logger,
	}
}

// Scan 執行 CSPM 掃描
func (s *CSPMScanner) Scan(ctx context.Context, task *schemas.ScanTaskPayload) ([]*schemas.FindingPayload, error) {
	var findings []*schemas.FindingPayload

	// 從 task metadata 或 URL 中提取 provider 信息
	provider := "generic"
	// 可以從 URL scheme 提取，例如 "aws://..." 或從 ScanTaskPayload 的 RepositoryInfo 中提取

	s.logger.Info("Starting CSMP scan",
		zap.String("provider", provider),
		zap.String("task_id", task.TaskID))

	// 根據提供商選擇掃描方法
	var misconfigs []CSPMMisconfig
	var err error

	switch provider {
	case "aws":
		misconfigs, err = s.scanAWS(ctx, task)
	case "azure":
		misconfigs, err = s.scanAzure(ctx, task)
	case "gcp":
		misconfigs, err = s.scanGCP(ctx, task)
	case "kubernetes", "k8s":
		misconfigs, err = s.scanKubernetes(ctx, task)
	default:
		misconfigs, err = s.scanGeneric(ctx, task)
	}

	if err != nil {
		return nil, fmt.Errorf("scan failed: %w", err)
	}

	// 轉換為 FindingPayload
	scanID := fmt.Sprintf("scan_cspm_%d", time.Now().UnixNano())
	for _, misconfig := range misconfigs {
		finding := s.createFinding(task.TaskID, scanID, provider, misconfig)
		findings = append(findings, &finding)
	}

	return findings, nil
}

// scanAWS 掃描 AWS 配置
func (s *CSPMScanner) scanAWS(ctx context.Context, task *schemas.ScanTaskPayload) ([]CSPMMisconfig, error) {
	return s.scanWithTrivy(ctx, task, "aws")
}

// scanAzure 掃描 Azure 配置
func (s *CSPMScanner) scanAzure(ctx context.Context, task *schemas.ScanTaskPayload) ([]CSPMMisconfig, error) {
	return s.scanWithTrivy(ctx, task, "azure")
}

// scanGCP 掃描 GCP 配置
func (s *CSPMScanner) scanGCP(ctx context.Context, task *schemas.ScanTaskPayload) ([]CSPMMisconfig, error) {
	return s.scanWithTrivy(ctx, task, "gcp")
}

// scanKubernetes 掃描 Kubernetes 配置
func (s *CSPMScanner) scanKubernetes(ctx context.Context, task *schemas.ScanTaskPayload) ([]CSPMMisconfig, error) {
	// 從 Target.URL 中提取配置路徑
	configPath := task.Target.URL.(string)
	if configPath == "" {
		return nil, fmt.Errorf("kubernetes config path is required")
	}

	// 使用 Trivy 掃描 Kubernetes manifests
	cmd := exec.CommandContext(ctx, "trivy", "config", "--format", "json", configPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Trivy 在發現問題時會返回非零退出碼，這是正常的
		s.logger.Warn("Trivy returned non-zero exit code", zap.Error(err))
	}

	return s.parseTrivyOutput(output)
}

// scanGeneric 掃描通用配置文件
func (s *CSPMScanner) scanGeneric(ctx context.Context, task *schemas.ScanTaskPayload) ([]CSPMMisconfig, error) {
	// 從 Target.URL 中提取配置路徑
	configPath := task.Target.URL.(string)
	if configPath == "" {
		configPath = "."
	}

	cmd := exec.CommandContext(ctx, "trivy", "config", "--format", "json", configPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		s.logger.Warn("Trivy returned non-zero exit code", zap.Error(err))
	}

	return s.parseTrivyOutput(output)
}

// scanWithTrivy 使用 Trivy 掃描雲端配置
func (s *CSPMScanner) scanWithTrivy(ctx context.Context, _ *schemas.ScanTaskPayload, provider string) ([]CSPMMisconfig, error) {
	// Trivy 雲端掃描命令
	cmd := exec.CommandContext(ctx, "trivy", provider, "--format", "json")

	output, err := cmd.CombinedOutput()
	if err != nil {
		s.logger.Warn("Trivy cloud scan returned error", zap.Error(err))
	}

	return s.parseTrivyOutput(output)
}

// TrivyResult Trivy 輸出結構
type TrivyResult struct {
	Results []struct {
		Target     string           `json:"Target"`
		Misconfigs []TrivyMisconfig `json:"Misconfigurations"`
	} `json:"Results"`
}

// TrivyMisconfig Trivy 錯誤配置
type TrivyMisconfig struct {
	ID          string `json:"ID"`
	Title       string `json:"Title"`
	Description string `json:"Description"`
	Severity    string `json:"Severity"`
	Resolution  string `json:"Resolution"`
	PrimaryURL  string `json:"PrimaryURL"`
	Status      string `json:"Status"`
}

// parseTrivyOutput 解析 Trivy 輸出
func (s *CSPMScanner) parseTrivyOutput(output []byte) ([]CSPMMisconfig, error) {
	var result TrivyResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse trivy output: %w", err)
	}

	var misconfigs []CSPMMisconfig
	for _, res := range result.Results {
		for _, mc := range res.Misconfigs {
			// 只報告失敗的配置
			if mc.Status != "FAIL" {
				continue
			}

			misconfigs = append(misconfigs, CSPMMisconfig{
				ID:          mc.ID,
				Title:       mc.Title,
				Description: mc.Description,
				Severity:    mc.Severity,
				Resolution:  mc.Resolution,
				FilePath:    res.Target,
			})
		}
	}

	return misconfigs, nil
}

// stringPtr 返回字符串指標
func stringPtr(s string) *string {
	return &s
}

// createFinding 建立 Finding
func (s *CSPMScanner) createFinding(
	taskID string,
	scanID string,
	provider string,
	misconfig CSPMMisconfig,
) schemas.FindingPayload {
	findingID := fmt.Sprintf("finding_cspm_%d", time.Now().UnixNano())

	// 映射嚴重性
	severity := mapSeverity(misconfig.Severity)
	businessImpact := fmt.Sprintf("%s 級別的雲端配置錯誤可能導致安全風險", severity)

	return schemas.FindingPayload{
		FindingID: findingID,
		TaskID:    taskID,
		ScanID:    scanID,
		Status:    "confirmed",
		Vulnerability: schemas.Vulnerability{
			Name:        misconfig.Title,
			CWE:         stringPtr(misconfig.ID),
			Severity:    severity,
			Confidence:  "firm",
			Description: stringPtr(misconfig.Description),
		},
		Target: schemas.Target{
			URL: misconfig.FilePath,
		},
		Evidence: &schemas.FindingEvidence{
			Proof: stringPtr(fmt.Sprintf("Rule ID: %s, Provider: %s", misconfig.ID, provider)),
		},
		Impact: &schemas.FindingImpact{
			BusinessImpact: stringPtr(businessImpact),
		},
		Recommendation: &schemas.FindingRecommendation{
			Fix: stringPtr(misconfig.Resolution),
		},
		Metadata: map[string]interface{}{
			"provider":    provider,
			"rule_id":     misconfig.ID,
			"file_path":   misconfig.FilePath,
			"resource_id": misconfig.ResourceID,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// mapSeverity 映射嚴重性
func mapSeverity(trivySeverity string) string {
	switch trivySeverity {
	case "CRITICAL":
		return "CRITICAL"
	case "HIGH":
		return "HIGH"
	case "MEDIUM":
		return "MEDIUM"
	case "LOW":
		return "LOW"
	default:
		return "INFORMATIONAL"
	}
}
