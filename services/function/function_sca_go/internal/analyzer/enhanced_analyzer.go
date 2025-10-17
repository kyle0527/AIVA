package analyzer

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/function/function_sca_go/internal/vulndb"
)

// EnhancedSCAAnalyzer 增強型 SCA 分析器
type EnhancedSCAAnalyzer struct {
	analyzer       *DependencyAnalyzer
	vulnDB         vulndb.VulnDatabase
	logger         *zap.Logger
	config         *SCAConfig
	maxConcurrency int
	cache          *vulnCache
}

// vulnCache 漏洞快取
type vulnCache struct {
	mu    sync.RWMutex
	cache map[string][]vulndb.Vulnerability
}

// ScanResult 掃描結果
type ScanResult struct {
	Dependencies    []Dependency           `json:"dependencies"`
	Vulnerabilities []vulndb.Vulnerability `json:"vulnerabilities"`
	Statistics      ScanStatistics         `json:"statistics"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ScanStatistics 掃描統計
type ScanStatistics struct {
	TotalDeps         int                     `json:"total_dependencies"`
	VulnerableDeps    int                     `json:"vulnerable_dependencies"`
	TotalVulns        int                     `json:"total_vulnerabilities"`
	SeverityBreakdown map[string]int          `json:"severity_breakdown"`
	LanguageStats     map[string]LanguageStat `json:"language_stats"`
}

// LanguageStat 語言統計
type LanguageStat struct {
	TotalDeps      int `json:"total_dependencies"`
	VulnerableDeps int `json:"vulnerable_dependencies"`
	DirectDeps     int `json:"direct_dependencies"`
	DevDeps        int `json:"dev_dependencies"`
	IndirectDeps   int `json:"indirect_dependencies"`
}

// NewEnhancedSCAAnalyzer 建立增強型分析器
func NewEnhancedSCAAnalyzer(logger *zap.Logger, config *SCAConfig, vulnDB vulndb.VulnDatabase) *EnhancedSCAAnalyzer {
	if config == nil {
		config = &SCAConfig{
			SupportedLangs: []string{"nodejs", "python", "go", "rust", "java", "dotnet", "php", "ruby"},
			EnableDeepScan: false,
			CacheResults:   true,
		}
	}

	return &EnhancedSCAAnalyzer{
		analyzer:       NewDependencyAnalyzer(logger, config),
		vulnDB:         vulnDB,
		logger:         logger,
		config:         config,
		maxConcurrency: 10,
		cache: &vulnCache{
			cache: make(map[string][]vulndb.Vulnerability),
		},
	}
}

// ScanProject 掃描專案
func (esa *EnhancedSCAAnalyzer) ScanProject(ctx context.Context, projectPath string) (*ScanResult, error) {
	esa.logger.Info("Starting enhanced SCA scan",
		zap.String("path", projectPath),
		zap.Bool("deep_scan", esa.config.EnableDeepScan))

	// 第一階段: 相依性分析
	deps, err := esa.analyzer.AnalyzeProject(projectPath)
	if err != nil {
		return nil, fmt.Errorf("dependency analysis failed: %w", err)
	}

	// 過濾開發相依（可選）
	if !esa.config.EnableDeepScan {
		deps = filterDevDependencies(deps)
	}

	// 第二階段: 漏洞檢查
	vulnDeps, allVulns, err := esa.checkVulnerabilities(ctx, deps)
	if err != nil {
		// 即使漏洞檢查失敗，也返回部分結果
		esa.logger.Error("Vulnerability check failed", zap.Error(err))
	}

	// 第三階段: 生成統計與結果
	result := &ScanResult{
		Dependencies:    deps, // 使用更新後的 deps
		Vulnerabilities: allVulns,
		Statistics:      esa.generateStatistics(deps, vulnDeps, allVulns),
		Metadata: map[string]interface{}{
			"scan_time":    time.Now().Format(time.RFC3339),
			"project_path": projectPath,
			"deep_scan":    esa.config.EnableDeepScan,
		},
	}

	// 檢查是否超時或取消
	if ctx.Err() != nil {
		result.Metadata["scan_status"] = "partial"
		result.Metadata["error"] = ctx.Err().Error()
	} else {
		result.Metadata["scan_status"] = "complete"
	}

	return result, nil
}

// checkVulnerabilities 檢查漏洞
func (esa *EnhancedSCAAnalyzer) checkVulnerabilities(
	ctx context.Context,
	deps []Dependency,
) ([]Dependency, []vulndb.Vulnerability, error) {
	vulnDeps := []Dependency{}
	allVulns := []vulndb.Vulnerability{}

	if len(deps) == 0 {
		return vulnDeps, allVulns, nil
	}

	// 創建工作佇列
	jobs := make(chan depJob, len(deps))
	results := make(chan depResult, len(deps))

	// 啟動 worker pool
	var wg sync.WaitGroup
	for i := 0; i < esa.maxConcurrency; i++ {
		wg.Add(1)
		go esa.vulnerabilityWorker(ctx, jobs, results, &wg)
	}

	// 發送工作
	for i, dep := range deps {
		jobs <- depJob{index: i, dep: dep}
	}
	close(jobs)

	// 等待所有 worker 完成
	go func() {
		wg.Wait()
		close(results)
	}()

	// 收集結果
	for result := range results {
		if len(result.vulns) > 0 {
			// 更新原始 deps 列表中的漏洞資訊
			deps[result.index].Vulnerabilities = convertVulns(result.vulns)
			vulnDeps = append(vulnDeps, deps[result.index])
			allVulns = append(allVulns, result.vulns...)

			esa.logger.Info("Vulnerabilities found",
				zap.String("package", result.dep.Name),
				zap.String("version", result.dep.Version),
				zap.Int("vuln_count", len(result.vulns)))
		}

		// 每 100 個相依項記錄一次進度
		if (result.index+1)%100 == 0 {
			esa.logger.Info("Vulnerability check progress",
				zap.Int("checked", result.index+1),
				zap.Int("total", len(deps)))
		}
	}

	// 檢查是否取消或超時
	if ctx.Err() != nil {
		return vulnDeps, allVulns, ctx.Err()
	}

	return vulnDeps, allVulns, nil
}

// depJob 相依項工作
type depJob struct {
	index int
	dep   Dependency
}

// depResult 相依項結果
type depResult struct {
	index int
	dep   Dependency
	vulns []vulndb.Vulnerability
}

// vulnerabilityWorker 漏洞檢查 worker
func (esa *EnhancedSCAAnalyzer) vulnerabilityWorker(
	ctx context.Context,
	jobs <-chan depJob,
	results chan<- depResult,
	wg *sync.WaitGroup,
) {
	defer wg.Done()

	for job := range jobs {
		// 檢查是否取消
		select {
		case <-ctx.Done():
			return
		default:
		}

		// 檢查快取
		cacheKey := fmt.Sprintf("%s:%s:%s", job.dep.Language, job.dep.Name, job.dep.Version)
		if esa.config.CacheResults {
			if vulns := esa.cache.get(cacheKey); vulns != nil {
				results <- depResult{index: job.index, dep: job.dep, vulns: vulns}
				continue
			}
		}

		// 查詢漏洞
		vulns, err := esa.vulnDB.QueryVulnerabilities(ctx, job.dep.Name, job.dep.Version, job.dep.Language)
		if err != nil {
			esa.logger.Warn("Failed to query vulnerabilities",
				zap.String("package", job.dep.Name),
				zap.Error(err))
			results <- depResult{index: job.index, dep: job.dep, vulns: []vulndb.Vulnerability{}}
			continue
		}

		// 過濾嚴重性
		if esa.config.VulnSeverityMin != "" {
			vulns = filterBySeverity(vulns, esa.config.VulnSeverityMin)
		}

		// 儲存快取
		if esa.config.CacheResults {
			esa.cache.set(cacheKey, vulns)
		}

		results <- depResult{index: job.index, dep: job.dep, vulns: vulns}
	}
}

// generateStatistics 生成統計資料
func (esa *EnhancedSCAAnalyzer) generateStatistics(
	allDeps []Dependency,
	vulnDeps []Dependency,
	allVulns []vulndb.Vulnerability,
) ScanStatistics {
	stats := ScanStatistics{
		TotalDeps:         len(allDeps),
		VulnerableDeps:    len(vulnDeps),
		TotalVulns:        len(allVulns),
		SeverityBreakdown: make(map[string]int),
		LanguageStats:     generateLanguageStats(allDeps),
	}

	// 統計嚴重性分佈
	for _, vuln := range allVulns {
		stats.SeverityBreakdown[vuln.Severity]++
	}

	return stats
}

// convertVulns 轉換漏洞格式
func convertVulns(vulns []vulndb.Vulnerability) []Vulnerability {
	result := make([]Vulnerability, len(vulns))
	for i, v := range vulns {
		result[i] = Vulnerability{
			ID:          v.ID,
			Severity:    v.Severity,
			Description: v.Description,
			CVSS:        v.CVSS,
			References:  v.References,
		}
	}
	return result
}

// filterDevDependencies 過濾開發相依
func filterDevDependencies(deps []Dependency) []Dependency {
	filtered := []Dependency{}
	for _, dep := range deps {
		if dep.Type != "dev" {
			filtered = append(filtered, dep)
		}
	}
	return filtered
}

// filterBySeverity 根據嚴重性過濾
func filterBySeverity(vulns []vulndb.Vulnerability, minSeverity string) []vulndb.Vulnerability {
	severityOrder := map[string]int{
		"LOW":      1,
		"MEDIUM":   2,
		"HIGH":     3,
		"CRITICAL": 4,
	}

	minLevel := severityOrder[minSeverity]
	if minLevel == 0 {
		return vulns
	}

	filtered := []vulndb.Vulnerability{}
	for _, vuln := range vulns {
		if severityOrder[vuln.Severity] >= minLevel {
			filtered = append(filtered, vuln)
		}
	}

	return filtered
}

// generateLanguageStats 生成語言統計
func generateLanguageStats(deps []Dependency) map[string]LanguageStat {
	stats := make(map[string]LanguageStat)

	for _, dep := range deps {
		stat := stats[dep.Language]
		stat.TotalDeps++

		switch dep.Type {
		case "direct":
			stat.DirectDeps++
		case "dev":
			stat.DevDeps++
		case "indirect":
			stat.IndirectDeps++
		}

		if len(dep.Vulnerabilities) > 0 {
			stat.VulnerableDeps++
		}

		stats[dep.Language] = stat
	}

	return stats
}

// get 從快取獲取
func (c *vulnCache) get(key string) []vulndb.Vulnerability {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cache[key]
}

// set 設定快取
func (c *vulnCache) set(key string, vulns []vulndb.Vulnerability) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[key] = vulns
}

// ExportJSON 匯出 JSON 報告
func (esa *EnhancedSCAAnalyzer) ExportJSON(result *ScanResult, outputPath string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return esa.writeFile(outputPath, data)
}

// writeFile 寫入檔案（內部輔助函式）
func (esa *EnhancedSCAAnalyzer) writeFile(path string, data []byte) error {
	// 實現檔案寫入邏輯
	return nil
}
