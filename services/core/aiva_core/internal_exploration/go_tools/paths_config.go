// paths_config.go - Go 路徑配置
// 自動從 Python paths.py 生成
package main

import (
	"os"
	"path/filepath"
)

// PathsConfig 存儲所有路徑配置
type PathsConfig struct {
	ProjectRoot           string
	ServicesRoot          string
	IntegrationRoot       string
	IntegrationDataRoot   string
	
	// Internal Exploration 路徑
	InternalExplorationData string
	AnalysisResultsDir      string
	AnalysisHistoryDir      string
	SelfHealingDir          string
	
	// 其他整合模組路徑
	AttackPathsDir      string
	ExperiencesDir      string
	TrainingDataDir     string
	
	// 環境變量控制
	UseIntegratedPaths bool
}

// GetPathsConfig 返回配置實例
func GetPathsConfig() *PathsConfig {
	// 獲取當前文件目錄
	// 當前文件: services/core/aiva_core/internal_exploration/go_tools/paths_config.go
	// 向上 5 層到達專案根目錄
	cwd, _ := os.Getwd()
	projectRoot := filepath.Join(cwd, "../../../../")
	projectRoot, _ = filepath.Abs(projectRoot)
	
	servicesRoot := filepath.Join(projectRoot, "services")
	integrationRoot := filepath.Join(servicesRoot, "integration")
	integrationDataRoot := filepath.Join(integrationRoot, "data")
	
	internalExplorationData := filepath.Join(integrationDataRoot, "internal_exploration")
	
	// 檢查環境變量
	useIntegratedPaths := os.Getenv("AIVA_USE_INTEGRATED_PATHS") != "false"
	
	return &PathsConfig{
		ProjectRoot:             projectRoot,
		ServicesRoot:            servicesRoot,
		IntegrationRoot:         integrationRoot,
		IntegrationDataRoot:     integrationDataRoot,
		InternalExplorationData: internalExplorationData,
		AnalysisResultsDir:      filepath.Join(internalExplorationData, "analysis_results"),
		AnalysisHistoryDir:      filepath.Join(internalExplorationData, "analysis_history"),
		SelfHealingDir:          filepath.Join(internalExplorationData, "self_healing"),
		AttackPathsDir:          filepath.Join(integrationDataRoot, "attack_paths"),
		ExperiencesDir:          filepath.Join(integrationDataRoot, "experiences"),
		TrainingDataDir:         filepath.Join(integrationDataRoot, "training"),
		UseIntegratedPaths:      useIntegratedPaths,
	}
}

// EnsureDirectories 確保所有目錄存在
func (pc *PathsConfig) EnsureDirectories() error {
	dirs := []string{
		pc.InternalExplorationData,
		pc.AnalysisResultsDir,
		pc.AnalysisHistoryDir,
		pc.SelfHealingDir,
		pc.AttackPathsDir,
		pc.ExperiencesDir,
		pc.TrainingDataDir,
	}
	
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	
	return nil
}

// GetAnalysisOutputDir 獲取特定工具的輸出目錄
func (pc *PathsConfig) GetAnalysisOutputDir(toolName string) string {
	if toolName == "self_healing" {
		return pc.SelfHealingDir
	}
	
	toolDir := filepath.Join(pc.AnalysisResultsDir, toolName)
	os.MkdirAll(toolDir, 0755)
	return toolDir
}

// GetDefaultOutputDir 獲取默認輸出目錄（向後兼容）
func (pc *PathsConfig) GetDefaultOutputDir() string {
	if pc.UseIntegratedPaths {
		pc.EnsureDirectories()
		return pc.GetAnalysisOutputDir("go")
	}
	
	// 向後兼容：使用舊路徑
	return "./analysis_output"
}
