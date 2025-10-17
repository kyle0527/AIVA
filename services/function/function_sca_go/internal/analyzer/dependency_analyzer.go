package analyzer

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"
)

// DependencyAnalyzer 相依性分析器
type DependencyAnalyzer struct {
	logger *zap.Logger
	config *SCAConfig
}

// SCAConfig SCA 配置
type SCAConfig struct {
	SupportedLangs  []string // 支援的語言清單
	EnableDeepScan  bool     // 啟用深度掃描
	VulnSeverityMin string   // 最小漏洞嚴重性
	CacheResults    bool     // 快取結果
	SkipDirs        []string // 跳過的目錄
}

// Dependency 相依項結構
type Dependency struct {
	Name            string          `json:"name"`
	Version         string          `json:"version"`
	Language        string          `json:"language"`
	Type            string          `json:"type"`   // direct, dev, indirect
	Source          string          `json:"source"` // 檔案路徑
	Vulnerabilities []Vulnerability `json:"vulnerabilities,omitempty"`
}

// Vulnerability 漏洞結構
type Vulnerability struct {
	ID          string   `json:"id"`
	Severity    string   `json:"severity"`
	Description string   `json:"description"`
	CVSS        float64  `json:"cvss,omitempty"`
	References  []string `json:"references,omitempty"`
}

// NewDependencyAnalyzer 建立相依性分析器
func NewDependencyAnalyzer(logger *zap.Logger, config *SCAConfig) *DependencyAnalyzer {
	if config == nil {
		config = &SCAConfig{
			SupportedLangs: []string{"nodejs", "python", "go", "rust", "java", "dotnet", "php", "ruby"},
			SkipDirs:       []string{"node_modules", ".git", "vendor", "target", "dist", "build", "__pycache__"},
		}
	}
	return &DependencyAnalyzer{
		logger: logger,
		config: config,
	}
}

// AnalyzeProject 分析專案相依性
func (da *DependencyAnalyzer) AnalyzeProject(projectPath string) ([]Dependency, error) {
	da.logger.Info("Starting dependency analysis", zap.String("path", projectPath))

	allDeps := []Dependency{}

	// 遍歷專案目錄
	err := filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			da.logger.Warn("Failed to access path", zap.String("path", path), zap.Error(err))
			return nil // 繼續處理其他檔案
		}

		// 跳過目錄
		if info.IsDir() {
			if da.shouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}

		// 分析檔案
		deps, err := da.analyzeFileByType(path)
		if err != nil {
			da.logger.Warn("Failed to analyze file",
				zap.String("path", path),
				zap.Error(err))
			return nil // 繼續處理其他檔案
		}

		allDeps = append(allDeps, deps...)
		return nil
	})

	if err != nil {
		da.logger.Error("Failed to walk project files",
			zap.String("projectPath", projectPath),
			zap.Error(err))
		return allDeps, err
	}

	// 過濾支援的語言
	if len(da.config.SupportedLangs) > 0 {
		filteredDeps := []Dependency{}
		langMap := make(map[string]bool)
		for _, lang := range da.config.SupportedLangs {
			langMap[lang] = true
		}

		for _, dep := range allDeps {
			if langMap[dep.Language] {
				filteredDeps = append(filteredDeps, dep)
			}
		}
		allDeps = filteredDeps
	}

	da.logger.Info("Dependency analysis completed",
		zap.Int("total_dependencies", len(allDeps)))

	return allDeps, nil
}

// shouldSkipDir 判斷是否跳過目錄
func (da *DependencyAnalyzer) shouldSkipDir(dirName string) bool {
	for _, skip := range da.config.SkipDirs {
		if dirName == skip {
			return true
		}
	}
	return false
}

// analyzeFileByType 根據檔案類型分析
func (da *DependencyAnalyzer) analyzeFileByType(filePath string) ([]Dependency, error) {
	base := filepath.Base(filePath)

	switch {
	// Node.js
	case base == "package.json":
		return da.analyzeNodeJS(filePath)
	case base == "package-lock.json":
		return da.analyzeNodeJSLock(filePath)

	// Python
	case base == "requirements.txt":
		return da.analyzePythonRequirements(filePath)
	case base == "Pipfile", base == "Pipfile.lock":
		return da.analyzePipfile(filePath)
	case base == "pyproject.toml":
		return da.analyzePyProject(filePath)

	// Go
	case base == "go.mod":
		return da.analyzeGoMod(filePath)
	case base == "go.sum":
		return da.analyzeGoSum(filePath)

	// Rust
	case base == "Cargo.toml":
		return da.analyzeCargoToml(filePath)
	case base == "Cargo.lock":
		return da.analyzeCargoLock(filePath)

	// Java
	case base == "pom.xml":
		return da.analyzeMavenPom(filePath)
	case base == "build.gradle", base == "build.gradle.kts":
		return da.analyzeGradle(filePath)

	// .NET
	case strings.HasSuffix(base, ".csproj"):
		return da.analyzeDotNet(filePath)

	// PHP
	case base == "composer.json", base == "composer.lock":
		return da.analyzeComposer(filePath)

	// Ruby
	case base == "Gemfile", base == "Gemfile.lock":
		return da.analyzeGemfile(filePath)

	default:
		// 未知類型，跳過
		da.logger.Debug("Skipping unknown file type", zap.String("file", base))
		return []Dependency{}, nil
	}
}

// analyzeNodeJS 分析 Node.js package.json
func (da *DependencyAnalyzer) analyzeNodeJS(filePath string) ([]Dependency, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var pkg struct {
		Dependencies    map[string]string `json:"dependencies"`
		DevDependencies map[string]string `json:"devDependencies"`
	}

	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil, err
	}

	deps := []Dependency{}

	// 生產相依
	for name, version := range pkg.Dependencies {
		deps = append(deps, Dependency{
			Name:     name,
			Version:  version,
			Language: "nodejs",
			Type:     "direct",
			Source:   filePath,
		})
	}

	// 開發相依
	for name, version := range pkg.DevDependencies {
		deps = append(deps, Dependency{
			Name:     name,
			Version:  version,
			Language: "nodejs",
			Type:     "dev",
			Source:   filePath,
		})
	}

	return deps, nil
}

// analyzeNodeJSLock 分析 Node.js package-lock.json
func (da *DependencyAnalyzer) analyzeNodeJSLock(filePath string) ([]Dependency, error) {
	if !da.config.EnableDeepScan {
		return []Dependency{}, nil
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var lock struct {
		Packages map[string]struct {
			Version  string `json:"version"`
			Dev      bool   `json:"dev"`
			Resolved string `json:"resolved"`
		} `json:"packages"`
	}

	if err := json.Unmarshal(data, &lock); err != nil {
		return nil, err
	}

	deps := []Dependency{}
	for pkgPath, pkg := range lock.Packages {
		// 跳過根套件
		if pkgPath == "" {
			continue
		}

		name := strings.TrimPrefix(pkgPath, "node_modules/")
		depType := "indirect"
		if pkg.Dev {
			depType = "dev"
		}

		deps = append(deps, Dependency{
			Name:     name,
			Version:  pkg.Version,
			Language: "nodejs",
			Type:     depType,
			Source:   filePath,
		})
	}

	return deps, nil
}

// analyzePythonRequirements 分析 Python requirements.txt
func (da *DependencyAnalyzer) analyzePythonRequirements(filePath string) ([]Dependency, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	deps := []Dependency{}
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// 跳過註解和空行
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// 解析套件名稱和版本
		dep := da.parsePythonRequirement(line)
		if dep != nil {
			dep.Source = filePath
			deps = append(deps, *dep)
		}
	}

	return deps, scanner.Err()
}

// parsePythonRequirement 解析 Python 相依項
func (da *DependencyAnalyzer) parsePythonRequirement(line string) *Dependency {
	// 處理格式: package==version, package>=version, package[extra]==version
	line = strings.Split(line, "#")[0] // 移除行尾註解
	line = strings.TrimSpace(line)

	// 移除 extras
	if idx := strings.Index(line, "["); idx != -1 {
		line = line[:idx] + line[strings.Index(line, "]")+1:]
	}

	// 解析名稱和版本
	for _, op := range []string{"==", ">=", "<=", "~=", "!=", ">", "<"} {
		if parts := strings.Split(line, op); len(parts) == 2 {
			return &Dependency{
				Name:     strings.TrimSpace(parts[0]),
				Version:  strings.TrimSpace(parts[1]),
				Language: "python",
				Type:     "direct",
			}
		}
	}

	// 沒有版本限制
	if line != "" {
		return &Dependency{
			Name:     line,
			Version:  "*",
			Language: "python",
			Type:     "direct",
		}
	}

	return nil
}

// analyzePipfile 分析 Python Pipfile
func (da *DependencyAnalyzer) analyzePipfile(filePath string) ([]Dependency, error) {
	// Pipfile 使用 TOML 格式，這裡簡化處理
	// 生產環境應使用 TOML 解析庫
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	deps := []Dependency{}
	scanner := bufio.NewScanner(file)
	section := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if strings.HasPrefix(line, "[packages]") {
			section = "packages"
			continue
		} else if strings.HasPrefix(line, "[dev-packages]") {
			section = "dev-packages"
			continue
		} else if strings.HasPrefix(line, "[") {
			section = ""
			continue
		}

		if section != "" && strings.Contains(line, "=") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				name := strings.TrimSpace(parts[0])
				version := strings.Trim(strings.TrimSpace(parts[1]), "\"'")

				depType := "direct"
				if section == "dev-packages" {
					depType = "dev"
				}

				deps = append(deps, Dependency{
					Name:     name,
					Version:  version,
					Language: "python",
					Type:     depType,
					Source:   filePath,
				})
			}
		}
	}

	return deps, scanner.Err()
}

// analyzePyProject 分析 Python pyproject.toml
func (da *DependencyAnalyzer) analyzePyProject(filePath string) ([]Dependency, error) {
	// 簡化實現，生產環境應使用 TOML 解析庫
	da.logger.Debug("PyProject.toml parsing not fully implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeGoMod 分析 Go go.mod
func (da *DependencyAnalyzer) analyzeGoMod(filePath string) ([]Dependency, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	deps := []Dependency{}
	scanner := bufio.NewScanner(file)
	inRequireBlock := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// 跳過註解
		if strings.HasPrefix(line, "//") {
			continue
		}

		// 檢查 require 區塊
		if strings.HasPrefix(line, "require (") {
			inRequireBlock = true
			continue
		}

		if inRequireBlock && line == ")" {
			inRequireBlock = false
			continue
		}

		// 解析相依項
		if strings.HasPrefix(line, "require ") || inRequireBlock {
			dep := da.parseGoRequire(line)
			if dep != nil {
				dep.Source = filePath
				deps = append(deps, *dep)
			}
		}
	}

	return deps, scanner.Err()
}

// parseGoRequire 解析 Go require 行
func (da *DependencyAnalyzer) parseGoRequire(line string) *Dependency {
	line = strings.TrimPrefix(line, "require ")
	line = strings.TrimSpace(line)

	// 移除註解
	if idx := strings.Index(line, "//"); idx != -1 {
		line = line[:idx]
	}

	parts := strings.Fields(line)
	if len(parts) >= 2 {
		depType := "direct"
		if strings.Contains(line, "// indirect") || len(parts) > 2 && parts[2] == "indirect" {
			depType = "indirect"
		}

		return &Dependency{
			Name:     parts[0],
			Version:  parts[1],
			Language: "go",
			Type:     depType,
		}
	}

	return nil
}

// analyzeGoSum 分析 Go go.sum
func (da *DependencyAnalyzer) analyzeGoSum(filePath string) ([]Dependency, error) {
	// go.sum 包含所有模組的校驗和，通常不需要單獨解析
	// 如果需要深度掃描，可以實現
	if !da.config.EnableDeepScan {
		return []Dependency{}, nil
	}

	da.logger.Debug("Go.sum deep scan not implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeCargoToml 分析 Rust Cargo.toml
func (da *DependencyAnalyzer) analyzeCargoToml(filePath string) ([]Dependency, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	deps := []Dependency{}
	scanner := bufio.NewScanner(file)
	section := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if strings.HasPrefix(line, "[dependencies]") {
			section = "dependencies"
			continue
		} else if strings.HasPrefix(line, "[dev-dependencies]") {
			section = "dev-dependencies"
			continue
		} else if strings.HasPrefix(line, "[") {
			section = ""
			continue
		}

		if section != "" && strings.Contains(line, "=") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				name := strings.TrimSpace(parts[0])
				version := strings.Trim(strings.TrimSpace(parts[1]), "\"'")

				depType := "direct"
				if section == "dev-dependencies" {
					depType = "dev"
				}

				deps = append(deps, Dependency{
					Name:     name,
					Version:  version,
					Language: "rust",
					Type:     depType,
					Source:   filePath,
				})
			}
		}
	}

	return deps, scanner.Err()
}

// analyzeCargoLock 分析 Rust Cargo.lock
func (da *DependencyAnalyzer) analyzeCargoLock(filePath string) ([]Dependency, error) {
	if !da.config.EnableDeepScan {
		return []Dependency{}, nil
	}

	da.logger.Debug("Cargo.lock deep scan not fully implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeMavenPom 分析 Java Maven pom.xml
func (da *DependencyAnalyzer) analyzeMavenPom(filePath string) ([]Dependency, error) {
	// 需要 XML 解析，這裡標記為未實現
	da.logger.Warn("Java Maven dependency analysis not implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeGradle 分析 Java Gradle build.gradle
func (da *DependencyAnalyzer) analyzeGradle(filePath string) ([]Dependency, error) {
	da.logger.Warn("Java Gradle dependency analysis not implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeDotNet 分析 .NET .csproj
func (da *DependencyAnalyzer) analyzeDotNet(filePath string) ([]Dependency, error) {
	// 需要 XML 解析
	da.logger.Warn(".NET dependency analysis not implemented", zap.String("file", filePath))
	return []Dependency{}, nil
}

// analyzeComposer 分析 PHP composer.json
func (da *DependencyAnalyzer) analyzeComposer(filePath string) ([]Dependency, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var composer struct {
		Require    map[string]string `json:"require"`
		RequireDev map[string]string `json:"require-dev"`
	}

	if err := json.Unmarshal(data, &composer); err != nil {
		return nil, err
	}

	deps := []Dependency{}

	for name, version := range composer.Require {
		deps = append(deps, Dependency{
			Name:     name,
			Version:  version,
			Language: "php",
			Type:     "direct",
			Source:   filePath,
		})
	}

	for name, version := range composer.RequireDev {
		deps = append(deps, Dependency{
			Name:     name,
			Version:  version,
			Language: "php",
			Type:     "dev",
			Source:   filePath,
		})
	}

	return deps, nil
}

// analyzeGemfile 分析 Ruby Gemfile
func (da *DependencyAnalyzer) analyzeGemfile(filePath string) ([]Dependency, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	deps := []Dependency{}
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// 跳過註解和空行
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// 解析 gem 聲明
		if strings.HasPrefix(line, "gem ") {
			dep := da.parseGemLine(line)
			if dep != nil {
				dep.Source = filePath
				deps = append(deps, *dep)
			}
		}
	}

	return deps, scanner.Err()
}

// parseGemLine 解析 Ruby gem 行
func (da *DependencyAnalyzer) parseGemLine(line string) *Dependency {
	// 格式: gem 'name', 'version'
	line = strings.TrimPrefix(line, "gem ")
	parts := strings.Split(line, ",")

	if len(parts) >= 1 {
		name := strings.Trim(strings.TrimSpace(parts[0]), "'\"")
		version := "*"

		if len(parts) >= 2 {
			version = strings.Trim(strings.TrimSpace(parts[1]), "'\"~>")
		}

		return &Dependency{
			Name:     name,
			Version:  version,
			Language: "ruby",
			Type:     "direct",
		}
	}

	return nil
}
