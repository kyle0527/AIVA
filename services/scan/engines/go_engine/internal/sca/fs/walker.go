package fs

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
)

// =====================================================================
// File System Walker - SCA 檔案系統遍歷
// =====================================================================
// 來源: C:\Users\User\Downloads\新增資料夾 (6)\walker.go
// 用途: 遍歷項目目錄，識別依賴定義檔
// =====================================================================

// DependencyFile 代表發現的依賴定義檔
type DependencyFile struct {
	Path string
	Type string // "gomod", "npm", "pip", "maven", "yarn"
}

// WalkResult 包含遍歷結果
type WalkResult struct {
	Files []DependencyFile
	Errors []error
}

// ScanDirectory 遍歷指定目錄尋找依賴檔
// rootPath: 項目根目錄路徑
// 返回: 遍歷結果（包含發現的所有依賴檔）
func ScanDirectory(rootPath string) (*WalkResult, error) {
	result := &WalkResult{
		Files:  make([]DependencyFile, 0),
		Errors: make([]error, 0),
	}

	logrus.Infof("[FS Walker] Starting scan of directory: %s", rootPath)

	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// 記錄錯誤但繼續遍歷
			logrus.Warnf("[FS Walker] Error accessing path %s: %v", path, err)
			result.Errors = append(result.Errors, err)
			return nil
		}

		// 跳過 .git 目錄 (節省時間)
		if info.IsDir() && info.Name() == ".git" {
			logrus.Debugf("[FS Walker] Skipping .git directory")
			return filepath.SkipDir
		}

		// 跳過 node_modules 目錄 (節省時間)
		if info.IsDir() && info.Name() == "node_modules" {
			logrus.Debugf("[FS Walker] Skipping node_modules directory")
			return filepath.SkipDir
		}

		// 跳過 vendor 目錄 (Go 依賴)
		if info.IsDir() && info.Name() == "vendor" {
			logrus.Debugf("[FS Walker] Skipping vendor directory")
			return filepath.SkipDir
		}

		// 只處理文件
		if !info.IsDir() {
			filename := strings.ToLower(info.Name())
			fileType := detectFileType(filename)
			
			if fileType != "" {
				logrus.Infof("[FS Walker] Found dependency file: %s (type: %s)", path, fileType)
				result.Files = append(result.Files, DependencyFile{
					Path: path,
					Type: fileType,
				})
			}
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	logrus.Infof("[FS Walker] Scan completed, found %d dependency files", len(result.Files))
	return result, nil
}

// detectFileType 根據文件名判斷依賴檔類型
func detectFileType(filename string) string {
	switch filename {
	case "go.mod":
		return "gomod"
	case "go.sum":
		return "gosum"
	case "package.json":
		return "npm"
	case "package-lock.json":
		return "npm-lock"
	case "yarn.lock":
		return "yarn"
	case "requirements.txt":
		return "pip"
	case "pipfile":
		return "pipfile"
	case "pipfile.lock":
		return "pipfile-lock"
	case "pom.xml":
		return "maven"
	case "build.gradle":
		return "gradle"
	case "build.gradle.kts":
		return "gradle-kotlin"
	case "composer.json":
		return "composer"
	case "gemfile":
		return "bundler"
	case "cargo.toml":
		return "cargo"
	}
	return ""
}

// GetSupportedTypes 返回所有支持的依賴檔類型
func GetSupportedTypes() []string {
	return []string{
		"gomod", "gosum",
		"npm", "npm-lock", "yarn",
		"pip", "pipfile", "pipfile-lock",
		"maven", "gradle", "gradle-kotlin",
		"composer", "bundler", "cargo",
	}
}
