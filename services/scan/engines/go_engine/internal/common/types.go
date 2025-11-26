// Common Types - 跨掃描器共享的類型定義
package common

import (
	"context"
	"time"
)

// ==================== 請求/響應結構 ====================

// ScanRequest - 統一掃描請求格式（與 Python go_adapter.py 約定）
type ScanRequest struct {
	// 基本配置
	ScanID      string   `json:"scan_id"`       // 掃描唯一 ID
	Targets     []string `json:"targets"`       // 目標列表
	ScannerType string   `json:"scanner_type"`  // 掃描器類型: ssrf/cspm/sca

	// 掃描控制
	MaxDepth    int `json:"max_depth"`    // 最大深度 (默認: 3)
	Timeout     int `json:"timeout"`      // 總超時秒數 (默認: 300)
	Concurrency int `json:"concurrency"`  // 並發數 (默認: 10)

	// HTTP 配置
	Headers    map[string]string `json:"headers"`     // 自定義 Headers
	UserAgent  string            `json:"user_agent"`  // User-Agent
	MaxRetries int               `json:"max_retries"` // 重試次數

	// 高級選項
	Options map[string]interface{} `json:"options"` // 掃描器特定選項

	// 可觀測性
	EnableProgressReport bool `json:"enable_progress_report"` // 是否輸出進度
	EnableVerboseLog     bool `json:"enable_verbose_log"`     // 詳細日誌
}

// ScanResult - 統一掃描結果格式（與 Python go_adapter.py 約定）
type ScanResult struct {
	// 掃描元數據
	ScanID      string `json:"scan_id"`
	ScannerType string `json:"scanner_type"`
	Status      string `json:"status"` // success/failed/partial/cancelled

	// 執行統計
	ExecutionTime  float64 `json:"execution_time"`   // 執行時間（秒）
	TargetsScanned int     `json:"targets_scanned"`  // 掃描目標數
	RequestsMade   int     `json:"requests_made"`    // 發出請求數
	SuccessCount   int     `json:"success_count"`    // 成功數
	FailureCount   int     `json:"failure_count"`    // 失敗數

	// 掃描結果
	Assets []Asset `json:"assets"` // 發現的資產（標準格式）

	// 錯誤信息
	Errors   []string `json:"errors"`   // 錯誤列表
	Warnings []string `json:"warnings"` // 警告列表

	// 可選：進度報告
	ProgressReports []ProgressReport `json:"progress_reports,omitempty"`
}

// Asset - 資產格式（與 aiva_common.schemas.Asset 一致）
type Asset struct {
	Type         string                 `json:"type"`          // web_vulnerability, api_endpoint, etc.
	Name         string                 `json:"name"`          // 資產名稱
	Severity     string                 `json:"severity"`      // critical/high/medium/low/info
	Confidence   string                 `json:"confidence"`    // high/medium/low
	Details      map[string]interface{} `json:"details"`       // 詳細信息
	SourceEngine string                 `json:"source_engine"` // "go"
	Metadata     map[string]interface{} `json:"metadata"`      // 額外元數據
}

// ProgressReport - 進度報告（可選，用於長時間運行任務）
type ProgressReport struct {
	Timestamp       string  `json:"timestamp"`
	TargetsScanned  int     `json:"targets_scanned"`
	TotalTargets    int     `json:"total_targets"`
	PercentComplete float64 `json:"percent_complete"`
	CurrentTarget   string  `json:"current_target"`
}

// ==================== 掃描器配置 ====================

// ScannerConfig - 掃描器運行時配置
type ScannerConfig struct {
	MaxDepth       int
	Timeout        time.Duration
	Concurrency    int
	Headers        map[string]string
	UserAgent      string
	MaxRetries     int
	EnableProgress bool
	EnableVerbose  bool
}

// ==================== 掃描器接口 ====================

// Scanner - 掃描器統一接口
type Scanner interface {
	// Scan - 執行掃描
	Scan(ctx context.Context, targets []string, config *ScannerConfig) ([]Asset, error)

	// Name - 掃描器名稱
	Name() string

	// Type - 掃描器類型
	Type() string
}

// ==================== Worker Pool 類型 ====================

// Task - Worker Pool 任務
type Task struct {
	ID      string          // 任務 ID
	Target  string          // 目標
	Scanner Scanner         // 掃描器實例
	Config  *ScannerConfig  // 配置
}

// TaskResult - 任務執行結果
type TaskResult struct {
	TaskID   string        // 任務 ID
	Target   string        // 目標
	Assets   []Asset       // 發現的資產
	Error    error         // 錯誤（如有）
	Duration time.Duration // 執行時長
}
