package schemas

import "time"

// CommonVulnerability 統一的漏洞結構定義
type CommonVulnerability struct {
	// 基本識別信息
	ID          string `json:"id"`          // 漏洞 ID (CVE-2023-1234, GHSA-xxxx-yyyy-zzzz)
	Type        string `json:"type"`        // 漏洞類型 (SCA, SAST, DAST, etc.)
	Name        string `json:"name"`        // 漏洞名稱
	Summary     string `json:"summary"`     // 簡要說明
	Description string `json:"description"` // 詳細描述

	// 嚴重性和評分
	Severity   string  `json:"severity"`              // 嚴重性 (CRITICAL, HIGH, MEDIUM, LOW)
	CVSS       float64 `json:"cvss,omitempty"`        // CVSS 分數
	CVSSVector string  `json:"cvss_vector,omitempty"` // CVSS 向量字串

	// 標準化識別碼
	CVEID  string   `json:"cve_id,omitempty"`  // CVE 編號
	GHSAID string   `json:"ghsa_id,omitempty"` // GitHub Security Advisory ID
	CWEIDs []string `json:"cwe_ids,omitempty"` // CWE 弱點分類
	OSVIDs []string `json:"osv_ids,omitempty"` // OSV 數據庫 ID

	// 影響範圍
	Affected   []string `json:"affected,omitempty"`   // 受影響的版本範圍
	Fixed      []string `json:"fixed,omitempty"`      // 已修復的版本
	Introduced []string `json:"introduced,omitempty"` // 引入漏洞的版本

	// 參考資料
	References []string `json:"references,omitempty"` // 參考連結
	Aliases    []string `json:"aliases,omitempty"`    // 其他別名

	// 時間戳記
	PublishedAt *time.Time `json:"published_at,omitempty"` // 公開日期
	ModifiedAt  *time.Time `json:"modified_at,omitempty"`  // 修改日期

	// 擴展元數據
	Metadata map[string]interface{} `json:"metadata,omitempty"` // 擴展資料
}

// CommonDependency 統一的依賴項結構定義
type CommonDependency struct {
	// 基本信息
	Name      string `json:"name"`      // 套件名稱
	Version   string `json:"version"`   // 版本號
	Language  string `json:"language"`  // 程式語言
	Ecosystem string `json:"ecosystem"` // 生態系統 (npm, PyPI, Maven, etc.)

	// 來源信息
	FilePath   string `json:"file_path"`   // 來源文件路徑
	LineNumber int    `json:"line_number"` // 行號

	// 依賴類型
	Type  string `json:"type"`            // 依賴類型 (direct, indirect, dev)
	Scope string `json:"scope,omitempty"` // 依賴範圍 (compile, runtime, test, etc.)

	// 漏洞信息
	Vulnerabilities []CommonVulnerability `json:"vulnerabilities,omitempty"`

	// 許可證信息
	License    string `json:"license,omitempty"`     // 許可證
	LicenseURL string `json:"license_url,omitempty"` // 許可證連結

	// 元數據
	Metadata map[string]string `json:"metadata,omitempty"`

	// 時間戳記
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
}

// ScanResult 掃描結果統一結構
type ScanResult struct {
	// 掃描基本信息
	ScanID   string `json:"scan_id"`   // 掃描 ID
	TaskID   string `json:"task_id"`   // 任務 ID
	ScanType string `json:"scan_type"` // 掃描類型 (SCA, SAST, DAST)

	// 目標信息
	Target ScanTarget `json:"target"` // 掃描目標

	// 掃描狀態
	Status   string  `json:"status"`   // 狀態 (running, completed, failed)
	Progress float64 `json:"progress"` // 進度 (0-100)

	// 結果統計
	Summary ScanSummary `json:"summary"` // 結果摘要

	// 詳細結果
	Dependencies []CommonDependency `json:"dependencies,omitempty"` // 發現的依賴
	Findings     []FindingPayload   `json:"findings,omitempty"`     // 發現的問題

	// 時間信息
	StartedAt   time.Time  `json:"started_at"`             // 開始時間
	CompletedAt *time.Time `json:"completed_at,omitempty"` // 完成時間
	Duration    int64      `json:"duration,omitempty"`     // 掃描時長(秒)

	// 配置和元數據
	Config   map[string]interface{} `json:"config,omitempty"`   // 掃描配置
	Metadata map[string]interface{} `json:"metadata,omitempty"` // 擴展元數據

	// 錯誤信息
	Error *ScanError `json:"error,omitempty"` // 錯誤詳情
}

// ScanTarget 掃描目標
type ScanTarget struct {
	Type       string            `json:"type"`                 // 目標類型 (repository, file, url)
	URL        string            `json:"url"`                  // 目標 URL
	Path       string            `json:"path,omitempty"`       // 本地路徑
	Branch     string            `json:"branch,omitempty"`     // Git 分支
	Commit     string            `json:"commit,omitempty"`     // Git 提交
	Method     string            `json:"method,omitempty"`     // HTTP 方法
	Headers    map[string]string `json:"headers,omitempty"`    // HTTP 標頭
	Parameters map[string]string `json:"parameters,omitempty"` // 參數
}

// ScanSummary 掃描結果摘要
type ScanSummary struct {
	// 統計數字
	TotalDependencies int `json:"total_dependencies"` // 總依賴數
	TotalFindings     int `json:"total_findings"`     // 總發現數

	// 按嚴重性分組
	Critical int `json:"critical"` // 嚴重漏洞數
	High     int `json:"high"`     // 高危漏洞數
	Medium   int `json:"medium"`   // 中危漏洞數
	Low      int `json:"low"`      // 低危漏洞數
	Info     int `json:"info"`     // 信息級別數

	// 按類型分組
	ByType map[string]int `json:"by_type,omitempty"` // 按漏洞類型分組

	// 按語言分組
	ByLanguage map[string]int `json:"by_language,omitempty"` // 按程式語言分組
}

// ScanError 掃描錯誤信息
type ScanError struct {
	Code        string    `json:"code"`              // 錯誤代碼
	Message     string    `json:"message"`           // 錯誤訊息
	Details     string    `json:"details,omitempty"` // 錯誤詳情
	Timestamp   time.Time `json:"timestamp"`         // 發生時間
	Recoverable bool      `json:"recoverable"`       // 是否可恢復
}

// FindingPayload 統一的發現載荷結構
type FindingPayload struct {
	// 基本識別
	FindingID string `json:"finding_id"`        // 發現 ID
	TaskID    string `json:"task_id"`           // 任務 ID
	ScanID    string `json:"scan_id,omitempty"` // 掃描 ID

	// 漏洞信息
	Vulnerability CommonVulnerability `json:"vulnerability"` // 漏洞詳情

	// 嚴重性和可信度
	Severity   string `json:"severity"`   // 嚴重性
	Confidence string `json:"confidence"` // 可信度 (CERTAIN, FIRM, TENTATIVE)

	// 目標和位置
	Target   FindingTarget   `json:"target"`             // 發現目標
	Location FindingLocation `json:"location,omitempty"` // 具體位置

	// 證據和影響
	Evidence       FindingEvidence       `json:"evidence"`       // 證據
	Impact         FindingImpact         `json:"impact"`         // 影響
	Recommendation FindingRecommendation `json:"recommendation"` // 建議

	// 分類和標籤
	Category string   `json:"category,omitempty"` // 分類
	Tags     []string `json:"tags,omitempty"`     // 標籤

	// 狀態管理
	Status     string     `json:"status"`                // 狀態 (new, confirmed, false_positive, fixed)
	AssignedTo string     `json:"assigned_to,omitempty"` // 負責人
	CreatedAt  time.Time  `json:"created_at"`            // 創建時間
	UpdatedAt  time.Time  `json:"updated_at"`            // 更新時間
	ResolvedAt *time.Time `json:"resolved_at,omitempty"` // 解決時間

	// 擴展信息
	Metadata map[string]interface{} `json:"metadata,omitempty"` // 元數據
}

// FindingTarget 發現目標 (與 models 保持一致)
type FindingTarget struct {
	URL       string `json:"url"`
	Method    string `json:"method,omitempty"`
	Parameter string `json:"parameter,omitempty"`
}

// FindingLocation 發現的具體位置
type FindingLocation struct {
	FilePath    string `json:"file_path,omitempty"`    // 文件路徑
	LineStart   int    `json:"line_start,omitempty"`   // 起始行
	LineEnd     int    `json:"line_end,omitempty"`     // 結束行
	ColumnStart int    `json:"column_start,omitempty"` // 起始列
	ColumnEnd   int    `json:"column_end,omitempty"`   // 結束列
	Function    string `json:"function,omitempty"`     // 函數名
	Class       string `json:"class,omitempty"`        // 類名
}

// FindingEvidence 證據 (與 models 保持一致)
type FindingEvidence struct {
	Request        string `json:"request"`
	Response       string `json:"response"`
	Payload        string `json:"payload,omitempty"`
	ProofOfConcept string `json:"proof_of_concept,omitempty"`
}

// FindingImpact 影響 (與 models 保持一致)
type FindingImpact struct {
	Confidentiality string `json:"confidentiality"`
	Integrity       string `json:"integrity"`
	Availability    string `json:"availability"`
	BusinessImpact  string `json:"business_impact"`
}

// FindingRecommendation 修復建議 (與 models 保持一致)
type FindingRecommendation struct {
	Remediation string   `json:"remediation"`
	References  []string `json:"references"`
}
