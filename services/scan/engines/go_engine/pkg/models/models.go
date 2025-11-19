package models

import "time"

// =====================================================================
// Phase 0 → Phase 1 數據橋接結構
// =====================================================================
// 這些結構定義了 Rust 引擎 (Phase 0) 傳遞給 Go 引擎 (Phase 1) 的數據格式
// 來源: C:\Users\User\Downloads\新增資料夾 (6)\rust_bridge.go
// =====================================================================

// RustScanResult 對應 Rust 引擎輸出的完整 JSON 結構
// 這是 Phase 0 到 Phase 1 的主要載體
type RustScanResult struct {
	Target       string         `json:"target"`
	ScanID       string         `json:"scan_id"`
	Timestamp    time.Time      `json:"timestamp"`
	Endpoints    []RustEndpoint `json:"endpoints"`
	DetectedTech TechStack      `json:"detected_tech"`
}

// RustEndpoint 描述 Rust 發現的單個端點及其特徵
type RustEndpoint struct {
	URL       string   `json:"url"`
	Method    string   `json:"method"`
	Params    []string `json:"params"`     // 發現的參數名，如 ["callback", "url"]
	RiskScore string   `json:"risk_score"` // high, medium, low
	Evidence  string   `json:"evidence"`   // 例如 "Reflected parameter found"
}

// TechStack 描述 Rust 識別出的技術堆疊
type TechStack struct {
	CloudProvider string `json:"cloud_provider"` // aws, gcp, azure, none
	Server        string `json:"server"`         // nginx, apache...
	Language      string `json:"language"`       // go, java, python...
}

// =====================================================================
// Go 引擎任務結構定義
// =====================================================================
// 下列結構定義了 Python Dispatcher 派發給 Go 引擎的具體任務
// =====================================================================

// SSRFVerifyTask 是指派給 SSRF Scanner 的任務
type SSRFVerifyTask struct {
	TaskID    string `json:"task_id"`
	TargetURL string `json:"target_url"`
	Method    string `json:"method"`
	ParamName string `json:"param_name"` // 需要進行 Fuzzing 的參數
	OOBToken  string `json:"oob_token"`  // 預先生成的 OOB 驗證 Token
}

// CSPMAuditTask 是指派給 CSPM Scanner 的任務
type CSPMAuditTask struct {
	TaskID       string   `json:"task_id"`
	Provider     string   `json:"provider"`      // aws, gcp, azure
	CredentialID string   `json:"credential_id"` // 用於從 Vault 獲取憑證的 ID
	Regions      []string `json:"regions"`       // 指定掃描區域
}

// SCADepCheckTask 是指派給 SCA Scanner 的任務
type SCADepCheckTask struct {
	TaskID     string `json:"task_id"`
	SourcePath string `json:"source_path"` // 原始碼路徑
	GitURL     string `json:"git_url"`     // 或 Git 倉庫地址
}

// =====================================================================
// Go 引擎結果結構定義
// =====================================================================
// 這些結構定義了 Go 引擎返回給 Python Dispatcher 的結果格式
// =====================================================================

// SSRFVerifyResult SSRF 驗證結果
type SSRFVerifyResult struct {
	TaskID         string                  `json:"task_id"`
	TargetURL      string                  `json:"target_url"`
	ParamName      string                  `json:"param_name"`
	Vulnerable     bool                    `json:"vulnerable"`
	Severity       string                  `json:"severity"`        // critical, high, medium, low
	TestedPayloads int                     `json:"tested_payloads"` // 測試的 Payload 數量
	Evidence       []SSRFEvidenceItem      `json:"evidence"`        // 證據列表
	OOBInteraction *OOBInteractionEvidence `json:"oob_interaction,omitempty"`
}

// SSRFEvidenceItem 單個 SSRF 證據
type SSRFEvidenceItem struct {
	Payload      string `json:"payload"`
	ResponseCode int    `json:"response_code"`
	ResponseSize int    `json:"response_size"`
	ResponseTime int    `json:"response_time_ms"`
	Matched      bool   `json:"matched"`
	MatchReason  string `json:"match_reason"` // 例如 "Metadata API response detected"
}

// OOBInteractionEvidence OOB 回連證據
type OOBInteractionEvidence struct {
	Token        string    `json:"token"`
	InteractType string    `json:"interact_type"` // http, dns
	SourceIP     string    `json:"source_ip"`
	Timestamp    time.Time `json:"timestamp"`
	RawData      string    `json:"raw_data"`
}

// CSPMAuditResult CSPM 審計結果
type CSPMAuditResult struct {
	TaskID     string            `json:"task_id"`
	Provider   string            `json:"provider"`
	Findings   []CSPMFinding     `json:"findings"`
	Summary    CSPMAuditSummary  `json:"summary"`
	Timestamp  time.Time         `json:"timestamp"`
}

// CSPMFinding 單個 CSPM 發現
type CSPMFinding struct {
	ResourceType string `json:"resource_type"` // s3_bucket, iam_user, security_group
	ResourceID   string `json:"resource_id"`
	RuleName     string `json:"rule_name"`     // 例如 "S3 Bucket Public Access"
	Severity     string `json:"severity"`      // critical, high, medium, low
	Description  string `json:"description"`
	Remediation  string `json:"remediation"`
}

// CSPMAuditSummary CSPM 審計摘要
type CSPMAuditSummary struct {
	TotalChecks     int `json:"total_checks"`
	PassedChecks    int `json:"passed_checks"`
	FailedChecks    int `json:"failed_checks"`
	CriticalCount   int `json:"critical_count"`
	HighCount       int `json:"high_count"`
	MediumCount     int `json:"medium_count"`
	LowCount        int `json:"low_count"`
}

// SCADepCheckResult SCA 依賴檢查結果
type SCADepCheckResult struct {
	TaskID       string                `json:"task_id"`
	SourcePath   string                `json:"source_path"`
	Dependencies []DependencyInfo      `json:"dependencies"`
	Vulnerabilities []VulnerabilityInfo `json:"vulnerabilities"`
	Summary      SCADepCheckSummary    `json:"summary"`
	Timestamp    time.Time             `json:"timestamp"`
}

// DependencyInfo 依賴信息
type DependencyInfo struct {
	Name         string `json:"name"`
	Version      string `json:"version"`
	DependencyFile string `json:"dependency_file"` // go.mod, package.json, requirements.txt
	PackageManager string `json:"package_manager"` // go, npm, pip
}

// VulnerabilityInfo 漏洞信息
type VulnerabilityInfo struct {
	DependencyName string `json:"dependency_name"`
	CVEID          string `json:"cve_id"`
	Severity       string `json:"severity"`
	Description    string `json:"description"`
	FixedVersion   string `json:"fixed_version"`
}

// SCADepCheckSummary SCA 檢查摘要
type SCADepCheckSummary struct {
	TotalDependencies int `json:"total_dependencies"`
	VulnerableCount   int `json:"vulnerable_count"`
	CriticalCount     int `json:"critical_count"`
	HighCount         int `json:"high_count"`
	MediumCount       int `json:"medium_count"`
	LowCount          int `json:"low_count"`
}
