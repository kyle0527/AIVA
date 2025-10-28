// AIVA Go Schema - 自動生成
// ===========================
//
// AIVA跨語言Schema統一定義 - 以手動維護版本為準
//
// ⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
// 📅 最後更新: 2025-10-28T10:24:34.374262
// 🔄 Schema 版本: 1.0.0

package schemas

import "time"

// ==================== 基礎類型 ====================

// MessageHeader 訊息標頭 - 用於所有訊息的統一標頭格式
type MessageHeader struct {
	MessageId     string    `json:"message_id"`               //
	TraceId       string    `json:"trace_id"`                 //
	CorrelationId *string   `json:"correlation_id,omitempty"` //
	SourceModule  string    `json:"source_module"`            // 來源模組名稱
	Timestamp     time.Time `json:"timestamp,omitempty"`      //
	Version       string    `json:"version,omitempty"`        //
}

// Target 目標資訊 - 漏洞所在位置
type Target struct {
	Url       interface{}            `json:"url"`                 //
	Parameter *string                `json:"parameter,omitempty"` //
	Method    *string                `json:"method,omitempty"`    //
	Headers   map[string]interface{} `json:"headers,omitempty"`   //
	Params    map[string]interface{} `json:"params,omitempty"`    //
	Body      *string                `json:"body,omitempty"`      //
}

// Vulnerability 漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP
type Vulnerability struct {
	Name          interface{} `json:"name"`                     //
	Cwe           *string     `json:"cwe,omitempty"`            // CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/
	Cve           *string     `json:"cve,omitempty"`            // CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/
	Severity      interface{} `json:"severity"`                 //
	Confidence    interface{} `json:"confidence"`               //
	Description   *string     `json:"description,omitempty"`    //
	CvssScore     interface{} `json:"cvss_score,omitempty"`     // CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/
	CvssVector    *string     `json:"cvss_vector,omitempty"`    // CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
	OwaspCategory *string     `json:"owasp_category,omitempty"` // OWASP Top 10 分類，例如: A03:2021-Injection
}

// Asset 資產基本資訊
type Asset struct {
	AssetId    string   `json:"asset_id"`             //
	Type       string   `json:"type"`                 //
	Value      string   `json:"value"`                //
	Parameters []string `json:"parameters,omitempty"` //
	HasForm    bool     `json:"has_form,omitempty"`   //
}

// Authentication 認證資訊
type Authentication struct {
	Method      string                 `json:"method,omitempty"`      //
	Credentials map[string]interface{} `json:"credentials,omitempty"` //
}

// ExecutionError 執行錯誤統一格式
type ExecutionError struct {
	ErrorId   string    `json:"error_id"`            //
	ErrorType string    `json:"error_type"`          //
	Message   string    `json:"message"`             //
	Payload   *string   `json:"payload,omitempty"`   //
	Vector    *string   `json:"vector,omitempty"`    //
	Timestamp time.Time `json:"timestamp,omitempty"` //
	Attempts  int       `json:"attempts,omitempty"`  //
}

// Fingerprints 技術指紋
type Fingerprints struct {
	WebServer   map[string]interface{} `json:"web_server,omitempty"`   //
	Framework   map[string]interface{} `json:"framework,omitempty"`    //
	Language    map[string]interface{} `json:"language,omitempty"`     //
	WafDetected bool                   `json:"waf_detected,omitempty"` //
	WafVendor   *string                `json:"waf_vendor,omitempty"`   //
}

// RateLimit 速率限制
type RateLimit struct {
	RequestsPerSecond int `json:"requests_per_second,omitempty"` //
	Burst             int `json:"burst,omitempty"`               //
}

// RiskFactor 風險因子
type RiskFactor struct {
	FactorName  string  `json:"factor_name"`           // 風險因子名稱
	Weight      float64 `json:"weight"`                // 權重
	Value       float64 `json:"value"`                 // 因子值
	Description *string `json:"description,omitempty"` // 因子描述
}

// ScanScope 掃描範圍
type ScanScope struct {
	Exclusions        []string `json:"exclusions,omitempty"`         //
	IncludeSubdomains bool     `json:"include_subdomains,omitempty"` //
	AllowedHosts      []string `json:"allowed_hosts,omitempty"`      //
}

// Summary 掃描摘要
type Summary struct {
	UrlsFound           int `json:"urls_found,omitempty"`            //
	FormsFound          int `json:"forms_found,omitempty"`           //
	ApisFound           int `json:"apis_found,omitempty"`            //
	ScanDurationSeconds int `json:"scan_duration_seconds,omitempty"` //
}

// TaskDependency 任務依賴
type TaskDependency struct {
	DependencyType  string  `json:"dependency_type"`     // 依賴類型
	DependentTaskId string  `json:"dependent_task_id"`   // 依賴任務ID
	Condition       *string `json:"condition,omitempty"` // 依賴條件
	Required        bool    `json:"required,omitempty"`  // 是否必需
}

// AIVerificationRequest AI 驅動漏洞驗證請求
type AIVerificationRequest struct {
	VerificationId    string                 `json:"verification_id"`             //
	FindingId         string                 `json:"finding_id"`                  //
	ScanId            string                 `json:"scan_id"`                     //
	VulnerabilityType interface{}            `json:"vulnerability_type"`          //
	Target            interface{}            `json:"target"`                      //
	Evidence          interface{}            `json:"evidence"`                    //
	VerificationMode  string                 `json:"verification_mode,omitempty"` //
	Context           map[string]interface{} `json:"context,omitempty"`           //
}

// AIVerificationResult AI 驅動漏洞驗證結果
type AIVerificationResult struct {
	VerificationId     string    `json:"verification_id"`           //
	FindingId          string    `json:"finding_id"`                //
	VerificationStatus string    `json:"verification_status"`       //
	ConfidenceScore    float64   `json:"confidence_score"`          //
	VerificationMethod string    `json:"verification_method"`       //
	TestSteps          []string  `json:"test_steps,omitempty"`      //
	Observations       []string  `json:"observations,omitempty"`    //
	Recommendations    []string  `json:"recommendations,omitempty"` //
	Timestamp          time.Time `json:"timestamp,omitempty"`       //
}

// CodeLevelRootCause 程式碼層面根因分析結果
type CodeLevelRootCause struct {
	AnalysisId           string   `json:"analysis_id"`                     //
	VulnerableComponent  string   `json:"vulnerable_component"`            //
	AffectedFindings     []string `json:"affected_findings"`               //
	CodeLocation         *string  `json:"code_location,omitempty"`         //
	VulnerabilityPattern *string  `json:"vulnerability_pattern,omitempty"` //
	FixRecommendation    *string  `json:"fix_recommendation,omitempty"`    //
}

// FindingEvidence 漏洞證據
type FindingEvidence struct {
	Payload           *string     `json:"payload,omitempty"`             //
	ResponseTimeDelta interface{} `json:"response_time_delta,omitempty"` //
	DbVersion         *string     `json:"db_version,omitempty"`          //
	Request           *string     `json:"request,omitempty"`             //
	Response          *string     `json:"response,omitempty"`            //
	Proof             *string     `json:"proof,omitempty"`               //
}

// FindingImpact 漏洞影響描述
type FindingImpact struct {
	Description     *string     `json:"description,omitempty"`      //
	BusinessImpact  *string     `json:"business_impact,omitempty"`  //
	TechnicalImpact *string     `json:"technical_impact,omitempty"` //
	AffectedUsers   interface{} `json:"affected_users,omitempty"`   //
	EstimatedCost   interface{} `json:"estimated_cost,omitempty"`   //
}

// FindingPayload 漏洞發現 Payload - 統一的漏洞報告格式
type FindingPayload struct {
	FindingId      string                 `json:"finding_id"`               //
	TaskId         string                 `json:"task_id"`                  //
	ScanId         string                 `json:"scan_id"`                  //
	Status         string                 `json:"status"`                   //
	Vulnerability  interface{}            `json:"vulnerability"`            //
	Target         interface{}            `json:"target"`                   //
	Strategy       *string                `json:"strategy,omitempty"`       //
	Evidence       interface{}            `json:"evidence,omitempty"`       //
	Impact         interface{}            `json:"impact,omitempty"`         //
	Recommendation interface{}            `json:"recommendation,omitempty"` //
	Metadata       map[string]interface{} `json:"metadata,omitempty"`       //
	CreatedAt      time.Time              `json:"created_at,omitempty"`     //
	UpdatedAt      time.Time              `json:"updated_at,omitempty"`     //
}

// FindingRecommendation 漏洞修復建議
type FindingRecommendation struct {
	Fix              *string  `json:"fix,omitempty"`               //
	Priority         *string  `json:"priority,omitempty"`          //
	RemediationSteps []string `json:"remediation_steps,omitempty"` //
	References       []string `json:"references,omitempty"`        //
}

// FindingTarget 目標資訊 - 漏洞所在位置
type FindingTarget struct {
	Url       interface{}            `json:"url"`                 //
	Parameter *string                `json:"parameter,omitempty"` //
	Method    *string                `json:"method,omitempty"`    //
	Headers   map[string]interface{} `json:"headers,omitempty"`   //
	Params    map[string]interface{} `json:"params,omitempty"`    //
	Body      *string                `json:"body,omitempty"`      //
}

// JavaScriptAnalysisResult JavaScript 分析結果
type JavaScriptAnalysisResult struct {
	AnalysisId         string                 `json:"analysis_id"`                   //
	Url                string                 `json:"url"`                           //
	SourceSizeBytes    int                    `json:"source_size_bytes"`             //
	DangerousFunctions []string               `json:"dangerous_functions,omitempty"` //
	ExternalResources  []string               `json:"external_resources,omitempty"`  //
	DataLeaks          map[string]interface{} `json:"data_leaks,omitempty"`          //
	Findings           []string               `json:"findings,omitempty"`            //
	ApisCalled         []string               `json:"apis_called,omitempty"`         //
	AjaxEndpoints      []string               `json:"ajax_endpoints,omitempty"`      //
	SuspiciousPatterns []string               `json:"suspicious_patterns,omitempty"` //
	RiskScore          float64                `json:"risk_score,omitempty"`          //
	SecurityScore      int                    `json:"security_score,omitempty"`      //
	Timestamp          time.Time              `json:"timestamp,omitempty"`           //
}

// SASTDASTCorrelation SAST-DAST 資料流關聯結果
type SASTDASTCorrelation struct {
	CorrelationId      string   `json:"correlation_id"`        //
	SastFindingId      string   `json:"sast_finding_id"`       //
	DastFindingId      string   `json:"dast_finding_id"`       //
	DataFlowPath       []string `json:"data_flow_path"`        //
	VerificationStatus string   `json:"verification_status"`   //
	ConfidenceScore    float64  `json:"confidence_score"`      //
	Explanation        *string  `json:"explanation,omitempty"` //
}

// SensitiveMatch 敏感資訊匹配結果
type SensitiveMatch struct {
	MatchId     string      `json:"match_id"`              //
	PatternName string      `json:"pattern_name"`          //
	MatchedText string      `json:"matched_text"`          //
	Context     string      `json:"context"`               //
	Confidence  float64     `json:"confidence"`            //
	LineNumber  interface{} `json:"line_number,omitempty"` //
	FilePath    *string     `json:"file_path,omitempty"`   //
	Url         *string     `json:"url,omitempty"`         //
	Severity    interface{} `json:"severity,omitempty"`    //
}

// VulnerabilityCorrelation 漏洞關聯分析結果
type VulnerabilityCorrelation struct {
	CorrelationId    string    `json:"correlation_id"`              //
	CorrelationType  string    `json:"correlation_type"`            //
	RelatedFindings  []string  `json:"related_findings"`            //
	ConfidenceScore  float64   `json:"confidence_score"`            //
	RootCause        *string   `json:"root_cause,omitempty"`        //
	CommonComponents []string  `json:"common_components,omitempty"` //
	Explanation      *string   `json:"explanation,omitempty"`       //
	Timestamp        time.Time `json:"timestamp,omitempty"`         //
}

// ==================== 訊息通訊 ====================

// AivaMessage AIVA統一訊息格式 - 所有跨服務通訊的標準信封
type AivaMessage struct {
	Header        MessageHeader          `json:"header"`         // 訊息標頭
	Topic         string                 `json:"topic"`          // 訊息主題
	SchemaVersion string                 `json:"schema_version"` // Schema版本
	Payload       map[string]interface{} `json:"payload"`        // 訊息載荷
}

// AIVARequest 統一請求格式 - 模組間請求通訊
type AIVARequest struct {
	RequestId      string                 `json:"request_id"`         // 請求識別碼
	SourceModule   string                 `json:"source_module"`      // 來源模組
	TargetModule   string                 `json:"target_module"`      // 目標模組
	RequestType    string                 `json:"request_type"`       // 請求類型
	Payload        map[string]interface{} `json:"payload"`            // 請求載荷
	TraceId        *string                `json:"trace_id,omitempty"` // 追蹤識別碼
	TimeoutSeconds int                    `json:"timeout_seconds"`    // 逾時秒數
	Metadata       map[string]interface{} `json:"metadata,omitempty"` // 中繼資料
	Timestamp      string                 `json:"timestamp"`          // 時間戳
}

// AIVAResponse 統一響應格式 - 模組間響應通訊
type AIVAResponse struct {
	RequestId    string                 `json:"request_id"`              // 對應的請求識別碼
	ResponseType string                 `json:"response_type"`           // 響應類型
	Success      bool                   `json:"success"`                 // 執行是否成功
	Payload      map[string]interface{} `json:"payload,omitempty"`       // 響應載荷
	ErrorCode    *string                `json:"error_code,omitempty"`    // 錯誤代碼
	ErrorMessage *string                `json:"error_message,omitempty"` // 錯誤訊息
	Metadata     map[string]interface{} `json:"metadata,omitempty"`      // 中繼資料
	Timestamp    string                 `json:"timestamp"`               // 時間戳
}

// ==================== 任務管理 ====================

// FunctionTaskPayload 功能任務載荷 - 掃描任務的標準格式
type FunctionTaskPayload struct {
	TaskId         string                 `json:"task_id"`                   // 任務識別碼
	ScanId         string                 `json:"scan_id"`                   // 掃描識別碼
	Priority       int                    `json:"priority"`                  // 任務優先級
	Target         FunctionTaskTarget     `json:"target"`                    // 掃描目標
	Context        FunctionTaskContext    `json:"context"`                   // 任務上下文
	Strategy       string                 `json:"strategy"`                  // 掃描策略
	CustomPayloads []string               `json:"custom_payloads,omitempty"` // 自訂載荷
	TestConfig     FunctionTaskTestConfig `json:"test_config"`               // 測試配置
}

// FunctionTaskTarget 功能任務目標
type FunctionTaskTarget struct {
	ParameterLocation string                 `json:"parameter_location"`  // 參數位置
	Cookies           map[string]string      `json:"cookies,omitempty"`   // Cookie資料
	FormData          map[string]interface{} `json:"form_data,omitempty"` // 表單資料
	JsonData          map[string]interface{} `json:"json_data,omitempty"` // JSON資料
}

// FunctionTaskContext 功能任務上下文
type FunctionTaskContext struct {
	DbTypeHint      *string  `json:"db_type_hint,omitempty"`     // 資料庫類型提示
	WafDetected     bool     `json:"waf_detected"`               // 是否檢測到WAF
	RelatedFindings []string `json:"related_findings,omitempty"` // 相關發現
}

// FunctionTaskTestConfig 功能任務測試配置
type FunctionTaskTestConfig struct {
	Payloads       []string `json:"payloads"`                  // 標準載荷列表
	CustomPayloads []string `json:"custom_payloads,omitempty"` // 自訂載荷列表
	BlindXss       bool     `json:"blind_xss"`                 // 是否進行Blind XSS測試
	DomTesting     bool     `json:"dom_testing"`               // 是否進行DOM測試
	Timeout        *float64 `json:"timeout,omitempty"`         // 請求逾時(秒)
}

// ScanTaskPayload 掃描任務載荷 - 用於SCA/SAST等需要項目URL的掃描任務
type ScanTaskPayload struct {
	TaskId         string                 `json:"task_id"`                   // 任務識別碼
	ScanId         string                 `json:"scan_id"`                   // 掃描識別碼
	Priority       int                    `json:"priority"`                  // 任務優先級
	Target         Target                 `json:"target"`                    // 掃描目標 (包含URL)
	ScanType       string                 `json:"scan_type"`                 // 掃描類型
	RepositoryInfo map[string]interface{} `json:"repository_info,omitempty"` // 代碼倉庫資訊 (分支、commit等)
	Timeout        *int                   `json:"timeout,omitempty"`         // 掃描逾時(秒)
}

// ==================== 發現結果 ====================
