// AIVA Go Schema - 自動生成
// ===========================
//
// AIVA跨語言Schema統一定義 - 以手動維護版本為準
//
// ⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
// 📅 最後更新: 2025-10-30T00:00:00.000000
// 🔄 Schema 版本: 1.1.0

package schemas

import "time"

// ==================== 枚舉類型 ====================

// Severity 漏洞嚴重程度枚舉
type Severity string

const (
    SeverityCritical               Severity = "critical"  // 嚴重漏洞
    SeverityHigh                   Severity = "high"  // 高風險漏洞
    SeverityMedium                 Severity = "medium"  // 中等風險漏洞
    SeverityLow                    Severity = "low"  // 低風險漏洞
    SeverityInfo                   Severity = "info"  // 資訊性發現
)

// Confidence 漏洞信心度枚舉
type Confidence string

const (
    ConfidenceConfirmed            Confidence = "confirmed"  // 已確認
    ConfidenceFirm                 Confidence = "firm"  // 確實
    ConfidenceTentative            Confidence = "tentative"  // 暫定
)

// FindingStatus 發現狀態枚舉
type FindingStatus string

const (
    FindingStatusNew               FindingStatus = "new"  // 新發現
    FindingStatusConfirmed         FindingStatus = "confirmed"  // 已確認
    FindingStatusResolved          FindingStatus = "resolved"  // 已解決
    FindingStatusFalse_Positive    FindingStatus = "false_positive"  // 誤報
)

// AsyncTaskStatus 異步任務狀態枚舉
type AsyncTaskStatus string

const (
    AsyncTaskStatusPending         AsyncTaskStatus = "pending"  // 等待中
    AsyncTaskStatusRunning         AsyncTaskStatus = "running"  // 執行中
    AsyncTaskStatusCompleted       AsyncTaskStatus = "completed"  // 已完成
    AsyncTaskStatusFailed          AsyncTaskStatus = "failed"  // 執行失敗
    AsyncTaskStatusCancelled       AsyncTaskStatus = "cancelled"  // 已取消
    AsyncTaskStatusTimeout         AsyncTaskStatus = "timeout"  // 執行超時
    AsyncTaskStatusRetrying        AsyncTaskStatus = "retrying"  // 重試中
)

// PluginStatus 插件狀態枚舉
type PluginStatus string

const (
    PluginStatusInactive           PluginStatus = "inactive"  // 未啟用
    PluginStatusActive             PluginStatus = "active"  // 已啟用
    PluginStatusLoading            PluginStatus = "loading"  // 載入中
    PluginStatusError              PluginStatus = "error"  // 錯誤狀態
    PluginStatusUpdating           PluginStatus = "updating"  // 更新中
)

// PluginType 插件類型枚舉
type PluginType string

const (
    PluginTypeScanner              PluginType = "scanner"  // 掃描器插件
    PluginTypeFilter               PluginType = "filter"  // 過濾器插件
    PluginTypeReporter             PluginType = "reporter"  // 報告器插件
    PluginTypeIntegration          PluginType = "integration"  // 整合插件
    PluginTypeUtility              PluginType = "utility"  // 工具插件
)

// ==================== 基礎類型 ====================

// MessageHeader 訊息標頭 - 用於所有訊息的統一標頭格式
type MessageHeader struct {
    MessageID            string                    `json:"message_id"`  // 
    TraceID              string                    `json:"trace_id"`  // 
    CorrelationID        *string                   `json:"correlation_id,omitempty"`  // 
    SourceModule         string                    `json:"source_module"`  // 來源模組名稱
    Timestamp            time.Time                 `json:"timestamp,omitempty"`  // 
    Version              string                    `json:"version,omitempty"`  // 
}

// Target 目標資訊 - 漏洞所在位置
type Target struct {
    URL                  interface{}               `json:"url"`  // 
    Parameter            *string                   `json:"parameter,omitempty"`  // 
    Method               *string                   `json:"method,omitempty"`  // 
    Headers              map[string]interface{}    `json:"headers,omitempty"`  // 
    Params               map[string]interface{}    `json:"params,omitempty"`  // 
    Body                 *string                   `json:"body,omitempty"`  // 
}

// Vulnerability 漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP
type Vulnerability struct {
    Name                 interface{}               `json:"name"`  // 
    CWE                  *string                   `json:"cwe,omitempty"`  // CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/
    CVE                  *string                   `json:"cve,omitempty"`  // CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/
    Severity             interface{}               `json:"severity"`  // 
    Confidence           interface{}               `json:"confidence"`  // 
    Description          *string                   `json:"description,omitempty"`  // 
    CvssScore            interface{}               `json:"cvss_score,omitempty"`  // CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/
    CvssVector           *string                   `json:"cvss_vector,omitempty"`  // CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
    OWASPCategory        *string                   `json:"owasp_category,omitempty"`  // OWASP Top 10 分類，例如: A03:2021-Injection
}

// Asset 資產基本資訊
type Asset struct {
    AssetID              string                    `json:"asset_id"`  // 
    Type                 string                    `json:"type"`  // 
    Value                string                    `json:"value"`  // 
    Parameters           []string                  `json:"parameters,omitempty"`  // 
    HasForm              bool                      `json:"has_form,omitempty"`  // 
}

// Authentication 認證資訊
type Authentication struct {
    Method               string                    `json:"method,omitempty"`  // 
    Credentials          map[string]interface{}    `json:"credentials,omitempty"`  // 
}

// ExecutionError 執行錯誤統一格式
type ExecutionError struct {
    ErrorID              string                    `json:"error_id"`  // 
    ErrorType            string                    `json:"error_type"`  // 
    Message              string                    `json:"message"`  // 
    Payload              *string                   `json:"payload,omitempty"`  // 
    Vector               *string                   `json:"vector,omitempty"`  // 
    Timestamp            time.Time                 `json:"timestamp,omitempty"`  // 
    Attempts             int                       `json:"attempts,omitempty"`  // 
}

// Fingerprints 技術指紋
type Fingerprints struct {
    WebServer            map[string]interface{}    `json:"web_server,omitempty"`  // 
    Framework            map[string]interface{}    `json:"framework,omitempty"`  // 
    Language             map[string]interface{}    `json:"language,omitempty"`  // 
    WafDetected          bool                      `json:"waf_detected,omitempty"`  // 
    WafVendor            *string                   `json:"waf_vendor,omitempty"`  // 
}

// RateLimit 速率限制
type RateLimit struct {
    RequestsPerSecond    int                       `json:"requests_per_second,omitempty"`  // 
    Burst                int                       `json:"burst,omitempty"`  // 
}

// RiskFactor 風險因子
type RiskFactor struct {
    FactorName           string                    `json:"factor_name"`  // 風險因子名稱
    Weight               float64                   `json:"weight"`  // 權重
    Value                float64                   `json:"value"`  // 因子值
    Description          *string                   `json:"description,omitempty"`  // 因子描述
}

// ScanScope 掃描範圍
type ScanScope struct {
    Exclusions           []string                  `json:"exclusions,omitempty"`  // 
    IncludeSubdomains    bool                      `json:"include_subdomains,omitempty"`  // 
    AllowedHosts         []string                  `json:"allowed_hosts,omitempty"`  // 
}

// Summary 掃描摘要
type Summary struct {
    UrlsFound            int                       `json:"urls_found,omitempty"`  // 
    FormsFound           int                       `json:"forms_found,omitempty"`  // 
    ApisFound            int                       `json:"apis_found,omitempty"`  // 
    ScanDurationSeconds  int                       `json:"scan_duration_seconds,omitempty"`  // 
}

// TaskDependency 任務依賴
type TaskDependency struct {
    DependencyType       string                    `json:"dependency_type"`  // 依賴類型
    DependentTaskID      string                    `json:"dependent_task_id"`  // 依賴任務ID
    Condition            *string                   `json:"condition,omitempty"`  // 依賴條件
    Required             bool                      `json:"required,omitempty"`  // 是否必需
}

// AIVerificationRequest AI 驅動漏洞驗證請求
type AIVerificationRequest struct {
    VerificationID       string                    `json:"verification_id"`  // 
    FindingID            string                    `json:"finding_id"`  // 
    ScanID               string                    `json:"scan_id"`  // 
    VulnerabilityType    interface{}               `json:"vulnerability_type"`  // 
    Target               interface{}               `json:"target"`  // 
    Evidence             interface{}               `json:"evidence"`  // 
    VerificationMode     string                    `json:"verification_mode,omitempty"`  // 
    Context              map[string]interface{}    `json:"context,omitempty"`  // 
}

// AIVerificationResult AI 驅動漏洞驗證結果
type AIVerificationResult struct {
    VerificationID       string                    `json:"verification_id"`  // 
    FindingID            string                    `json:"finding_id"`  // 
    VerificationStatus   string                    `json:"verification_status"`  // 
    ConfidenceScore      float64                   `json:"confidence_score"`  // 
    VerificationMethod   string                    `json:"verification_method"`  // 
    TestSteps            []string                  `json:"test_steps,omitempty"`  // 
    Observations         []string                  `json:"observations,omitempty"`  // 
    Recommendations      []string                  `json:"recommendations,omitempty"`  // 
    Timestamp            time.Time                 `json:"timestamp,omitempty"`  // 
}

// CodeLevelRootCause 程式碼層面根因分析結果
type CodeLevelRootCause struct {
    AnalysisID           string                    `json:"analysis_id"`  // 
    VulnerableComponent  string                    `json:"vulnerable_component"`  // 
    AffectedFindings     []string                  `json:"affected_findings"`  // 
    CodeLocation         *string                   `json:"code_location,omitempty"`  // 
    VulnerabilityPattern *string                   `json:"vulnerability_pattern,omitempty"`  // 
    FixRecommendation    *string                   `json:"fix_recommendation,omitempty"`  // 
}

// FindingTarget 目標資訊 - 漏洞所在位置
type FindingTarget struct {
    URL                  interface{}               `json:"url"`  // 
    Parameter            *string                   `json:"parameter,omitempty"`  // 
    Method               *string                   `json:"method,omitempty"`  // 
    Headers              map[string]interface{}    `json:"headers,omitempty"`  // 
    Params               map[string]interface{}    `json:"params,omitempty"`  // 
    Body                 *string                   `json:"body,omitempty"`  // 
}

// JavaScriptAnalysisResult JavaScript 分析結果
type JavaScriptAnalysisResult struct {
    AnalysisID           string                    `json:"analysis_id"`  // 
    URL                  string                    `json:"url"`  // 
    SourceSizeBytes      int                       `json:"source_size_bytes"`  // 
    DangerousFunctions   []string                  `json:"dangerous_functions,omitempty"`  // 
    ExternalResources    []string                  `json:"external_resources,omitempty"`  // 
    DataLeaks            map[string]interface{}    `json:"data_leaks,omitempty"`  // 
    Findings             []string                  `json:"findings,omitempty"`  // 
    ApisCalled           []string                  `json:"apis_called,omitempty"`  // 
    AjaxEndpoints        []string                  `json:"ajax_endpoints,omitempty"`  // 
    SuspiciousPatterns   []string                  `json:"suspicious_patterns,omitempty"`  // 
    RiskScore            float64                   `json:"risk_score,omitempty"`  // 
    SecurityScore        int                       `json:"security_score,omitempty"`  // 
    Timestamp            time.Time                 `json:"timestamp,omitempty"`  // 
}

// SASTDASTCorrelation SAST-DAST 資料流關聯結果
type SASTDASTCorrelation struct {
    CorrelationID        string                    `json:"correlation_id"`  // 
    SastFindingID        string                    `json:"sast_finding_id"`  // 
    DastFindingID        string                    `json:"dast_finding_id"`  // 
    DataFlowPath         []string                  `json:"data_flow_path"`  // 
    VerificationStatus   string                    `json:"verification_status"`  // 
    ConfidenceScore      float64                   `json:"confidence_score"`  // 
    Explanation          *string                   `json:"explanation,omitempty"`  // 
}

// SensitiveMatch 敏感資訊匹配結果
type SensitiveMatch struct {
    MatchID              string                    `json:"match_id"`  // 
    PatternName          string                    `json:"pattern_name"`  // 
    MatchedText          string                    `json:"matched_text"`  // 
    Context              string                    `json:"context"`  // 
    Confidence           float64                   `json:"confidence"`  // 
    LineNumber           interface{}               `json:"line_number,omitempty"`  // 
    FilePath             *string                   `json:"file_path,omitempty"`  // 
    URL                  *string                   `json:"url,omitempty"`  // 
    Severity             interface{}               `json:"severity,omitempty"`  // 
}

// VulnerabilityCorrelation 漏洞關聯分析結果
type VulnerabilityCorrelation struct {
    CorrelationID        string                    `json:"correlation_id"`  // 
    CorrelationType      string                    `json:"correlation_type"`  // 
    RelatedFindings      []string                  `json:"related_findings"`  // 
    ConfidenceScore      float64                   `json:"confidence_score"`  // 
    RootCause            *string                   `json:"root_cause,omitempty"`  // 
    CommonComponents     []string                  `json:"common_components,omitempty"`  // 
    Explanation          *string                   `json:"explanation,omitempty"`  // 
    Timestamp            time.Time                 `json:"timestamp,omitempty"`  // 
}

// ==================== 訊息通訊 ====================

// AivaMessage AIVA統一訊息格式 - 所有跨服務通訊的標準信封
type AivaMessage struct {
    Header               MessageHeader             `json:"header"`  // 訊息標頭
    Topic                string                    `json:"topic"`  // 訊息主題
    SchemaVersion        string                    `json:"schema_version"`  // Schema版本
    Payload              map[string]interface{}    `json:"payload"`  // 訊息載荷
}

// AIVARequest 統一請求格式 - 模組間請求通訊
type AIVARequest struct {
    RequestID            string                    `json:"request_id"`  // 請求識別碼
    SourceModule         string                    `json:"source_module"`  // 來源模組
    TargetModule         string                    `json:"target_module"`  // 目標模組
    RequestType          string                    `json:"request_type"`  // 請求類型
    Payload              map[string]interface{}    `json:"payload"`  // 請求載荷
    TraceID              *string                   `json:"trace_id,omitempty"`  // 追蹤識別碼
    TimeoutSeconds       int                       `json:"timeout_seconds"`  // 逾時秒數
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 中繼資料
    Timestamp            string                    `json:"timestamp"`  // 時間戳
}

// AIVAResponse 統一響應格式 - 模組間響應通訊
type AIVAResponse struct {
    RequestID            string                    `json:"request_id"`  // 對應的請求識別碼
    ResponseType         string                    `json:"response_type"`  // 響應類型
    Success              bool                      `json:"success"`  // 執行是否成功
    Payload              map[string]interface{}    `json:"payload,omitempty"`  // 響應載荷
    ErrorCode            *string                   `json:"error_code,omitempty"`  // 錯誤代碼
    ErrorMessage         *string                   `json:"error_message,omitempty"`  // 錯誤訊息
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 中繼資料
    Timestamp            string                    `json:"timestamp"`  // 時間戳
}

// ==================== 任務管理 ====================

// FunctionTaskPayload 功能任務載荷 - 掃描任務的標準格式
type FunctionTaskPayload struct {
    TaskID               string                    `json:"task_id"`  // 任務識別碼
    ScanID               string                    `json:"scan_id"`  // 掃描識別碼
    Priority             int                       `json:"priority"`  // 任務優先級
    Target               FunctionTaskTarget        `json:"target"`  // 掃描目標
    Context              FunctionTaskContext       `json:"context"`  // 任務上下文
    Strategy             string                    `json:"strategy"`  // 掃描策略
    CustomPayloads       []string                  `json:"custom_payloads,omitempty"`  // 自訂載荷
    TestConfig           FunctionTaskTestConfig    `json:"test_config"`  // 測試配置
}

// FunctionTaskTarget 功能任務目標
type FunctionTaskTarget struct {
    URL                  interface{}               `json:"url"`  // 
    Parameter            *string                   `json:"parameter,omitempty"`  // 
    Method               *string                   `json:"method,omitempty"`  // 
    Headers              map[string]interface{}    `json:"headers,omitempty"`  // 
    Params               map[string]interface{}    `json:"params,omitempty"`  // 
    Body                 *string                   `json:"body,omitempty"`  // 
    ParameterLocation    string                    `json:"parameter_location"`  // 參數位置
    Cookies              map[string]string         `json:"cookies,omitempty"`  // Cookie資料
    FormData             map[string]interface{}    `json:"form_data,omitempty"`  // 表單資料
    JSONData             map[string]interface{}    `json:"json_data,omitempty"`  // JSON資料
}

// FunctionTaskContext 功能任務上下文
type FunctionTaskContext struct {
    DBTypeHint           *string                   `json:"db_type_hint,omitempty"`  // 資料庫類型提示
    WafDetected          bool                      `json:"waf_detected"`  // 是否檢測到WAF
    RelatedFindings      []string                  `json:"related_findings,omitempty"`  // 相關發現
}

// FunctionTaskTestConfig 功能任務測試配置
type FunctionTaskTestConfig struct {
    Payloads             []string                  `json:"payloads"`  // 標準載荷列表
    CustomPayloads       []string                  `json:"custom_payloads,omitempty"`  // 自訂載荷列表
    BlindXss             bool                      `json:"blind_xss"`  // 是否進行Blind XSS測試
    DomTesting           bool                      `json:"dom_testing"`  // 是否進行DOM測試
    Timeout              *float64                  `json:"timeout,omitempty"`  // 請求逾時(秒)
}

// ScanTaskPayload 掃描任務載荷 - 用於SCA/SAST等需要項目URL的掃描任務
type ScanTaskPayload struct {
    TaskID               string                    `json:"task_id"`  // 任務識別碼
    ScanID               string                    `json:"scan_id"`  // 掃描識別碼
    Priority             int                       `json:"priority"`  // 任務優先級
    Target               Target                    `json:"target"`  // 掃描目標 (包含URL)
    ScanType             string                    `json:"scan_type"`  // 掃描類型
    RepositoryInfo       map[string]interface{}    `json:"repository_info,omitempty"`  // 代碼倉庫資訊 (分支、commit等)
    Timeout              *int                      `json:"timeout,omitempty"`  // 掃描逾時(秒)
}

// ==================== 發現結果 ====================

// FindingPayload 漏洞發現載荷 - 掃描結果的標準格式
type FindingPayload struct {
    FindingID            string                    `json:"finding_id"`  // 發現識別碼
    TaskID               string                    `json:"task_id"`  // 任務識別碼
    ScanID               string                    `json:"scan_id"`  // 掃描識別碼
    Status               string                    `json:"status"`  // 發現狀態
    Vulnerability        Vulnerability             `json:"vulnerability"`  // 漏洞資訊
    Target               Target                    `json:"target"`  // 目標資訊
    Strategy             *string                   `json:"strategy,omitempty"`  // 使用的策略
    Evidence             *FindingEvidence          `json:"evidence,omitempty"`  // 證據資料
    Impact               *FindingImpact            `json:"impact,omitempty"`  // 影響評估
    Recommendation       *FindingRecommendation    `json:"recommendation,omitempty"`  // 修復建議
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 中繼資料
    CreatedAt            time.Time                 `json:"created_at"`  // 建立時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// FindingEvidence 漏洞證據
type FindingEvidence struct {
    Payload              *string                   `json:"payload,omitempty"`  // 攻擊載荷
    ResponseTimeDelta    *float64                  `json:"response_time_delta,omitempty"`  // 響應時間差異
    DBVersion            *string                   `json:"db_version,omitempty"`  // 資料庫版本
    Request              *string                   `json:"request,omitempty"`  // HTTP請求
    Response             *string                   `json:"response,omitempty"`  // HTTP響應
    Proof                *string                   `json:"proof,omitempty"`  // 證明資料
}

// FindingImpact 漏洞影響評估
type FindingImpact struct {
    Description          *string                   `json:"description,omitempty"`  // 影響描述
    BusinessImpact       *string                   `json:"business_impact,omitempty"`  // 業務影響
    TechnicalImpact      *string                   `json:"technical_impact,omitempty"`  // 技術影響
    AffectedUsers        *int                      `json:"affected_users,omitempty"`  // 受影響用戶數
    EstimatedCost        *float64                  `json:"estimated_cost,omitempty"`  // 估計成本
}

// FindingRecommendation 漏洞修復建議
type FindingRecommendation struct {
    Fix                  *string                   `json:"fix,omitempty"`  // 修復方法
    Priority             *string                   `json:"priority,omitempty"`  // 修復優先級
    RemediationSteps     []string                  `json:"remediation_steps,omitempty"`  // 修復步驟
    References           []string                  `json:"references,omitempty"`  // 參考資料
}

// TokenTestResult Token 測試結果
type TokenTestResult struct {
    Vulnerable           bool                      `json:"vulnerable"`  // 是否存在漏洞
    TokenType            string                    `json:"token_type"`  // Token 類型 (jwt, session, api, etc.)
    Issue                string                    `json:"issue"`  // 發現的問題
    Details              string                    `json:"details"`  // 詳細描述
    DecodedPayload       map[string]interface{}    `json:"decoded_payload,omitempty"`  // 解碼後的載荷內容
    Severity             string                    `json:"severity,omitempty"`  // 漏洞嚴重程度
    TestType             string                    `json:"test_type"`  // 測試類型
}

// ==================== 異步工具 ====================

// RetryConfig 重試配置
type RetryConfig struct {
    MaxAttempts          int                       `json:"max_attempts"`  // 最大重試次數
    BackoffBase          float64                   `json:"backoff_base"`  // 退避基礎時間(秒)
    BackoffFactor        float64                   `json:"backoff_factor"`  // 退避倍數
    MaxBackoff           float64                   `json:"max_backoff"`  // 最大退避時間(秒)
    ExponentialBackoff   bool                      `json:"exponential_backoff"`  // 是否使用指數退避
}

// ResourceLimits 資源限制配置
type ResourceLimits struct {
    MaxMemoryMb          *int                      `json:"max_memory_mb,omitempty"`  // 最大內存限制(MB)
    MaxCPUPercent        *float64                  `json:"max_cpu_percent,omitempty"`  // 最大CPU使用率(%)
    MaxExecutionTime     *int                      `json:"max_execution_time,omitempty"`  // 最大執行時間(秒)
    MaxConcurrentTasks   int                       `json:"max_concurrent_tasks"`  // 最大並發任務數
}

// AsyncTaskConfig 異步任務配置
type AsyncTaskConfig struct {
    TaskName             string                    `json:"task_name"`  // 任務名稱
    TimeoutSeconds       int                       `json:"timeout_seconds"`  // 超時時間(秒)
    RetryConfig          RetryConfig               `json:"retry_config"`  // 重試配置
    Priority             int                       `json:"priority"`  // 任務優先級
    ResourceLimits       ResourceLimits            `json:"resource_limits"`  // 資源限制
    Tags                 []string                  `json:"tags,omitempty"`  // 任務標籤
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 任務元數據
}

// AsyncTaskResult 異步任務結果
type AsyncTaskResult struct {
    TaskID               string                    `json:"task_id"`  // 任務ID
    TaskName             string                    `json:"task_name"`  // 任務名稱
    Status               string                    `json:"status"`  // 任務狀態
    Result               map[string]interface{}    `json:"result,omitempty"`  // 執行結果
    ErrorMessage         *string                   `json:"error_message,omitempty"`  // 錯誤信息
    ExecutionTimeMs      float64                   `json:"execution_time_ms"`  // 執行時間(毫秒)
    StartTime            time.Time                 `json:"start_time"`  // 開始時間
    EndTime              *time.Time                `json:"end_time,omitempty"`  // 結束時間
    RetryCount           int                       `json:"retry_count"`  // 重試次數
    ResourceUsage        map[string]interface{}    `json:"resource_usage,omitempty"`  // 資源使用情況
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 結果元數據
}

// AsyncBatchConfig 異步批次任務配置
type AsyncBatchConfig struct {
    BatchID              string                    `json:"batch_id"`  // 批次ID
    BatchName            string                    `json:"batch_name"`  // 批次名稱
    Tasks                []AsyncTaskConfig         `json:"tasks"`  // 任務列表
    MaxConcurrent        int                       `json:"max_concurrent"`  // 最大並發數
    StopOnFirstError     bool                      `json:"stop_on_first_error"`  // 遇到第一個錯誤時停止
    BatchTimeoutSeconds  int                       `json:"batch_timeout_seconds"`  // 批次超時時間(秒)
}

// AsyncBatchResult 異步批次任務結果
type AsyncBatchResult struct {
    BatchID              string                    `json:"batch_id"`  // 批次ID
    BatchName            string                    `json:"batch_name"`  // 批次名稱
    TotalTasks           int                       `json:"total_tasks"`  // 總任務數
    CompletedTasks       int                       `json:"completed_tasks"`  // 已完成任務數
    FailedTasks          int                       `json:"failed_tasks"`  // 失敗任務數
    TaskResults          []AsyncTaskResult         `json:"task_results,omitempty"`  // 任務結果列表
    BatchStatus          string                    `json:"batch_status"`  // 批次狀態
    StartTime            time.Time                 `json:"start_time"`  // 開始時間
    EndTime              *time.Time                `json:"end_time,omitempty"`  // 結束時間
    TotalExecutionTimeMs float64                   `json:"total_execution_time_ms"`  // 總執行時間(毫秒)
}

// ==================== 插件管理 ====================

// PluginManifest 插件清單
type PluginManifest struct {
    PluginID             string                    `json:"plugin_id"`  // 插件唯一標識符
    Name                 string                    `json:"name"`  // 插件名稱
    Version              string                    `json:"version"`  // 插件版本
    Author               string                    `json:"author"`  // 插件作者
    Description          string                    `json:"description"`  // 插件描述
    PluginType           string                    `json:"plugin_type"`  // 插件類型
    Dependencies         []string                  `json:"dependencies,omitempty"`  // 依賴插件列表
    Permissions          []string                  `json:"permissions,omitempty"`  // 所需權限列表
    ConfigSchema         map[string]interface{}    `json:"config_schema,omitempty"`  // 配置 Schema
    MinAivaVersion       string                    `json:"min_aiva_version"`  // 最低AIVA版本要求
    MaxAivaVersion       *string                   `json:"max_aiva_version,omitempty"`  // 最高AIVA版本要求
    EntryPoint           string                    `json:"entry_point"`  // 插件入口點
    Homepage             *string                   `json:"homepage,omitempty"`  // 插件主頁
    Repository           *string                   `json:"repository,omitempty"`  // 源碼倉庫
    License              string                    `json:"license"`  // 許可證
    Keywords             []string                  `json:"keywords,omitempty"`  // 關鍵詞
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// PluginExecutionContext 插件執行上下文
type PluginExecutionContext struct {
    PluginID             string                    `json:"plugin_id"`  // 插件ID
    ExecutionID          string                    `json:"execution_id"`  // 執行ID
    InputData            map[string]interface{}    `json:"input_data"`  // 輸入數據
    Context              map[string]interface{}    `json:"context,omitempty"`  // 執行上下文
    TimeoutSeconds       int                       `json:"timeout_seconds"`  // 執行超時時間(秒)
    Environment          map[string]string         `json:"environment,omitempty"`  // 環境變數
    WorkingDirectory     *string                   `json:"working_directory,omitempty"`  // 工作目錄
    UserID               *string                   `json:"user_id,omitempty"`  // 執行用戶ID
    SessionID            *string                   `json:"session_id,omitempty"`  // 會話ID
    TraceID              *string                   `json:"trace_id,omitempty"`  // 追蹤ID
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 元數據
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
}

// PluginExecutionResult 插件執行結果
type PluginExecutionResult struct {
    ExecutionID          string                    `json:"execution_id"`  // 執行ID
    PluginID             string                    `json:"plugin_id"`  // 插件ID
    Success              bool                      `json:"success"`  // 執行是否成功
    ResultData           map[string]interface{}    `json:"result_data,omitempty"`  // 結果數據
    ErrorMessage         *string                   `json:"error_message,omitempty"`  // 錯誤信息
    ErrorCode            *string                   `json:"error_code,omitempty"`  // 錯誤代碼
    ExecutionTimeMs      float64                   `json:"execution_time_ms"`  // 執行時間(毫秒)
    MemoryUsageMb        *float64                  `json:"memory_usage_mb,omitempty"`  // 內存使用量(MB)
    OutputLogs           []string                  `json:"output_logs,omitempty"`  // 輸出日誌
    Warnings             []string                  `json:"warnings,omitempty"`  // 警告信息
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 結果元數據
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
}

// PluginConfig 插件配置
type PluginConfig struct {
    PluginID             string                    `json:"plugin_id"`  // 插件ID
    Enabled              bool                      `json:"enabled"`  // 是否啟用
    Configuration        map[string]interface{}    `json:"configuration,omitempty"`  // 配置參數
    Priority             int                       `json:"priority"`  // 執行優先級
    AutoStart            bool                      `json:"auto_start"`  // 是否自動啟動
    MaxInstances         int                       `json:"max_instances"`  // 最大實例數
    ResourceLimits       map[string]interface{}    `json:"resource_limits,omitempty"`  // 資源限制
    EnvironmentVariables map[string]string         `json:"environment_variables,omitempty"`  // 環境變數
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// PluginRegistry 插件註冊表
type PluginRegistry struct {
    RegistryID           string                    `json:"registry_id"`  // 註冊表ID
    Name                 string                    `json:"name"`  // 註冊表名稱
    Plugins              map[string]PluginManifest `json:"plugins,omitempty"`  // 已註冊插件
    TotalPlugins         int                       `json:"total_plugins"`  // 插件總數
    ActivePlugins        int                       `json:"active_plugins"`  // 活躍插件數
    RegistryVersion      string                    `json:"registry_version"`  // 註冊表版本
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// PluginHealthCheck 插件健康檢查
type PluginHealthCheck struct {
    PluginID             string                    `json:"plugin_id"`  // 插件ID
    Status               string                    `json:"status"`  // 插件狀態
    LastCheckTime        time.Time                 `json:"last_check_time"`  // 最後檢查時間
    ResponseTimeMs       *float64                  `json:"response_time_ms,omitempty"`  // 響應時間(毫秒)
    ErrorMessage         *string                   `json:"error_message,omitempty"`  // 錯誤信息
    HealthScore          float64                   `json:"health_score"`  // 健康分數
    UptimePercentage     float64                   `json:"uptime_percentage"`  // 運行時間百分比
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 健康檢查元數據
}

// ==================== CLI 界面 ====================

// CLIParameter CLI 參數定義
type CLIParameter struct {
    Name                 string                    `json:"name"`  // 參數名稱
    Type                 string                    `json:"type"`  // 參數類型
    Description          string                    `json:"description"`  // 參數描述
    Required             bool                      `json:"required"`  // 是否必需
    DefaultValue         *interface{}              `json:"default_value,omitempty"`  // 默認值
    Choices              []string                  `json:"choices,omitempty"`  // 可選值列表
    MinValue             *float64                  `json:"min_value,omitempty"`  // 最小值
    MaxValue             *float64                  `json:"max_value,omitempty"`  // 最大值
    Pattern              *string                   `json:"pattern,omitempty"`  // 正則表達式模式
    HelpText             *string                   `json:"help_text,omitempty"`  // 幫助文本
}

// CLICommand CLI 命令定義
type CLICommand struct {
    CommandName          string                    `json:"command_name"`  // 命令名稱
    Description          string                    `json:"description"`  // 命令描述
    Category             string                    `json:"category"`  // 命令分類
    Parameters           []CLIParameter            `json:"parameters,omitempty"`  // 命令參數列表
    Examples             []string                  `json:"examples,omitempty"`  // 使用示例
    Aliases              []string                  `json:"aliases,omitempty"`  // 命令別名
    Deprecated           bool                      `json:"deprecated"`  // 是否已棄用
    MinArgs              int                       `json:"min_args"`  // 最少參數數量
    MaxArgs              *int                      `json:"max_args,omitempty"`  // 最多參數數量
    RequiresAuth         bool                      `json:"requires_auth"`  // 是否需要認證
    Permissions          []string                  `json:"permissions,omitempty"`  // 所需權限
    Tags                 []string                  `json:"tags,omitempty"`  // 標籤
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// CLIExecutionResult CLI 執行結果
type CLIExecutionResult struct {
    Command              string                    `json:"command"`  // 執行的命令
    Arguments            []string                  `json:"arguments,omitempty"`  // 命令參數
    ExitCode             int                       `json:"exit_code"`  // 退出代碼
    Stdout               string                    `json:"stdout"`  // 標準輸出
    Stderr               string                    `json:"stderr"`  // 標準錯誤
    ExecutionTimeMs      float64                   `json:"execution_time_ms"`  // 執行時間(毫秒)
    StartTime            time.Time                 `json:"start_time"`  // 開始時間
    EndTime              *time.Time                `json:"end_time,omitempty"`  // 結束時間
    UserID               *string                   `json:"user_id,omitempty"`  // 執行用戶ID
    SessionID            *string                   `json:"session_id,omitempty"`  // 會話ID
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 執行元數據
}

// CLISession CLI 會話
type CLISession struct {
    SessionID            string                    `json:"session_id"`  // 會話ID
    UserID               *string                   `json:"user_id,omitempty"`  // 用戶ID
    StartTime            time.Time                 `json:"start_time"`  // 開始時間
    EndTime              *time.Time                `json:"end_time,omitempty"`  // 結束時間
    CommandHistory       []string                  `json:"command_history,omitempty"`  // 命令歷史
    Environment          map[string]string         `json:"environment,omitempty"`  // 環境變數
    WorkingDirectory     string                    `json:"working_directory"`  // 工作目錄
    Active               bool                      `json:"active"`  // 會話是否活躍
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 會話元數據
}

// CLIConfiguration CLI 配置
type CLIConfiguration struct {
    ConfigID             string                    `json:"config_id"`  // 配置ID
    Name                 string                    `json:"name"`  // 配置名稱
    Settings             map[string]interface{}    `json:"settings,omitempty"`  // 配置設定
    AutoCompletion       bool                      `json:"auto_completion"`  // 是否啟用自動完成
    HistorySize          int                       `json:"history_size"`  // 歷史記錄大小
    PromptStyle          string                    `json:"prompt_style"`  // 提示符樣式
    ColorScheme          string                    `json:"color_scheme"`  // 顏色方案
    TimeoutSeconds       int                       `json:"timeout_seconds"`  // 命令超時時間(秒)
    CreatedAt            time.Time                 `json:"created_at"`  // 創建時間
    UpdatedAt            time.Time                 `json:"updated_at"`  // 更新時間
}

// CLIMetrics CLI 使用指標
type CLIMetrics struct {
    MetricID             string                    `json:"metric_id"`  // 指標ID
    CommandCount         int                       `json:"command_count"`  // 命令執行總數
    SuccessfulCommands   int                       `json:"successful_commands"`  // 成功執行的命令數
    FailedCommands       int                       `json:"failed_commands"`  // 失敗的命令數
    AverageExecutionTimeMs float64                   `json:"average_execution_time_ms"`  // 平均執行時間(毫秒)
    MostUsedCommands     []string                  `json:"most_used_commands,omitempty"`  // 最常用命令列表
    PeakUsageTime        *time.Time                `json:"peak_usage_time,omitempty"`  // 峰值使用時間
    CollectionPeriodStart time.Time                 `json:"collection_period_start"`  // 統計開始時間
    CollectionPeriodEnd  *time.Time                `json:"collection_period_end,omitempty"`  // 統計結束時間
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 統計元數據
}
