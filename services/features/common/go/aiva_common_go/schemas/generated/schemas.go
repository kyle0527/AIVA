// AIVA Go Schema - 自動生成
// ===========================
//
// AIVA跨語言Schema統一定義
//
// ⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
// 📅 最後更新: 2025-10-23T00:00:00Z
// 🔄 Schema 版本: 1.0.0

package schemas
import "time"

// ==================== 基礎類型 ====================

// MessageHeader 統一訊息標頭 - 所有跨服務通訊的基礎
type MessageHeader struct {
    MessageId            string                    `json:"message_id"`  // 唯一訊息識別碼
    TraceId              string                    `json:"trace_id"`  // 分散式追蹤識別碼
    CorrelationId        *string                   `json:"correlation_id,omitempty"`  // 關聯識別碼 - 用於請求-響應配對
    SourceModule         string                    `json:"source_module"`  // 來源模組名稱
    Timestamp            time.Time                 `json:"timestamp"`  // 訊息時間戳
    Version              string                    `json:"version"`  // Schema版本號
}

// Target 掃描/攻擊目標定義
type Target struct {
    Url                  string                    `json:"url"`  // 目標URL
    Parameter            *string                   `json:"parameter,omitempty"`  // 目標參數名稱
    Method               *string                   `json:"method,omitempty"`  // HTTP方法
    Headers              map[string]string         `json:"headers,omitempty"`  // HTTP標頭
    Params               map[string]interface{}    `json:"params,omitempty"`  // HTTP參數
    Body                 *string                   `json:"body,omitempty"`  // HTTP請求體
}

// Vulnerability 漏洞資訊定義
type Vulnerability struct {
    Name                 string                    `json:"name"`  // 漏洞名稱
    Cwe                  *string                   `json:"cwe,omitempty"`  // CWE編號
    Severity             string                    `json:"severity"`  // 嚴重程度
    Confidence           string                    `json:"confidence"`  // 信心度
    Description          *string                   `json:"description,omitempty"`  // 漏洞描述
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
    RequestId            string                    `json:"request_id"`  // 請求識別碼
    SourceModule         string                    `json:"source_module"`  // 來源模組
    TargetModule         string                    `json:"target_module"`  // 目標模組
    RequestType          string                    `json:"request_type"`  // 請求類型
    Payload              map[string]interface{}    `json:"payload"`  // 請求載荷
    TraceId              *string                   `json:"trace_id,omitempty"`  // 追蹤識別碼
    TimeoutSeconds       int                       `json:"timeout_seconds"`  // 逾時秒數
    Metadata             map[string]interface{}    `json:"metadata,omitempty"`  // 中繼資料
    Timestamp            string                    `json:"timestamp"`  // 時間戳
}

// AIVAResponse 統一響應格式 - 模組間響應通訊
type AIVAResponse struct {
    RequestId            string                    `json:"request_id"`  // 對應的請求識別碼
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
    TaskId               string                    `json:"task_id"`  // 任務識別碼
    ScanId               string                    `json:"scan_id"`  // 掃描識別碼
    Priority             int                       `json:"priority"`  // 任務優先級
    Target               FunctionTaskTarget        `json:"target"`  // 掃描目標
    Context              FunctionTaskContext       `json:"context"`  // 任務上下文
    Strategy             string                    `json:"strategy"`  // 掃描策略
    CustomPayloads       []string                  `json:"custom_payloads,omitempty"`  // 自訂載荷
    TestConfig           FunctionTaskTestConfig    `json:"test_config"`  // 測試配置
}

// FunctionTaskTarget 功能任務目標
type FunctionTaskTarget struct {
    ParameterLocation    string                    `json:"parameter_location"`  // 參數位置
    Cookies              map[string]string         `json:"cookies,omitempty"`  // Cookie資料
    FormData             map[string]interface{}    `json:"form_data,omitempty"`  // 表單資料
    JsonData             map[string]interface{}    `json:"json_data,omitempty"`  // JSON資料
}

// FunctionTaskContext 功能任務上下文
type FunctionTaskContext struct {
    DbTypeHint           *string                   `json:"db_type_hint,omitempty"`  // 資料庫類型提示
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
    TaskId               string                    `json:"task_id"`  // 任務識別碼
    ScanId               string                    `json:"scan_id"`  // 掃描識別碼
    Priority             int                       `json:"priority"`  // 任務優先級
    Target               Target                    `json:"target"`  // 掃描目標 (包含URL)
    ScanType             string                    `json:"scan_type"`  // 掃描類型
    RepositoryInfo       map[string]interface{}    `json:"repository_info,omitempty"`  // 代碼倉庫資訊 (分支、commit等)
    Timeout              *int                      `json:"timeout,omitempty"`  // 掃描逾時(秒)
}

// ==================== 發現結果 ====================

// FindingPayload 漏洞發現載荷 - 掃描結果的標準格式
type FindingPayload struct {
    FindingId            string                    `json:"finding_id"`  // 發現識別碼
    TaskId               string                    `json:"task_id"`  // 任務識別碼
    ScanId               string                    `json:"scan_id"`  // 掃描識別碼
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
    DbVersion            *string                   `json:"db_version,omitempty"`  // 資料庫版本
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
