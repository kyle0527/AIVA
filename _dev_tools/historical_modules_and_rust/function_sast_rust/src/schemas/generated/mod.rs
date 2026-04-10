// AIVA Rust Schema - 自動生成
// 版本: 1.0.0
// 生成時間: N/A
// 
// 完整的 Rust Schema 實現，包含序列化/反序列化支持

#![allow(dead_code)] // Generated schemas for future cross-service communication
#![allow(unused_imports)] // Standard imports for generated code

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
// Note: uuid::Uuid and other imports will be used by generated schemas

/// 漏洞嚴重程度枚舉
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    /// 嚴重漏洞
    CRITICAL,
    /// 高風險漏洞
    HIGH,
    /// 中等風險漏洞
    MEDIUM,
    /// 低風險漏洞
    LOW,
    /// 資訊性發現
    INFO,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::CRITICAL => write!(f, "critical"),
            Severity::HIGH => write!(f, "high"),
            Severity::MEDIUM => write!(f, "medium"),
            Severity::LOW => write!(f, "low"),
            Severity::INFO => write!(f, "info"),
        }
    }
}

impl std::str::FromStr for Severity {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "CRITICAL" => Ok(Severity::CRITICAL),
            "HIGH" => Ok(Severity::HIGH),
            "MEDIUM" => Ok(Severity::MEDIUM),
            "LOW" => Ok(Severity::LOW),
            "INFO" => Ok(Severity::INFO),
            _ => Err(format!("Invalid Severity: {}", s)),
        }
    }
}

/// 漏洞信心度枚舉
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Confidence {
    /// 已確認
    CONFIRMED,
    /// 確實
    FIRM,
    /// 暫定
    TENTATIVE,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::CONFIRMED => write!(f, "confirmed"),
            Confidence::FIRM => write!(f, "firm"),
            Confidence::TENTATIVE => write!(f, "tentative"),
        }
    }
}

impl std::str::FromStr for Confidence {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "CONFIRMED" => Ok(Confidence::CONFIRMED),
            "FIRM" => Ok(Confidence::FIRM),
            "TENTATIVE" => Ok(Confidence::TENTATIVE),
            _ => Err(format!("Invalid Confidence: {}", s)),
        }
    }
}

/// 發現狀態枚舉
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(dead_code)] // Generated schema for future use
pub enum FindingStatus {
    /// 新發現
    NEW,
    /// 已確認
    CONFIRMED,
    /// 已解決
    RESOLVED,
    /// 誤報
    FalsePositive,
}

impl std::fmt::Display for FindingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FindingStatus::NEW => write!(f, "new"),
            FindingStatus::CONFIRMED => write!(f, "confirmed"),
            FindingStatus::RESOLVED => write!(f, "resolved"),
            FindingStatus::FalsePositive => write!(f, "false_positive"),
        }
    }
}

impl std::str::FromStr for FindingStatus {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NEW" => Ok(FindingStatus::NEW),
            "CONFIRMED" => Ok(FindingStatus::CONFIRMED),
            "RESOLVED" => Ok(FindingStatus::RESOLVED),
            "FALSE_POSITIVE" => Ok(FindingStatus::FalsePositive),
            _ => Err(format!("Invalid FindingStatus: {}", s)),
        }
    }
}

/// 統一訊息標頭 - 所有跨服務通訊的基礎
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)] // Generated schema for cross-service communication
pub struct MessageHeader {
    /// 唯一訊息識別碼
    pub message_id: String,
    /// 分散式追蹤識別碼
    pub trace_id: String,
    /// 關聯識別碼 - 用於請求-響應配對
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    /// 來源模組名稱
    pub source_module: String,
    /// 訊息時間戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Schema版本號
    pub version: String,
}

impl MessageHeader {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            message_id: String::new(),
            trace_id: String::new(),
            correlation_id: None,
            source_module: String::new(),
            timestamp: chrono::Utc::now(),
            version: "1.0".to_string(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for MessageHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// 掃描/攻擊目標定義
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Target {
    /// 目標URL
    pub url: String,
    /// 目標參數名稱
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    /// HTTP方法
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// HTTP標頭
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// HTTP參數
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// HTTP請求體
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl Target {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            url: String::new(),
            parameter: None,
            method: None,
            headers: None,
            params: None,
            body: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Target {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞資訊定義
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Vulnerability {
    /// 漏洞名稱
    pub name: String,
    /// CWE編號
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwe: Option<String>,
    /// 嚴重程度
    pub severity: String,
    /// 信心度
    pub confidence: String,
    /// 漏洞描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Vulnerability {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            name: String::new(),
            cwe: None,
            severity: String::new(),
            confidence: String::new(),
            description: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Vulnerability {
    fn default() -> Self {
        Self::new()
    }
}

/// AIVA統一訊息格式 - 所有跨服務通訊的標準信封
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AivaMessage {
    /// 訊息標頭
    pub header: MessageHeader,
    /// 訊息主題
    pub topic: String,
    /// Schema版本
    pub schema_version: String,
    /// 訊息載荷
    pub payload: std::collections::HashMap<String, serde_json::Value>,
}

impl AivaMessage {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            header: MessageHeader::default(),
            topic: String::new(),
            schema_version: "1.0".to_string(),
            payload: std::collections::HashMap::new(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for AivaMessage {
    fn default() -> Self {
        Self::new()
    }
}

/// 統一請求格式 - 模組間請求通訊
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AIVARequest {
    /// 請求識別碼
    pub request_id: String,
    /// 來源模組
    pub source_module: String,
    /// 目標模組
    pub target_module: String,
    /// 請求類型
    pub request_type: String,
    /// 請求載荷
    pub payload: std::collections::HashMap<String, serde_json::Value>,
    /// 追蹤識別碼
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// 逾時秒數
    pub timeout_seconds: i32,
    /// 中繼資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 時間戳
    pub timestamp: String,
}

impl AIVARequest {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            request_id: String::new(),
            source_module: String::new(),
            target_module: String::new(),
            request_type: String::new(),
            payload: std::collections::HashMap::new(),
            trace_id: None,
            timeout_seconds: 30,
            metadata: None,
            timestamp: String::new(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for AIVARequest {
    fn default() -> Self {
        Self::new()
    }
}

/// 統一響應格式 - 模組間響應通訊
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AIVAResponse {
    /// 對應的請求識別碼
    pub request_id: String,
    /// 響應類型
    pub response_type: String,
    /// 執行是否成功
    pub success: bool,
    /// 響應載荷
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 錯誤代碼
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// 錯誤訊息
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// 中繼資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 時間戳
    pub timestamp: String,
}

impl AIVAResponse {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            request_id: String::new(),
            response_type: String::new(),
            success: false,
            payload: None,
            error_code: None,
            error_message: None,
            metadata: None,
            timestamp: String::new(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for AIVAResponse {
    fn default() -> Self {
        Self::new()
    }
}

/// 功能任務載荷 - 掃描任務的標準格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionTaskPayload {
    /// 任務識別碼
    pub task_id: String,
    /// 掃描識別碼
    pub scan_id: String,
    /// 任務優先級
    pub priority: i32,
    /// 掃描目標
    pub target: FunctionTaskTarget,
    /// 任務上下文
    pub context: FunctionTaskContext,
    /// 掃描策略
    pub strategy: String,
    /// 自訂載荷
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_payloads: Option<Vec<String>>,
    /// 測試配置
    pub test_config: FunctionTaskTestConfig,
}

impl FunctionTaskPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            task_id: String::new(),
            scan_id: String::new(),
            priority: 0,
            target: FunctionTaskTarget::default(),
            context: FunctionTaskContext::default(),
            strategy: String::new(),
            custom_payloads: None,
            test_config: FunctionTaskTestConfig::default(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FunctionTaskPayload {
    fn default() -> Self {
        Self::new()
    }
}

/// 功能任務目標
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionTaskTarget {
}

impl FunctionTaskTarget {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FunctionTaskTarget {
    fn default() -> Self {
        Self::new()
    }
}

/// 功能任務上下文
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionTaskContext {
    /// 資料庫類型提示
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub db_type_hint: Option<String>,
    /// 是否檢測到WAF
    pub waf_detected: bool,
    /// 相關發現
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_findings: Option<Vec<String>>,
}

impl FunctionTaskContext {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            db_type_hint: None,
            waf_detected: false,
            related_findings: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FunctionTaskContext {
    fn default() -> Self {
        Self::new()
    }
}

/// 功能任務測試配置
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionTaskTestConfig {
    /// 標準載荷列表
    pub payloads: Vec<String>,
    /// 自訂載荷列表
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_payloads: Option<Vec<String>>,
    /// 是否進行Blind XSS測試
    pub blind_xss: bool,
    /// 是否進行DOM測試
    pub dom_testing: bool,
    /// 請求逾時(秒)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

impl FunctionTaskTestConfig {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            payloads: Vec::new(),
            custom_payloads: None,
            blind_xss: false,
            dom_testing: false,
            timeout: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FunctionTaskTestConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// 掃描任務載荷 - 用於SCA/SAST等需要項目URL的掃描任務
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScanTaskPayload {
    /// 任務識別碼
    pub task_id: String,
    /// 掃描識別碼
    pub scan_id: String,
    /// 任務優先級
    pub priority: i32,
    /// 掃描目標 (包含URL)
    pub target: Target,
    /// 掃描類型
    pub scan_type: String,
    /// 代碼倉庫資訊 (分支、commit等)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_info: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 掃描逾時(秒)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<i32>,
}

impl ScanTaskPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            task_id: String::new(),
            scan_id: String::new(),
            priority: 0,
            target: Target::default(),
            scan_type: String::new(),
            repository_info: None,
            timeout: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for ScanTaskPayload {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞發現載荷 - 掃描結果的標準格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FindingPayload {
    /// 發現識別碼
    pub finding_id: String,
    /// 任務識別碼
    pub task_id: String,
    /// 掃描識別碼
    pub scan_id: String,
    /// 發現狀態
    pub status: String,
    /// 漏洞資訊
    pub vulnerability: Vulnerability,
    /// 目標資訊
    pub target: Target,
    /// 使用的策略
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// 證據資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<FindingEvidence>,
    /// 影響評估
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<FindingImpact>,
    /// 修復建議
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<FindingRecommendation>,
    /// 中繼資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 建立時間
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 更新時間
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl FindingPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            finding_id: String::new(),
            task_id: String::new(),
            scan_id: String::new(),
            status: String::new(),
            vulnerability: Vulnerability::default(),
            target: Target::default(),
            strategy: None,
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FindingPayload {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞證據
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FindingEvidence {
    /// 攻擊載荷
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    /// 響應時間差異
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_time_delta: Option<f64>,
    /// 資料庫版本
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub db_version: Option<String>,
    /// HTTP請求
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    /// HTTP響應
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    /// 證明資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,
}

impl FindingEvidence {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            payload: None,
            response_time_delta: None,
            db_version: None,
            request: None,
            response: None,
            proof: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FindingEvidence {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞影響評估
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FindingImpact {
    /// 影響描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 業務影響
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub business_impact: Option<String>,
    /// 技術影響
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub technical_impact: Option<String>,
    /// 受影響用戶數
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affected_users: Option<i32>,
    /// 估計成本
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
}

impl FindingImpact {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            description: None,
            business_impact: None,
            technical_impact: None,
            affected_users: None,
            estimated_cost: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FindingImpact {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞修復建議
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FindingRecommendation {
    /// 修復方法
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// 修復優先級
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// 修復步驟
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remediation_steps: Option<Vec<String>>,
    /// 參考資料
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<String>>,
}

impl FindingRecommendation {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            fix: None,
            priority: None,
            remediation_steps: None,
            references: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for FindingRecommendation {
    fn default() -> Self {
        Self::new()
    }
}

