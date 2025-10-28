// AIVA Rust Schema - 自動生成
// 版本: 1.0.0
// 生成時間: N/A
// 
// 完整的 Rust Schema 實現，包含序列化/反序列化支持

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// 可選依賴 - 根據實際使用情況啟用
#[cfg(feature = "uuid")]
use uuid::Uuid;

#[cfg(feature = "url")]
use url::Url;

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
pub enum FindingStatus {
    /// 新發現
    NEW,
    /// 已確認
    CONFIRMED,
    /// 已解決
    RESOLVED,
    /// 誤報
    FALSE_POSITIVE,
}

impl std::fmt::Display for FindingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FindingStatus::NEW => write!(f, "new"),
            FindingStatus::CONFIRMED => write!(f, "confirmed"),
            FindingStatus::RESOLVED => write!(f, "resolved"),
            FindingStatus::FALSE_POSITIVE => write!(f, "false_positive"),
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
            "FALSE_POSITIVE" => Ok(FindingStatus::FALSE_POSITIVE),
            _ => Err(format!("Invalid FindingStatus: {}", s)),
        }
    }
}

/// 訊息標頭 - 用於所有訊息的統一標頭格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MessageHeader {
    /// 
    pub message_id: String,
    /// 
    pub trace_id: String,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    /// 來源模組名稱
    pub source_module: String,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl MessageHeader {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            message_id: String::new(),
            trace_id: String::new(),
            correlation_id: None,
            source_module: String::new(),
            timestamp: None,
            version: None,
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

/// 目標資訊 - 漏洞所在位置
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Target {
    /// 
    pub url: serde_json::Value,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl Target {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            url: serde_json::Value::Null,
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

/// 漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Vulnerability {
    /// 
    pub name: serde_json::Value,
    /// CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwe: Option<String>,
    /// CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cve: Option<String>,
    /// 
    pub severity: serde_json::Value,
    /// 
    pub confidence: serde_json::Value,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cvss_score: Option<serde_json::Value>,
    /// CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cvss_vector: Option<String>,
    /// OWASP Top 10 分類，例如: A03:2021-Injection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owasp_category: Option<String>,
}

impl Vulnerability {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            name: serde_json::Value::Null,
            cwe: None,
            cve: None,
            severity: serde_json::Value::Null,
            confidence: serde_json::Value::Null,
            description: None,
            cvss_score: None,
            cvss_vector: None,
            owasp_category: None,
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

/// 資產基本資訊
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Asset {
    /// 
    pub asset_id: String,
    /// 
    pub type: String,
    /// 
    pub value: String,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_form: Option<bool>,
}

impl Asset {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            asset_id: String::new(),
            type: String::new(),
            value: String::new(),
            parameters: None,
            has_form: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Asset {
    fn default() -> Self {
        Self::new()
    }
}

/// 認證資訊
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Authentication {
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl Authentication {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            method: None,
            credentials: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Authentication {
    fn default() -> Self {
        Self::new()
    }
}

/// 執行錯誤統一格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecutionError {
    /// 
    pub error_id: String,
    /// 
    pub error_type: String,
    /// 
    pub message: String,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts: Option<i32>,
}

impl ExecutionError {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            error_id: String::new(),
            error_type: String::new(),
            message: String::new(),
            payload: None,
            vector: None,
            timestamp: None,
            attempts: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for ExecutionError {
    fn default() -> Self {
        Self::new()
    }
}

/// 技術指紋
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Fingerprints {
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_server: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framework: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waf_detected: Option<bool>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waf_vendor: Option<String>,
}

impl Fingerprints {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            web_server: None,
            framework: None,
            language: None,
            waf_detected: None,
            waf_vendor: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Fingerprints {
    fn default() -> Self {
        Self::new()
    }
}

/// 速率限制
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RateLimit {
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_second: Option<i32>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub burst: Option<i32>,
}

impl RateLimit {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            requests_per_second: None,
            burst: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for RateLimit {
    fn default() -> Self {
        Self::new()
    }
}

/// 風險因子
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RiskFactor {
    /// 風險因子名稱
    pub factor_name: String,
    /// 權重
    pub weight: f64,
    /// 因子值
    pub value: f64,
    /// 因子描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl RiskFactor {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            factor_name: String::new(),
            weight: 0.0,
            value: 0.0,
            description: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for RiskFactor {
    fn default() -> Self {
        Self::new()
    }
}

/// 掃描範圍
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScanScope {
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusions: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_subdomains: Option<bool>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_hosts: Option<Vec<String>>,
}

impl ScanScope {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            exclusions: None,
            include_subdomains: None,
            allowed_hosts: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for ScanScope {
    fn default() -> Self {
        Self::new()
    }
}

/// 掃描摘要
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Summary {
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub urls_found: Option<i32>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forms_found: Option<i32>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apis_found: Option<i32>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_duration_seconds: Option<i32>,
}

impl Summary {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            urls_found: None,
            forms_found: None,
            apis_found: None,
            scan_duration_seconds: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Summary {
    fn default() -> Self {
        Self::new()
    }
}

/// 任務依賴
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaskDependency {
    /// 依賴類型
    pub dependency_type: String,
    /// 依賴任務ID
    pub dependent_task_id: String,
    /// 依賴條件
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// 是否必需
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
}

impl TaskDependency {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            dependency_type: String::new(),
            dependent_task_id: String::new(),
            condition: None,
            required: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for TaskDependency {
    fn default() -> Self {
        Self::new()
    }
}

/// AI 驅動漏洞驗證請求
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AIVerificationRequest {
    /// 
    pub verification_id: String,
    /// 
    pub finding_id: String,
    /// 
    pub scan_id: String,
    /// 
    pub vulnerability_type: serde_json::Value,
    /// 
    pub target: serde_json::Value,
    /// 
    pub evidence: serde_json::Value,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_mode: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl AIVerificationRequest {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            verification_id: String::new(),
            finding_id: String::new(),
            scan_id: String::new(),
            vulnerability_type: serde_json::Value::Null,
            target: serde_json::Value::Null,
            evidence: serde_json::Value::Null,
            verification_mode: None,
            context: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for AIVerificationRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// AI 驅動漏洞驗證結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AIVerificationResult {
    /// 
    pub verification_id: String,
    /// 
    pub finding_id: String,
    /// 
    pub verification_status: String,
    /// 
    pub confidence_score: f64,
    /// 
    pub verification_method: String,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_steps: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observations: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendations: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl AIVerificationResult {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            verification_id: String::new(),
            finding_id: String::new(),
            verification_status: String::new(),
            confidence_score: 0.0,
            verification_method: String::new(),
            test_steps: None,
            observations: None,
            recommendations: None,
            timestamp: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for AIVerificationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// 程式碼層面根因分析結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CodeLevelRootCause {
    /// 
    pub analysis_id: String,
    /// 
    pub vulnerable_component: String,
    /// 
    pub affected_findings: Vec<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_location: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulnerability_pattern: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix_recommendation: Option<String>,
}

impl CodeLevelRootCause {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            analysis_id: String::new(),
            vulnerable_component: String::new(),
            affected_findings: Vec::new(),
            code_location: None,
            vulnerability_pattern: None,
            fix_recommendation: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for CodeLevelRootCause {
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

/// 目標資訊 - 漏洞所在位置
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FindingTarget {
    /// 
    pub url: serde_json::Value,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

impl FindingTarget {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            url: serde_json::Value::Null,
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

impl Default for FindingTarget {
    fn default() -> Self {
        Self::new()
    }
}

/// JavaScript 分析結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct JavaScriptAnalysisResult {
    /// 
    pub analysis_id: String,
    /// 
    pub url: String,
    /// 
    pub source_size_bytes: i32,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dangerous_functions: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_resources: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_leaks: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub findings: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apis_called: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ajax_endpoints: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspicious_patterns: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_score: Option<f64>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security_score: Option<i32>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl JavaScriptAnalysisResult {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            analysis_id: String::new(),
            url: String::new(),
            source_size_bytes: 0,
            dangerous_functions: None,
            external_resources: None,
            data_leaks: None,
            findings: None,
            apis_called: None,
            ajax_endpoints: None,
            suspicious_patterns: None,
            risk_score: None,
            security_score: None,
            timestamp: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for JavaScriptAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// SAST-DAST 資料流關聯結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SASTDASTCorrelation {
    /// 
    pub correlation_id: String,
    /// 
    pub sast_finding_id: String,
    /// 
    pub dast_finding_id: String,
    /// 
    pub data_flow_path: Vec<String>,
    /// 
    pub verification_status: String,
    /// 
    pub confidence_score: f64,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

impl SASTDASTCorrelation {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            correlation_id: String::new(),
            sast_finding_id: String::new(),
            dast_finding_id: String::new(),
            data_flow_path: Vec::new(),
            verification_status: String::new(),
            confidence_score: 0.0,
            explanation: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for SASTDASTCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

/// 敏感資訊匹配結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SensitiveMatch {
    /// 
    pub match_id: String,
    /// 
    pub pattern_name: String,
    /// 
    pub matched_text: String,
    /// 
    pub context: String,
    /// 
    pub confidence: f64,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_number: Option<serde_json::Value>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<serde_json::Value>,
}

impl SensitiveMatch {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            match_id: String::new(),
            pattern_name: String::new(),
            matched_text: String::new(),
            context: String::new(),
            confidence: 0.0,
            line_number: None,
            file_path: None,
            url: None,
            severity: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for SensitiveMatch {
    fn default() -> Self {
        Self::new()
    }
}

/// 漏洞關聯分析結果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VulnerabilityCorrelation {
    /// 
    pub correlation_id: String,
    /// 
    pub correlation_type: String,
    /// 
    pub related_findings: Vec<String>,
    /// 
    pub confidence_score: f64,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_cause: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub common_components: Option<Vec<String>>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    /// 
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl VulnerabilityCorrelation {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            correlation_id: String::new(),
            correlation_type: String::new(),
            related_findings: Vec::new(),
            confidence_score: 0.0,
            root_cause: None,
            common_components: None,
            explanation: None,
            timestamp: None,
        }
    }
    
    /// 驗證結構體數據
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for VulnerabilityCorrelation {
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

