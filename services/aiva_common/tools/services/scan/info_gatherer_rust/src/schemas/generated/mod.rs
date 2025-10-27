// AIVA Rust Schema - 自動生成
// 版本: 1.0.0
// 生成時間: N/A
// 
// 完整的 Rust Schema 實現，包含序列化/反序列化支持

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// 功能任務載荷 - 掃描任務的標準格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionTaskPayload {
    /// 任務識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// 掃描識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_id: Option<String>,
    /// 任務優先級
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// 掃描目標
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// 任務上下文
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// 掃描策略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// 自訂載荷
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_payloads: Option<String>,
    /// 測試配置
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_config: Option<String>,
}

impl FunctionTaskPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            task_id: None,
            scan_id: None,
            priority: None,
            target: None,
            context: None,
            strategy: None,
            custom_payloads: None,
            test_config: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_type_hint: Option<String>,
    /// 是否檢測到WAF
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waf_detected: Option<String>,
    /// 相關發現
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_findings: Option<String>,
}

impl FunctionTaskContext {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            db_type_hint: None,
            waf_detected: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payloads: Option<String>,
    /// 自訂載荷列表
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_payloads: Option<String>,
    /// 是否進行Blind XSS測試
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blind_xss: Option<String>,
    /// 是否進行DOM測試
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dom_testing: Option<String>,
    /// 請求逾時(秒)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
}

impl FunctionTaskTestConfig {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            payloads: None,
            custom_payloads: None,
            blind_xss: None,
            dom_testing: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// 掃描識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_id: Option<String>,
    /// 任務優先級
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// 掃描目標 (包含URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// 掃描類型
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_type: Option<String>,
    /// 代碼倉庫資訊 (分支、commit等)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository_info: Option<String>,
    /// 掃描逾時(秒)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
}

impl ScanTaskPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            task_id: None,
            scan_id: None,
            priority: None,
            target: None,
            scan_type: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finding_id: Option<String>,
    /// 任務識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// 掃描識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_id: Option<String>,
    /// 發現狀態
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// 漏洞資訊
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vulnerability: Option<String>,
    /// 目標資訊
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// 使用的策略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    /// 證據資料
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    /// 影響評估
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact: Option<String>,
    /// 修復建議
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<String>,
    /// 中繼資料
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// 建立時間
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// 更新時間
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

impl FindingPayload {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            finding_id: None,
            task_id: None,
            scan_id: None,
            status: None,
            vulnerability: None,
            target: None,
            strategy: None,
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: None,
            created_at: None,
            updated_at: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    /// 響應時間差異
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time_delta: Option<String>,
    /// 資料庫版本
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_version: Option<String>,
    /// HTTP請求
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    /// HTTP響應
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    /// 證明資料
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 業務影響
    #[serde(skip_serializing_if = "Option::is_none")]
    pub business_impact: Option<String>,
    /// 技術影響
    #[serde(skip_serializing_if = "Option::is_none")]
    pub technical_impact: Option<String>,
    /// 受影響用戶數
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affected_users: Option<String>,
    /// 估計成本
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// 修復優先級
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// 修復步驟
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remediation_steps: Option<String>,
    /// 參考資料
    #[serde(skip_serializing_if = "Option::is_none")]
    pub references: Option<String>,
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

/// AIVA統一訊息格式 - 所有跨服務通訊的標準信封
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AivaMessage {
    /// 訊息標頭
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// 訊息主題
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    /// Schema版本
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    /// 訊息載荷
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
}

impl AivaMessage {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            header: None,
            topic: None,
            schema_version: None,
            payload: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// 來源模組
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_module: Option<String>,
    /// 目標模組
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_module: Option<String>,
    /// 請求類型
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_type: Option<String>,
    /// 請求載荷
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    /// 追蹤識別碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// 逾時秒數
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<String>,
    /// 中繼資料
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// 時間戳
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl AIVARequest {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            request_id: None,
            source_module: None,
            target_module: None,
            request_type: None,
            payload: None,
            trace_id: None,
            timeout_seconds: None,
            metadata: None,
            timestamp: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// 響應類型
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_type: Option<String>,
    /// 執行是否成功
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<String>,
    /// 響應載荷
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    /// 錯誤代碼
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// 錯誤訊息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// 中繼資料
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// 時間戳
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl AIVAResponse {
    /// 創建新的實例
    pub fn new() -> Self {
        Self {
            request_id: None,
            response_type: None,
            success: None,
            payload: None,
            error_code: None,
            error_message: None,
            metadata: None,
            timestamp: None,
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

