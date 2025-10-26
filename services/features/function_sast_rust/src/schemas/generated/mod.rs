// AIVA Rust Schema for function_sast_rust - 自動生成
// 版本: 1.0.0
// 基於 Python aiva_common 作為單一事實來源
// 此文件與 services/aiva_common/schemas/ 保持完全一致性

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// ==================== 枚舉定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    #[serde(rename = "Critical")]
    Critical,
    #[serde(rename = "High")]
    High,
    #[serde(rename = "Medium")]
    Medium,
    #[serde(rename = "Low")]
    Low,
    #[serde(rename = "Informational")]
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Confidence {
    #[serde(rename = "Certain")]
    Certain,
    #[serde(rename = "Firm")]
    Firm,
    #[serde(rename = "Possible")]
    Possible,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VulnerabilityType {
    #[serde(rename = "XSS")]
    Xss,
    #[serde(rename = "SQL Injection")]
    SqlInjection,
    #[serde(rename = "SSRF")]
    Ssrf,
    #[serde(rename = "IDOR")]
    Idor,
    #[serde(rename = "BOLA")]
    Bola,
    #[serde(rename = "Information Leak")]
    InformationLeak,
    #[serde(rename = "Weak Authentication")]
    WeakAuthentication,
    #[serde(rename = "Remote Code Execution")]
    RemoteCodeExecution,
    #[serde(rename = "Authentication Bypass")]
    AuthenticationBypass,
    #[serde(rename = "Price Manipulation")]
    PriceManipulation,
    #[serde(rename = "Workflow Bypass")]
    WorkflowBypass,
    #[serde(rename = "Race Condition")]
    RaceCondition,
    #[serde(rename = "Forced Browsing")]
    ForcedBrowsing,
    #[serde(rename = "State Manipulation")]
    StateManipulation,
    #[serde(rename = "SAST")]
    Sast,
}

// ==================== 核心結構定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub name: VulnerabilityType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwe: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cve: Option<String>,
    pub severity: Severity,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cvss_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cvss_vector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owasp_category: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub url: serde_json::Value, // Accept arbitrary URL-like values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,

    // SAST 特定欄位
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_snippet: Option<String>,
}

// 向後相容別名
pub type FindingTarget = Target;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,

    // SAST 特定欄位
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingImpact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub business_impact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub technical_impact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affected_users: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exploitability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingRecommendation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    #[serde(default)]
    pub remediation_steps: Vec<String>,
    #[serde(default)]
    pub references: Vec<String>,
}

// ==================== FindingPayload - 主要結構 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingPayload {
    pub finding_id: String,
    pub task_id: String,
    pub scan_id: String,
    pub status: String,
    pub vulnerability: Vulnerability,
    pub target: Target,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<FindingEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impact: Option<FindingImpact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<FindingRecommendation>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl FindingPayload {
    /// 創建新的 FindingPayload 實例
    pub fn new(
        finding_id: String,
        task_id: String,
        scan_id: String,
        status: String,
        vulnerability: Vulnerability,
        target: Target,
    ) -> Self {
        let now = Utc::now();
        Self {
            finding_id,
            task_id,
            scan_id,
            status,
            vulnerability,
            target,
            strategy: Some("SAST".to_string()),
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 驗證 finding_id 格式
    pub fn validate_finding_id(&self) -> Result<(), String> {
        if !self.finding_id.starts_with("finding_") {
            return Err("finding_id must start with 'finding_'".to_string());
        }
        Ok(())
    }

    /// 驗證 task_id 格式
    pub fn validate_task_id(&self) -> Result<(), String> {
        if !self.task_id.starts_with("task_") {
            return Err("task_id must start with 'task_'".to_string());
        }
        Ok(())
    }

    /// 驗證 scan_id 格式
    pub fn validate_scan_id(&self) -> Result<(), String> {
        if !self.scan_id.starts_with("scan_") {
            return Err("scan_id must start with 'scan_'".to_string());
        }
        Ok(())
    }

    /// 驗證 status 值
    pub fn validate_status(&self) -> Result<(), String> {
        let allowed = ["confirmed", "potential", "false_positive", "needs_review"];
        if !allowed.contains(&self.status.as_str()) {
            return Err(format!(
                "Invalid status: {}. Must be one of {:?}",
                self.status, allowed
            ));
        }
        Ok(())
    }

    /// 完整驗證
    pub fn validate(&self) -> Result<(), String> {
        self.validate_finding_id()?;
        self.validate_task_id()?;
        self.validate_scan_id()?;
        self.validate_status()?;
        Ok(())
    }

    /// 更新時間戳
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}