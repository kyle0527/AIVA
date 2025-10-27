// AIVA Rust Schema - 自動生成
// 版本: 1.0.0
// 基於 core_schema_sot.yaml 作為單一事實來源
// 此文件與 Python aiva_common 保持完全一致性
//
// ⚠️  此檔案自動生成，請勿手動修改
// 📅 最後更新: 2025-10-27T08:15:28.157056

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// ==================== 枚舉定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    #[serde(rename = "critical")]
    Critical,
    #[serde(rename = "high")]
    High,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "info")]
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Confidence {
    #[serde(rename = "confirmed")]
    Confirmed,
    #[serde(rename = "firm")]
    Firm,
    #[serde(rename = "tentative")]
    Tentative,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FindingStatus {
    #[serde(rename = "new")]
    New,
    #[serde(rename = "confirmed")]
    Confirmed,
    #[serde(rename = "false_positive")]
    FalsePositive,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "ignored")]
    Ignored,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HttpMethod {
    #[serde(rename = "GET")]
    Get,
    #[serde(rename = "POST")]
    Post,
    #[serde(rename = "PUT")]
    Put,
    #[serde(rename = "DELETE")]
    Delete,
    #[serde(rename = "PATCH")]
    Patch,
    #[serde(rename = "HEAD")]
    Head,
    #[serde(rename = "OPTIONS")]
    Options,
}


// ==================== 核心結構定義 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    pub message_id: String,
    pub trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub source_module: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "default_version")]
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    #[serde(default = "default_get_method")]
    pub method: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwe: Option<String>,
    pub severity: Severity,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

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
    pub affected_users: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
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

// ==================== 主要 Payload 結構 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingPayload {
    pub finding_id: String,
    pub task_id: String,
    pub scan_id: String,
    pub status: FindingStatus,
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

// ==================== 輔助函數 ====================

fn default_version() -> String {
    "1.0".to_string()
}

fn default_get_method() -> String {
    "GET".to_string()
}

impl FindingPayload {
    /// 創建新的 FindingPayload 實例
    pub fn new(
        finding_id: String,
        task_id: String,
        scan_id: String,
        status: FindingStatus,
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
            strategy: None,
            evidence: None,
            impact: None,
            recommendation: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 驗證必要字段格式
    pub fn validate(&self) -> Result<(), String> {
        if !self.finding_id.starts_with("finding_") {
            return Err("finding_id must start with 'finding_'".to_string());
        }
        if !self.task_id.starts_with("task_") {
            return Err("task_id must start with 'task_'".to_string());
        }
        if !self.scan_id.starts_with("scan_") {
            return Err("scan_id must start with 'scan_'".to_string());
        }
        Ok(())
    }

    /// 更新時間戳
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

impl Default for FindingStatus {
    fn default() -> Self {
        FindingStatus::New
    }
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Medium
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Confidence::Tentative
    }
}
