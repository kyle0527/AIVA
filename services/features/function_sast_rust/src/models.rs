// SAST 數據模型 - 使用標準 schema

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::schemas::generated::{FindingPayload, Vulnerability, Target, FindingEvidence, FindingImpact, FindingRecommendation, Severity, Confidence};
// FindingStatus and ScanTaskPayload are reserved for future use

// 現在使用標準的 ScanTaskPayload，因為 SAST 是掃描類型的服務

// 不再需要自訂TaskTarget和TaskOptions，使用標準ScanTaskPayload中的Target

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SastIssue {
    pub rule_id: String,
    pub rule_name: String,
    pub cwe: String,
    pub severity: String,
    pub confidence: String,
    pub file_path: String,
    pub line_number: u32,
    pub function_name: Option<String>,
    pub code_snippet: String,
    pub matched_pattern: String,
    pub description: String,
    pub recommendation: String,
}

impl SastIssue {
    pub fn to_finding(&self, task_id: &str, scan_id: &str) -> FindingPayload {
        // 解析嚴重性
        let severity = match self.severity.to_uppercase().as_str() {
            "CRITICAL" => Severity::CRITICAL,
            "HIGH" => Severity::HIGH,
            "MEDIUM" => Severity::MEDIUM,
            "LOW" => Severity::LOW,
            _ => Severity::INFO,
        };

        // 解析可信度
        let confidence = match self.confidence.to_uppercase().as_str() {
            "CONFIRMED" => Confidence::CONFIRMED,
            "FIRM" => Confidence::FIRM,
            _ => Confidence::TENTATIVE,
        };

        FindingPayload {
            finding_id: format!("finding_sast_{}", Uuid::new_v4()),
            task_id: task_id.to_string(),
            scan_id: scan_id.to_string(),
            status: "confirmed".to_string(),
            vulnerability: Vulnerability {
                name: "SAST".to_string(),  // 使用字符串而不是枚舉
                cwe: Some(self.cwe.clone()),
                severity: format!("{}", severity),  // 轉換為字符串
                confidence: format!("{}", confidence),  // 轉換為字符串
                description: Some(self.description.clone()),
            },
            target: Target {
                url: format!("file://{}", self.file_path),
                parameter: None,
                method: None,
                headers: Some(std::collections::HashMap::new()),
                params: Some(std::collections::HashMap::new()),
                body: None,
            },
            strategy: Some("SAST".to_string()),
            evidence: Some(FindingEvidence {
                payload: None,
                response_time_delta: None,
                db_version: None,
                request: None,
                response: None,
                proof: Some(format!("SAST rule {} matched at line {}: {}", self.rule_id, self.line_number, self.code_snippet)),
            }),
            impact: Some(FindingImpact {
                description: Some(self.description.clone()),
                business_impact: Some(format!("SAST 級漏洞可能導致嚴重安全問題")),
                technical_impact: Some(format!("代碼安全漏洞，可能被攻擊者利用")),
                affected_users: None,
                estimated_cost: None,
            }),
            recommendation: Some(FindingRecommendation {
                fix: Some(self.recommendation.clone()),
                priority: Some(format!("{}", severity)),
                remediation_steps: Some(vec![self.recommendation.clone()]),
                references: Some(vec![]),
            }),
            metadata: Some({
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("rule_id".to_string(), serde_json::Value::String(self.rule_id.clone()));
                metadata.insert("file_path".to_string(), serde_json::Value::String(self.file_path.clone()));
                metadata.insert("line_number".to_string(), serde_json::Value::Number(serde_json::Number::from(self.line_number)));
                metadata.insert("rule_name".to_string(), serde_json::Value::String(self.rule_name.clone()));
                metadata
            }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}
