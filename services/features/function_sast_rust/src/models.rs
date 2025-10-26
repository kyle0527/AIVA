// SAST 數據模型 - 使用標準 schema

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::schemas::generated::{FindingPayload, Vulnerability, Target, FindingEvidence, FindingImpact, FindingRecommendation, VulnerabilityType, Severity, Confidence};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTaskPayload {
    pub task_id: String,
    pub function_type: String,
    pub target: TaskTarget,
    pub options: Option<TaskOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTarget {
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub repository: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOptions {
    pub language: Option<String>,
    pub rules: Option<Vec<String>>,
    pub severity_threshold: Option<String>,
}

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
            "CRITICAL" => Severity::Critical,
            "HIGH" => Severity::High,
            "MEDIUM" => Severity::Medium,
            "LOW" => Severity::Low,
            _ => Severity::Informational,
        };

        // 解析可信度
        let confidence = match self.confidence.to_uppercase().as_str() {
            "CERTAIN" => Confidence::Certain,
            "FIRM" => Confidence::Firm,
            _ => Confidence::Possible,
        };

        FindingPayload {
            finding_id: format!("finding_sast_{}", Uuid::new_v4()),
            task_id: task_id.to_string(),
            scan_id: scan_id.to_string(),
            status: "confirmed".to_string(),
            vulnerability: Vulnerability {
                name: VulnerabilityType::Sast,
                cwe: Some(self.cwe.clone()),
                cve: None,
                severity,
                confidence,
                description: Some(self.description.clone()),
                cvss_score: None,
                cvss_vector: None,
                owasp_category: None,
            },
            target: Target {
                url: serde_json::Value::String(format!("file://{}", self.file_path)),
                parameter: None,
                method: None,
                headers: std::collections::HashMap::new(),
                params: std::collections::HashMap::new(),
                body: None,
                file_path: Some(self.file_path.clone()),
                line_number: Some(self.line_number),
                function_name: self.function_name.clone(),
                code_snippet: Some(self.code_snippet.clone()),
            },
            strategy: Some("SAST".to_string()),
            evidence: Some(FindingEvidence {
                payload: None,
                response_time_delta: None,
                db_version: None,
                request: None,
                response: None,
                proof: None,
                rule_id: Some(self.rule_id.clone()),
                pattern: Some(self.matched_pattern.clone()),
                matched_code: Some(self.code_snippet.clone()),
                context: Some(format!("Line {}: {}", self.line_number, self.code_snippet)),
            }),
            impact: Some(FindingImpact {
                description: Some(self.description.clone()),
                business_impact: Some(format!("{:?} 級漏洞可能導致嚴重安全問題", severity)),
                technical_impact: None,
                affected_users: None,
                estimated_cost: None,
                exploitability: Some(match severity {
                    Severity::Critical => "極高".to_string(),
                    Severity::High => "高".to_string(),
                    Severity::Medium => "中".to_string(),
                    _ => "低".to_string(),
                }),
            }),
            recommendation: Some(FindingRecommendation {
                fix: Some(self.recommendation.clone()),
                priority: Some(format!("{:?}", severity)),
                remediation_steps: vec![self.recommendation.clone()],
                references: vec![],
            }),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("rule_id".to_string(), serde_json::Value::String(self.rule_id.clone()));
                metadata.insert("file_path".to_string(), serde_json::Value::String(self.file_path.clone()));
                metadata.insert("line_number".to_string(), serde_json::Value::Number(serde_json::Number::from(self.line_number)));
                metadata.insert("rule_name".to_string(), serde_json::Value::String(self.rule_name.clone()));
                metadata
            },
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}
