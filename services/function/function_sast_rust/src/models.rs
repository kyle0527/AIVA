// SAST 數據模型

use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
pub struct FindingPayload {
    pub finding_id: String,
    pub task_id: String,
    pub scan_id: String,
    pub status: String,
    pub vulnerability: Vulnerability,
    pub target: FindingTarget,
    pub evidence: FindingEvidence,
    pub impact: FindingImpact,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub name: String,
    pub cwe: String,
    pub severity: String,
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingTarget {
    pub file_path: String,
    pub line_number: u32,
    pub function_name: Option<String>,
    pub code_snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingEvidence {
    pub rule_id: String,
    pub pattern: String,
    pub matched_code: String,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingImpact {
    pub description: String,
    pub business_impact: String,
    pub exploitability: String,
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
        FindingPayload {
            finding_id: format!("finding_sast_{}", Uuid::new_v4()),
            task_id: task_id.to_string(),
            scan_id: scan_id.to_string(),
            status: "CONFIRMED".to_string(),
            vulnerability: Vulnerability {
                name: self.rule_name.clone(),
                cwe: self.cwe.clone(),
                severity: self.severity.clone(),
                confidence: self.confidence.clone(),
            },
            target: FindingTarget {
                file_path: self.file_path.clone(),
                line_number: self.line_number,
                function_name: self.function_name.clone(),
                code_snippet: self.code_snippet.clone(),
            },
            evidence: FindingEvidence {
                rule_id: self.rule_id.clone(),
                pattern: self.matched_pattern.clone(),
                matched_code: self.code_snippet.clone(),
                context: format!("Line {}: {}", self.line_number, self.code_snippet),
            },
            impact: FindingImpact {
                description: self.description.clone(),
                business_impact: format!("{} 級漏洞可能導致嚴重安全問題", self.severity),
                exploitability: match self.severity.as_str() {
                    "CRITICAL" => "極高".to_string(),
                    "HIGH" => "高".to_string(),
                    "MEDIUM" => "中".to_string(),
                    _ => "低".to_string(),
                },
            },
            recommendation: self.recommendation.clone(),
        }
    }
}
