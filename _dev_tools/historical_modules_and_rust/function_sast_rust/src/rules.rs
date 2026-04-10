// SAST 規則引擎

use crate::models::SastIssue;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SastRule {
    pub id: String,
    pub name: String,
    pub cwe: String,
    pub severity: String,
    pub confidence: String,
    pub pattern: String,
    pub description: String,
    pub recommendation: String,
    pub languages: Vec<String>,
}

pub struct RuleEngine {
    rules: Vec<SastRule>,
    regex_cache: HashMap<String, Regex>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Self::load_default_rules(),
            regex_cache: HashMap::new(),
        }
    }
    
    fn load_default_rules() -> Vec<SastRule> {
        vec![
            // SQL 注入
            SastRule {
                id: "SAST-001".to_string(),
                name: "SQL Injection".to_string(),
                cwe: "CWE-89".to_string(),
                severity: "CRITICAL".to_string(),
                confidence: "HIGH".to_string(),
                pattern: r#"(execute|cursor\.execute|db\.query)\s*\([^)]*\+[^)]*\)"#.to_string(),
                description: "檢測到可能的 SQL 注入漏洞，使用字串拼接構建 SQL 查詢".to_string(),
                recommendation: "使用參數化查詢或 ORM 框架，永不直接拼接使用者輸入".to_string(),
                languages: vec!["python".to_string(), "java".to_string()],
            },
            // 命令注入
            SastRule {
                id: "SAST-002".to_string(),
                name: "Command Injection".to_string(),
                cwe: "CWE-78".to_string(),
                severity: "CRITICAL".to_string(),
                confidence: "HIGH".to_string(),
                pattern: r#"(os\.system|subprocess\.call|exec|eval)\s*\([^)]*\)"#.to_string(),
                description: "檢測到可能的命令注入漏洞，直接執行系統命令".to_string(),
                recommendation: "避免使用 os.system/exec，使用 subprocess 並進行嚴格的輸入驗證".to_string(),
                languages: vec!["python".to_string(), "javascript".to_string()],
            },
            // 硬編碼憑證
            SastRule {
                id: "SAST-003".to_string(),
                name: "Hardcoded Credentials".to_string(),
                cwe: "CWE-798".to_string(),
                severity: "HIGH".to_string(),
                confidence: "MEDIUM".to_string(),
                pattern: r#"(password|secret|api_key|token)\s*=\s*['""][^'""]+['""]"#.to_string(),
                description: "檢測到硬編碼的憑證資訊".to_string(),
                recommendation: "使用環境變數或密鑰管理系統儲存敏感資訊".to_string(),
                languages: vec!["python".to_string(), "javascript".to_string(), "go".to_string(), "java".to_string()],
            },
            // XSS
            SastRule {
                id: "SAST-004".to_string(),
                name: "Cross-Site Scripting (XSS)".to_string(),
                cwe: "CWE-79".to_string(),
                severity: "HIGH".to_string(),
                confidence: "MEDIUM".to_string(),
                pattern: r#"(innerHTML|outerHTML|document\.write)\s*=\s*"#.to_string(),
                description: "檢測到可能的 XSS 漏洞，直接操作 DOM".to_string(),
                recommendation: "使用安全的 DOM API (textContent) 或進行 HTML 編碼".to_string(),
                languages: vec!["javascript".to_string()],
            },
            // 不安全的隨機數
            SastRule {
                id: "SAST-005".to_string(),
                name: "Insecure Random".to_string(),
                cwe: "CWE-338".to_string(),
                severity: "MEDIUM".to_string(),
                confidence: "HIGH".to_string(),
                pattern: r#"(random\.random|Math\.random)\s*\(\)"#.to_string(),
                description: "檢測到使用不安全的隨機數生成器".to_string(),
                recommendation: "對安全敏感的操作使用 secrets 模組或 crypto.getRandomValues".to_string(),
                languages: vec!["python".to_string(), "javascript".to_string()],
            },
        ]
    }
    
    pub fn check_code(&mut self, source_code: &str, language: &str) -> Vec<SastIssue> {
        let mut issues = Vec::new();
        
        for rule in &self.rules {
            // 檢查規則是否適用於此語言
            if !rule.languages.iter().any(|lang| lang.to_lowercase() == language.to_lowercase()) {
                continue;
            }
            
            // 獲取或編譯正則表達式
            let regex = self.regex_cache.entry(rule.id.clone()).or_insert_with(|| {
                Regex::new(&rule.pattern).expect("Invalid regex pattern")
            });
            
            // 掃描每一行
            for (line_num, line) in source_code.lines().enumerate() {
                if let Some(captures) = regex.captures(line) {
                    let matched = captures.get(0).map_or("", |m| m.as_str());
                    
                    issues.push(SastIssue {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        cwe: rule.cwe.clone(),
                        severity: rule.severity.clone(),
                        confidence: rule.confidence.clone(),
                        file_path: String::new(), // 由外部填充
                        line_number: (line_num + 1) as u32,
                        function_name: None,
                        code_snippet: line.trim().to_string(),
                        matched_pattern: matched.to_string(),
                        description: rule.description.clone(),
                        recommendation: rule.recommendation.clone(),
                    });
                }
            }
        }
        
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sql_injection_detection() {
        let mut engine = RuleEngine::new();
        let code = r#"
query = "SELECT * FROM users WHERE id = " + user_input
cursor.execute(query)
        "#;
        
        let issues = engine.check_code(code, "python");
        assert!(issues.len() > 0);
        assert_eq!(issues[0].rule_id, "SAST-001");
    }
    
    #[test]
    fn test_hardcoded_credentials() {
        let mut engine = RuleEngine::new();
        let code = r#"password = "admin123""#;
        
        let issues = engine.check_code(code, "python");
        assert!(issues.len() > 0);
        assert_eq!(issues[0].rule_id, "SAST-003");
    }
}
