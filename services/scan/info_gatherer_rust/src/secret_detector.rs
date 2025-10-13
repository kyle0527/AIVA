use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn};

/// 憑證洩漏檢測器
pub struct SecretDetector {
    /// 高熵字串檢測器
    entropy_detector: EntropyDetector,
    /// 規則匹配器
    rule_matchers: Vec<SecretRule>,
}

/// 憑證檢測規則
#[derive(Clone, Debug)]
pub struct SecretRule {
    pub name: String,
    pub regex: Regex,
    pub severity: String,
    pub description: String,
}

/// 熵值檢測器
pub struct EntropyDetector {
    threshold: f64,
    min_length: usize,
}

/// 憑證發現結果
#[derive(Debug, Serialize, Deserialize)]
pub struct SecretFinding {
    pub rule_name: String,
    pub matched_text: String,
    pub file_path: String,
    pub line_number: usize,
    pub severity: String,
    pub entropy: Option<f64>,
    pub context: String,
}

impl SecretDetector {
    /// 建立新的憑證檢測器
    pub fn new() -> Self {
        Self {
            entropy_detector: EntropyDetector::new(4.5, 20),
            rule_matchers: Self::load_default_rules(),
        }
    }

    /// 載入預設規則
    fn load_default_rules() -> Vec<SecretRule> {
        vec![
            // AWS Access Key
            SecretRule {
                name: "AWS Access Key ID".to_string(),
                regex: Regex::new(r"(?i)(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "AWS Access Key ID detected".to_string(),
            },
            // AWS Secret Access Key
            SecretRule {
                name: "AWS Secret Access Key".to_string(),
                regex: Regex::new(r#"(?i)aws(.{0,20})?(?-i)['\"][0-9a-zA-Z/+]{40}['"]"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "AWS Secret Access Key detected".to_string(),
            },
            // GitHub Token
            SecretRule {
                name: "GitHub Personal Access Token".to_string(),
                regex: Regex::new(r"ghp_[0-9a-zA-Z]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub Personal Access Token detected".to_string(),
            },
            // GitHub OAuth Token
            SecretRule {
                name: "GitHub OAuth Token".to_string(),
                regex: Regex::new(r"gho_[0-9a-zA-Z]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub OAuth Access Token detected".to_string(),
            },
            // Slack Token
            SecretRule {
                name: "Slack Token".to_string(),
                regex: Regex::new(r"xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[0-9a-zA-Z]{24,32}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Slack Token detected".to_string(),
            },
            // Google API Key
            SecretRule {
                name: "Google API Key".to_string(),
                regex: Regex::new(r"AIza[0-9A-Za-z-_]{35}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Google API Key detected".to_string(),
            },
            // Generic API Key
            SecretRule {
                name: "Generic API Key".to_string(),
                regex: Regex::new(r#"(?i)(api[_-]?key|apikey)['"\s]*[:=]['"\s]*['"]([0-9a-zA-Z\-_]{16,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic API Key pattern detected".to_string(),
            },
            // Generic Secret
            SecretRule {
                name: "Generic Secret".to_string(),
                regex: Regex::new(r#"(?i)(secret|password|passwd|pwd)['"\s]*[:=]['"\s]*['"]([^\s'"]{8,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic Secret pattern detected".to_string(),
            },
            // Private Key
            SecretRule {
                name: "Private Key".to_string(),
                regex: Regex::new(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Private Key detected".to_string(),
            },
            // JWT Token
            SecretRule {
                name: "JWT Token".to_string(),
                regex: Regex::new(r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*").unwrap(),
                severity: "MEDIUM".to_string(),
                description: "JWT Token detected".to_string(),
            },
            // Database Connection String
            SecretRule {
                name: "Database Connection String".to_string(),
                regex: Regex::new(r#"(?i)(mongodb|mysql|postgres|postgresql)://[^\s'"]+:[^\s'"]+@[^\s'"]+(?::\d+)?(?:/[^\s'"]*)?"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Database connection string with credentials detected".to_string(),
            },
            // Docker Auth
            SecretRule {
                name: "Docker Auth Config".to_string(),
                regex: Regex::new(r#"(?i)"auth":\s*"[A-Za-z0-9+/=]{20,}""#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Docker authentication config detected".to_string(),
            },
            // NPM Token
            SecretRule {
                name: "NPM Token".to_string(),
                regex: Regex::new(r"npm_[a-zA-Z0-9]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "NPM access token detected".to_string(),
            },
            // Stripe API Key
            SecretRule {
                name: "Stripe API Key".to_string(),
                regex: Regex::new(r"sk_live_[0-9a-zA-Z]{24,}").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Stripe Live Secret Key detected".to_string(),
            },
            // Twilio API Key
            SecretRule {
                name: "Twilio API Key".to_string(),
                regex: Regex::new(r"SK[0-9a-fA-F]{32}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Twilio API Key detected".to_string(),
            },
        ]
    }

    /// 掃描檔案內容
    pub fn scan_content(&self, content: &str, file_path: &str) -> Vec<SecretFinding> {
        let mut findings = Vec::new();

        // 1. 規則匹配
        for (line_num, line) in content.lines().enumerate() {
            for rule in &self.rule_matchers {
                if let Some(captures) = rule.regex.captures(line) {
                    let matched_text = captures.get(0).map(|m| m.as_str()).unwrap_or("");
                    
                    // 提取上下文（前後各 50 字元）
                    let context = Self::extract_context(line, matched_text, 50);

                    findings.push(SecretFinding {
                        rule_name: rule.name.clone(),
                        matched_text: Self::redact_secret(matched_text),
                        file_path: file_path.to_string(),
                        line_number: line_num + 1,
                        severity: rule.severity.clone(),
                        entropy: None,
                        context,
                    });
                }
            }

            // 2. 高熵字串檢測
            if let Some(high_entropy_strings) = self.entropy_detector.detect_line(line) {
                for (text, entropy) in high_entropy_strings {
                    let context = Self::extract_context(line, &text, 50);
                    
                    findings.push(SecretFinding {
                        rule_name: "High Entropy String".to_string(),
                        matched_text: Self::redact_secret(&text),
                        file_path: file_path.to_string(),
                        line_number: line_num + 1,
                        severity: "MEDIUM".to_string(),
                        entropy: Some(entropy),
                        context,
                    });
                }
            }
        }

        findings
    }

    /// 提取上下文
    fn extract_context(line: &str, matched_text: &str, max_length: usize) -> String {
        if let Some(pos) = line.find(matched_text) {
            let start = pos.saturating_sub(max_length);
            let end = (pos + matched_text.len() + max_length).min(line.len());
            line[start..end].to_string()
        } else {
            line.to_string()
        }
    }

    /// 遮蔽敏感資訊
    fn redact_secret(text: &str) -> String {
        if text.len() <= 8 {
            "*".repeat(text.len())
        } else {
            format!("{}***{}", &text[..4], &text[text.len()-4..])
        }
    }
}

impl EntropyDetector {
    /// 建立熵值檢測器
    pub fn new(threshold: f64, min_length: usize) -> Self {
        Self {
            threshold,
            min_length,
        }
    }

    /// 檢測一行中的高熵字串
    pub fn detect_line(&self, line: &str) -> Option<Vec<(String, f64)>> {
        let mut findings = Vec::new();

        // 提取可能的 token（引號內的字串、等號後的值等）
        let token_regex = Regex::new(r#"['"]([^'"]{20,})['"]|=\s*([^\s'"]{20,})"#).unwrap();

        for captures in token_regex.captures_iter(line) {
            let token = captures.get(1)
                .or_else(|| captures.get(2))
                .map(|m| m.as_str())
                .unwrap_or("");

            if token.len() >= self.min_length {
                let entropy = self.calculate_entropy(token);
                if entropy >= self.threshold {
                    findings.push((token.to_string(), entropy));
                }
            }
        }

        if findings.is_empty() {
            None
        } else {
            Some(findings)
        }
    }

    /// 計算 Shannon 熵
    pub fn calculate_entropy(&self, text: &str) -> f64 {
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        
        for c in text.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }

        let len = text.len() as f64;
        let mut entropy = 0.0;

        for count in char_counts.values() {
            let probability = *count as f64 / len;
            entropy -= probability * probability.log2();
        }

        entropy
    }
}

impl Default for SecretDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_key_detection() {
        let detector = SecretDetector::new();
        let content = r#"
            AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
            AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        "#;

        let findings = detector.scan_content(content, "test.env");
        assert!(!findings.is_empty());
        assert!(findings.iter().any(|f| f.rule_name.contains("AWS")));
    }

    #[test]
    fn test_github_token_detection() {
        let detector = SecretDetector::new();
        let content = "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz";

        let findings = detector.scan_content(content, "test.txt");
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_entropy_calculation() {
        let detector = EntropyDetector::new(4.0, 10);
        
        // 低熵字串（重複字元）
        let low_entropy = detector.calculate_entropy("aaaaaaaaaa");
        assert!(low_entropy < 2.0);

        // 高熵字串（隨機字元）
        let high_entropy = detector.calculate_entropy("a1B2c3D4e5");
        assert!(high_entropy > 3.0);
    }

    #[test]
    fn test_private_key_detection() {
        let detector = SecretDetector::new();
        let content = r#"
            -----BEGIN RSA PRIVATE KEY-----
            MIIEpAIBAAKCAQEA...
            -----END RSA PRIVATE KEY-----
        "#;

        let findings = detector.scan_content(content, "key.pem");
        assert!(findings.iter().any(|f| f.rule_name.contains("Private Key")));
    }

    #[test]
    fn test_redact_secret() {
        let text = "AKIAIOSFODNN7EXAMPLE";
        let redacted = SecretDetector::redact_secret(text);
        assert_eq!(redacted, "AKIA***MPLE");
    }
}
