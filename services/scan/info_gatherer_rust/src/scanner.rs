// Sensitive Info Scanner - 核心掃描邏輯
use aho_corasick::AhoCorasick;
use regex::Regex;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SensitiveInfo {
    pub info_type: String,
    pub value: String,
    pub confidence: f32,
    pub location: String,
}

pub struct SensitiveInfoScanner {
    patterns: Vec<Pattern>,
    keyword_matcher: AhoCorasick,
}

struct Pattern {
    name: &'static str,
    regex: Regex,
    confidence: f32,
}

impl SensitiveInfoScanner {
    pub fn new() -> Self {
        // 編譯正則表達式 (一次性成本)
        let patterns = vec![
            Pattern {
                name: "AWS Access Key",
                regex: Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
                confidence: 0.95,
            },
            Pattern {
                name: "AWS Secret Key",
                regex: Regex::new(r"(?i)aws(.{0,20})?['\"][0-9a-zA-Z/+]{40}['\"]").unwrap(),
                confidence: 0.85,
            },
            Pattern {
                name: "GitHub Token",
                regex: Regex::new(r"ghp_[0-9a-zA-Z]{36}").unwrap(),
                confidence: 0.98,
            },
            Pattern {
                name: "Generic API Key",
                regex: Regex::new(r"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?([0-9a-zA-Z\-_]{20,})").unwrap(),
                confidence: 0.75,
            },
            Pattern {
                name: "Private Key",
                regex: Regex::new(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----").unwrap(),
                confidence: 0.99,
            },
            Pattern {
                name: "Email",
                regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
                confidence: 0.90,
            },
            Pattern {
                name: "IP Address",
                regex: Regex::new(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b").unwrap(),
                confidence: 0.70,
            },
            Pattern {
                name: "JWT Token",
                regex: Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}").unwrap(),
                confidence: 0.95,
            },
            Pattern {
                name: "Password in Code",
                regex: Regex::new(r#"(?i)password['"]?\s*[:=]\s*['"]([^'"]{6,})"#).unwrap(),
                confidence: 0.80,
            },
            Pattern {
                name: "Database Connection String",
                regex: Regex::new(r"(?i)(mysql|postgres|mongodb)://[^\s]+").unwrap(),
                confidence: 0.90,
            },
        ];

        // 關鍵字快速匹配 (Aho-Corasick 算法)
        let keywords = vec![
            "password",
            "api_key",
            "secret",
            "token",
            "private_key",
            "access_key",
            "credential",
        ];

        let keyword_matcher = AhoCorasick::new(keywords).unwrap();

        Self {
            patterns,
            keyword_matcher,
        }
    }

    pub fn scan(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
        // 先用關鍵字快速過濾
        if !self.keyword_matcher.is_match(content) {
            return Vec::new();
        }

        // 使用 Rayon 並行掃描
        self.patterns
            .par_iter()
            .flat_map(|pattern| {
                pattern
                    .regex
                    .find_iter(content)
                    .map(|m| {
                        let matched_text = m.as_str();
                        let position = m.start();

                        SensitiveInfo {
                            info_type: pattern.name.to_string(),
                            value: Self::mask_sensitive(matched_text),
                            confidence: pattern.confidence,
                            location: format!("{}:{}", source_url, position),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn mask_sensitive(value: &str) -> String {
        if value.len() <= 8 {
            return "*".repeat(value.len());
        }

        let visible_len = 4;
        let prefix = &value[..visible_len];
        let suffix = &value[value.len() - visible_len..];
        let masked_len = value.len() - 2 * visible_len;

        format!("{}{}...{}", prefix, "*".repeat(masked_len.min(10)), suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_aws_key() {
        let scanner = SensitiveInfoScanner::new();
        let content = "AKIAIOSFODNN7EXAMPLE";

        let findings = scanner.scan(content, "test.js");

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].info_type, "AWS Access Key");
    }

    #[test]
    fn test_scan_multiple_patterns() {
        let scanner = SensitiveInfoScanner::new();
        let content = r#"
            const api_key = "sk_test_1234567890abcdefghij";
            const email = "admin@example.com";
            const password = "SuperSecret123!";
        "#;

        let findings = scanner.scan(content, "config.js");

        assert!(findings.len() >= 2);
    }

    #[test]
    fn test_masking() {
        let masked = SensitiveInfoScanner::mask_sensitive("AKIAIOSFODNN7EXAMPLE");
        assert!(masked.starts_with("AKIA"));
        assert!(masked.contains("*"));
        assert!(masked.ends_with("MPLE"));
    }
}
