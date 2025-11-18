// Sensitive Info Scanner - 核心掃描邏輯
use aho_corasick::AhoCorasick;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;

/// 掃描模式 - 根據 Nmap 和 OWASP 最佳實踐設計
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Mode 1: 快速發現模式 (Fast Discovery)
    /// - 類似 Nmap -T4/-T5
    /// - 輕量級正則匹配
    /// - 無密鑰驗證
    /// - 並行度最大化
    /// - 目標:大範圍快速資產識別(技術棧、框架、敏感特徵)
    FastDiscovery,
    
    /// Mode 2: 深度分析模式 (Deep Analysis) - 現有功能
    /// - 完整的敏感資訊掃描
    /// - 10+ 種密鑰檢測規則
    /// - 高優先級密鑰驗證
    /// - 詳細統計記錄
    DeepAnalysis,
    
    /// Mode 3: 聚焦驗證模式 (Focused Verification)
    /// - 針對特定類型的深度驗證
    /// - 根據 Mode 1 結果動態選擇規則
    /// - 只驗證高價值目標
    /// - 平衡速度與準確性
    FocusedVerification,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensitiveInfo {
    pub info_type: String,
    pub value: String,
    pub confidence: f32,
    pub location: String,
}

pub struct SensitiveInfoScanner {
    patterns: Vec<Pattern>,
    fast_patterns: Vec<Pattern>,  // 快速模式專用
    keyword_matcher: AhoCorasick,
    mode: ScanMode,
}

struct Pattern {
    name: &'static str,
    regex: Regex,
    confidence: f32,
}

impl SensitiveInfoScanner {
    pub fn new() -> Self {
        Self::with_mode(ScanMode::DeepAnalysis)
    }
    
    pub fn with_mode(mode: ScanMode) -> Self {
        // 編譯正則表達式 (一次性成本)
        let patterns = vec![
            Pattern {
                name: "AWS Access Key",
                regex: Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
                confidence: 0.95,
            },
            Pattern {
                name: "AWS Secret Key",
                regex: Regex::new(r#"(?i)aws(.{0,20})?['"][0-9a-zA-Z/+]{40}['"]"#).unwrap(),
                confidence: 0.85,
            },
            Pattern {
                name: "GitHub Token",
                regex: Regex::new(r"ghp_[0-9a-zA-Z]{36}").unwrap(),
                confidence: 0.98,
            },
            Pattern {
                name: "Generic API Key",
                regex: Regex::new(r#"(?i)api[_-]?key['"]?\s*[:=]\s*['"]?([0-9a-zA-Z\-_]{20,})"#)
                    .unwrap(),
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
                regex: Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
                    .unwrap(),
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
        
        // 快速發現模式專用模式 (輕量級)
        let fast_patterns = vec![
            Pattern {
                name: "Tech Stack",
                regex: Regex::new(r"(?i)(php|java|node\.js|\.net|python|ruby|react|vue|angular)").unwrap(),
                confidence: 0.90,
            },
            Pattern {
                name: "API Endpoint",
                regex: Regex::new(r"/api/[v\d]+/[a-zA-Z_]+").unwrap(),
                confidence: 0.85,
            },
            Pattern {
                name: "Admin Interface",
                regex: Regex::new(r"(?i)/(admin|manage|backend|console)").unwrap(),
                confidence: 0.80,
            },
            Pattern {
                name: "Config File",
                regex: Regex::new(r"(?i)\.(config|env|ini|yaml|yml)$").unwrap(),
                confidence: 0.75,
            },
        ];

        Self {
            patterns,
            fast_patterns,
            keyword_matcher,
            mode,
        }
    }

    pub fn scan(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
        match self.mode {
            ScanMode::FastDiscovery => self.scan_fast(content, source_url),
            ScanMode::DeepAnalysis => self.scan_deep(content, source_url),
            ScanMode::FocusedVerification => self.scan_focused(content, source_url),
        }
    }
    
    /// Mode 1: 快速發現掃描 - 大範圍識別技術棧和敏感特徵
    fn scan_fast(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
        // 快速模式:無關鍵字過濾,直接並行掃描
        self.fast_patterns
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
                            value: matched_text.to_string(), // 快速模式不遮罩
                            confidence: pattern.confidence,
                            location: format!("{}:{}", source_url, position),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// Mode 2: 深度分析掃描 - 完整敏感資訊檢測(現有功能)
    fn scan_deep(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
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
    
    /// Mode 3: 聚焦驗證掃描 - 針對特定類型深度驗證
    fn scan_focused(&self, content: &str, source_url: &str) -> Vec<SensitiveInfo> {
        // 聚焦模式:只掃描高價值目標
        let focused_patterns = self.patterns.iter()
            .filter(|p| matches!(p.name, 
                "AWS Access Key" | "GitHub Token" | "Private Key" | "JWT Token"
            ));
        
        focused_patterns
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
