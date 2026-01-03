use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    #[allow(dead_code)] // Reserved for future detailed reporting
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

    /// 載入預設規則 (50+ 密鑰檢測規則)
    fn load_default_rules() -> Vec<SecretRule> {
        vec![
            // ==================== AWS ====================
            SecretRule {
                name: "AWS Access Key ID".to_string(),
                regex: Regex::new(r"(?i)(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "AWS Access Key ID detected".to_string(),
            },
            SecretRule {
                name: "AWS Secret Access Key".to_string(),
                regex: Regex::new(r#"(?i)aws(.{0,20})?(?-i)['\"][0-9a-zA-Z/+]{40}['"]"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "AWS Secret Access Key detected".to_string(),
            },
            SecretRule {
                name: "AWS Session Token".to_string(),
                regex: Regex::new(r#"(?i)aws[_-]?session[_-]?token['"\s]*[:=]['"\s]*['"]([A-Za-z0-9+/=]{100,})['"]"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "AWS Session Token detected".to_string(),
            },

            // ==================== Azure ====================
            SecretRule {
                name: "Azure Storage Account Key".to_string(),
                regex: Regex::new(r"(?i)AccountKey=[A-Za-z0-9+/=]{88}").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Azure Storage Account Key detected".to_string(),
            },
            SecretRule {
                name: "Azure Connection String".to_string(),
                regex: Regex::new(r#"(?i)DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]{88}"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Azure Storage Connection String detected".to_string(),
            },
            SecretRule {
                name: "Azure SAS Token".to_string(),
                regex: Regex::new(r"(?i)\?sv=\d{4}-\d{2}-\d{2}&ss=[bfqt]+&srt=[sco]+&sp=[rwdlacup]+&se=[\d-]+T[\d:]+Z&st=[\d-]+T[\d:]+Z&spr=https&sig=[A-Za-z0-9%]+").unwrap(),
                severity: "HIGH".to_string(),
                description: "Azure SAS (Shared Access Signature) Token detected".to_string(),
            },
            SecretRule {
                name: "Azure Client Secret".to_string(),
                regex: Regex::new(r#"(?i)client[_-]?secret['"\s]*[:=]['"\s]*['"]([A-Za-z0-9~._-]{32,})['"]"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Azure AD Application Client Secret detected".to_string(),
            },

            // ==================== GitHub ====================
            SecretRule {
                name: "GitHub Personal Access Token".to_string(),
                regex: Regex::new(r"ghp_[0-9a-zA-Z]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub Personal Access Token detected".to_string(),
            },
            SecretRule {
                name: "GitHub OAuth Token".to_string(),
                regex: Regex::new(r"gho_[0-9a-zA-Z]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub OAuth Access Token detected".to_string(),
            },
            SecretRule {
                name: "GitHub App Token".to_string(),
                regex: Regex::new(r"(ghu|ghs)_[0-9a-zA-Z]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub App/Server Token detected".to_string(),
            },
            SecretRule {
                name: "GitHub Refresh Token".to_string(),
                regex: Regex::new(r"ghr_[0-9a-zA-Z]{76}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitHub Refresh Token detected".to_string(),
            },

            // ==================== GitLab ====================
            SecretRule {
                name: "GitLab Personal Access Token".to_string(),
                regex: Regex::new(r"glpat-[0-9a-zA-Z_-]{20}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitLab Personal Access Token detected".to_string(),
            },
            SecretRule {
                name: "GitLab Pipeline Trigger Token".to_string(),
                regex: Regex::new(r"glptt-[0-9a-f]{40}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitLab Pipeline Trigger Token detected".to_string(),
            },
            SecretRule {
                name: "GitLab Runner Token".to_string(),
                regex: Regex::new(r"GR1348941[0-9a-zA-Z_-]{20}").unwrap(),
                severity: "HIGH".to_string(),
                description: "GitLab CI Runner Registration Token detected".to_string(),
            },

            // ==================== Slack ====================
            SecretRule {
                name: "Slack Token".to_string(),
                regex: Regex::new(r"xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[0-9a-zA-Z]{24,32}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Slack Token detected".to_string(),
            },
            SecretRule {
                name: "Slack Webhook".to_string(),
                regex: Regex::new(r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24}").unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Slack Webhook URL detected".to_string(),
            },

            // ==================== Google ====================
            SecretRule {
                name: "Google API Key".to_string(),
                regex: Regex::new(r"AIza[0-9A-Za-z-_]{35}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Google API Key detected".to_string(),
            },
            SecretRule {
                name: "Google OAuth Access Token".to_string(),
                regex: Regex::new(r"ya29\.[0-9A-Za-z\-_]+").unwrap(),
                severity: "HIGH".to_string(),
                description: "Google OAuth Access Token detected".to_string(),
            },
            SecretRule {
                name: "Google Cloud Service Account".to_string(),
                regex: Regex::new(r#"(?i)"type":\s*"service_account"[^}]*"private_key":\s*"-----BEGIN PRIVATE KEY-----"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Google Cloud Service Account JSON detected".to_string(),
            },

            // ==================== Stripe ====================
            SecretRule {
                name: "Stripe Live Secret Key".to_string(),
                regex: Regex::new(r"sk_live_[0-9a-zA-Z]{24,}").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Stripe Live Secret Key detected".to_string(),
            },
            SecretRule {
                name: "Stripe Test Secret Key".to_string(),
                regex: Regex::new(r"sk_test_[0-9a-zA-Z]{24,}").unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Stripe Test Secret Key detected".to_string(),
            },
            SecretRule {
                name: "Stripe Restricted Key".to_string(),
                regex: Regex::new(r"rk_live_[0-9a-zA-Z]{24,}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Stripe Restricted API Key detected".to_string(),
            },

            // ==================== Twilio ====================
            SecretRule {
                name: "Twilio API Key".to_string(),
                regex: Regex::new(r"SK[0-9a-fA-F]{32}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Twilio API Key detected".to_string(),
            },
            SecretRule {
                name: "Twilio Account SID".to_string(),
                regex: Regex::new(r"AC[a-z0-9]{32}").unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Twilio Account SID detected".to_string(),
            },

            // ==================== SendGrid ====================
            SecretRule {
                name: "SendGrid API Key".to_string(),
                regex: Regex::new(r"SG\.[0-9A-Za-z\-_]{22}\.[0-9A-Za-z\-_]{43}").unwrap(),
                severity: "HIGH".to_string(),
                description: "SendGrid API Key detected".to_string(),
            },

            // ==================== Mailgun ====================
            SecretRule {
                name: "Mailgun API Key".to_string(),
                regex: Regex::new(r"key-[0-9a-zA-Z]{32}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Mailgun API Key detected".to_string(),
            },

            // ==================== Firebase ====================
            SecretRule {
                name: "Firebase API Key".to_string(),
                regex: Regex::new(r#"(?i)firebase[_-]?api[_-]?key['"\s]*[:=]['"\s]*['"]AIza[0-9A-Za-z_-]{35}['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Firebase API Key detected".to_string(),
            },
            SecretRule {
                name: "Firebase Cloud Messaging Token".to_string(),
                regex: Regex::new(r#"(?i)fcm[_-]?token['"\s]*[:=]['"\s]*['"]([a-zA-Z0-9:_-]{140,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Firebase Cloud Messaging Token detected".to_string(),
            },

            // ==================== Heroku ====================
            SecretRule {
                name: "Heroku API Key".to_string(),
                regex: Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Heroku API Key (UUID format) detected".to_string(),
            },

            // ==================== NPM ====================
            SecretRule {
                name: "NPM Access Token".to_string(),
                regex: Regex::new(r"npm_[a-zA-Z0-9]{36}").unwrap(),
                severity: "HIGH".to_string(),
                description: "NPM access token detected".to_string(),
            },

            // ==================== PyPI ====================
            SecretRule {
                name: "PyPI Upload Token".to_string(),
                regex: Regex::new(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}").unwrap(),
                severity: "HIGH".to_string(),
                description: "PyPI Upload Token detected".to_string(),
            },

            // ==================== Docker Hub ====================
            SecretRule {
                name: "Docker Hub Access Token".to_string(),
                regex: Regex::new(r"dckr_pat_[a-zA-Z0-9_-]{32,}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Docker Hub Personal Access Token detected".to_string(),
            },
            SecretRule {
                name: "Docker Auth Config".to_string(),
                regex: Regex::new(r#"(?i)"auth":\s*"[A-Za-z0-9+/=]{20,}""#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Docker authentication config detected".to_string(),
            },

            // ==================== CI/CD Platforms ====================
            SecretRule {
                name: "CircleCI Personal Token".to_string(),
                regex: Regex::new(r#"circleci[_-]?token['"\s]*[:=]['"\s]*['"]([a-f0-9]{40})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "CircleCI Personal API Token detected".to_string(),
            },
            SecretRule {
                name: "Travis CI Token".to_string(),
                regex: Regex::new(r#"travis[_-]?token['"\s]*[:=]['"\s]*['"]([a-zA-Z0-9]{22})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Travis CI Access Token detected".to_string(),
            },
            SecretRule {
                name: "Jenkins API Token".to_string(),
                regex: Regex::new(r#"jenkins[_-]?token['"\s]*[:=]['"\s]*['"]([a-f0-9]{32,})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Jenkins API Token detected".to_string(),
            },

            // ==================== Bitbucket ====================
            SecretRule {
                name: "Bitbucket Personal Access Token".to_string(),
                regex: Regex::new(r#"(?i)bitbucket[_-]?token['"\s]*[:=]['"\s]*['"]([A-Za-z0-9]{43})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Bitbucket Personal Access Token detected".to_string(),
            },

            // ==================== Dropbox ====================
            SecretRule {
                name: "Dropbox Access Token".to_string(),
                regex: Regex::new(r"sl\.[A-Za-z0-9\-_=]{135,}").unwrap(),
                severity: "HIGH".to_string(),
                description: "Dropbox Access Token detected".to_string(),
            },
            SecretRule {
                name: "Dropbox API Key".to_string(),
                regex: Regex::new(r#"(?i)dropbox[_-]?api[_-]?key['"\s]*[:=]['"\s]*['"]([a-z0-9]{15})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Dropbox API Key detected".to_string(),
            },

            // ==================== DigitalOcean ====================
            SecretRule {
                name: "DigitalOcean Personal Access Token".to_string(),
                regex: Regex::new(r"dop_v1_[a-f0-9]{64}").unwrap(),
                severity: "HIGH".to_string(),
                description: "DigitalOcean Personal Access Token detected".to_string(),
            },
            SecretRule {
                name: "DigitalOcean OAuth Token".to_string(),
                regex: Regex::new(r"doo_v1_[a-f0-9]{64}").unwrap(),
                severity: "HIGH".to_string(),
                description: "DigitalOcean OAuth Token detected".to_string(),
            },

            // ==================== Cloudflare ====================
            SecretRule {
                name: "Cloudflare API Key".to_string(),
                regex: Regex::new(r#"(?i)cloudflare[_-]?api[_-]?key['"\s]*[:=]['"\s]*['"]([a-z0-9]{37})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Cloudflare API Key detected".to_string(),
            },
            SecretRule {
                name: "Cloudflare API Token".to_string(),
                regex: Regex::new(r#"(?i)cloudflare[_-]?token['"\s]*[:=]['"\s]*['"]([A-Za-z0-9_-]{40})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Cloudflare API Token detected".to_string(),
            },

            // ==================== Datadog ====================
            SecretRule {
                name: "Datadog API Key".to_string(),
                regex: Regex::new(r#"(?i)datadog[_-]?api[_-]?key['"\s]*[:=]['"\s]*['"]([a-f0-9]{32})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Datadog API Key detected".to_string(),
            },
            SecretRule {
                name: "Datadog Application Key".to_string(),
                regex: Regex::new(r#"(?i)datadog[_-]?app[_-]?key['"\s]*[:=]['"\s]*['"]([a-f0-9]{40})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Datadog Application Key detected".to_string(),
            },

            // ==================== MongoDB ====================
            SecretRule {
                name: "MongoDB Connection String".to_string(),
                regex: Regex::new(r#"(?i)mongodb(\+srv)?://[^\s'"]+:[^\s'"]+@[^\s'"]+(?::\d+)?(?:/[^\s'"]*)?"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "MongoDB connection string with credentials detected".to_string(),
            },
            SecretRule {
                name: "MongoDB Atlas API Key".to_string(),
                regex: Regex::new(r#"(?i)mongodb[_-]?atlas[_-]?api['"\s]*[:=]['"\s]*['"]([a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12})['"]"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "MongoDB Atlas API Key detected".to_string(),
            },

            // ==================== Redis ====================
            SecretRule {
                name: "Redis Connection String".to_string(),
                regex: Regex::new(r#"(?i)redis://:[^\s'"]+@[^\s'"]+(?::\d+)?"#).unwrap(),
                severity: "HIGH".to_string(),
                description: "Redis connection string with password detected".to_string(),
            },

            // ==================== PostgreSQL / MySQL ====================
            SecretRule {
                name: "PostgreSQL Connection String".to_string(),
                regex: Regex::new(r#"(?i)postgres(ql)?://[^\s'"]+:[^\s'"]+@[^\s'"]+(?::\d+)?(?:/[^\s'"]*)?"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "PostgreSQL connection string with credentials detected".to_string(),
            },
            SecretRule {
                name: "MySQL Connection String".to_string(),
                regex: Regex::new(r#"(?i)mysql://[^\s'"]+:[^\s'"]+@[^\s'"]+(?::\d+)?(?:/[^\s'"]*)?"#).unwrap(),
                severity: "CRITICAL".to_string(),
                description: "MySQL connection string with credentials detected".to_string(),
            },

            // ==================== Private Keys ====================
            SecretRule {
                name: "RSA Private Key".to_string(),
                regex: Regex::new(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "Private Key detected".to_string(),
            },
            SecretRule {
                name: "PGP Private Key".to_string(),
                regex: Regex::new(r"-----BEGIN PGP PRIVATE KEY BLOCK-----").unwrap(),
                severity: "CRITICAL".to_string(),
                description: "PGP Private Key Block detected".to_string(),
            },

            // ==================== JWT ====================
            SecretRule {
                name: "JWT Token".to_string(),
                regex: Regex::new(r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*").unwrap(),
                severity: "MEDIUM".to_string(),
                description: "JWT Token detected".to_string(),
            },

            // ==================== Generic Patterns ====================
            SecretRule {
                name: "Generic API Key".to_string(),
                regex: Regex::new(r#"(?i)(api[_-]?key|apikey)['"\s]*[:=]['"\s]*['"]([0-9a-zA-Z\-_]{16,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic API Key pattern detected".to_string(),
            },
            SecretRule {
                name: "Generic Secret".to_string(),
                regex: Regex::new(r#"(?i)(secret|password|passwd|pwd)['"\s]*[:=]['"\s]*['"]([^\s'"]{8,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic Secret pattern detected".to_string(),
            },
            SecretRule {
                name: "Generic Access Token".to_string(),
                regex: Regex::new(r#"(?i)(access[_-]?token|accesstoken)['"\s]*[:=]['"\s]*['"]([0-9a-zA-Z\-_]{20,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic Access Token pattern detected".to_string(),
            },
            SecretRule {
                name: "Generic Auth Token".to_string(),
                regex: Regex::new(r#"(?i)(auth[_-]?token|authtoken|bearer[_-]?token)['"\s]*[:=]['"\s]*['"]([0-9a-zA-Z\-_]{20,})['"]"#).unwrap(),
                severity: "MEDIUM".to_string(),
                description: "Generic Auth/Bearer Token pattern detected".to_string(),
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
            format!("{}***{}", &text[..4], &text[text.len() - 4..])
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
            let token = captures
                .get(1)
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
