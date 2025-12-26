// JavaScript 密碼學分析器
// 分析下載的 JavaScript 文件中的密碼學問題

use crate::Finding;
use regex::Regex;

pub fn scan_javascript(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // 1. 硬編碼 API Key 檢測
    findings.extend(detect_hardcoded_keys(content));

    // 2. 弱加密算法使用
    findings.extend(detect_weak_crypto_usage(content));

    // 3. 不安全的隨機數生成
    findings.extend(detect_weak_random(content));

    // 4. JWT 安全問題
    findings.extend(detect_jwt_issues(content));

    // 5. 本地存儲敏感數據
    findings.extend(detect_insecure_storage(content));

    findings
}

fn detect_hardcoded_keys(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Stripe Keys
    let stripe_regex = Regex::new(r"(?i)(pk|sk)_(live|test)_[0-9a-zA-Z]{24,}").unwrap();
    for cap in stripe_regex.captures_iter(content) {
        let key = cap.get(0).unwrap().as_str();
        let is_live = key.contains("_live_");
        findings.push(Finding {
            issue_type: "HARDCODED_API_KEY".to_string(),
            severity: if is_live { "CRITICAL" } else { "HIGH" }.to_string(),
            title: format!("Hardcoded {} Stripe API Key", if is_live { "Live" } else { "Test" }),
            description: format!(
                "Found hardcoded Stripe API key: {}...{}",
                &key[..15],
                &key[key.len()-4..]
            ),
            recommendation: "Remove API keys from client-side code. Use environment variables and backend proxy.".to_string(),
            location: Some(format!("Line {}", find_line_number(content, key))),
            evidence: Some(mask_key(key)),
        });
    }

    // AWS Access Keys
    let aws_regex = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
    for mat in aws_regex.find_iter(content) {
        let key = mat.as_str();
        findings.push(Finding {
            issue_type: "HARDCODED_AWS_KEY".to_string(),
            severity: "CRITICAL".to_string(),
            title: "Hardcoded AWS Access Key".to_string(),
            description:
                "AWS Access Key found in JavaScript. This allows unauthorized AWS API access."
                    .to_string(),
            recommendation:
                "Immediately revoke this key and use IAM roles or Cognito for client access."
                    .to_string(),
            location: Some(format!("Line {}", find_line_number(content, key))),
            evidence: Some(mask_key(key)),
        });
    }

    // Google API Keys
    let google_regex = Regex::new(r"AIza[0-9A-Za-z\-_]{35}").unwrap();
    for mat in google_regex.find_iter(content) {
        let key = mat.as_str();
        findings.push(Finding {
            issue_type: "HARDCODED_GOOGLE_KEY".to_string(),
            severity: "HIGH".to_string(),
            title: "Hardcoded Google API Key".to_string(),
            description: "Google API key exposed in client-side code.".to_string(),
            recommendation: "Use API key restrictions (HTTP referrers, IP addresses) and consider backend proxy.".to_string(),
            location: Some(format!("Line {}", find_line_number(content, key))),
            evidence: Some(mask_key(key)),
        });
    }

    // Generic API Keys
    let generic_regex = Regex::new(
        r#"(?i)(?:api[_-]?key|apikey|api_secret|access[_-]?token)['"\s:=]+['"]?([0-9a-zA-Z\-_]{20,})['"]?"#
    ).unwrap();
    for cap in generic_regex.captures_iter(content) {
        if let Some(key_match) = cap.get(1) {
            let key = key_match.as_str();
            findings.push(Finding {
                issue_type: "HARDCODED_API_KEY".to_string(),
                severity: "HIGH".to_string(),
                title: "Hardcoded API Key or Token".to_string(),
                description: "Sensitive credential found in JavaScript code.".to_string(),
                recommendation:
                    "Move credentials to backend. Never expose secrets in client-side code."
                        .to_string(),
                location: Some(format!("Line {}", find_line_number(content, key))),
                evidence: Some(mask_key(key)),
            });
        }
    }

    findings
}

fn detect_weak_crypto_usage(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // 使用更精確的正則表達式檢測實際的函數調用，避免誤報註釋和字符串
    
    // MD5 使用 - 檢測函數調用或 crypto API 使用
    let md5_regex = Regex::new(r#"(?i)\b(?:crypto\.createHash\s*\(\s*['"]md5['"]|require\s*\(['"][^'"]*md5['"]\)|import\s+.*\bmd5\b|md5\s*\()"#).unwrap();
    if md5_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "WEAK_HASH_ALGORITHM".to_string(),
            severity: "MEDIUM".to_string(),
            title: "MD5 Hash Algorithm Used".to_string(),
            description:
                "MD5 is cryptographically broken and should not be used for security purposes."
                    .to_string(),
            recommendation: "Use SHA-256 or SHA-3 for hashing. Use bcrypt/argon2 for passwords."
                .to_string(),
            location: None,
            evidence: None,
        });
    }

    // SHA-1 使用 - 檢測函數調用
    let sha1_regex = Regex::new(r#"(?i)\b(?:crypto\.createHash\s*\(\s*['"]sha-?1['"]|require\s*\(['"][^'"]*sha-?1['"]\)|import\s+.*\bsha1\b|sha1\s*\()"#).unwrap();
    if sha1_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "WEAK_HASH_ALGORITHM".to_string(),
            severity: "MEDIUM".to_string(),
            title: "SHA-1 Hash Algorithm Used".to_string(),
            description: "SHA-1 is deprecated due to collision attacks.".to_string(),
            recommendation: "Upgrade to SHA-256 or SHA-3.".to_string(),
            location: None,
            evidence: None,
        });
    }

    // DES/3DES 使用 - 檢測加密 API 調用，避免誤報 "describe", "design" 等詞
    let des_regex = Regex::new(r#"(?i)\b(?:crypto\.createCipher(?:iv)?\s*\(\s*['"](?:des|3des|tripledes)['"]|require\s*\(['"][^'"]*\bdes\b['"]\)|import\s+.*\bdes\b)"#).unwrap();
    if des_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            severity: "HIGH".to_string(),
            title: "Weak Encryption Algorithm (DES/3DES)".to_string(),
            description: "DES and 3DES are deprecated encryption algorithms.".to_string(),
            recommendation: "Use AES-256-GCM for encryption.".to_string(),
            location: None,
            evidence: None,
        });
    }

    // RC4 使用 - 檢測加密 API 調用
    let rc4_regex = Regex::new(r#"(?i)\b(?:crypto\.createCipher(?:iv)?\s*\(\s*['"](?:rc4|arcfour)['"]|require\s*\(['"][^'"]*\brc4\b['"]\)|import\s+.*\brc4\b)"#).unwrap();
    if rc4_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            severity: "HIGH".to_string(),
            title: "RC4 Cipher Used".to_string(),
            description: "RC4 has known biases and is insecure.".to_string(),
            recommendation: "Use AES-GCM or ChaCha20-Poly1305.".to_string(),
            location: None,
            evidence: None,
        });
    }

    findings
}

fn detect_weak_random(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Math.random() 用於安全目的
    let math_random_regex =
        Regex::new(r"(?i)Math\.random\(\).*(?:token|key|id|secret|password|session)").unwrap();

    if math_random_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "WEAK_RANDOM".to_string(),
            severity: "MEDIUM".to_string(),
            title: "Insecure Random Number Generation".to_string(),
            description: "Math.random() is not cryptographically secure and should not be used for security-sensitive values.".to_string(),
            recommendation: "Use crypto.getRandomValues() or crypto.randomUUID() for security purposes.".to_string(),
            location: None,
            evidence: Some("Math.random() used for security-sensitive value generation".to_string()),
        });
    }

    findings
}

fn detect_jwt_issues(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // JWT 'none' algorithm
    if content.contains(r#""alg":"none""#) || content.contains(r#"'alg':'none'"#) {
        findings.push(Finding {
            issue_type: "JWT_NONE_ALGORITHM".to_string(),
            severity: "CRITICAL".to_string(),
            title: "JWT 'none' Algorithm Accepted".to_string(),
            description: "JWT configured to accept 'none' algorithm, allowing unsigned tokens."
                .to_string(),
            recommendation:
                "Explicitly whitelist only secure algorithms (RS256, ES256). Never accept 'none'."
                    .to_string(),
            location: None,
            evidence: Some("JWT 'none' algorithm detected in code".to_string()),
        });
    }

    // JWT stored in localStorage
    if content.contains("localStorage") && (content.contains("jwt") || content.contains("token")) {
        findings.push(Finding {
            issue_type: "INSECURE_TOKEN_STORAGE".to_string(),
            severity: "MEDIUM".to_string(),
            title: "JWT/Token Stored in localStorage".to_string(),
            description: "localStorage is vulnerable to XSS attacks. Tokens stored here can be stolen.".to_string(),
            recommendation: "Store tokens in httpOnly cookies or use sessionStorage with additional protections.".to_string(),
            location: None,
            evidence: None,
        });
    }

    findings
}

fn detect_insecure_storage(content: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Sensitive data in localStorage
    let sensitive_regex = Regex::new(
        r#"localStorage\.setItem\s*\(\s*['"]([^'"]*(?:password|secret|key|token|auth)[^'"]*)['"]\s*,"#
    ).unwrap();

    if sensitive_regex.is_match(content) {
        findings.push(Finding {
            issue_type: "INSECURE_STORAGE".to_string(),
            severity: "MEDIUM".to_string(),
            title: "Sensitive Data in localStorage".to_string(),
            description: "Storing sensitive data in localStorage exposes it to XSS attacks.".to_string(),
            recommendation: "Never store sensitive data in localStorage. Use httpOnly cookies or encrypted sessionStorage.".to_string(),
            location: None,
            evidence: None,
        });
    }

    findings
}

fn find_line_number(content: &str, search: &str) -> usize {
    content[..content.find(search).unwrap_or(0)].lines().count() + 1
}

fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        return "***".to_string();
    }
    format!("{}...{}", &key[..4], &key[key.len() - 4..])
}
