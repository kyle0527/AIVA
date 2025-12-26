// HTTP 安全標頭分析器（密碼學相關）
// 分析 HTTP 響應標頭中的安全配置

use crate::Finding;
use serde_json::Value;

pub fn analyze_headers(headers_json: &str, url: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    let is_https = url.starts_with("https://");

    // 解析標頭 JSON
    let headers: Value = match serde_json::from_str(headers_json) {
        Ok(h) => h,
        Err(_) => return findings,
    };

    let headers_obj = match headers.as_object() {
        Some(obj) => obj,
        None => return findings,
    };

    // 將標頭鍵轉為小寫以便比較
    let headers_lower: std::collections::HashMap<String, String> = headers_obj
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.as_str().unwrap_or("").to_string()))
        .collect();

    // 1. 檢查 HSTS（Strict-Transport-Security）
    if is_https {
        if let Some(hsts) = headers_lower.get("strict-transport-security") {
            // 檢查 max-age
            if !hsts.contains("max-age") {
                findings.push(Finding {
                    issue_type: "WEAK_HSTS".to_string(),
                    severity: "MEDIUM".to_string(),
                    title: "HSTS Header Missing max-age".to_string(),
                    description: "HSTS header present but missing max-age directive.".to_string(),
                    recommendation:
                        "Add max-age directive: 'Strict-Transport-Security: max-age=31536000'"
                            .to_string(),
                    location: None,
                    evidence: Some(hsts.clone()),
                });
            } else {
                // 檢查 max-age 值是否足夠長
                if let Some(max_age) = extract_max_age(hsts) {
                    if max_age < 31536000 {
                        // 少於 1 年
                        findings.push(Finding {
                            issue_type: "WEAK_HSTS".to_string(),
                            severity: "LOW".to_string(),
                            title: "HSTS max-age Too Short".to_string(),
                            description: format!("HSTS max-age is {} seconds (< 1 year)", max_age),
                            recommendation: "Increase max-age to at least 31536000 (1 year)."
                                .to_string(),
                            location: None,
                            evidence: Some(hsts.clone()),
                        });
                    }
                }
            }

            // 檢查 includeSubDomains
            if !hsts.contains("includeSubDomains") {
                findings.push(Finding {
                    issue_type: "WEAK_HSTS".to_string(),
                    severity: "LOW".to_string(),
                    title: "HSTS Missing includeSubDomains".to_string(),
                    description: "HSTS does not cover subdomains.".to_string(),
                    recommendation: "Add 'includeSubDomains' directive for complete protection."
                        .to_string(),
                    location: None,
                    evidence: Some(hsts.clone()),
                });
            }
        } else {
            findings.push(Finding {
                issue_type: "MISSING_HSTS".to_string(),
                severity: "MEDIUM".to_string(),
                title: "Missing HSTS Header".to_string(),
                description: "HTTPS site without HSTS allows protocol downgrade attacks."
                    .to_string(),
                recommendation:
                    "Add 'Strict-Transport-Security: max-age=31536000; includeSubDomains' header."
                        .to_string(),
                location: None,
                evidence: None,
            });
        }
    }

    // 2. 檢查 CSP（Content-Security-Policy）- 密碼學相關部分
    if let Some(csp) = headers_lower.get("content-security-policy") {
        // 檢查 upgrade-insecure-requests
        if !csp.contains("upgrade-insecure-requests") && is_https {
            findings.push(Finding {
                issue_type: "WEAK_CSP".to_string(),
                severity: "LOW".to_string(),
                title: "CSP Missing upgrade-insecure-requests".to_string(),
                description: "CSP does not upgrade insecure requests to HTTPS.".to_string(),
                recommendation: "Add 'upgrade-insecure-requests' directive to CSP.".to_string(),
                location: None,
                evidence: Some(csp.clone()),
            });
        }

        // 檢查 unsafe-inline 和 unsafe-eval（可能繞過 CSP）
        if csp.contains("'unsafe-inline'") || csp.contains("'unsafe-eval'") {
            findings.push(Finding {
                issue_type: "WEAK_CSP".to_string(),
                severity: "MEDIUM".to_string(),
                title: "CSP Contains 'unsafe-inline' or 'unsafe-eval'".to_string(),
                description: "CSP weakened by unsafe directives, may allow code injection."
                    .to_string(),
                recommendation: "Remove 'unsafe-inline' and 'unsafe-eval'. Use nonces or hashes."
                    .to_string(),
                location: None,
                evidence: Some(csp.clone()),
            });
        }
    } else if is_https {
        findings.push(Finding {
            issue_type: "MISSING_CSP".to_string(),
            severity: "LOW".to_string(),
            title: "Missing Content-Security-Policy Header".to_string(),
            description: "No CSP header to prevent code injection attacks.".to_string(),
            recommendation: "Implement Content-Security-Policy header.".to_string(),
            location: None,
            evidence: None,
        });
    }

    // 3. 檢查 X-Content-Type-Options
    if !headers_lower.contains_key("x-content-type-options") {
        findings.push(Finding {
            issue_type: "MISSING_NOSNIFF".to_string(),
            severity: "LOW".to_string(),
            title: "Missing X-Content-Type-Options Header".to_string(),
            description:
                "Browser may MIME-sniff responses, potentially executing malicious content."
                    .to_string(),
            recommendation: "Add 'X-Content-Type-Options: nosniff' header.".to_string(),
            location: None,
            evidence: None,
        });
    }

    findings
}

fn extract_max_age(hsts: &str) -> Option<u64> {
    for part in hsts.split(';') {
        let part = part.trim();
        if let Some(stripped) = part.strip_prefix("max-age=") {
            if let Ok(age) = stripped.trim().parse::<u64>() {
                return Some(age);
            }
        }
    }
    None
}
