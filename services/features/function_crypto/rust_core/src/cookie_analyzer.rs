// Cookie 安全性分析器
// 分析 HTTP 響應中的 Cookie 安全屬性

use crate::Finding;
use serde_json::Value;

pub fn analyze_cookies(cookies_json: &str, url: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    let is_https = url.starts_with("https://");

    // 解析 Cookie JSON
    let cookies: Vec<Value> = match serde_json::from_str(cookies_json) {
        Ok(c) => c,
        Err(_) => return findings,
    };

    for cookie in cookies {
        let cookie_str = cookie.as_str().unwrap_or("");
        let cookie_name = extract_cookie_name(cookie_str);

        // 檢查 Secure 標誌
        if is_https && !cookie_str.to_lowercase().contains("secure") {
            findings.push(Finding {
                issue_type: "MISSING_SECURE_FLAG".to_string(),
                severity: "MEDIUM".to_string(),
                title: format!("Cookie '{}' Missing Secure Flag", cookie_name),
                description: "Cookie transmitted over HTTPS without Secure flag can be intercepted over HTTP.".to_string(),
                recommendation: "Add 'Secure' flag to all cookies on HTTPS sites.".to_string(),
                location: Some(cookie_name.clone()),
                evidence: Some(cookie_str.to_string()),
            });
        }

        // 檢查 HttpOnly 標誌
        if is_sensitive_cookie(&cookie_name) && !cookie_str.to_lowercase().contains("httponly") {
            findings.push(Finding {
                issue_type: "MISSING_HTTPONLY_FLAG".to_string(),
                severity: "MEDIUM".to_string(),
                title: format!("Sensitive Cookie '{}' Missing HttpOnly Flag", cookie_name),
                description: "Cookie accessible to JavaScript, vulnerable to XSS attacks."
                    .to_string(),
                recommendation: "Add 'HttpOnly' flag to prevent JavaScript access.".to_string(),
                location: Some(cookie_name.clone()),
                evidence: Some(cookie_str.to_string()),
            });
        }

        // 檢查 SameSite 標誌
        if !cookie_str.to_lowercase().contains("samesite") {
            findings.push(Finding {
                issue_type: "MISSING_SAMESITE_FLAG".to_string(),
                severity: "LOW".to_string(),
                title: format!("Cookie '{}' Missing SameSite Flag", cookie_name),
                description: "Cookie without SameSite protection is vulnerable to CSRF attacks."
                    .to_string(),
                recommendation: "Add 'SameSite=Strict' or 'SameSite=Lax' flag.".to_string(),
                location: Some(cookie_name),
                evidence: Some(cookie_str.to_string()),
            });
        }
    }

    findings
}

fn extract_cookie_name(cookie_str: &str) -> String {
    cookie_str
        .split(';')
        .next()
        .unwrap_or("")
        .split('=')
        .next()
        .unwrap_or("unknown")
        .trim()
        .to_string()
}

fn is_sensitive_cookie(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("session")
        || lower.contains("auth")
        || lower.contains("token")
        || lower.contains("jwt")
        || lower.contains("csrf")
}
