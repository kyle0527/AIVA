// JavaScript Analyzer - Phase0 核心功能
// HackerOne 實戰: 從 JS 文件提取 API 端點、檢測 API Key 洩漏

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, info};

/// JS 分析發現
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsFinding {
    pub file_path: String,
    pub finding_type: JsFindingType,
    pub value: String,
    pub severity: String,
    pub line_number: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsFindingType {
    ApiEndpoint,        // API 端點
    ApiKeyLeak,         // API Key 洩漏
    InternalDomain,     // 內部域名
    SensitiveComment,   // 敏感註釋
    ConfigData,         // 配置數據
}

/// JS 分析器
pub struct JsAnalyzer {
    // API 端點模式
    api_endpoint_regex: Regex,
    fetch_regex: Regex,
    axios_regex: Regex,
    
    // API Key 洩漏模式 (高危！)
    stripe_key_regex: Regex,
    aws_key_regex: Regex,
    google_api_regex: Regex,
    generic_api_key_regex: Regex,
    
    // 內部域名/IP
    internal_domain_regex: Regex,
    private_ip_regex: Regex,
    
    // 敏感註釋
    sensitive_comment_regex: Regex,
}

impl JsAnalyzer {
    pub fn new() -> Self {
        Self {
            // API 端點提取
            api_endpoint_regex: Regex::new(
                r#"['"`](/(?:api|admin|auth|graphql|upload|download)[^'"`\s]{0,100})['"`]"#
            ).unwrap(),
            
            fetch_regex: Regex::new(
                r#"fetch\s*\(\s*['"`]([^'"`]+)['"`]"#
            ).unwrap(),
            
            axios_regex: Regex::new(
                r#"axios\.\w+\s*\(\s*['"`]([^'"`]+)['"`]"#
            ).unwrap(),
            
            // API Key 洩漏 (CRITICAL!)
            stripe_key_regex: Regex::new(
                r#"(?:pk|sk)_(?:live|test)_[0-9a-zA-Z]{24,}"#
            ).unwrap(),
            
            aws_key_regex: Regex::new(
                r#"AKIA[0-9A-Z]{16}"#
            ).unwrap(),
            
            google_api_regex: Regex::new(
                r#"AIza[0-9A-Za-z\-_]{35}"#
            ).unwrap(),
            
            generic_api_key_regex: Regex::new(
                r#"(?i)(?:api[_-]?key|apikey|api_secret)['"\s:=]+['"]?([0-9a-zA-Z\-_]{20,})['"]?"#
            ).unwrap(),
            
            // 內部域名/IP
            internal_domain_regex: Regex::new(
                r#"(?:https?://)?([a-zA-Z0-9\-]+\.(?:internal|local|corp|dev|staging)[a-zA-Z0-9\-\.]*)"#
            ).unwrap(),
            
            private_ip_regex: Regex::new(
                r#"\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}\b"#
            ).unwrap(),
            
            // 敏感註釋
            sensitive_comment_regex: Regex::new(
                r#"//.*?(?i)(password|secret|token|key|credential|admin).*?[:=]\s*['"]?([^'"\n]+)"#
            ).unwrap(),
        }
    }

    /// 分析 JS 文件 (主入口)
    pub fn analyze(&self, js_content: &str, file_path: &str) -> Vec<JsFinding> {
        info!("📜 分析 JS 文件: {}", file_path);
        
        let mut findings = Vec::new();
        
        // 1. API 端點提取
        findings.extend(self.extract_api_endpoints(js_content, file_path));
        
        // 2. API Key 洩漏檢測 (高危！)
        findings.extend(self.detect_api_key_leaks(js_content, file_path));
        
        // 3. 內部域名/IP 檢測
        findings.extend(self.detect_internal_domains(js_content, file_path));
        
        // 4. 敏感註釋檢測
        findings.extend(self.detect_sensitive_comments(js_content, file_path));
        
        info!("  ✅ 發現 {} 個 JS finding", findings.len());
        findings
    }

    /// 提取 API 端點
    fn extract_api_endpoints(&self, content: &str, file_path: &str) -> Vec<JsFinding> {
        let mut endpoints = HashSet::new();
        let mut findings = Vec::new();
        
        // 方法 1: 通用 API 路徑模式
        for cap in self.api_endpoint_regex.captures_iter(content) {
            if let Some(path) = cap.get(1) {
                endpoints.insert(path.as_str().to_string());
            }
        }
        
        // 方法 2: fetch() 調用
        for cap in self.fetch_regex.captures_iter(content) {
            if let Some(url) = cap.get(1) {
                let url_str = url.as_str();
                if url_str.starts_with('/') {
                    endpoints.insert(url_str.to_string());
                }
            }
        }
        
        // 方法 3: axios 調用
        for cap in self.axios_regex.captures_iter(content) {
            if let Some(url) = cap.get(1) {
                let url_str = url.as_str();
                if url_str.starts_with('/') {
                    endpoints.insert(url_str.to_string());
                }
            }
        }
        
        // 轉換為 Finding
        for endpoint in endpoints {
            let line_num = Self::find_line_number(content, &endpoint);
            
            findings.push(JsFinding {
                file_path: file_path.to_string(),
                finding_type: JsFindingType::ApiEndpoint,
                value: endpoint,
                severity: "INFO".to_string(),
                line_number: line_num,
                confidence: 0.90,
            });
        }
        
        debug!("  📍 API 端點: {} 個", findings.len());
        findings
    }

    /// 檢測 API Key 洩漏 (CRITICAL!)
    fn detect_api_key_leaks(&self, content: &str, file_path: &str) -> Vec<JsFinding> {
        let mut findings = Vec::new();
        
        // Stripe Keys (pk_live_xxx, sk_live_xxx)
        for mat in self.stripe_key_regex.find_iter(content) {
            let key = mat.as_str();
            let line_num = Self::find_line_number(content, key);
            let is_live = key.contains("_live_");
            
            findings.push(JsFinding {
                file_path: file_path.to_string(),
                finding_type: JsFindingType::ApiKeyLeak,
                value: Self::mask_key(key),
                severity: if is_live { "CRITICAL" } else { "HIGH" }.to_string(),
                line_number: line_num,
                confidence: 0.98,
            });
        }
        
        // AWS Keys (AKIA...)
        for mat in self.aws_key_regex.find_iter(content) {
            let key = mat.as_str();
            let line_num = Self::find_line_number(content, key);
            
            findings.push(JsFinding {
                file_path: file_path.to_string(),
                finding_type: JsFindingType::ApiKeyLeak,
                value: Self::mask_key(key),
                severity: "CRITICAL".to_string(),
                line_number: line_num,
                confidence: 0.99,
            });
        }
        
        // Google API Keys (AIza...)
        for mat in self.google_api_regex.find_iter(content) {
            let key = mat.as_str();
            let line_num = Self::find_line_number(content, key);
            
            findings.push(JsFinding {
                file_path: file_path.to_string(),
                finding_type: JsFindingType::ApiKeyLeak,
                value: Self::mask_key(key),
                severity: "HIGH".to_string(),
                line_number: line_num,
                confidence: 0.95,
            });
        }
        
        // Generic API Keys
        for cap in self.generic_api_key_regex.captures_iter(content) {
            if let Some(key) = cap.get(1) {
                let key_str = key.as_str();
                let line_num = Self::find_line_number(content, key_str);
                
                findings.push(JsFinding {
                    file_path: file_path.to_string(),
                    finding_type: JsFindingType::ApiKeyLeak,
                    value: Self::mask_key(key_str),
                    severity: "MEDIUM".to_string(),
                    line_number: line_num,
                    confidence: 0.75,
                });
            }
        }
        
        if !findings.is_empty() {
            debug!("  🔑 API Key 洩漏: {} 個 (CRITICAL!)", findings.len());
        }
        
        findings
    }

    /// 檢測內部域名/IP
    fn detect_internal_domains(&self, content: &str, file_path: &str) -> Vec<JsFinding> {
        let mut findings = Vec::new();
        
        // 內部域名
        for cap in self.internal_domain_regex.captures_iter(content) {
            if let Some(domain) = cap.get(1) {
                let domain_str = domain.as_str();
                let line_num = Self::find_line_number(content, domain_str);
                
                findings.push(JsFinding {
                    file_path: file_path.to_string(),
                    finding_type: JsFindingType::InternalDomain,
                    value: domain_str.to_string(),
                    severity: "MEDIUM".to_string(),
                    line_number: line_num,
                    confidence: 0.85,
                });
            }
        }
        
        // 私有 IP
        for mat in self.private_ip_regex.find_iter(content) {
            let ip = mat.as_str();
            let line_num = Self::find_line_number(content, ip);
            
            findings.push(JsFinding {
                file_path: file_path.to_string(),
                finding_type: JsFindingType::InternalDomain,
                value: ip.to_string(),
                severity: "LOW".to_string(),
                line_number: line_num,
                confidence: 0.80,
            });
        }
        
        if !findings.is_empty() {
            debug!("  🌐 內部域名/IP: {} 個", findings.len());
        }
        
        findings
    }

    /// 檢測敏感註釋
    fn detect_sensitive_comments(&self, content: &str, file_path: &str) -> Vec<JsFinding> {
        let mut findings = Vec::new();
        
        for cap in self.sensitive_comment_regex.captures_iter(content) {
            if let Some(value) = cap.get(2) {
                let value_str = value.as_str();
                let line_num = Self::find_line_number(content, value_str);
                
                findings.push(JsFinding {
                    file_path: file_path.to_string(),
                    finding_type: JsFindingType::SensitiveComment,
                    value: Self::mask_key(value_str),
                    severity: "LOW".to_string(),
                    line_number: line_num,
                    confidence: 0.70,
                });
            }
        }
        
        if !findings.is_empty() {
            debug!("  💬 敏感註釋: {} 個", findings.len());
        }
        
        findings
    }

    /// 查找行號
    fn find_line_number(content: &str, target: &str) -> usize {
        content[..content.find(target).unwrap_or(0)]
            .lines()
            .count() + 1
    }

    /// 遮罩敏感 Key
    fn mask_key(key: &str) -> String {
        if key.len() <= 8 {
            return "*".repeat(key.len());
        }
        
        let prefix = &key[..4];
        let suffix = &key[key.len()-4..];
        format!("{}****{}", prefix, suffix)
    }
}
