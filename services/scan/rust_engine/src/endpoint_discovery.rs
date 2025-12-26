// Endpoint Discovery Module - Phase0 核心功能
// HackerOne 實戰: 發現 API 端點、管理界面、敏感路徑

use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;
use tracing::{debug, info};

/// 端點發現結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEndpoint {
    pub path: String,
    pub method: String,
    pub status_code: u16,
    pub discovered_by: DiscoveryMethod,
    pub risk_level: RiskLevel,
    pub response_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiscoveryMethod {
    Dictionary,    // 字典爆破
    JsAnalysis,    // JS 文件分析
    Sitemap,       // Sitemap.xml
    Robots,        // Robots.txt
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// 端點發現器
pub struct EndpointDiscoverer {
    client: Client,
    common_paths: Vec<&'static str>,
    js_endpoint_regex: Regex,
    timeout: Duration,
}

impl EndpointDiscoverer {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .danger_accept_invalid_certs(true)
            .build()
            .expect("Failed to build HTTP client");

        // 常見路徑 - 基於 SecLists 和 HackerOne 實戰經驗
        let common_paths = vec![
            // API 端點
            "/api",
            "/api/v1",
            "/api/v2",
            "/api/users",
            "/api/config",
            "/graphql",
            "/graphiql",
            
            // 管理界面
            "/admin",
            "/admin/login",
            "/administrator",
            "/management",
            "/console",
            "/backend",
            
            // 認證相關
            "/auth",
            "/login",
            "/signin",
            "/logout",
            "/register",
            "/reset-password",
            
            // 文檔與配置
            "/swagger.json",
            "/swagger-ui",
            "/openapi.json",
            "/api-docs",
            "/.well-known/security.txt",
            "/security.txt",
            
            // 開發相關
            "/debug",
            "/test",
            "/dev",
            "/_debug",
            
            // 文件操作
            "/upload",
            "/download",
            "/files",
            
            // 敏感目錄 (低優先級，但保留接口)
            "/.env",
            "/.git/config",
            "/config.json",
            "/package.json",
            
            // 其他
            "/sitemap.xml",
            "/robots.txt",
            "/crossdomain.xml",
        ];

        // JS 中的 API 端點提取
        let js_endpoint_regex = Regex::new(
            r#"['"`](/(?:api|admin|auth|upload|download|graphql)[^'"`\s]*?)['"`]"#
        ).expect("Invalid regex");

        Self {
            client,
            common_paths,
            js_endpoint_regex,
            timeout: Duration::from_secs(10),
        }
    }

    /// 發現端點 (主入口)
    pub async fn discover(&self, base_url: &str) -> Vec<DiscoveredEndpoint> {
        info!("🔍 開始端點發現: {}", base_url);
        
        let mut all_endpoints = Vec::new();
        let mut seen_paths = HashSet::new();

        // 方法 1: 字典掃描
        let dict_endpoints = self.scan_common_paths(base_url).await;
        for endpoint in dict_endpoints {
            if seen_paths.insert(endpoint.path.clone()) {
                all_endpoints.push(endpoint);
            }
        }

        // 方法 2: Robots.txt 分析
        if let Some(robots_endpoints) = self.analyze_robots(base_url).await {
            for endpoint in robots_endpoints {
                if seen_paths.insert(endpoint.path.clone()) {
                    all_endpoints.push(endpoint);
                }
            }
        }

        // 方法 3: Sitemap.xml 分析
        if let Some(sitemap_endpoints) = self.analyze_sitemap(base_url).await {
            for endpoint in sitemap_endpoints {
                if seen_paths.insert(endpoint.path.clone()) {
                    all_endpoints.push(endpoint);
                }
            }
        }

        info!("✅ 端點發現完成: 共 {} 個", all_endpoints.len());
        all_endpoints
    }

    /// 字典掃描常見路徑
    async fn scan_common_paths(&self, base_url: &str) -> Vec<DiscoveredEndpoint> {
        debug!("📖 字典掃描開始 ({} 個路徑)", self.common_paths.len());
        
        let mut endpoints = Vec::new();
        
        for path in &self.common_paths {
            let url = format!("{}{}", base_url.trim_end_matches('/'), path);
            
            match self.client.get(&url).send().await {
                Ok(response) => {
                    let status = response.status().as_u16();
                    let size = response.content_length().unwrap_or(0) as usize;
                    
                    // 只記錄有效響應 (排除 404)
                    if status != 404 {
                        let risk = Self::calculate_path_risk(path);
                        
                        debug!("  ✅ {} [{}] ({} bytes)", path, status, size);
                        
                        endpoints.push(DiscoveredEndpoint {
                            path: path.to_string(),
                            method: "GET".to_string(),
                            status_code: status,
                            discovered_by: DiscoveryMethod::Dictionary,
                            risk_level: risk,
                            response_size: size,
                        });
                    }
                }
                Err(e) => {
                    debug!("  ❌ {} - Error: {:?}", path, e);
                }
            }
        }
        
        debug!("📖 字典掃描完成: {} 個有效端點", endpoints.len());
        endpoints
    }

    /// 分析 robots.txt
    async fn analyze_robots(&self, base_url: &str) -> Option<Vec<DiscoveredEndpoint>> {
        let robots_url = format!("{}/robots.txt", base_url.trim_end_matches('/'));
        
        match self.client.get(&robots_url).send().await {
            Ok(response) if response.status().is_success() => {
                let text = response.text().await.ok()?;
                let mut endpoints = Vec::new();
                
                for line in text.lines() {
                    if let Some(path) = Self::extract_robots_path(line) {
                        // 驗證路徑是否存在
                        let url = format!("{}{}", base_url.trim_end_matches('/'), path);
                        if let Ok(resp) = self.client.head(&url).send().await {
                            if resp.status().as_u16() != 404 {
                                endpoints.push(DiscoveredEndpoint {
                                    path: path.to_string(),
                                    method: "GET".to_string(),
                                    status_code: resp.status().as_u16(),
                                    discovered_by: DiscoveryMethod::Robots,
                                    risk_level: Self::calculate_path_risk(&path),
                                    response_size: 0,
                                });
                            }
                        }
                    }
                }
                
                if !endpoints.is_empty() {
                    info!("🤖 Robots.txt: 發現 {} 個路徑", endpoints.len());
                    Some(endpoints)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// 分析 sitemap.xml
    async fn analyze_sitemap(&self, base_url: &str) -> Option<Vec<DiscoveredEndpoint>> {
        let sitemap_url = format!("{}/sitemap.xml", base_url.trim_end_matches('/'));
        
        match self.client.get(&sitemap_url).send().await {
            Ok(response) if response.status().is_success() => {
                let text = response.text().await.ok()?;
                let urls = Self::extract_sitemap_urls(&text);
                
                if !urls.is_empty() {
                    info!("🗺️  Sitemap.xml: 發現 {} 個 URL", urls.len());
                    
                    // 轉換為端點 (只取路徑)
                    let endpoints = urls.iter()
                        .filter_map(|url| {
                            url.strip_prefix(base_url).map(|path| {
                                DiscoveredEndpoint {
                                    path: path.to_string(),
                                    method: "GET".to_string(),
                                    status_code: 200,
                                    discovered_by: DiscoveryMethod::Sitemap,
                                    risk_level: Self::calculate_path_risk(path),
                                    response_size: 0,
                                }
                            })
                        })
                        .collect();
                    
                    Some(endpoints)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// 從 JS 文件提取端點 (供 JsAnalyzer 調用)
    pub fn extract_endpoints_from_js(&self, js_content: &str) -> Vec<String> {
        let mut endpoints = HashSet::new();
        
        for cap in self.js_endpoint_regex.captures_iter(js_content) {
            if let Some(path) = cap.get(1) {
                endpoints.insert(path.as_str().to_string());
            }
        }
        
        endpoints.into_iter().collect()
    }

    /// 計算路徑風險等級
    fn calculate_path_risk(path: &str) -> RiskLevel {
        let path_lower = path.to_lowercase();
        
        // 關鍵字匹配
        if path_lower.contains("/admin") || path_lower.contains("/management") {
            return RiskLevel::Critical;
        }
        
        if path_lower.contains("/upload") || path_lower.contains("/download") {
            return RiskLevel::High;
        }
        
        if path_lower.contains("/api") || path_lower.contains("/auth") {
            return RiskLevel::High;
        }
        
        if path_lower.contains("/graphql") || path_lower.contains("/swagger") {
            return RiskLevel::Medium;
        }
        
        if path_lower.contains("/debug") || path_lower.contains("/test") {
            return RiskLevel::Medium;
        }
        
        if path_lower.ends_with(".json") || path_lower.ends_with(".xml") {
            return RiskLevel::Low;
        }
        
        RiskLevel::Info
    }

    /// 從 robots.txt 行提取路徑
    fn extract_robots_path(line: &str) -> Option<String> {
        let line = line.trim();
        
        if line.starts_with("Disallow:") {
            line.strip_prefix("Disallow:")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.starts_with('/'))
        } else if line.starts_with("Allow:") {
            line.strip_prefix("Allow:")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.starts_with('/'))
        } else {
            None
        }
    }

    /// 從 sitemap.xml 提取 URL
    fn extract_sitemap_urls(xml: &str) -> Vec<String> {
        let url_regex = Regex::new(r"<loc>(.*?)</loc>").unwrap();
        
        url_regex.captures_iter(xml)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_path_risk() {
        assert_eq!(
            EndpointDiscoverer::calculate_path_risk("/admin"),
            RiskLevel::Critical
        );
        assert_eq!(
            EndpointDiscoverer::calculate_path_risk("/api/users"),
            RiskLevel::High
        );
        assert_eq!(
            EndpointDiscoverer::calculate_path_risk("/config.json"),
            RiskLevel::Low
        );
    }

    #[test]
    fn test_extract_robots_path() {
        assert_eq!(
            EndpointDiscoverer::extract_robots_path("Disallow: /admin"),
            Some("/admin".to_string())
        );
        assert_eq!(
            EndpointDiscoverer::extract_robots_path("Allow: /api"),
            Some("/api".to_string())
        );
        assert_eq!(
            EndpointDiscoverer::extract_robots_path("# Comment"),
            None
        );
    }

    #[test]
    fn test_js_endpoint_extraction() {
        let discoverer = EndpointDiscoverer::new();
        let js_code = r#"
            fetch('/api/users')
            axios.get("/admin/config")
            $.ajax({url: '/graphql'})
        "#;
        
        let endpoints = discoverer.extract_endpoints_from_js(js_code);
        assert!(endpoints.contains(&"/api/users".to_string()));
        assert!(endpoints.contains(&"/admin/config".to_string()));
        assert!(endpoints.contains(&"/graphql".to_string()));
    }
}
