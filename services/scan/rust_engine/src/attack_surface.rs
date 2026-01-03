// Attack Surface Assessment - Phase0 核心功能
// HackerOne 實戰: 風險評分、優先級排序、Phase1 引擎建議

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::endpoint_discovery::{DiscoveredEndpoint, RiskLevel};
use crate::js_analyzer::JsFinding;

/// 攻擊面評估結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSurfaceReport {
    pub total_endpoints: usize,
    pub total_js_findings: usize,
    pub risk_distribution: HashMap<String, usize>,
    pub high_risk_endpoints: Vec<PriorityTarget>,
    pub recommended_engines: Vec<String>,
    pub summary: String,
}

/// 優先級目標
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityTarget {
    pub endpoint: String,
    pub risk_score: u32,
    pub risk_level: String,
    pub reasons: Vec<String>,
    pub recommended_tests: Vec<String>,
}

/// 攻擊面評估器
pub struct AttackSurfaceAssessor {
    risk_calculator: RiskCalculator,
    engine_recommender: EngineRecommender,
}

impl AttackSurfaceAssessor {
    pub fn new() -> Self {
        Self {
            risk_calculator: RiskCalculator::new(),
            engine_recommender: EngineRecommender::new(),
        }
    }

    /// 評估攻擊面 (主入口)
    pub fn assess(
        &self,
        endpoints: &[DiscoveredEndpoint],
        js_findings: &[JsFinding],
    ) -> AttackSurfaceReport {
        info!("🎯 開始攻擊面評估");
        info!("  📊 端點: {} 個", endpoints.len());
        info!("  📜 JS Findings: {} 個", js_findings.len());

        // 1. 計算風險分布
        let risk_distribution = self.calculate_risk_distribution(endpoints);

        // 2. 識別高風險目標
        let high_risk_endpoints = self.identify_high_risk_targets(endpoints);

        // 3. Phase1 引擎建議
        let recommended_engines = self.engine_recommender.recommend(endpoints, js_findings);

        // 4. 生成摘要
        let summary = self.generate_summary(endpoints, js_findings, &high_risk_endpoints);

        info!("✅ 攻擊面評估完成");
        info!("  ⚠️  高風險端點: {} 個", high_risk_endpoints.len());
        info!("  🔧 建議引擎: {:?}", recommended_engines);

        AttackSurfaceReport {
            total_endpoints: endpoints.len(),
            total_js_findings: js_findings.len(),
            risk_distribution,
            high_risk_endpoints,
            recommended_engines,
            summary,
        }
    }

    /// 計算風險分布
    fn calculate_risk_distribution(&self, endpoints: &[DiscoveredEndpoint]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        distribution.insert("critical".to_string(), 0);
        distribution.insert("high".to_string(), 0);
        distribution.insert("medium".to_string(), 0);
        distribution.insert("low".to_string(), 0);
        distribution.insert("info".to_string(), 0);

        for endpoint in endpoints {
            let level = match endpoint.risk_level {
                RiskLevel::Critical => "critical",
                RiskLevel::High => "high",
                RiskLevel::Medium => "medium",
                RiskLevel::Low => "low",
                RiskLevel::Info => "info",
            };
            *distribution.get_mut(level).unwrap() += 1;
        }

        distribution
    }

    /// 識別高風險目標
    fn identify_high_risk_targets(&self, endpoints: &[DiscoveredEndpoint]) -> Vec<PriorityTarget> {
        let mut targets: Vec<_> = endpoints
            .iter()
            .filter_map(|endpoint| {
                let risk_score = self.risk_calculator.calculate(endpoint);
                if risk_score.value >= 15 {
                    // 高風險閾值
                    Some(PriorityTarget {
                        endpoint: endpoint.path.clone(),
                        risk_score: risk_score.value,
                        risk_level: risk_score.level.clone(),
                        reasons: risk_score.reasons,
                        recommended_tests: self.recommend_tests(endpoint),
                    })
                } else {
                    None
                }
            })
            .collect();

        // 按風險分數排序
        targets.sort_by(|a, b| b.risk_score.cmp(&a.risk_score));

        // 只返回前 20 個
        targets.truncate(20);
        targets
    }

    /// 推薦測試類型
    fn recommend_tests(&self, endpoint: &DiscoveredEndpoint) -> Vec<String> {
        let mut tests = Vec::new();
        let path_lower = endpoint.path.to_lowercase();

        if path_lower.contains("/admin") || path_lower.contains("/management") {
            tests.push("權限繞過 (Authorization Bypass)".to_string());
            tests.push("IDOR (Insecure Direct Object Reference)".to_string());
        }

        if path_lower.contains("/api") {
            tests.push("SQL 注入 (SQL Injection)".to_string());
            tests.push("NoSQL 注入 (NoSQL Injection)".to_string());
            tests.push("大量賦值 (Mass Assignment)".to_string());
        }

        if path_lower.contains("/upload") {
            tests.push("文件上傳漏洞 (File Upload)".to_string());
            tests.push("路徑遍歷 (Path Traversal)".to_string());
        }

        if path_lower.contains("/download") {
            tests.push("路徑遍歷 (Path Traversal)".to_string());
            tests.push("IDOR".to_string());
        }

        if path_lower.contains("/graphql") {
            tests.push("GraphQL Introspection".to_string());
            tests.push("GraphQL 深度查詢攻擊".to_string());
        }

        if path_lower.contains("/auth") || path_lower.contains("/login") {
            tests.push("暴力破解 (Brute Force)".to_string());
            tests.push("憑證填充 (Credential Stuffing)".to_string());
        }

        if tests.is_empty() {
            tests.push("XSS (Cross-Site Scripting)".to_string());
            tests.push("CSRF (Cross-Site Request Forgery)".to_string());
        }

        tests
    }

    /// 生成摘要
    fn generate_summary(
        &self,
        endpoints: &[DiscoveredEndpoint],
        js_findings: &[JsFinding],
        high_risk: &[PriorityTarget],
    ) -> String {
        format!(
            "發現 {} 個端點、{} 個 JS Finding。{} 個高風險目標需優先測試。",
            endpoints.len(),
            js_findings.len(),
            high_risk.len()
        )
    }
}

/// 風險計算器
struct RiskCalculator;

#[derive(Debug, Clone)]
struct RiskScore {
    value: u32,
    level: String,
    reasons: Vec<String>,
}

impl RiskCalculator {
    fn new() -> Self {
        Self
    }

    /// 計算端點風險分數
    fn calculate(&self, endpoint: &DiscoveredEndpoint) -> RiskScore {
        let mut score = 0u32;
        let mut reasons = Vec::new();

        let path_lower = endpoint.path.to_lowercase();

        // 基礎風險 (來自 endpoint_discovery)
        let base_risk = match endpoint.risk_level {
            RiskLevel::Critical => 25,
            RiskLevel::High => 20,
            RiskLevel::Medium => 10,
            RiskLevel::Low => 5,
            RiskLevel::Info => 0,
        };
        score += base_risk;

        // 管理功能 (+25)
        if path_lower.contains("/admin") || path_lower.contains("/management") {
            score += 25;
            reasons.push("管理界面".to_string());
        }

        // 文件操作 (+20)
        if path_lower.contains("/upload") {
            score += 20;
            reasons.push("文件上傳".to_string());
        }

        // 認證相關 (+20)
        if path_lower.contains("/auth") || path_lower.contains("/login") {
            score += 20;
            reasons.push("認證端點".to_string());
        }

        // API 端點 (+15)
        if path_lower.contains("/api") {
            score += 15;
            reasons.push("API 端點".to_string());
        }

        // 用戶輸入 (+10)
        if path_lower.contains("?") || path_lower.contains("=") {
            score += 10;
            reasons.push("包含參數".to_string());
        }

        // GraphQL (+15)
        if path_lower.contains("/graphql") {
            score += 15;
            reasons.push("GraphQL 端點".to_string());
        }

        // 下載功能 (+15)
        if path_lower.contains("/download") {
            score += 15;
            reasons.push("文件下載".to_string());
        }

        // 開發/測試端點 (+10)
        if path_lower.contains("/debug") || path_lower.contains("/test") {
            score += 10;
            reasons.push("開發端點".to_string());
        }

        let level = if score >= 40 {
            "CRITICAL"
        } else if score >= 25 {
            "HIGH"
        } else if score >= 15 {
            "MEDIUM"
        } else {
            "LOW"
        };

        debug!("  🎯 {} => {} ({})", endpoint.path, score, level);

        RiskScore {
            value: score,
            level: level.to_string(),
            reasons,
        }
    }
}

/// Phase1 引擎推薦器
struct EngineRecommender;

impl EngineRecommender {
    fn new() -> Self {
        Self
    }

    /// 推薦 Phase1 引擎
    fn recommend(
        &self,
        endpoints: &[DiscoveredEndpoint],
        js_findings: &[JsFinding],
    ) -> Vec<String> {
        let mut engines = Vec::new();

        // Python: 大量靜態端點 (爬蟲)
        if endpoints.len() > 10 {
            engines.push("python".to_string());
            debug!("  🐍 Python: {} 個端點需爬取", endpoints.len());
        }

        // TypeScript: 檢測到 JS/SPA 特徵
        if !js_findings.is_empty() || self.has_spa_indicators(endpoints) {
            engines.push("typescript".to_string());
            debug!("  📘 TypeScript: 檢測到 {} 個 JS Finding", js_findings.len());
        }

        // Go: SSRF/Cloud/CSPM 特徵
        if self.has_cloud_indicators(endpoints) {
            engines.push("go".to_string());
            debug!("  🔵 Go: 檢測到雲端/SSRF 特徵");
        }

        // 如果沒有特殊需求，預設使用 Python
        if engines.is_empty() {
            engines.push("python".to_string());
            debug!("  🐍 Python: 預設選擇");
        }

        engines
    }

    /// 檢測 SPA 特徵
    fn has_spa_indicators(&self, endpoints: &[DiscoveredEndpoint]) -> bool {
        endpoints.iter().any(|e| {
            let path = e.path.to_lowercase();
            path.ends_with(".js")
                || path.contains("/app.")
                || path.contains("/main.")
                || path.contains("/bundle.")
        })
    }

    /// 檢測雲端/SSRF 特徵
    fn has_cloud_indicators(&self, endpoints: &[DiscoveredEndpoint]) -> bool {
        endpoints.iter().any(|e| {
            let path = e.path.to_lowercase();
            path.contains("metadata")
                || path.contains("ssrf")
                || path.contains("cloud")
                || path.contains("s3")
                || path.contains("bucket")
        })
    }
}
