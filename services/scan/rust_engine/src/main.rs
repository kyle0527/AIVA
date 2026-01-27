// AIVA Rust Scanner - CLI Implementation
// 日期: 2025-11-19
// 功能: 高性能多目標掃描引擎，支持兩種模式 (fast/deep)

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

mod scanner;
mod schemas;
mod secret_detector;
mod verifier;

// Phase0 核心模組
mod endpoint_discovery;
mod js_analyzer;
mod attack_surface;

// OWASP 漏洞檢測模組 - Enhanced Version
mod smuggling_detector_v2;  // OWASP A05 - HTTP Request Smuggling (Enhanced)
mod auth_brute_v2;           // OWASP A07 - Credential Spraying (Enhanced)

use endpoint_discovery::{EndpointDiscoverer, DiscoveredEndpoint};
use js_analyzer::JsAnalyzer;
use attack_surface::AttackSurfaceAssessor;
use scanner::{SensitiveInfoScanner, ScanMode};
use smuggling_detector_v2::SmugglingDetector;
use auth_brute_v2::{AuthBruteForcer, AuthProtocol, RateLimitStrategy};

/// 掃描模式
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScanModeArg {
    /// 快速偵察模式 (Phase0) - 用於獲取基礎資料給四種語言引擎搭配使用
    Fast,
    /// 深度分析模式 (Phase1) - 由 AI 決定如何搭配，協調模組執行
    Deep,
}

impl From<ScanModeArg> for ScanMode {
    fn from(mode: ScanModeArg) -> Self {
        match mode {
            ScanModeArg::Fast => ScanMode::FastDiscovery,
            ScanModeArg::Deep => ScanMode::DeepAnalysis,
        }
    }
}

/// CLI 參數
#[derive(Parser, Debug)]
#[command(name = "rust_scanner")]
#[command(about = "AIVA Rust 高性能掃描引擎", long_about = None)]
struct Args {
    /// 掃描子命令
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// 執行掃描
    Scan {
        /// 目標 URL (支持多個目標)
        #[arg(long, required = true, num_args = 1..)]
        url: Vec<String>,
        
        /// 掃描模式
        #[arg(long, value_enum, default_value = "fast")]
        mode: ScanModeArg,
        
        /// 輸出格式
        #[arg(long, default_value = "json")]
        format: String,
        
        /// 超時時間 (秒)
        #[arg(long, default_value = "30")]
        timeout: u64,
        
        /// 掃描深度
        #[arg(long, default_value = "1")]
        depth: usize,
    },
    /// 顯示版本
    Version,
}

/// 掃描結果 (統一輸出格式)
#[derive(Debug, Serialize, Deserialize)]
struct ScanResult {
    /// 掃描模式
    mode: String,
    /// 掃描的目標列表
    targets: Vec<TargetResult>,
    /// 總執行時間 (毫秒)
    total_execution_time_ms: f64,
    /// 掃描摘要
    summary: ScanSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct TargetResult {
    /// 目標 URL
    url: String,
    /// 掃描成功與否
    success: bool,
    /// 錯誤信息 (如果有)
    error: Option<String>,
    /// 發現的端點
    endpoints: Vec<DiscoveredEndpoint>,
    /// JS 分析發現
    js_findings: Vec<String>,  // 簡化為字符串，避免類型耦合
    /// 敏感信息
    sensitive_info: Vec<SensitiveInfo>,
    /// 攻擊面評估
    attack_surface: Option<String>,  // 簡化為字符串
    /// 技術棧指紋
    technologies: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SensitiveInfo {
    info_type: String,
    value: String,
    location: String,
    confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanSummary {
    /// 總目標數
    total_targets: usize,
    /// 成功掃描數
    successful_scans: usize,
    /// 失敗掃描數
    failed_scans: usize,
    /// 發現的端點總數
    total_endpoints: usize,
    /// 發現的敏感信息總數
    total_sensitive_info: usize,
    /// 平均風險評分
    average_risk_score: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日誌
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    match args.command {
        Commands::Scan {
            url,
            mode,
            format,
            timeout,
            depth,
        } => {
            info!("🚀 AIVA Rust Scanner 啟動");
            info!("📋 掃描模式: {:?}", mode);
            info!("🎯 目標數量: {}", url.len());
            
            let result = scan_targets(
                url,
                mode.into(),
                timeout,
                depth,
            )
            .await?;
            
            // 輸出結果
            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
                _ => {
                    eprintln!("不支持的輸出格式: {}", format);
                    std::process::exit(1);
                }
            }
        }
        Commands::Version => {
            println!("AIVA Rust Scanner v{}", env!("CARGO_PKG_VERSION"));
        }
    }

    Ok(())
}

/// 掃描多個目標 (並發執行)
async fn scan_targets(
    urls: Vec<String>,
    mode: ScanMode,
    timeout_secs: u64,
    depth: usize,
) -> Result<ScanResult, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let timeout_duration = Duration::from_secs(timeout_secs);
    
    info!("開始掃描 {} 個目標 (模式: {:?})", urls.len(), mode);
    
    // 並發掃描所有目標
    let mut tasks = Vec::new();
    
    for url in urls.iter() {
        let url_clone = url.clone();
        let task = tokio::spawn(async move {
            match timeout(
                timeout_duration,
                scan_single_target(url_clone.clone(), mode, depth)
            ).await {
                Ok(Ok(result)) => result,
                Ok(Err(e)) => {
                    error!("掃描失敗 {}: {}", url_clone, e);
                    TargetResult {
                        url: url_clone,
                        success: false,
                        error: Some(e.to_string()),
                        endpoints: Vec::new(),
                        js_findings: Vec::new(),
                        sensitive_info: Vec::new(),
                        attack_surface: None,
                        technologies: Vec::new(),
                    }
                }
                Err(_) => {
                    error!("掃描超時: {}", url_clone);
                    TargetResult {
                        url: url_clone,
                        success: false,
                        error: Some("Scan timeout".to_string()),
                        endpoints: Vec::new(),
                        js_findings: Vec::new(),
                        sensitive_info: Vec::new(),
                        attack_surface: None,
                        technologies: Vec::new(),
                    }
                }
            }
        });
        tasks.push(task);
    }
    
    // 等待所有任務完成
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await?);
    }
    
    // 生成摘要
    let summary = generate_summary(&results);
    
    let elapsed = start_time.elapsed();
    info!("掃描完成，耗時 {:.2}秒", elapsed.as_secs_f64());
    
    Ok(ScanResult {
        mode: format!("{:?}", mode),
        targets: results,
        total_execution_time_ms: elapsed.as_millis() as f64,
        summary,
    })
}

/// 掃描單個目標
async fn scan_single_target(
    url: String,
    mode: ScanMode,
    _depth: usize,
) -> Result<TargetResult, Box<dyn std::error::Error>> {
    info!("掃描目標: {}", url);
    
    let mut endpoints = Vec::new();
    let mut js_findings_list = Vec::new();
    let mut sensitive_info = Vec::new();
    let mut technologies = Vec::new();
    let mut attack_surface = None;
    
    // 根據模式執行不同的掃描
    match mode {
        ScanMode::FastDiscovery => {
            // Phase0: 快速偵察 - 為四種語言引擎提供基礎數據
            info!("[Phase0] 快速偵察模式 - 掃描 {}", url);
            
            // 1. 實際端點發現 - 使用 EndpointDiscoverer 真正掃描靶場
            let discoverer = EndpointDiscoverer::new();
            endpoints = discoverer.discover(&url).await;
            info!("  ✅ 發現 {} 個端點", endpoints.len());
            
            // 2. 下載並分析真實 JS 文件
            let js_analyzer = JsAnalyzer::new();
            let js_urls = vec![
                format!("{}/main.js", url),
                format!("{}/runtime.js", url),
                format!("{}/vendor.js", url),
            ];
            
            for js_url in js_urls {
                match fetch_page_content(&js_url).await {
                    Ok(js_content) => {
                        let findings = js_analyzer.analyze(&js_content, &js_url);
                        let count = findings.len();
                        for finding in findings {
                            js_findings_list.push(format!("{:?}", finding));
                        }
                        if count > 0 {
                            info!("    - {}: {} findings", js_url.split('/').last().unwrap_or("unknown"), count);
                        }
                    }
                    Err(e) => {
                        eprintln!("⚠️  無法下載 {}: {}", js_url, e);
                    }
                }
            }
            info!("  ✅ JS分析完成: {} 個發現", js_findings_list.len());
            
            // 3. 獲取首頁並掃描敏感信息
            if let Ok(page_content) = fetch_page_content(&url).await {
                let scanner = SensitiveInfoScanner::with_mode(ScanMode::FastDiscovery);
                let findings = scanner.scan(&page_content, &url);
                sensitive_info = findings
                    .into_iter()
                    .map(|f| SensitiveInfo {
                        info_type: "sensitive_data".to_string(),
                        value: f.value,
                        location: f.location,
                        confidence: f.confidence,
                    })
                    .collect();
                info!("  ✅ 敏感信息: {} 個", sensitive_info.len());
                
                // 4. 技術棧檢測
                technologies = detect_technologies_from_content(&page_content);
                info!("  ✅ 檢測到 {} 個技術", technologies.len());
            }
        }
        
        ScanMode::DeepAnalysis => {
            // Phase1: 深度分析 - 由 AI 決定，協調模組執行
            info!("[Phase1] 深度分析模式 - 掃描 {}", url);
            
            // 1. 完整端點發現 - 使用 EndpointDiscoverer 深度掃描
            let discoverer = EndpointDiscoverer::new();
            endpoints = discoverer.discover(&url).await;
            info!("  ✅ 發現 {} 個端點", endpoints.len());
            
            // 2. 下載並深度分析所有 JS 文件
            let js_analyzer = JsAnalyzer::new();
            let mut all_js_findings = Vec::new();
            let js_urls = vec![
                format!("{}/main.js", url),
                format!("{}/runtime.js", url),
                format!("{}/polyfills.js", url),
                format!("{}/vendor.js", url),
            ];
            
            for js_url in js_urls {
                match fetch_page_content(&js_url).await {
                    Ok(js_content) => {
                        let findings = js_analyzer.analyze(&js_content, &js_url);
                        let count = findings.len();
                        for finding in &findings {
                            js_findings_list.push(format!("{:?}", finding));
                        }
                        all_js_findings.extend(findings);
                        if count > 0 {
                            info!("    - {}: {} findings", js_url.split('/').last().unwrap_or("unknown"), count);
                        }
                    }
                    Err(e) => {
                        eprintln!("⚠️  無法下載 {}: {}", js_url, e);
                    }
                }
            }
            info!("  ✅ JS深度分析: {} 個發現", js_findings_list.len());
            
            // 3. 獲取首頁並深度掃描
            if let Ok(page_content) = fetch_page_content(&url).await {
                let scanner = SensitiveInfoScanner::with_mode(ScanMode::DeepAnalysis);
                let findings = scanner.scan(&page_content, &url);
                sensitive_info = findings
                    .into_iter()
                    .map(|f| SensitiveInfo {
                        info_type: "sensitive_data".to_string(),
                        value: f.value,
                        location: f.location,
                        confidence: f.confidence,
                    })
                    .collect();
                info!("  ✅ 敏感信息: {} 個", sensitive_info.len());
                
                technologies = detect_technologies_from_content(&page_content);
                info!("  ✅ 技術棧: {} 個", technologies.len());
            }
            
            // 4. 攻擊面評估 - 使用實際的 JS findings
            let assessor = AttackSurfaceAssessor::new();
            let assessment_report = assessor.assess(&endpoints, &all_js_findings);
            info!("  ✅ 攻擊面評估: {} 個高風險目標", 
                  assessment_report.high_risk_endpoints.len());
            attack_surface = Some(format!("{:?}", assessment_report));
            
            info!(
                "✅ 深度分析完成: {} 端點, {} JS發現, {} 敏感信息",
                endpoints.len(),
                js_findings_list.len(),
                sensitive_info.len()
            );
        }
        
        ScanMode::FocusedVerification => {
            // 聚焦驗證模式 (保留，暫不實現)
            warn!("聚焦驗證模式暫未實現");
        }
    }
    
    // A4: 去除重複的 JS findings
    js_findings_list = deduplicate_findings(js_findings_list);
    
    Ok(TargetResult {
        url,
        success: true,
        error: None,
        endpoints,
        js_findings: js_findings_list,
        sensitive_info,
        attack_surface,
        technologies,
    })
}

/// 獲取頁面內容
async fn fetch_page_content(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    
    let response = client.get(url).send().await?;
    let content = response.text().await?;
    
    Ok(content)
}

/// 從內容檢測技術棧
fn detect_technologies_from_content(content: &str) -> Vec<String> {
    let mut technologies = Vec::new();
    
    // 檢測常見框架和庫
    if content.contains("WordPress") || content.contains("wp-content") {
        technologies.push("WordPress".to_string());
    }
    if content.contains("React") || content.contains("react") {
        technologies.push("React".to_string());
    }
    if content.contains("Angular") || content.contains("ng-") {
        technologies.push("Angular".to_string());
    }
    if content.contains("Vue") || content.contains("vue") {
        technologies.push("Vue.js".to_string());
    }
    if content.contains("jQuery") || content.contains("jquery") {
        technologies.push("jQuery".to_string());
    }
    if content.contains("Express") {
        technologies.push("Express".to_string());
    }
    if content.contains("Next.js") || content.contains("_next") {
        technologies.push("Next.js".to_string());
    }
    if content.contains("Bootstrap") || content.contains("bootstrap") {
        technologies.push("Bootstrap".to_string());
    }
    
    technologies
}

/// 去除 JS Findings 中的重複項目
/// 根據完整字符串去重(簡單有效)
fn deduplicate_findings(findings: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    findings
        .into_iter()
        .filter(|f| seen.insert(f.clone()))
        .collect()
}

/// 生成掃描摘要
fn generate_summary(results: &[TargetResult]) -> ScanSummary {
    let total_targets = results.len();
    let successful_scans = results.iter().filter(|r| r.success).count();
    let failed_scans = total_targets - successful_scans;
    
    let total_endpoints: usize = results.iter().map(|r| r.endpoints.len()).sum();
    let total_sensitive_info: usize = results.iter().map(|r| r.sensitive_info.len()).sum();
    
    // TODO: 實現實際的風險評分解析邏輯
    // attack_surface 現在是 String 型別,需要解析或重構為結構化型別
    let average_risk_score = 0.0f32;
    
    ScanSummary {
        total_targets,
        successful_scans,
        failed_scans,
        total_endpoints,
        total_sensitive_info,
        average_risk_score,
    }
}
