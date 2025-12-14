// AIVA Crypto Scanner v2.0 - 網路層密碼學配置掃描器
// 專注於黑盒測試實際能檢測的內容
// 日期: 2025-12-12

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

mod cookie_analyzer;
mod header_analyzer;
mod js_crypto_analyzer;
mod tls_analyzer;

#[derive(Parser)]
#[command(name = "crypto-scanner")]
#[command(version = "2.0.0")]
#[command(about = "Network-based cryptography configuration scanner", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 分析 JavaScript 中的密碼學問題（從下載的 JS 文件）
    ScanJs {
        /// JavaScript 文件內容
        #[arg(long)]
        content: String,

        /// JavaScript 文件 URL（可選，用於報告）
        #[arg(long)]
        url: Option<String>,
    },

    /// 分析 TLS/SSL 配置（從網路握手）
    AnalyzeTls {
        /// 目標域名
        #[arg(long)]
        target: String,

        /// 目標端口（默認 443）
        #[arg(long, default_value = "443")]
        port: u16,
    },

    /// 分析 Cookie 安全性（從 HTTP 響應）
    AnalyzeCookies {
        /// Set-Cookie 標頭的 JSON 數組
        #[arg(long)]
        cookies_json: String,

        /// 目標 URL（用於判斷 HTTPS）
        #[arg(long)]
        url: String,
    },

    /// 分析 HTTP 安全標頭（密碼學相關）
    AnalyzeHeaders {
        /// HTTP 響應標頭的 JSON 對象
        #[arg(long)]
        headers_json: String,

        /// 目標 URL
        #[arg(long)]
        url: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanResult {
    scan_type: String,
    target: String,
    findings: Vec<Finding>,
    summary: Summary,
}

#[derive(Debug, Serialize, Deserialize)]
struct Finding {
    issue_type: String,
    severity: String,
    title: String,
    description: String,
    recommendation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Summary {
    total_findings: usize,
    critical: usize,
    high: usize,
    medium: usize,
    low: usize,
    info: usize,
}

impl Summary {
    fn from_findings(findings: &[Finding]) -> Self {
        let mut summary = Summary {
            total_findings: findings.len(),
            critical: 0,
            high: 0,
            medium: 0,
            low: 0,
            info: 0,
        };

        for finding in findings {
            match finding.severity.as_str() {
                "CRITICAL" => summary.critical += 1,
                "HIGH" => summary.high += 1,
                "MEDIUM" => summary.medium += 1,
                "LOW" => summary.low += 1,
                "INFO" => summary.info += 1,
                _ => {}
            }
        }

        summary
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::ScanJs { content, url } => {
            let findings = js_crypto_analyzer::scan_javascript(&content);
            ScanResult {
                scan_type: "javascript_crypto".to_string(),
                target: url.unwrap_or_else(|| "unknown".to_string()),
                summary: Summary::from_findings(&findings),
                findings,
            }
        }

        Commands::AnalyzeTls { target, port } => {
            let findings = tls_analyzer::analyze_tls(&target, port).await;
            ScanResult {
                scan_type: "tls_configuration".to_string(),
                target: format!("{}:{}", target, port),
                summary: Summary::from_findings(&findings),
                findings,
            }
        }

        Commands::AnalyzeCookies { cookies_json, url } => {
            let findings = cookie_analyzer::analyze_cookies(&cookies_json, &url);
            ScanResult {
                scan_type: "cookie_security".to_string(),
                target: url,
                summary: Summary::from_findings(&findings),
                findings,
            }
        }

        Commands::AnalyzeHeaders { headers_json, url } => {
            let findings = header_analyzer::analyze_headers(&headers_json, &url);
            ScanResult {
                scan_type: "security_headers".to_string(),
                target: url,
                summary: Summary::from_findings(&findings),
                findings,
            }
        }
    };

    // 輸出 JSON 結果
    match serde_json::to_string_pretty(&result) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing result: {}", e),
    }
}
