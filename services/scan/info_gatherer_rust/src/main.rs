// AIVA Info Gatherer - Rust Implementation
// 日期: 2025-10-13
// 功能: 高性能敏感資訊掃描器

use futures_lite::stream::StreamExt;
use lapin::{
    options::*, types::FieldTable, Connection, ConnectionProperties,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, error};
use tracing_subscriber;

mod scanner;
mod secret_detector;
mod git_history_scanner;
mod verifier;
mod schemas;

use scanner::SensitiveInfoScanner;
use secret_detector::SecretDetector;
use git_history_scanner::GitHistoryScanner;
use verifier::Verifier;
use schemas::generated::{FindingPayload, Vulnerability, VulnerabilityType, Severity, Confidence, Target, FindingEvidence};

const RABBITMQ_URL: &str = "amqp://aiva:dev_password@localhost:5672";
const TASK_QUEUE: &str = "task.scan.sensitive_info";
const RESULT_QUEUE: &str = "results.scan.sensitive_info";

#[derive(Debug, Deserialize)]
struct ScanTask {
    task_id: String,
    content: String,
    source_url: String,
}

// 移除自定義 Finding 結構，使用標準 FindingPayload
// 保持向後兼容的輔助函數
fn create_finding_payload(
    task_id: &str,
    scan_id: &str,
    info_type: &str,
    value: &str,
    location: &str,
    severity: Option<&str>,
    confidence_level: Confidence,
) -> FindingPayload {
    let finding_id = format!("finding_{}_{}", task_id, uuid::Uuid::new_v4().to_string());
    
    let vulnerability_type = match info_type {
        "secret" | "git_secret" => VulnerabilityType::WeakAuthentication,
        "sensitive_info" => VulnerabilityType::InformationLeak,
        _ => VulnerabilityType::InformationLeak,
    };
    
    let severity_enum = match severity.unwrap_or("medium") {
        "CRITICAL" => Severity::Critical,
        "HIGH" => Severity::High,
        "MEDIUM" => Severity::Medium,
        "LOW" => Severity::Low,
        _ => Severity::Medium,
    };
    
    let vulnerability = Vulnerability {
        name: vulnerability_type,
        cwe: Some("CWE-200".to_string()), // Information Exposure
        cve: None,
        severity: severity_enum,
        confidence: confidence_level,
        description: Some(format!("Sensitive information detected: {}", info_type)),
        cvss_score: None,
        cvss_vector: None,
        owasp_category: None,
    };
    
    let target = Target {
        url: serde_json::Value::String(location.to_string()),
        parameter: None,
        method: None,
        headers: std::collections::HashMap::new(),
        params: std::collections::HashMap::new(),
        body: None,
    };
    
    let evidence = FindingEvidence {
        payload: Some(value.to_string()),
        response_time_delta: None,
        db_version: None,
        request: None,
        response: None,
        proof: Some(format!("Found {} at {}", info_type, location)),
    };
    
    let mut finding = FindingPayload::new(
        finding_id,
        task_id.to_string(),
        scan_id.to_string(),
        "confirmed".to_string(),
        vulnerability,
        target,
    );
    
    finding.evidence = Some(evidence);
    finding.strategy = Some("sensitive_info_detection".to_string());
    finding
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 初始化日誌
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 AIVA Sensitive Info Gatherer 啟動中...");

    // 連接 RabbitMQ
    info!("📡 連接 RabbitMQ...");
    let conn = Connection::connect(
        RABBITMQ_URL,
        ConnectionProperties::default(),
    )
    .await?;

    let channel = conn.create_channel().await?;

    // 聲明任務隊列
    channel
        .queue_declare(
            TASK_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await?;

    // 設置 prefetch
    channel
        .basic_qos(1, BasicQosOptions::default())
        .await?;

    info!("✅ 初始化完成,開始監聽任務...");

    // 初始化掃描器
    let scanner = Arc::new(SensitiveInfoScanner::new());
    
    // 初始化驗證器
    let verifier = Arc::new(Verifier::new());

    // 消費訊息
    let mut consumer = channel
        .basic_consume(
            TASK_QUEUE,
            "rust_info_gatherer",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await?;

    while let Some(delivery) = consumer.next().await {
        let delivery = delivery?;
        let scanner = Arc::clone(&scanner);
        let verifier = Arc::clone(&verifier);
        let channel = channel.clone();

        tokio::spawn(async move {
            match process_task(&delivery.data, scanner, verifier, &channel).await {
                Ok(_) => {
                    delivery
                        .ack(BasicAckOptions::default())
                        .await
                        .expect("Failed to ack");
                }
                Err(e) => {
                    error!("處理任務失敗: {:?}", e);
                    delivery
                        .nack(BasicNackOptions {
                            requeue: true,
                            ..Default::default()
                        })
                        .await
                        .expect("Failed to nack");
                }
            }
        });
    }

    Ok(())
}

async fn process_task(
    data: &[u8],
    scanner: Arc<SensitiveInfoScanner>,
    verifier: Arc<Verifier>,
    channel: &lapin::Channel,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let task: ScanTask = serde_json::from_slice(data)?;
    info!("📥 收到敏感資訊掃描任務: {}", task.task_id);

    let scan_id = format!("scan_{}", uuid::Uuid::new_v4().to_string());
    let mut all_findings = Vec::<FindingPayload>::new();

    // 1. 原有的敏感資訊掃描
    let sensitive_findings = scanner.scan(&task.content, &task.source_url);
    info!("  📊 敏感資訊掃描: 發現 {} 個結果", sensitive_findings.len());
    
    for finding in sensitive_findings {
        let confidence_level = if finding.confidence >= 0.8 {
            Confidence::Certain
        } else if finding.confidence >= 0.6 {
            Confidence::Firm
        } else {
            Confidence::Possible
        };
        
        let finding_payload = create_finding_payload(
            &task.task_id,
            &scan_id,
            "sensitive_info",
            &finding.value,
            &finding.location,
            None,
            confidence_level,
        );
        
        all_findings.push(finding_payload);
    }

    // 2. 密鑰檢測掃描
    let secret_detector = SecretDetector::new();
    let secret_findings = secret_detector.scan_content(&task.content, &task.source_url);
    info!("  🔐 密鑰檢測掃描: 發現 {} 個密鑰", secret_findings.len());
    
    // 驗證檢測到的密鑰
    for finding in secret_findings {
        // 僅對高優先級密鑰進行驗證
        let should_verify = matches!(
            finding.severity.as_str(),
            "CRITICAL" | "HIGH"
        );
        
        let (verified, verification_message, verification_metadata) = if should_verify {
            info!("  🔍 驗證密鑰: {} ...", finding.rule_name);
            let result = verifier.verify(&finding.rule_name, &finding.matched_text).await;
            
            use verifier::VerificationStatus;
            let verified = match result.status {
                VerificationStatus::Valid => Some(true),
                VerificationStatus::Invalid => Some(false),
                _ => None,
            };
            
            (verified, Some(result.message), Some(result.metadata))
        } else {
            (None, None, None)
        };
        
        let mut finding_payload = create_finding_payload(
            &task.task_id,
            &scan_id,
            "secret",
            &finding.matched_text,
            &format!("{}:{}", finding.file_path, finding.line_number),
            Some(&finding.severity),
            Confidence::Firm, // 密鑰匹配高信心度
        );
        
        // 添加密鑰檢測專用的元數據
        finding_payload.metadata.insert(
            "entropy".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(finding.entropy.unwrap_or(0.0))
                    .unwrap_or(serde_json::Number::from(0))
            )
        );
        finding_payload.metadata.insert(
            "rule_name".to_string(),
            serde_json::Value::String(finding.rule_name)
        );
        
        if let Some(verified) = verified {
            finding_payload.metadata.insert(
                "verified".to_string(),
                serde_json::Value::Bool(verified)
            );
        }
        
        if let Some(msg) = verification_message {
            finding_payload.metadata.insert(
                "verification_message".to_string(),
                serde_json::Value::String(msg)
            );
        }
        
        all_findings.push(finding_payload);
    }

    // 3. Git 歷史掃描（僅當 source_url 看起來像 Git 倉庫時）
    if task.source_url.contains(".git") || 
       task.source_url.starts_with("http") || 
       task.source_url.starts_with("git@") {
        
        info!("  📜 檢測到 Git 倉庫，啟動歷史掃描...");
        let git_scanner = GitHistoryScanner::new(1000); // 掃描最近 1000 個提交
        
        // 注意：這裡假設 source_url 是本地路徑或已克隆的倉庫
        // 實際使用時可能需要先克隆遠程倉庫
        if let Ok(git_findings) = git_scanner.scan_repository(std::path::Path::new(&task.source_url)) {
            info!("  🔍 Git 歷史掃描: 發現 {} 個密鑰", git_findings.len());
            
            for finding in git_findings {
                let location = format!("commit:{} {}:{}", 
                    &finding.commit_hash[..8], 
                    finding.finding.file_path, 
                    finding.finding.line_number
                );
                
                let mut finding_payload = create_finding_payload(
                    &task.task_id,
                    &scan_id,
                    "git_secret",
                    &finding.finding.matched_text,
                    &location,
                    Some(&finding.finding.severity),
                    Confidence::Possible, // Git 歷史匹配稍低信心度
                );
                
                // 添加 Git 專用元數據
                finding_payload.metadata.insert(
                    "commit_hash".to_string(),
                    serde_json::Value::String(finding.commit_hash)
                );
                finding_payload.metadata.insert(
                    "entropy".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(finding.finding.entropy.unwrap_or(0.0))
                            .unwrap_or(serde_json::Number::from(0))
                    )
                );
                finding_payload.metadata.insert(
                    "rule_name".to_string(),
                    serde_json::Value::String(finding.finding.rule_name)
                );
                finding_payload.metadata.insert(
                    "git_historical".to_string(),
                    serde_json::Value::Bool(true)
                );
                
                all_findings.push(finding_payload);
            }
        } else {
            info!("  ⚠️  Git 歷史掃描跳過（可能不是有效的 Git 倉庫）");
        }
    }

    info!(
        "✅ 掃描完成: {} (總計發現 {} 個結果)",
        task.task_id,
        all_findings.len()
    );

    // 發送結果
    for finding in all_findings {
        let payload = serde_json::to_vec(&finding)?;

        // 聲明結果隊列
        channel
            .queue_declare(
                RESULT_QUEUE,
                QueueDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await?;

        channel
            .basic_publish(
                "",
                RESULT_QUEUE,
                BasicPublishOptions::default(),
                &payload,
                lapin::BasicProperties::default(),
            )
            .await?;
    }

    Ok(())
}
