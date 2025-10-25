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
use scanner::SensitiveInfoScanner;
use secret_detector::SecretDetector;
use git_history_scanner::GitHistoryScanner;

const RABBITMQ_URL: &str = "amqp://aiva:dev_password@localhost:5672";
const TASK_QUEUE: &str = "task.scan.sensitive_info";
const RESULT_QUEUE: &str = "results.scan.sensitive_info";

#[derive(Debug, Deserialize)]
struct ScanTask {
    task_id: String,
    content: String,
    source_url: String,
}

#[derive(Debug, Serialize)]
struct Finding {
    task_id: String,
    info_type: String,
    value: String,
    confidence: f32,
    location: String,
    severity: Option<String>,      // 新增：密鑰嚴重性
    entropy: Option<f64>,          // 新增：熵值
    rule_name: Option<String>,     // 新增：觸發的規則名稱
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
        let channel = channel.clone();

        tokio::spawn(async move {
            match process_task(&delivery.data, scanner, &channel).await {
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
    channel: &lapin::Channel,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let task: ScanTask = serde_json::from_slice(data)?;
    info!("📥 收到敏感資訊掃描任務: {}", task.task_id);

    let mut all_findings = Vec::new();

    // 1. 原有的敏感資訊掃描
    let sensitive_findings = scanner.scan(&task.content, &task.source_url);
    info!("  📊 敏感資訊掃描: 發現 {} 個結果", sensitive_findings.len());
    
    for finding in sensitive_findings {
        all_findings.push(Finding {
            task_id: task.task_id.clone(),
            info_type: finding.info_type,
            value: finding.value,
            confidence: finding.confidence,
            location: finding.location,
            severity: None,
            entropy: None,
            rule_name: None,
        });
    }

    // 2. 密鑰檢測掃描
    let secret_detector = SecretDetector::new();
    let secret_findings = secret_detector.scan_content(&task.content, &task.source_url);
    info!("  🔐 密鑰檢測掃描: 發現 {} 個密鑰", secret_findings.len());
    
    for finding in secret_findings {
        all_findings.push(Finding {
            task_id: task.task_id.clone(),
            info_type: "secret".to_string(),
            value: finding.matched_text.clone(),
            confidence: 0.9, // 密鑰匹配高信心度
            location: format!("{}:{}", finding.file_path, finding.line_number),
            severity: Some(finding.severity),
            entropy: finding.entropy,
            rule_name: Some(finding.rule_name),
        });
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
                all_findings.push(Finding {
                    task_id: task.task_id.clone(),
                    info_type: "git_secret".to_string(),
                    value: finding.finding.matched_text.clone(),
                    confidence: 0.85, // Git 歷史匹配稍低信心度
                    location: format!("commit:{} {}:{}", 
                        &finding.commit_hash[..8], 
                        finding.finding.file_path, 
                        finding.finding.line_number
                    ),
                    severity: Some(finding.finding.severity),
                    entropy: finding.finding.entropy,
                    rule_name: Some(finding.finding.rule_name),
                });
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
