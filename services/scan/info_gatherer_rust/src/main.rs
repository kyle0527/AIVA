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

    // 執行掃描
    let findings = scanner.scan(&task.content, &task.source_url);

    info!(
        "✅ 掃描完成: {} (發現 {} 個敏感資訊)",
        task.task_id,
        findings.len()
    );

    // 發送結果
    for finding in findings {
        let finding_with_id = Finding {
            task_id: task.task_id.clone(),
            info_type: finding.info_type,
            value: finding.value,
            confidence: finding.confidence,
            location: finding.location,
        };

        let payload = serde_json::to_vec(&finding_with_id)?;

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
