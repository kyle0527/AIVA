// SAST Worker - RabbitMQ 消費者

use crate::analyzers::StaticAnalyzer;
use crate::schemas::generated::FindingPayload;
use crate::schemas::generated::ScanTaskPayload;
use anyhow::{Context, Result};
use lapin::{
    options::*, types::FieldTable, Channel, Connection, ConnectionProperties,
    message::Delivery,
};
use futures_lite::stream::StreamExt;
use std::env;
use uuid::Uuid;

const TASK_QUEUE: &str = "tasks.function.sast";
const FINDING_QUEUE: &str = "findings.new";

/// 檢查消息是否應該重試
/// 實施統一的重試策略，防止 poison pill 消息無限循環
fn should_retry_message(delivery: &Delivery, _error: &dyn std::error::Error) -> bool {
    const MAX_RETRY_ATTEMPTS: i32 = 3;
    
    // 檢查消息頭部中的重試次數
    let retry_count = delivery
        .properties
        .headers()
        .as_ref()
        .and_then(|headers| headers.inner().get("x-aiva-retry-count"))
        .and_then(|val| {
            if let lapin::types::AMQPValue::LongInt(count) = val {
                Some(*count)
            } else {
                None
            }
        })
        .unwrap_or(0);
    
    if retry_count >= MAX_RETRY_ATTEMPTS {
        tracing::error!(
            "消息已達到最大重試次數 {}, 發送到死信隊列",
            MAX_RETRY_ATTEMPTS
        );
        false
    } else {
        tracing::warn!(
            "消息重試 {}/{}: 錯誤處理中",
            retry_count + 1,
            MAX_RETRY_ATTEMPTS
        );
        true
    }
}

pub struct SastWorker {
    connection: Connection,
    #[allow(dead_code)] // Analyzer reserved for future static analysis features
    analyzer: StaticAnalyzer,
}

impl SastWorker {
    pub async fn new() -> Result<Self> {
        // 遵循 12-factor app 原則和 aiva_common 配置標準
        let rabbitmq_url = env::var("RABBITMQ_URL")
            .or_else(|_| {
                // 如果沒有 URL，則組合各個部件
                let host = env::var("RABBITMQ_HOST").unwrap_or_else(|_| "localhost".to_string());
                let port = env::var("RABBITMQ_PORT").unwrap_or_else(|_| "5672".to_string());
                let user = env::var("RABBITMQ_USER")?;
                let password = env::var("RABBITMQ_PASSWORD")?;
                let vhost = env::var("RABBITMQ_VHOST").unwrap_or_else(|_| "/".to_string());
                
                Ok::<String, anyhow::Error>(format!("amqp://{}:{}@{}:{}{}", user, password, host, port, vhost))
            })
            .context("RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set")?;
        
        let connection = Connection::connect(&rabbitmq_url, ConnectionProperties::default())
            .await
            .context("Failed to connect to RabbitMQ")?;
        
        Ok(Self {
            connection,
            analyzer: StaticAnalyzer::new(),
        })
    }
    
    pub async fn run(&self) -> Result<()> {
        let channel = self.connection.create_channel().await?;
        
        // 宣告任務佇列
        channel
            .queue_declare(
                TASK_QUEUE,
                QueueDeclareOptions::default(),
                FieldTable::default(),
            )
            .await?;
        
        tracing::info!("📬 Waiting for SAST tasks on queue: {}", TASK_QUEUE);
        
        // 消費訊息
        let mut consumer = channel
            .basic_consume(
                TASK_QUEUE,
                "sast_consumer",
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await?;
        
        while let Some(delivery) = consumer.next().await {
            let delivery = delivery.context("Failed to get delivery")?;
            
            match self.handle_task(&channel, &delivery.data).await {
                Ok(_) => {
                    delivery
                        .ack(BasicAckOptions::default())
                        .await
                        .context("Failed to ack message")?;
                }
                Err(e) => {
                    tracing::error!("Task failed: {}", e);
                    
                    // 實施重試邏輯，防止 poison pill 消息無限循環
                    let should_requeue = should_retry_message(&delivery, e.as_ref());
                    
                    if should_requeue {
                        tracing::warn!("重新入隊消息進行重試");
                        delivery
                            .nack(BasicNackOptions {
                                requeue: true,
                                ..Default::default()
                            })
                            .await
                            .context("Failed to nack for retry")?;
                    } else {
                        tracing::error!("達到最大重試次數，發送到死信隊列");
                        delivery
                            .nack(BasicNackOptions {
                                requeue: false,  // 不重新入隊，發送到死信隊列
                                ..Default::default()
                            })
                            .await
                            .context("Failed to nack to dead letter")?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn handle_task(&self, channel: &Channel, data: &[u8]) -> Result<()> {
        let task: ScanTaskPayload = serde_json::from_slice(data)
            .context("Failed to deserialize task")?;
        
        let task_id = &task.task_id;
        tracing::info!("🔍 Processing SAST task: {}", task_id);
        
        let target_url = &task.target.url;
        tracing::info!("🎯 Target: {}", target_url);
        
        // 從 URL 中提取路徑（假設是 file:// URL）
        let target_path = target_url.strip_prefix("file://").unwrap_or(target_url);
        
        // 執行靜態分析
        let mut analyzer = StaticAnalyzer::new();
        let issues = if std::path::Path::new(&target_path).is_dir() {
            analyzer.analyze_directory(&target_path).await?
        } else {
            analyzer.analyze_file(&target_path).await?
        };
        
        tracing::info!("Found {} SAST issues", issues.len());
        
        // 發布 Findings
        let scan_id = format!("scan_sast_{}", Uuid::new_v4());
        for issue in issues {
            let finding = issue.to_finding(task_id, &scan_id);
            self.publish_finding(channel, finding).await?;
        }
        
        Ok(())
    }
    
    async fn publish_finding(&self, channel: &Channel, finding: FindingPayload) -> Result<()> {
        let payload = serde_json::to_vec(&finding)?;
        
        channel
            .basic_publish(
                "",
                FINDING_QUEUE,
                BasicPublishOptions::default(),
                &payload,
                lapin::BasicProperties::default(),
            )
            .await?
            .await?;
        
        Ok(())
    }
}
