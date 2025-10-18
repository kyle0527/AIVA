// SAST Worker - RabbitMQ 消費者

use crate::analyzers::StaticAnalyzer;
use crate::models::{FindingPayload, FunctionTaskPayload};
use anyhow::{Context, Result};
use lapin::{
    options::*, types::FieldTable, Channel, Connection, ConnectionProperties,
};
use std::env;
use uuid::Uuid;

const TASK_QUEUE: &str = "tasks.function.sast";
const FINDING_QUEUE: &str = "findings";

pub struct SastWorker {
    connection: Connection,
    analyzer: StaticAnalyzer,
}

impl SastWorker {
    pub async fn new() -> Result<Self> {
        let rabbitmq_url = env::var("RABBITMQ_URL")
            .unwrap_or_else(|_| "amqp://guest:guest@localhost:5672/%2f".to_string());
        
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
                    delivery
                        .nack(BasicNackOptions::default())
                        .await
                        .context("Failed to nack message")?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn handle_task(&self, channel: &Channel, data: &[u8]) -> Result<()> {
        let task: FunctionTaskPayload = serde_json::from_slice(data)
            .context("Failed to deserialize task")?;
        
        tracing::info!("🔍 Processing SAST task: {}", task.task_id);
        
        let target_path = task.target.file_path
            .or(task.target.repository.clone())
            .context("No target path provided")?;
        
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
            let finding = issue.to_finding(&task.task_id, &scan_id);
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
