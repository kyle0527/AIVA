// Function-SAST: Static Application Security Testing
// 靜態應用程式安全測試模組

mod analyzers;
mod models;
mod parsers;
mod rules;
mod worker;

use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日誌
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "function_sast_rust=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🔍 Starting AIVA Function-SAST Worker (Rust)");

    // 啟動 Worker
    let worker = worker::SastWorker::new().await?;
    worker.run().await?;

    Ok(())
}
