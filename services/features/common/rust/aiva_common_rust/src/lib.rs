// AIVA Common Rust 模組
// 日期: 2025-01-07
// 目的: 提供 AIVA 平台的 Rust 共用功能

pub mod metrics;

// 重新導出主要類型以便使用
pub use metrics::{
    MetricsCollector, 
    SeverityLevel, 
    initialize_metrics, 
    cleanup_metrics,
    record_task_received,
    record_task_completed,
    record_task_failed,
    record_vulnerability_found,
    update_system_metrics,
};