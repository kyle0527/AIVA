// AIVA Common Rust 模組
// 日期: 2025-10-28
// 目的: 提供 AIVA 平台的 Rust 共用功能和標準化 Schema

pub mod metrics;
pub mod schemas;

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

// 重新導出 Schema 類型
pub use schemas::{
    // 基礎類型
    MessageHeader,
    Target,
    Vulnerability,
    
    // 訊息通訊
    AivaMessage,
    
    // 任務管理
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskContext,
    FunctionTaskTestConfig,
    
    // 發現結果
    FindingPayload,
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    
    // 枚舉類型 - 從generated模組導出
    Severity,
    Confidence,
    FindingStatus,
};