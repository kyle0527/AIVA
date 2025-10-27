// AIVA 統一統計收集模組 - Rust 實現
// 日期: 2025-01-07
// 目的: 提供跨語言一致的性能監控和統計收集功能

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::fs::OpenOptions;
use std::io::Write;
use log::{debug, error, info, warn};

/// 指標類型枚舉
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Duration,
}

/// 嚴重程度級別
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeverityLevel {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl SeverityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SeverityLevel::Critical => "critical",
            SeverityLevel::High => "high",
            SeverityLevel::Medium => "medium",
            SeverityLevel::Low => "low",
            SeverityLevel::Info => "info",
        }
    }
}

/// 指標數據結構
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    pub name: String,
    pub value: f64,
    pub metric_type: MetricType,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Worker 統計指標集合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub worker_id: String,
    
    // 任務處理統計
    pub tasks_received: u64,
    pub tasks_processed: u64,
    pub tasks_failed: u64,
    pub tasks_retried: u64,
    pub total_processing_time: f64,  // 秒
    pub total_queue_wait_time: f64,  // 秒
    
    // 檢測結果統計
    pub findings_created: u64,
    pub vulnerabilities_found: u64,
    pub false_positives: u64,
    
    // 嚴重程度分佈
    pub severity_distribution: HashMap<String, u64>,
    
    // 系統資源 (瞬時值)
    pub current_memory_usage: f64,   // 位元組
    pub current_cpu_usage: f64,      // 百分比
    pub active_connections: u64,
}

impl WorkerMetrics {
    pub fn new(worker_id: String) -> Self {
        let mut severity_distribution = HashMap::new();
        severity_distribution.insert("critical".to_string(), 0);
        severity_distribution.insert("high".to_string(), 0);
        severity_distribution.insert("medium".to_string(), 0);
        severity_distribution.insert("low".to_string(), 0);
        severity_distribution.insert("info".to_string(), 0);
        
        Self {
            worker_id,
            tasks_received: 0,
            tasks_processed: 0,
            tasks_failed: 0,
            tasks_retried: 0,
            total_processing_time: 0.0,
            total_queue_wait_time: 0.0,
            findings_created: 0,
            vulnerabilities_found: 0,
            false_positives: 0,
            severity_distribution,
            current_memory_usage: 0.0,
            current_cpu_usage: 0.0,
            active_connections: 0,
        }
    }
    
    pub fn to_dict(&self) -> serde_json::Value {
        let avg_processing_time = if self.tasks_processed > 0 {
            self.total_processing_time / self.tasks_processed as f64
        } else {
            0.0
        };
        
        let avg_queue_wait_time = if self.tasks_received > 0 {
            self.total_queue_wait_time / self.tasks_received as f64
        } else {
            0.0
        };
        
        serde_json::json!({
            "worker_id": self.worker_id,
            "timestamp": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            "task_metrics": {
                "tasks_received": self.tasks_received,
                "tasks_processed": self.tasks_processed,
                "tasks_failed": self.tasks_failed,
                "tasks_retried": self.tasks_retried,
                "average_processing_time": avg_processing_time,
                "average_queue_wait_time": avg_queue_wait_time
            },
            "detection_metrics": {
                "findings_created": self.findings_created,
                "vulnerabilities_found": self.vulnerabilities_found,
                "false_positives": self.false_positives,
                "severity_distribution": self.severity_distribution
            },
            "system_metrics": {
                "memory_usage": self.current_memory_usage,
                "cpu_usage": self.current_cpu_usage,
                "active_connections": self.active_connections
            }
        })
    }
}

/// 指標導出器 trait
pub trait MetricsExporter: Send + Sync {
    fn export(&self, metrics: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>>;
}

/// JSON 格式指標導出器
pub struct JSONMetricsExporter {
    output_file: Option<String>,
    metrics_history: Arc<Mutex<Vec<serde_json::Value>>>,
    max_history_size: usize,
}

impl JSONMetricsExporter {
    pub fn new(output_file: Option<String>) -> Self {
        Self {
            output_file,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            max_history_size: 1000,
        }
    }
    
    pub fn get_recent_metrics(&self, count: usize) -> Vec<serde_json::Value> {
        let history = self.metrics_history.lock().unwrap();
        let start = if count >= history.len() { 0 } else { history.len() - count };
        history[start..].to_vec()
    }
}

impl MetricsExporter for JSONMetricsExporter {
    fn export(&self, metrics: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        // 添加導出時間戳
        let mut metrics_with_timestamp = metrics.clone();
        if let serde_json::Value::Object(ref mut map) = metrics_with_timestamp {
            map.insert(
                "exported_at".to_string(),
                serde_json::Value::String(
                    chrono::Utc::now().to_rfc3339()
                )
            );
        }
        
        // 加入歷史記錄
        {
            let mut history = self.metrics_history.lock().unwrap();
            history.push(metrics_with_timestamp.clone());
            if history.len() > self.max_history_size {
                history.remove(0);
            }
        }
        
        // 如果指定了輸出文件，寫入文件
        if let Some(ref output_file) = self.output_file {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(output_file)?;
            
            writeln!(file, "{}", serde_json::to_string(&metrics_with_timestamp)?)?;
        }
        
        debug!("Metrics exported: {}", serde_json::to_string_pretty(&metrics_with_timestamp)?);
        Ok(())
    }
}

/// 統一指標收集器
pub struct MetricsCollector {
    worker_id: String,
    collection_interval: Duration,
    exporters: Vec<Box<dyn MetricsExporter>>,
    
    metrics: Arc<RwLock<WorkerMetrics>>,
    
    // 性能追蹤
    task_start_times: Arc<Mutex<HashMap<String, Instant>>>,
    last_export_time: Arc<Mutex<Instant>>,
    
    // 後台導出控制
    stop_sender: Option<std::sync::mpsc::Sender<()>>,
    export_handle: Option<thread::JoinHandle<()>>,
}

impl MetricsCollector {
    pub fn new(
        worker_id: String,
        collection_interval: Duration,
        exporters: Vec<Box<dyn MetricsExporter>>,
    ) -> Self {
        let exporters = if exporters.is_empty() {
            vec![Box::new(JSONMetricsExporter::new(None)) as Box<dyn MetricsExporter>]
        } else {
            exporters
        };
        
        Self {
            worker_id: worker_id.clone(),
            collection_interval,
            exporters,
            metrics: Arc::new(RwLock::new(WorkerMetrics::new(worker_id))),
            task_start_times: Arc::new(Mutex::new(HashMap::new())),
            last_export_time: Arc::new(Mutex::new(Instant::now())),
            stop_sender: None,
            export_handle: None,
        }
    }
    
    pub fn start_background_export(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.stop_sender.is_some() {
            warn!("Background export already running");
            return Ok(());
        }
        
        let (tx, rx) = std::sync::mpsc::channel();
        self.stop_sender = Some(tx);
        
        let metrics = Arc::clone(&self.metrics);
        let exporters = Arc::new(Mutex::new(std::mem::take(&mut self.exporters)));
        let interval = self.collection_interval;
        
        let handle = thread::spawn(move || {
            loop {
                match rx.recv_timeout(interval) {
                    Ok(()) => break, // 收到停止信號
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // 導出指標
                        let metrics_data = {
                            let metrics_guard = metrics.read().unwrap();
                            metrics_guard.to_dict()
                        };
                        
                        let exporters_guard = exporters.lock().unwrap();
                        for exporter in exporters_guard.iter() {
                            if let Err(e) = exporter.export(&metrics_data) {
                                error!("Failed to export metrics: {}", e);
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });
        
        self.export_handle = Some(handle);
        info!("Background metrics export started");
        Ok(())
    }
    
    pub fn stop_background_export(&mut self) {
        if let Some(sender) = self.stop_sender.take() {
            let _ = sender.send(());
        }
        
        if let Some(handle) = self.export_handle.take() {
            let _ = handle.join();
            info!("Background metrics export stopped");
        }
    }
    
    // 任務處理指標
    pub fn record_task_received(&self, task_id: String) {
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.tasks_received += 1;
        }
        
        {
            let mut start_times = self.task_start_times.lock().unwrap();
            start_times.insert(task_id, Instant::now());
        }
    }
    
    pub fn record_task_completed(&self, task_id: String, findings_count: u64) {
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.tasks_processed += 1;
            metrics.findings_created += findings_count;
        }
        
        // 計算處理時間
        {
            let mut start_times = self.task_start_times.lock().unwrap();
            if let Some(start_time) = start_times.remove(&task_id) {
                let processing_time = start_time.elapsed().as_secs_f64();
                let mut metrics = self.metrics.write().unwrap();
                metrics.total_processing_time += processing_time;
            }
        }
    }
    
    pub fn record_task_failed(&self, task_id: String, will_retry: bool) {
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.tasks_failed += 1;
            if will_retry {
                metrics.tasks_retried += 1;
            }
        }
        
        // 清理開始時間
        {
            let mut start_times = self.task_start_times.lock().unwrap();
            start_times.remove(&task_id);
        }
    }
    
    // 檢測結果指標
    pub fn record_vulnerability_found(&self, severity: SeverityLevel) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.vulnerabilities_found += 1;
        if let Some(count) = metrics.severity_distribution.get_mut(severity.as_str()) {
            *count += 1;
        }
    }
    
    pub fn record_false_positive(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.false_positives += 1;
    }
    
    // 系統資源指標
    pub fn update_system_metrics(
        &self,
        memory_usage: Option<f64>,
        cpu_usage: Option<f64>,
        active_connections: Option<u64>,
    ) {
        let mut metrics = self.metrics.write().unwrap();
        
        if let Some(memory) = memory_usage {
            metrics.current_memory_usage = memory;
        }
        if let Some(cpu) = cpu_usage {
            metrics.current_cpu_usage = cpu;
        }
        if let Some(connections) = active_connections {
            metrics.active_connections = connections;
        }
    }
    
    pub fn get_current_metrics(&self) -> serde_json::Value {
        let metrics = self.metrics.read().unwrap();
        metrics.to_dict()
    }
    
    pub fn export_metrics(&self) -> Result<(), Box<dyn std::error::Error>> {
        let metrics_data = self.get_current_metrics();
        
        for exporter in &self.exporters {
            exporter.export(&metrics_data)?;
        }
        
        *self.last_export_time.lock().unwrap() = Instant::now();
        Ok(())
    }
    
    pub fn reset_counters(&self) {
        let mut metrics = self.metrics.write().unwrap();
        
        metrics.tasks_received = 0;
        metrics.tasks_processed = 0;
        metrics.tasks_failed = 0;
        metrics.tasks_retried = 0;
        metrics.total_processing_time = 0.0;
        metrics.total_queue_wait_time = 0.0;
        metrics.findings_created = 0;
        metrics.vulnerabilities_found = 0;
        metrics.false_positives = 0;
        
        for (_, count) in metrics.severity_distribution.iter_mut() {
            *count = 0;
        }
        
        info!("Metrics counters reset");
    }
    
    pub fn get_summary_report(&self) -> serde_json::Value {
        let metrics = self.metrics.read().unwrap();
        
        let total_tasks = metrics.tasks_received;
        let success_rate = if total_tasks > 0 {
            metrics.tasks_processed as f64 / total_tasks as f64
        } else {
            0.0
        };
        
        let failure_rate = if total_tasks > 0 {
            metrics.tasks_failed as f64 / total_tasks as f64
        } else {
            0.0
        };
        
        let retry_rate = if total_tasks > 0 {
            metrics.tasks_retried as f64 / total_tasks as f64
        } else {
            0.0
        };
        
        let avg_processing_time_ms = if metrics.tasks_processed > 0 {
            (metrics.total_processing_time / metrics.tasks_processed as f64) * 1000.0
        } else {
            0.0
        };
        
        let false_positive_rate = if metrics.findings_created > 0 {
            metrics.false_positives as f64 / metrics.findings_created as f64
        } else {
            0.0
        };
        
        serde_json::json!({
            "worker_id": metrics.worker_id,
            "report_generated_at": chrono::Utc::now().to_rfc3339(),
            "summary": {
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "failure_rate": failure_rate,
                "retry_rate": retry_rate,
                "average_processing_time_ms": avg_processing_time_ms,
                "total_findings": metrics.findings_created,
                "total_vulnerabilities": metrics.vulnerabilities_found,
                "false_positive_rate": false_positive_rate
            },
            "current_system_status": {
                "memory_usage_mb": metrics.current_memory_usage / (1024.0 * 1024.0),
                "cpu_usage_percent": metrics.current_cpu_usage,
                "active_connections": metrics.active_connections
            },
            "vulnerability_distribution": metrics.severity_distribution
        })
    }
}

impl Drop for MetricsCollector {
    fn drop(&mut self) {
        self.stop_background_export();
        let _ = self.export_metrics(); // 最後一次導出
    }
}

// 全局收集器實例 (單例模式)
static GLOBAL_COLLECTOR: std::sync::OnceLock<Arc<Mutex<Option<MetricsCollector>>>> = std::sync::OnceLock::new();

pub fn get_metrics_collector() -> Option<Arc<Mutex<Option<MetricsCollector>>>> {
    GLOBAL_COLLECTOR.get().cloned()
}

pub fn initialize_metrics(
    worker_id: String,
    collection_interval: Duration,
    output_file: Option<String>,
    start_background_export: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let global = GLOBAL_COLLECTOR.get_or_init(|| Arc::new(Mutex::new(None)));
    let mut collector_guard = global.lock().unwrap();
    
    if collector_guard.is_some() {
        warn!("Metrics already initialized");
        return Ok(());
    }
    
    // 創建導出器
    let exporters: Vec<Box<dyn MetricsExporter>> = vec![
        Box::new(JSONMetricsExporter::new(output_file))
    ];
    
    // 創建收集器
    let mut collector = MetricsCollector::new(worker_id.clone(), collection_interval, exporters);
    
    if start_background_export {
        collector.start_background_export()?;
    }
    
    *collector_guard = Some(collector);
    info!("Global metrics collector initialized for {}", worker_id);
    Ok(())
}

pub fn cleanup_metrics() {
    if let Some(global) = GLOBAL_COLLECTOR.get() {
        let mut collector_guard = global.lock().unwrap();
        if let Some(mut collector) = collector_guard.take() {
            collector.stop_background_export();
            let _ = collector.export_metrics(); // 最後一次導出
            info!("Metrics collector cleaned up");
        }
    }
}

// 便捷函數
pub fn record_task_received(task_id: String) {
    if let Some(global) = get_metrics_collector() {
        let collector_guard = global.lock().unwrap();
        if let Some(ref collector) = *collector_guard {
            collector.record_task_received(task_id);
        }
    }
}

pub fn record_task_completed(task_id: String, findings_count: u64) {
    if let Some(global) = get_metrics_collector() {
        let collector_guard = global.lock().unwrap();
        if let Some(ref collector) = *collector_guard {
            collector.record_task_completed(task_id, findings_count);
        }
    }
}

pub fn record_task_failed(task_id: String, will_retry: bool) {
    if let Some(global) = get_metrics_collector() {
        let collector_guard = global.lock().unwrap();
        if let Some(ref collector) = *collector_guard {
            collector.record_task_failed(task_id, will_retry);
        }
    }
}

pub fn record_vulnerability_found(severity: SeverityLevel) {
    if let Some(global) = get_metrics_collector() {
        let collector_guard = global.lock().unwrap();
        if let Some(ref collector) = *collector_guard {
            collector.record_vulnerability_found(severity);
        }
    }
}

pub fn update_system_metrics(
    memory_usage: Option<f64>,
    cpu_usage: Option<f64>,
    active_connections: Option<u64>,
) {
    if let Some(global) = get_metrics_collector() {
        let collector_guard = global.lock().unwrap();
        if let Some(ref collector) = *collector_guard {
            collector.update_system_metrics(memory_usage, cpu_usage, active_connections);
        }
    }
}