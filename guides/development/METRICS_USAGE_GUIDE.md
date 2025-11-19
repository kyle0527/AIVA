# AIVA 統一統計收集系統使用指南

## 📑 目錄

- [📋 概述](#-概述)
- [🔧 支援的語言](#-支援的語言)
- [⚡ 基本使用](#-基本使用)
- [📊 高級功能](#-高級功能)
- [🐛 故障排除](#-故障排除)
- [📈 最佳實踐](#-最佳實踐)
- [🔗 相關資源](#-相關資源)

## 概述

本文檔說明如何在 AIVA 的各個 Worker 中使用統一的統計收集系統。該系統提供跨語言一致的性能監控和統計收集功能。

## 支援的語言

- **Python**: `aiva_common.metrics`
- **Go**: `aiva_common_go/metrics`
- **Rust**: `aiva_common_rust::metrics`

## 基本使用

### Python 示例

```python
from aiva_common.metrics import (
    initialize_metrics, 
    record_task_received, 
    record_task_completed,
    record_vulnerability_found,
    SeverityLevel,
    cleanup_metrics
)

# 1. 初始化指標收集器 (在 worker 啟動時)
def main():
    worker_id = "function_sast_python"
    metrics_file = "/var/log/aiva/metrics.jsonl"
    
    # 初始化 (會自動啟動後台導出)
    collector = initialize_metrics(
        worker_id=worker_id,
        collection_interval=60,  # 60秒導出一次
        output_file=metrics_file,
        start_background_export=True
    )
    
    try:
        # 啟動 worker 邏輯
        run_worker()
    finally:
        # 清理 (程序結束時)
        cleanup_metrics()

# 2. 在任務處理中使用
def process_task(task_data):
    task_id = task_data.get("task_id", "unknown")
    
    # 記錄任務開始
    record_task_received(task_id)
    
    try:
        # 執行掃描邏輯
        results = perform_scan(task_data)
        
        # 記錄發現的漏洞
        for finding in results:
            if finding.severity == "high":
                record_vulnerability_found(SeverityLevel.HIGH)
        
        # 記錄任務完成
        record_task_completed(task_id, findings_count=len(results))
        
        return results
        
    except Exception as e:
        # 記錄任務失敗
        record_task_failed(task_id, will_retry=True)
        raise

# 3. 使用裝飾器 (更簡潔的方式)
from aiva_common.metrics import track_task_metrics

@track_task_metrics(lambda task_data: task_data["task_id"])
def scan_with_metrics(task_data):
    # 裝飾器會自動處理指標記錄
    results = perform_scan(task_data)
    return {"findings_count": len(results), "results": results}
```

### Go 示例

```go
package main

import (
    "log"
    "time"
    
    "your_project/aiva_common_go/metrics"
)

func main() {
    // 1. 初始化指標收集器
    workerID := "function_ssrf_go"
    metricsFile := "/var/log/aiva/metrics.jsonl"
    
    collector := metrics.InitializeMetrics(
        workerID,
        60*time.Second,  // 60秒導出一次
        metricsFile,
        true,            // 啟動後台導出
    )
    defer metrics.CleanupMetrics()
    
    // 啟動 worker
    runWorker()
}

func processTask(taskData map[string]interface{}) error {
    taskID := taskData["task_id"].(string)
    
    // 2. 記錄任務開始
    metrics.RecordTaskReceived(taskID)
    
    // 執行掃描
    results, err := performScan(taskData)
    if err != nil {
        // 記錄失敗
        metrics.RecordTaskFailed(taskID, true) // will retry
        return err
    }
    
    // 記錄發現的漏洞
    for _, finding := range results {
        if finding.Severity == "high" {
            metrics.RecordVulnerabilityFound(metrics.SeverityHigh)
        }
    }
    
    // 記錄任務完成
    metrics.RecordTaskCompleted(taskID, int64(len(results)))
    
    return nil
}

// 3. 定期更新系統指標
func updateSystemMetrics() {
    memUsage := getMemoryUsage()    // 獲取記憶體使用量
    cpuUsage := getCPUUsage()       // 獲取 CPU 使用率
    connections := int64(10)        // 活動連接數
    
    metrics.UpdateSystemMetrics(&memUsage, &cpuUsage, &connections)
}
```

### Rust 示例

```rust
use aiva_common_rust::{
    initialize_metrics, cleanup_metrics, 
    record_task_received, record_task_completed, record_task_failed,
    record_vulnerability_found, SeverityLevel
};
use std::time::Duration;
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 初始化指標收集器
    let worker_id = "info_gatherer_rust".to_string();
    let metrics_file = Some("/var/log/aiva/metrics.jsonl".to_string());
    
    initialize_metrics(
        worker_id,
        Duration::from_secs(60),  // 60秒導出一次
        metrics_file,
        true,                     // 啟動後台導出
    )?;
    
    // 確保程序結束時清理
    let _cleanup_guard = CleanupGuard;
    
    // 啟動 worker
    run_worker()?;
    
    Ok(())
}

// 清理保護 (RAII 模式)
struct CleanupGuard;
impl Drop for CleanupGuard {
    fn drop(&mut self) {
        cleanup_metrics();
    }
}

fn process_task(task_data: &TaskData) -> Result<Vec<Finding>, Box<dyn std::error::Error>> {
    let task_id = task_data.task_id.clone();
    
    // 2. 記錄任務開始
    record_task_received(task_id.clone());
    
    match perform_scan(task_data) {
        Ok(results) => {
            // 記錄漏洞發現
            for finding in &results {
                match finding.severity.as_str() {
                    "critical" => record_vulnerability_found(SeverityLevel::Critical),
                    "high" => record_vulnerability_found(SeverityLevel::High),
                    "medium" => record_vulnerability_found(SeverityLevel::Medium),
                    "low" => record_vulnerability_found(SeverityLevel::Low),
                    _ => record_vulnerability_found(SeverityLevel::Info),
                }
            }
            
            // 記錄任務完成
            record_task_completed(task_id, results.len() as u64);
            Ok(results)
        }
        Err(e) => {
            // 記錄任務失敗
            record_task_failed(task_id, true); // will retry
            Err(e)
        }
    }
}
```

## 導出格式

### JSON Lines 格式 (預設)

每行一個 JSON 物件，包含完整的指標數據：

```json
{
  "worker_id": "function_sast_rust",
  "timestamp": 1673024400,
  "task_metrics": {
    "tasks_received": 45,
    "tasks_processed": 42,
    "tasks_failed": 3,
    "tasks_retried": 2,
    "average_processing_time": 2.5,
    "average_queue_wait_time": 0.8
  },
  "detection_metrics": {
    "findings_created": 18,
    "vulnerabilities_found": 12,
    "false_positives": 2,
    "severity_distribution": {
      "critical": 1,
      "high": 4,
      "medium": 5,
      "low": 2,
      "info": 0
    }
  },
  "system_metrics": {
    "memory_usage": 104857600,
    "cpu_usage": 15.5,
    "active_connections": 3
  },
  "exported_at": "2025-01-07T10:30:00Z"
}
```

## 配置建議

### 配置說明

**研發階段**：無需設置環境變數，使用預設配置。

**生產環境**：部署時才需要設置相關監控參數。

```bash
# 統計收集配置
AIVA_METRICS_ENABLED=true
AIVA_METRICS_INTERVAL=60
AIVA_METRICS_OUTPUT_FILE=/var/log/aiva/metrics/${WORKER_ID}.jsonl
AIVA_METRICS_RETENTION_DAYS=30
```

### 生產環境設定

1. **收集間隔**: 建議 60-300 秒
2. **日誌輪轉**: 使用 logrotate 管理日誌檔案
3. **監控整合**: 可與 Prometheus、Grafana 等監控系統整合
4. **磁碟空間**: 監控日誌檔案大小，設定適當的保留策略

### Docker 配置

```dockerfile
# 在 Dockerfile 中創建日誌目錄
RUN mkdir -p /var/log/aiva/metrics

# 掛載外部卷用於持久化指標數據
VOLUME ["/var/log/aiva"]
```

## 最佳實踐

1. **初始化**: 在 worker 啟動時立即初始化指標收集器
2. **清理**: 確保程序結束時呼叫清理函數
3. **錯誤處理**: 指標收集失敗不應影響主要業務邏輯
4. **系統指標**: 定期更新系統資源使用情況
5. **標準化**: 使用一致的任務 ID 和嚴重程度級別

## 監控儀表板

���議建立統一的監控儀表板來檢視：

- 各 Worker 的處理能力
- 錯誤率和重試率
- 漏洞發現趨勢
- 系統資源使用情況
- 效能瓶頸分析

## 故障排除

### 常見問題

1. **指標未導出**: 檢查文件權限和磁碟空間
2. **記憶體洩漏**: 確保呼叫清理函數
3. **效能影響**: 調整收集間隔或關閉不必要的指標
4. **資料不一致**: 檢查時間同步和任務 ID 唯一性