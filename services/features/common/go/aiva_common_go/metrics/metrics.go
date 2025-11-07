// Package metrics 提供 AIVA 統一統計收集功能 - Go 實現
// 日期: 2025-01-07
// 目的: 提供跨語言一致的性能監控和統計收集功能
package metrics

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

// MetricType 指標類型
type MetricType string

const (
	MetricTypeCounter   MetricType = "counter"
	MetricTypeGauge     MetricType = "gauge"
	MetricTypeHistogram MetricType = "histogram"
	MetricTypeDuration  MetricType = "duration"
)

// SeverityLevel 嚴重程度級別
type SeverityLevel string

const (
	SeverityCritical SeverityLevel = "critical"
	SeverityHigh     SeverityLevel = "high"
	SeverityMedium   SeverityLevel = "medium"
	SeverityLow      SeverityLevel = "low"
	SeverityInfo     SeverityLevel = "info"
)

// MetricData 指標數據結構
type MetricData struct {
	Name       string            `json:"name"`
	Value      float64           `json:"value"`
	MetricType MetricType        `json:"metric_type"`
	Timestamp  int64             `json:"timestamp"`
	Labels     map[string]string `json:"labels,omitempty"`
	Unit       string            `json:"unit,omitempty"`
}

// WorkerMetrics Worker 統計指標集合
type WorkerMetrics struct {
	WorkerID string `json:"worker_id"`

	// 任務處理統計
	TasksReceived       int64   `json:"tasks_received"`
	TasksProcessed      int64   `json:"tasks_processed"`
	TasksFailed         int64   `json:"tasks_failed"`
	TasksRetried        int64   `json:"tasks_retried"`
	TotalProcessingTime float64 `json:"total_processing_time"` // 秒
	TotalQueueWaitTime  float64 `json:"total_queue_wait_time"` // 秒

	// 檢測結果統計
	FindingsCreated      int64 `json:"findings_created"`
	VulnerabilitiesFound int64 `json:"vulnerabilities_found"`
	FalsePositives       int64 `json:"false_positives"`

	// 嚴重程度分佈
	SeverityDistribution map[string]int64 `json:"severity_distribution"`

	// 系統資源 (瞬時值)
	CurrentMemoryUsage float64 `json:"current_memory_usage"` // 位元組
	CurrentCPUUsage    float64 `json:"current_cpu_usage"`    // 百分比
	ActiveConnections  int64   `json:"active_connections"`
}

// NewWorkerMetrics 創建新的 Worker 指標實例
func NewWorkerMetrics(workerID string) *WorkerMetrics {
	return &WorkerMetrics{
		WorkerID: workerID,
		SeverityDistribution: map[string]int64{
			"critical": 0,
			"high":     0,
			"medium":   0,
			"low":      0,
			"info":     0,
		},
	}
}

// ToDict 轉換為結構化數據
func (wm *WorkerMetrics) ToDict() map[string]interface{} {
	avgProcessingTime := float64(0)
	if wm.TasksProcessed > 0 {
		avgProcessingTime = wm.TotalProcessingTime / float64(wm.TasksProcessed)
	}

	avgQueueWaitTime := float64(0)
	if wm.TasksReceived > 0 {
		avgQueueWaitTime = wm.TotalQueueWaitTime / float64(wm.TasksReceived)
	}

	return map[string]interface{}{
		"worker_id": wm.WorkerID,
		"timestamp": time.Now().Unix(),
		"task_metrics": map[string]interface{}{
			"tasks_received":          wm.TasksReceived,
			"tasks_processed":         wm.TasksProcessed,
			"tasks_failed":            wm.TasksFailed,
			"tasks_retried":           wm.TasksRetried,
			"average_processing_time": avgProcessingTime,
			"average_queue_wait_time": avgQueueWaitTime,
		},
		"detection_metrics": map[string]interface{}{
			"findings_created":      wm.FindingsCreated,
			"vulnerabilities_found": wm.VulnerabilitiesFound,
			"false_positives":       wm.FalsePositives,
			"severity_distribution": wm.SeverityDistribution,
		},
		"system_metrics": map[string]interface{}{
			"memory_usage":       wm.CurrentMemoryUsage,
			"cpu_usage":          wm.CurrentCPUUsage,
			"active_connections": wm.ActiveConnections,
		},
	}
}

// MetricsExporter 指標導出器介面
type MetricsExporter interface {
	Export(metrics map[string]interface{}) error
}

// JSONMetricsExporter JSON 格式指標導出器
type JSONMetricsExporter struct {
	OutputFile     string
	metricsHistory []map[string]interface{}
	historyLock    sync.RWMutex
	maxHistorySize int
}

// NewJSONMetricsExporter 創建 JSON 導出器
func NewJSONMetricsExporter(outputFile string) *JSONMetricsExporter {
	return &JSONMetricsExporter{
		OutputFile:     outputFile,
		metricsHistory: make([]map[string]interface{}, 0),
		maxHistorySize: 1000,
	}
}

// Export 導出指標為 JSON 格式
func (je *JSONMetricsExporter) Export(metrics map[string]interface{}) error {
	// 添加導出時間戳
	metricsWithTimestamp := make(map[string]interface{})
	for k, v := range metrics {
		metricsWithTimestamp[k] = v
	}
	metricsWithTimestamp["exported_at"] = time.Now().Format(time.RFC3339)

	// 加入歷史記錄
	je.historyLock.Lock()
	je.metricsHistory = append(je.metricsHistory, metricsWithTimestamp)
	if len(je.metricsHistory) > je.maxHistorySize {
		je.metricsHistory = je.metricsHistory[1:]
	}
	je.historyLock.Unlock()

	// 如果指定了輸出文件，寫入文件
	if je.OutputFile != "" {
		file, err := os.OpenFile(je.OutputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("failed to open metrics file: %v", err)
		}
		defer file.Close()

		jsonData, err := json.Marshal(metricsWithTimestamp)
		if err != nil {
			return fmt.Errorf("failed to marshal metrics: %v", err)
		}

		if _, err := file.Write(append(jsonData, '\n')); err != nil {
			return fmt.Errorf("failed to write metrics: %v", err)
		}
	}

	log.Printf("Metrics exported: %v", metricsWithTimestamp)
	return nil
}

// GetRecentMetrics 獲取最近的指標記錄
func (je *JSONMetricsExporter) GetRecentMetrics(count int) []map[string]interface{} {
	je.historyLock.RLock()
	defer je.historyLock.RUnlock()

	if count >= len(je.metricsHistory) {
		result := make([]map[string]interface{}, len(je.metricsHistory))
		copy(result, je.metricsHistory)
		return result
	}

	start := len(je.metricsHistory) - count
	result := make([]map[string]interface{}, count)
	copy(result, je.metricsHistory[start:])
	return result
}

// MetricsCollector 統一指標收集器
type MetricsCollector struct {
	workerID           string
	collectionInterval time.Duration
	exporters          []MetricsExporter

	metrics *WorkerMetrics
	lock    sync.RWMutex

	// 性能追蹤
	taskStartTimes map[string]time.Time
	taskTimesLock  sync.RWMutex
	lastExportTime time.Time

	// 後台導出
	stopChannel   chan struct{}
	exportRunning bool
	exportLock    sync.Mutex
}

// NewMetricsCollector 創建指標收集器
func NewMetricsCollector(workerID string, collectionInterval time.Duration, exporters []MetricsExporter) *MetricsCollector {
	if exporters == nil {
		exporters = []MetricsExporter{NewJSONMetricsExporter("")}
	}

	return &MetricsCollector{
		workerID:           workerID,
		collectionInterval: collectionInterval,
		exporters:          exporters,
		metrics:            NewWorkerMetrics(workerID),
		taskStartTimes:     make(map[string]time.Time),
		lastExportTime:     time.Now(),
		stopChannel:        make(chan struct{}),
	}
}

// StartBackgroundExport 啟動後台指標導出
func (mc *MetricsCollector) StartBackgroundExport() {
	mc.exportLock.Lock()
	defer mc.exportLock.Unlock()

	if mc.exportRunning {
		log.Println("Background export already running")
		return
	}

	mc.exportRunning = true
	go mc.exportLoop()
	log.Println("Background metrics export started")
}

// StopBackgroundExport 停止後台指標導出
func (mc *MetricsCollector) StopBackgroundExport() {
	mc.exportLock.Lock()
	defer mc.exportLock.Unlock()

	if !mc.exportRunning {
		return
	}

	close(mc.stopChannel)
	mc.exportRunning = false
	log.Println("Background metrics export stopped")
}

// exportLoop 導出循環
func (mc *MetricsCollector) exportLoop() {
	ticker := time.NewTicker(mc.collectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := mc.ExportMetrics(); err != nil {
				log.Printf("Error exporting metrics: %v", err)
			}
		case <-mc.stopChannel:
			return
		}
	}
}

// RecordTaskReceived 記錄接收到任務
func (mc *MetricsCollector) RecordTaskReceived(taskID string) {
	mc.lock.Lock()
	mc.metrics.TasksReceived++
	mc.lock.Unlock()

	mc.taskTimesLock.Lock()
	mc.taskStartTimes[taskID] = time.Now()
	mc.taskTimesLock.Unlock()
}

// RecordTaskCompleted 記錄任務完成
func (mc *MetricsCollector) RecordTaskCompleted(taskID string, findingsCount int64) {
	mc.lock.Lock()
	mc.metrics.TasksProcessed++
	mc.metrics.FindingsCreated += findingsCount
	mc.lock.Unlock()

	// 計算處理時間
	mc.taskTimesLock.Lock()
	if startTime, exists := mc.taskStartTimes[taskID]; exists {
		processingTime := time.Since(startTime).Seconds()
		mc.lock.Lock()
		mc.metrics.TotalProcessingTime += processingTime
		mc.lock.Unlock()
		delete(mc.taskStartTimes, taskID)
	}
	mc.taskTimesLock.Unlock()
}

// RecordTaskFailed 記錄任務失敗
func (mc *MetricsCollector) RecordTaskFailed(taskID string, willRetry bool) {
	mc.lock.Lock()
	mc.metrics.TasksFailed++
	if willRetry {
		mc.metrics.TasksRetried++
	}
	mc.lock.Unlock()

	// 清理開始時間
	mc.taskTimesLock.Lock()
	delete(mc.taskStartTimes, taskID)
	mc.taskTimesLock.Unlock()
}

// RecordVulnerabilityFound 記錄發現漏洞
func (mc *MetricsCollector) RecordVulnerabilityFound(severity SeverityLevel) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.VulnerabilitiesFound++
	mc.metrics.SeverityDistribution[string(severity)]++
}

// RecordFalsePositive 記錄誤報
func (mc *MetricsCollector) RecordFalsePositive() {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.FalsePositives++
}

// UpdateSystemMetrics 更新系統資源指標
func (mc *MetricsCollector) UpdateSystemMetrics(memoryUsage, cpuUsage *float64, activeConnections *int64) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	if memoryUsage != nil {
		mc.metrics.CurrentMemoryUsage = *memoryUsage
	}
	if cpuUsage != nil {
		mc.metrics.CurrentCPUUsage = *cpuUsage
	}
	if activeConnections != nil {
		mc.metrics.ActiveConnections = *activeConnections
	}
}

// GetCurrentMetrics 獲取當前指標快照
func (mc *MetricsCollector) GetCurrentMetrics() map[string]interface{} {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	return mc.metrics.ToDict()
}

// ExportMetrics 導出指標到所有配置的導出器
func (mc *MetricsCollector) ExportMetrics() error {
	metricsData := mc.GetCurrentMetrics()

	var lastErr error
	for _, exporter := range mc.exporters {
		if err := exporter.Export(metricsData); err != nil {
			lastErr = err
			log.Printf("Exporter failed: %v", err)
		}
	}

	mc.lock.Lock()
	mc.lastExportTime = time.Now()
	mc.lock.Unlock()

	return lastErr
}

// ResetCounters 重置計數器 (保留瞬時值)
func (mc *MetricsCollector) ResetCounters() {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.TasksReceived = 0
	mc.metrics.TasksProcessed = 0
	mc.metrics.TasksFailed = 0
	mc.metrics.TasksRetried = 0
	mc.metrics.TotalProcessingTime = 0.0
	mc.metrics.TotalQueueWaitTime = 0.0
	mc.metrics.FindingsCreated = 0
	mc.metrics.VulnerabilitiesFound = 0
	mc.metrics.FalsePositives = 0

	for key := range mc.metrics.SeverityDistribution {
		mc.metrics.SeverityDistribution[key] = 0
	}

	log.Println("Metrics counters reset")
}

// GetSummaryReport 生成摘要報告
func (mc *MetricsCollector) GetSummaryReport() map[string]interface{} {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	totalTasks := mc.metrics.TasksReceived
	successRate := float64(0)
	if totalTasks > 0 {
		successRate = float64(mc.metrics.TasksProcessed) / float64(totalTasks)
	}

	failureRate := float64(0)
	if totalTasks > 0 {
		failureRate = float64(mc.metrics.TasksFailed) / float64(totalTasks)
	}

	retryRate := float64(0)
	if totalTasks > 0 {
		retryRate = float64(mc.metrics.TasksRetried) / float64(totalTasks)
	}

	avgProcessingTimeMs := float64(0)
	if mc.metrics.TasksProcessed > 0 {
		avgProcessingTimeMs = (mc.metrics.TotalProcessingTime / float64(mc.metrics.TasksProcessed)) * 1000
	}

	falsePositiveRate := float64(0)
	if mc.metrics.FindingsCreated > 0 {
		falsePositiveRate = float64(mc.metrics.FalsePositives) / float64(mc.metrics.FindingsCreated)
	}

	return map[string]interface{}{
		"worker_id":           mc.workerID,
		"report_generated_at": time.Now().Format(time.RFC3339),
		"summary": map[string]interface{}{
			"total_tasks":                totalTasks,
			"success_rate":               successRate,
			"failure_rate":               failureRate,
			"retry_rate":                 retryRate,
			"average_processing_time_ms": avgProcessingTimeMs,
			"total_findings":             mc.metrics.FindingsCreated,
			"total_vulnerabilities":      mc.metrics.VulnerabilitiesFound,
			"false_positive_rate":        falsePositiveRate,
		},
		"current_system_status": map[string]interface{}{
			"memory_usage_mb":    mc.metrics.CurrentMemoryUsage / (1024 * 1024),
			"cpu_usage_percent":  mc.metrics.CurrentCPUUsage,
			"active_connections": mc.metrics.ActiveConnections,
		},
		"vulnerability_distribution": mc.metrics.SeverityDistribution,
	}
}

// 全局收集器實例 (單例模式)
var (
	globalCollector *MetricsCollector
	collectorLock   sync.Mutex
)

// GetMetricsCollector 獲取全局指標收集器實例
func GetMetricsCollector() *MetricsCollector {
	collectorLock.Lock()
	defer collectorLock.Unlock()

	return globalCollector
}

// InitializeMetrics 初始化全局指標收集器
func InitializeMetrics(workerID string, collectionInterval time.Duration, outputFile string, startBackgroundExport bool) *MetricsCollector {
	collectorLock.Lock()
	defer collectorLock.Unlock()

	if globalCollector != nil {
		log.Printf("Metrics already initialized for %s", globalCollector.workerID)
		return globalCollector
	}

	// 創建導出器
	exporters := []MetricsExporter{NewJSONMetricsExporter(outputFile)}

	// 創建收集器
	globalCollector = NewMetricsCollector(workerID, collectionInterval, exporters)

	if startBackgroundExport {
		globalCollector.StartBackgroundExport()
	}

	log.Printf("Global metrics collector initialized for %s", workerID)
	return globalCollector
}

// CleanupMetrics 清理指標收集器
func CleanupMetrics() {
	collectorLock.Lock()
	defer collectorLock.Unlock()

	if globalCollector != nil {
		globalCollector.StopBackgroundExport()
		globalCollector.ExportMetrics() // 最後一次導出
		globalCollector = nil
		log.Println("Metrics collector cleaned up")
	}
}

// 便捷函數
func RecordTaskReceived(taskID string) {
	if collector := GetMetricsCollector(); collector != nil {
		collector.RecordTaskReceived(taskID)
	}
}

func RecordTaskCompleted(taskID string, findingsCount int64) {
	if collector := GetMetricsCollector(); collector != nil {
		collector.RecordTaskCompleted(taskID, findingsCount)
	}
}

func RecordTaskFailed(taskID string, willRetry bool) {
	if collector := GetMetricsCollector(); collector != nil {
		collector.RecordTaskFailed(taskID, willRetry)
	}
}

func RecordVulnerabilityFound(severity SeverityLevel) {
	if collector := GetMetricsCollector(); collector != nil {
		collector.RecordVulnerabilityFound(severity)
	}
}

func UpdateSystemMetrics(memoryUsage, cpuUsage *float64, activeConnections *int64) {
	if collector := GetMetricsCollector(); collector != nil {
		collector.UpdateSystemMetrics(memoryUsage, cpuUsage, activeConnections)
	}
}
