package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kyle0527/aiva/services/scan/go_engine/internal/common"
	"github.com/kyle0527/aiva/services/scan/go_engine/internal/ssrf/detector"
	"log"
)

const (
	VERSION      = "1.0.0"
	SCANNER_TYPE = "ssrf"
)

func main() {
	// 初始化 Logger (Development 模式顯示 Debug 日誌)
	// Using standard log package
	// Using standard log package

	// 設置信號處理（優雅關閉）
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		log.Println("[INFO] Received signal, shutting down")
		cancel()
	}()

	// 讀取 JSON 輸入
	var request common.ScanRequest
	decoder := json.NewDecoder(os.Stdin)
	if err := decoder.Decode(&request); err != nil {
		outputError(logger, "Failed to decode input", err)
		os.Exit(1)
	}

	// 驗證輸入
	if len(request.Targets) == 0 {
		outputError(logger, "No targets provided", fmt.Errorf("targets list is empty"))
		os.Exit(1)
	}

	// 設置默認值
	if request.Concurrency == 0 {
		request.Concurrency = 10
	}
	if request.Timeout == 0 {
		request.Timeout = 300
	}
	if request.MaxDepth == 0 {
		request.MaxDepth = 3
	}
	if request.UserAgent == "" {
		request.UserAgent = "AIVA-SSRF-Scanner/1.0"
	}

	log.Println("[INFO] SSRF Scanner started")
		zap.String("scan_id", request.ScanID),
		zap.Int("targets", len(request.Targets)),
		zap.Int("concurrency", request.Concurrency),
		zap.Int("timeout", request.Timeout),
	)

	// 創建掃描器
	ssrfDetector := detector.NewSSRFDetector()

	// 構建配置
	config := &common.ScannerConfig{
		MaxDepth:       request.MaxDepth,
		Timeout:        time.Duration(request.Timeout) * time.Second,
		Concurrency:    request.Concurrency,
		Headers:        request.Headers,
		UserAgent:      request.UserAgent,
		MaxRetries:     request.MaxRetries,
		EnableProgress: request.EnableProgressReport,
		EnableVerbose:  request.EnableVerboseLog,
	}

	// 執行掃描
	startTime := time.Now()
	result := executeScan(ctx, ssrfDetector, &request, config, logger)
	result.ExecutionTime = time.Since(startTime).Seconds()

	// 輸出結果
	outputResult(result)

	log.Println("[INFO] SSRF Scanner completed"),
		zap.String("status", result.Status),
		zap.Int("assets_found", len(result.Assets)),
		zap.Float64("execution_time", result.ExecutionTime),
	)
}

func executeScan(
	ctx context.Context,
	scanner common.Scanner,
	request *common.ScanRequest,
	config *common.ScannerConfig,
	logger *zap.Logger,
) *common.ScanResult {

	result := &common.ScanResult{
		ScanID:      request.ScanID,
		ScannerType: SCANNER_TYPE,
		Status:      "success",
		Assets:      []common.Asset{},
		Errors:      []string{},
		Warnings:    []string{},
	}

	// 創建 Worker Pool
	pool := common.NewWorkerPool($1)
	defer pool.Shutdown(5 * time.Second)

	// 提交任務
	for i, target := range request.Targets {
		task := &common.Task{
			ID:      fmt.Sprintf("%s-%d", request.ScanID, i),
			Target:  target,
			Scanner: scanner,
			Config:  config,
		}

		if err := pool.Submit(task); err != nil {
			log.Println("[ERROR] Failed to submit task"), zap.String("target", target))
			result.Errors = append(result.Errors, fmt.Sprintf("Failed to submit task for %s: %v", target, err))
			continue
		}
	}

	// 收集結果
	resultCount := 0
	totalTasks := len(request.Targets)

	for resultCount < totalTasks {
		select {
		case <-ctx.Done():
			result.Status = "cancelled"
			result.Warnings = append(result.Warnings, "Scan cancelled by user")
			return result

		case taskResult, ok := <-pool.Results():
			if !ok {
				break
			}

			resultCount++
			result.TargetsScanned++

			if taskResult.Error != nil {
				result.FailureCount++
				result.Errors = append(result.Errors,
					fmt.Sprintf("Target %s failed: %v", taskResult.Target, taskResult.Error))
				log.Println("[WARN] Task failed"),
					zap.Error(taskResult.Error),
				)
				continue
			}

			result.SuccessCount++
			result.Assets = append(result.Assets, taskResult.Assets...)
			
			// 🔧 修復: 收集請求統計信息
			if ssrfDetector, ok := scanner.(*detector.SSRFDetector); ok {
				stats := ssrfDetector.GetStats()
				result.RequestsMade += int(stats["sent"])
			}

			// 進度報告（輸出到 stderr，不影響 stdout）
			if config.EnableProgress && resultCount%10 == 0 {
				progress := common.ProgressReport{
					Timestamp:       time.Now().Format(time.RFC3339),
					TargetsScanned:  resultCount,
					TotalTargets:    totalTasks,
					PercentComplete: float64(resultCount) / float64(totalTasks) * 100,
					CurrentTarget:   taskResult.Target,
				}
				result.ProgressReports = append(result.ProgressReports, progress)

				progressJSON, _ := json.Marshal(progress)
				fmt.Fprintf(os.Stderr, "PROGRESS: %s\n", string(progressJSON))
			}
		}
	}

	if result.FailureCount > 0 {
		result.Status = "partial"
	}

	return result
}

func outputResult(result *common.ScanResult) {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to encode result: %v\n", err)
		os.Exit(1)
	}
}

func outputError(logger *zap.Logger, message string, err error) {
	logger.Error(message, zap.Error(err))

	result := &common.ScanResult{
		Status: "failed",
		Errors: []string{fmt.Sprintf("%s: %v", message, err)},
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	encoder.Encode(result)
}

