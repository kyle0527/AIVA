package main

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"syscall"

	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/config"
	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/logger"
	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/mq"
	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
	"github.com/kyle0527/aiva/services/function/function_cspm_go/internal/scanner"
	"go.uber.org/zap"
)

func main() {
	// 載入配置
	cfg, err := config.LoadConfig("function-cspm")
	if err != nil {
		panic(err)
	}

	// 初始化日誌
	log, err := logger.NewLogger(cfg.ServiceName, "cspm_worker")
	if err != nil {
		panic(err)
	}
	defer log.Sync()

	log.Info("🔍 Starting AIVA Function-CSPM Worker (Go)",
		zap.String("service", cfg.ServiceName),
		zap.String("version", "2.0.0-unified"))

	// 建立 MQ 客戶端
	mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
	if err != nil {
		log.Fatal("Failed to create MQ client", zap.Error(err))
	}
	defer mqClient.Close()

	// 建立掃描器
	cspmScanner := scanner.NewCSPMScanner(log)

	// 啟動消費循環
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 處理優雅關閉
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Info("Shutting down gracefully...")
		cancel()
	}()

	// 開始消費任務
	queueName := "tasks.function.cspm"
	err = mqClient.Consume(queueName, func(body []byte) error {
		return handleTask(ctx, body, cspmScanner, mqClient, log)
	})

	if err != nil {
		log.Fatal("Consumer error", zap.Error(err))
	}
}

func handleTask(
	ctx context.Context,
	taskData []byte,
	scanner *scanner.CSPMScanner,
	mqClient *mq.MQClient,
	log *zap.Logger,
) error {
	// 解析任務
	var task schemas.FunctionTaskPayload
	if err := json.Unmarshal(taskData, &task); err != nil {
		log.Error("Failed to parse task", zap.Error(err))
		return err
	}

	log.Info("Processing CSPM task",
		zap.String("task_id", task.TaskID),
		zap.String("scan_id", task.ScanID))

	// 執行 CSPM 掃描
	findings, err := scanner.Scan(ctx, &task)
	if err != nil {
		log.Error("Scan failed", zap.Error(err))
		return err
	}

	log.Info("CSPM scan completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings_count", len(findings)))

	// 發布 Findings
	for _, finding := range findings {
		findingData, err := json.Marshal(finding)
		if err != nil {
			log.Error("Failed to marshal finding", zap.Error(err))
			continue
		}

		if err := mqClient.Publish("findings.cspm", findingData); err != nil {
			log.Error("Failed to publish finding", zap.Error(err))
		}
	}

	return nil
}
