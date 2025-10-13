package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/kyle0527/aiva/services/function/function_cspm_go/internal/scanner"
	"github.com/kyle0527/aiva/services/function/function_cspm_go/pkg/messaging"
	"go.uber.org/zap"
)

func main() {
	// 初始化日誌
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	logger.Info("🔍 Starting AIVA Function-CSPM Worker (Go)")

	// RabbitMQ URL
	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://guest:guest@localhost:5672/"
	}

	// 建立掃描器
	cspmScanner := scanner.NewCSPMScanner(logger)

	// 建立 RabbitMQ 消費者
	consumer, err := messaging.NewConsumer(rabbitmqURL, "tasks.function.cspm", logger)
	if err != nil {
		logger.Fatal("Failed to create consumer", zap.Error(err))
	}
	defer consumer.Close()

	// 建立 RabbitMQ 發布者
	publisher, err := messaging.NewPublisher(rabbitmqURL, logger)
	if err != nil {
		logger.Fatal("Failed to create publisher", zap.Error(err))
	}
	defer publisher.Close()

	// 啟動消費循環
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 處理優雅關閉
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("Shutting down gracefully...")
		cancel()
	}()

	// 開始消費任務
	err = consumer.Consume(ctx, func(taskData []byte) error {
		return handleTask(ctx, taskData, cspmScanner, publisher, logger)
	})

	if err != nil {
		logger.Fatal("Consumer error", zap.Error(err))
	}
}

func handleTask(
	ctx context.Context,
	taskData []byte,
	scanner *scanner.CSPMScanner,
	publisher *messaging.Publisher,
	logger *zap.Logger,
) error {
	// 解析任務
	task, err := messaging.ParseTask(taskData)
	if err != nil {
		return err
	}

	logger.Info("Processing CSPM task", zap.String("task_id", task.TaskID))

	// 執行 CSPM 掃描
	findings, err := scanner.Scan(ctx, task)
	if err != nil {
		logger.Error("Scan failed", zap.Error(err))
		return err
	}

	logger.Info("CSPM scan completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings_count", len(findings)))

	// 發布 Findings
	for _, finding := range findings {
		if err := publisher.PublishFinding(finding); err != nil {
			logger.Error("Failed to publish finding", zap.Error(err))
		}
	}

	return nil
}
