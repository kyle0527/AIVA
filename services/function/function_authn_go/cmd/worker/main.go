package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/brute_force"
	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/token_test"
	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/weak_config"
	"github.com/kyle0527/aiva/services/function/function_authn_go/pkg/messaging"
	"go.uber.org/zap"
)

func main() {
	// 初始化日誌
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	logger.Info("🔐 Starting AIVA Function-AuthN Worker (Go)")

	// RabbitMQ URL
	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://guest:guest@localhost:5672/"
	}

	// 建立測試器
	bruteForcer := brute_force.NewBruteForcer(logger)
	weakConfigTester := weak_config.NewWeakConfigTester(logger)
	tokenAnalyzer := token_test.NewTokenAnalyzer(logger)

	// 建立 RabbitMQ 消費者
	consumer, err := messaging.NewConsumer(rabbitmqURL, "tasks.function.authn", logger)
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
		return handleTask(ctx, taskData, bruteForcer, weakConfigTester, tokenAnalyzer, publisher, logger)
	})

	if err != nil {
		logger.Fatal("Consumer error", zap.Error(err))
	}
}

func handleTask(
	ctx context.Context,
	taskData []byte,
	bruteForcer *brute_force.BruteForcer,
	weakConfigTester *weak_config.WeakConfigTester,
	tokenAnalyzer *token_test.TokenAnalyzer,
	publisher *messaging.Publisher,
	logger *zap.Logger,
) error {
	// 解析任務
	task, err := messaging.ParseTask(taskData)
	if err != nil {
		return err
	}

	logger.Info("Processing AuthN task", zap.String("task_id", task.TaskID))

	var findings []interface{}

	// 根據測試類型執行
	testType := task.Options.TestType
	if testType == "" {
		testType = "all"
	}

	switch testType {
	case "brute_force", "all":
		bf, err := bruteForcer.Test(ctx, task)
		if err != nil {
			logger.Error("Brute force test failed", zap.Error(err))
		} else {
			findings = append(findings, bf...)
		}

	case "weak_config", "all":
		wc, err := weakConfigTester.Test(ctx, task)
		if err != nil {
			logger.Error("Weak config test failed", zap.Error(err))
		} else {
			findings = append(findings, wc...)
		}

	case "token", "all":
		tk, err := tokenAnalyzer.Test(ctx, task)
		if err != nil {
			logger.Error("Token test failed", zap.Error(err))
		} else {
			findings = append(findings, tk...)
		}
	}

	logger.Info("AuthN test completed",
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
