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
	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/brute_force"
	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/token_test"
	"github.com/kyle0527/aiva/services/function/function_authn_go/internal/weak_config"
	"go.uber.org/zap"
)

func main() {
	// 載入配置
	cfg, err := config.LoadConfig("function-authn")
	if err != nil {
		panic(err)
	}

	// 初始化日誌
	log, err := logger.NewLogger(cfg.ServiceName, "authn_worker")
	if err != nil {
		panic(err)
	}
	defer log.Sync()

	log.Info("🔐 Starting AIVA Function-AuthN Worker (Go)",
		zap.String("service", cfg.ServiceName),
		zap.String("version", "2.0.0-unified"))

	// 建立 MQ 客戶端
	mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
	if err != nil {
		log.Fatal("Failed to create MQ client", zap.Error(err))
	}
	defer mqClient.Close()

	// 建立測試器
	bruteForcer := brute_force.NewBruteForcer(log)
	weakConfigTester := weak_config.NewWeakConfigTester(log)
	tokenAnalyzer := token_test.NewTokenAnalyzer(log)

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
	queueName := "tasks.function.authn"
	err = mqClient.Consume(queueName, func(body []byte) error {
		return handleTask(ctx, body, bruteForcer, weakConfigTester, tokenAnalyzer, mqClient, log)
	})

	if err != nil {
		log.Fatal("Consumer error", zap.Error(err))
	}
}

func handleTask(
	ctx context.Context,
	taskData []byte,
	bruteForcer *brute_force.BruteForcer,
	weakConfigTester *weak_config.WeakConfigTester,
	tokenAnalyzer *token_test.TokenAnalyzer,
	mqClient *mq.MQClient,
	log *zap.Logger,
) error {
	// 解析任務
	var task schemas.FunctionTaskPayload
	if err := json.Unmarshal(taskData, &task); err != nil {
		log.Error("Failed to parse task", zap.Error(err))
		return err
	}

	log.Info("Processing AuthN task", zap.String("task_id", task.TaskID))

	var findings []*schemas.FindingPayload

	// 根據策略執行測試類型 (使用 Strategy 字段)
	testType := "all"
	if task.Strategy != "" {
		testType = task.Strategy
	}

	// 執行暴力破解測試
	if testType == "brute_force" || testType == "all" {
		bf, err := bruteForcer.Test(ctx, &task)
		if err != nil {
			log.Error("Brute force test failed", zap.Error(err))
		} else {
			findings = append(findings, bf...)
		}
	}

	// 執行弱配置測試
	if testType == "weak_config" || testType == "all" {
		wc, err := weakConfigTester.Test(ctx, &task)
		if err != nil {
			log.Error("Weak config test failed", zap.Error(err))
		} else {
			findings = append(findings, wc...)
		}
	}

	// 執行 Token 分析測試
	if testType == "token" || testType == "all" {
		tk, err := tokenAnalyzer.Test(ctx, task)
		if err != nil {
			log.Error("Token test failed", zap.Error(err))
		} else {
			// 轉換 []interface{} 為 []*schemas.FindingPayload
			for _, finding := range tk {
				if findingPayload, ok := finding.(*schemas.FindingPayload); ok {
					findings = append(findings, findingPayload)
				}
			}
		}
	}

	log.Info("AuthN test completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings_count", len(findings)))

	// 發布 Findings
	resultQueue := "findings.new"
	for _, finding := range findings {
		if err := mqClient.Publish(resultQueue, finding); err != nil {
			log.Error("Failed to publish finding", zap.Error(err))
		}
	}

	return nil
}
