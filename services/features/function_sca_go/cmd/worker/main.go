package main

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/config"
	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/logger"
	"github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/mq"
	schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"
	"github.com/kyle0527/aiva/services/function/function_sca_go/internal/scanner"
)

func main() {
	// 載入配置
	cfg, err := config.LoadConfig("sca")
	if err != nil {
		panic(err)
	}

	// 初始化日誌
	log, err := logger.NewLogger(cfg.ServiceName, "sca_worker")
	if err != nil {
		panic(err)
	}
	defer log.Sync()

	// 建立 MQ 客戶端
	mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
	if err != nil {
		log.Fatal("Failed to create MQ client", zap.Error(err))
	}
	defer mqClient.Close()

	// 建立 SCA 掃描器
	scaScanner := scanner.NewSCAScanner(log)

	log.Info("SCA Worker started, waiting for tasks...")

	// 優雅關閉
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Info("Shutting down gracefully...")
		cancel()
	}()

	// 消費任務
	go func() {
		err := mqClient.Consume(
			"tasks.function.sca",
			func(body []byte) error {
				return processMessage(ctx, body, scaScanner, mqClient, log)
			},
		)

		if err != nil {
			log.Fatal("Consumer stopped with error", zap.Error(err))
		}
	}()

	// 等待取消信號
	<-ctx.Done()
	log.Info("Worker stopped")
}

func processMessage(
	ctx context.Context,
	body []byte,
	scaScanner *scanner.SCAScanner,
	mqClient *mq.MQClient,
	log *zap.Logger,
) error {
	// 解析任務 - 使用 ScanTaskPayload (包含 URL)
	var task schemas.ScanTaskPayload
	if err := json.Unmarshal(body, &task); err != nil {
		log.Error("Failed to parse task", zap.Error(err))
		return err
	}

	log.Info("Received scan task",
		zap.String("task_id", task.TaskID),
		zap.String("scan_type", task.ScanType),
		zap.String("target_url", task.Target.URL.(string)),
	)

	// 執行掃描
	startTime := time.Now()
	findings, err := scaScanner.Scan(ctx, task)
	duration := time.Since(startTime)

	if err != nil {
		log.Error("Scan failed",
			zap.String("task_id", task.TaskID),
			zap.Error(err),
		)
		return err
	}

	log.Info("Scan completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings_count", len(findings)),
		zap.Duration("duration", duration),
	)

	// 發送結果
	for _, finding := range findings {
		if err := mqClient.Publish("results.finding", finding); err != nil {
			log.Error("Failed to publish finding",
				zap.String("finding_id", finding.FindingID),
				zap.Error(err),
			)
			continue
		}

		log.Info("Published finding",
			zap.String("finding_id", finding.FindingID),
			zap.String("severity", finding.Vulnerability.Severity.(string)),
		)
	}

	return nil
}
