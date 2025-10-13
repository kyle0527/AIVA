package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/function/function_sca_go/internal/scanner"
	"github.com/kyle0527/aiva/services/function/function_sca_go/pkg/messaging"
	"github.com/kyle0527/aiva/services/function/function_sca_go/pkg/models"
)

func main() {
	// 初始化日誌
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logger.Sync()

	// 讀取配置
	config := loadConfig()

	// 連接 RabbitMQ
	conn, err := amqp.Dial(config.RabbitMQURL)
	if err != nil {
		logger.Fatal("Failed to connect to RabbitMQ", zap.Error(err))
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		logger.Fatal("Failed to open channel", zap.Error(err))
	}
	defer ch.Close()

	// 宣告佇列
	queue, err := ch.QueueDeclare(
		"tasks.function.sca", // 佇列名稱
		true,                 // durable
		false,                // delete when unused
		false,                // exclusive
		false,                // no-wait
		nil,                  // arguments
	)
	if err != nil {
		logger.Fatal("Failed to declare queue", zap.Error(err))
	}

	// 設定 QoS
	err = ch.Qos(
		1,     // prefetch count
		0,     // prefetch size
		false, // global
	)
	if err != nil {
		logger.Fatal("Failed to set QoS", zap.Error(err))
	}

	// 開始消費訊息
	msgs, err := ch.Consume(
		queue.Name, // queue
		"",         // consumer
		false,      // auto-ack
		false,      // exclusive
		false,      // no-local
		false,      // no-wait
		nil,        // args
	)
	if err != nil {
		logger.Fatal("Failed to register consumer", zap.Error(err))
	}

	// 建立 SCA 掃描器
	scaScanner := scanner.NewSCAScanner(logger)

	// 建立訊息發送器
	publisher := messaging.NewPublisher(ch, logger)

	logger.Info("SCA Worker started, waiting for tasks...")

	// 處理訊息
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 優雅關閉
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("Shutting down gracefully...")
		cancel()
	}()

	// 消費迴圈
	for {
		select {
		case <-ctx.Done():
			logger.Info("Context cancelled, exiting...")
			return
		case msg, ok := <-msgs:
			if !ok {
				logger.Warn("Message channel closed")
				return
			}
			processMessage(ctx, msg, scaScanner, publisher, logger)
		}
	}
}

func processMessage(
	ctx context.Context,
	msg amqp.Delivery,
	scaScanner *scanner.SCAScanner,
	publisher *messaging.Publisher,
	logger *zap.Logger,
) {
	logger.Info("Received task", zap.String("message_id", msg.MessageId))

	// 解析任務
	var task models.FunctionTaskPayload
	if err := json.Unmarshal(msg.Body, &task); err != nil {
		logger.Error("Failed to parse task", zap.Error(err))
		msg.Nack(false, false) // 不重新排隊
		return
	}

	// 執行掃描
	startTime := time.Now()
	findings, err := scaScanner.Scan(ctx, task)
	duration := time.Since(startTime)

	if err != nil {
		logger.Error("Scan failed",
			zap.String("task_id", task.TaskID),
			zap.Error(err),
		)
		// 發送錯誤通知（可選）
		msg.Nack(false, false)
		return
	}

	logger.Info("Scan completed",
		zap.String("task_id", task.TaskID),
		zap.Int("findings_count", len(findings)),
		zap.Duration("duration", duration),
	)

	// 發送結果
	for _, finding := range findings {
		if err := publisher.PublishFinding(ctx, finding); err != nil {
			logger.Error("Failed to publish finding",
				zap.String("finding_id", finding.FindingID),
				zap.Error(err),
			)
			// 繼續處理其他結果
		}
	}

	// 確認訊息
	msg.Ack(false)
}

type Config struct {
	RabbitMQURL string
	WorkerCount int
}

func loadConfig() Config {
	rabbitMQURL := os.Getenv("RABBITMQ_URL")
	if rabbitMQURL == "" {
		rabbitMQURL = "amqp://guest:guest@localhost:5672/"
	}

	return Config{
		RabbitMQURL: rabbitMQURL,
		WorkerCount: 5,
	}
}
