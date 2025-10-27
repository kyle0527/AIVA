// AIVA SSRF Detector - Go Implementation
// 日期: 2025-10-13
// 功能: 高性能 SSRF 漏洞檢測

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kyle0527/aiva/services/function/function_ssrf_go/internal/detector"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

const (
	taskQueue   = "tasks.function.ssrf"
	resultQueue = "findings.new"
)

var logger *zap.Logger

// getRabbitMQURL 獲取 RabbitMQ 連接 URL，遵循 12-factor app 原則
func getRabbitMQURL() string {
	// 優先使用完整 URL
	if url := os.Getenv("AIVA_RABBITMQ_URL"); url != "" {
		return url
	}

	// 組合式配置
	host := getEnv("AIVA_RABBITMQ_HOST", "localhost")
	port := getEnv("AIVA_RABBITMQ_PORT", "5672")
	user := os.Getenv("AIVA_RABBITMQ_USER")
	password := os.Getenv("AIVA_RABBITMQ_PASSWORD")
	vhost := getEnv("AIVA_RABBITMQ_VHOST", "/")

	if user == "" || password == "" {
		return ""
	}

	return fmt.Sprintf("amqp://%s:%s@%s:%s%s", user, password, host, port, vhost)
}

// getEnv 獲取環境變數，提供預設值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// shouldRetryMessage 檢查消息是否應該重試
// 實施統一的重試策略，防止 poison pill 消息無限循環
func shouldRetryMessage(delivery amqp.Delivery, err error) bool {
	const maxRetryAttempts = 3

	// 檢查消息頭部中的重試次數
	retryCount := 0
	if delivery.Headers != nil {
		if count, ok := delivery.Headers["x-aiva-retry-count"]; ok {
			if val, isInt := count.(int32); isInt {
				retryCount = int(val)
			}
		}
	}

	if retryCount >= maxRetryAttempts {
		logger.Error("消息已達到最大重試次數，發送到死信隊列",
			zap.Int("retry_count", retryCount),
			zap.Int("max_attempts", maxRetryAttempts),
			zap.Error(err))
		return false
	}

	logger.Warn("消息重試",
		zap.Int("attempt", retryCount+1),
		zap.Int("max_attempts", maxRetryAttempts),
		zap.Error(err))
	return true
}

func main() {
	// 初始化 Logger
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		panic(err)
	}
	defer logger.Sync()

	logger.Info("🚀 AIVA SSRF Detector 啟動中...")

	// 連接 RabbitMQ - 遵循 12-factor app 原則
	logger.Info("📡 連接 RabbitMQ...")
	rabbitmqURL := getRabbitMQURL()
	if rabbitmqURL == "" {
		logger.Fatal("AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set")
	}

	conn, err := amqp.Dial(rabbitmqURL)
	if err != nil {
		logger.Fatal("無法連接 RabbitMQ", zap.Error(err))
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		logger.Fatal("無法創建 Channel", zap.Error(err))
	}
	defer ch.Close()

	// 聲明任務隊列
	q, err := ch.QueueDeclare(
		taskQueue, // name
		true,      // durable
		false,     // delete when unused
		false,     // exclusive
		false,     // no-wait
		nil,       // arguments
	)
	if err != nil {
		logger.Fatal("無法聲明隊列", zap.Error(err))
	}

	// 設置 prefetch (一次只處理一個任務)
	err = ch.Qos(1, 0, false)
	if err != nil {
		logger.Fatal("無法設置 QoS", zap.Error(err))
	}

	// 消費訊息
	msgs, err := ch.Consume(
		q.Name, // queue
		"",     // consumer
		false,  // auto-ack
		false,  // exclusive
		false,  // no-local
		false,  // no-wait
		nil,    // args
	)
	if err != nil {
		logger.Fatal("無法開始消費訊息", zap.Error(err))
	}

	logger.Info("✅ 初始化完成,開始監聽任務...")

	// 初始化 SSRF 檢測器
	ssrfDetector := detector.NewSSRFDetector(logger)

	// 優雅關閉
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("🛑 收到關閉信號,正在關閉...")
		os.Exit(0)
	}()

	// 處理訊息
	forever := make(chan bool)

	go func() {
		for d := range msgs {
			var task detector.ScanTask
			err := json.Unmarshal(d.Body, &task)
			if err != nil {
				logger.Error("無法解析任務", zap.Error(err))
				d.Nack(false, false) // 拒絕並丟棄
				continue
			}

			logger.Info("📥 收到 SSRF 檢測任務", zap.String("task_id", task.TaskID))

			// 執行檢測
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			findings, err := ssrfDetector.Scan(ctx, &task)
			cancel()

			if err != nil {
				logger.Error("檢測失敗", zap.Error(err))

				// 實施重試邏輯，防止 poison pill 消息無限循環
				shouldRequeue := shouldRetryMessage(d, err)

				if shouldRequeue {
					logger.Warn("重新入隊消息進行重試")
					d.Nack(false, true) // 拒絕並重新排隊
				} else {
					logger.Error("達到最大重試次數，發送到死信隊列")
					d.Nack(false, false) // 拒絕並發送到死信隊列
				}
				continue
			}

			logger.Info("✅ 檢測完成",
				zap.String("task_id", task.TaskID),
				zap.Int("findings", len(findings)),
			)

			// 發送結果
			for _, finding := range findings {
				resultJSON, _ := json.Marshal(finding)

				// 聲明結果隊列
				_, err := ch.QueueDeclare(resultQueue, true, false, false, false, nil)
				if err != nil {
					logger.Error("無法聲明結果隊列", zap.Error(err))
					continue
				}

				err = ch.Publish(
					"",          // exchange
					resultQueue, // routing key
					false,       // mandatory
					false,       // immediate
					amqp.Publishing{
						ContentType: "application/json",
						Body:        resultJSON,
					},
				)

				if err != nil {
					logger.Error("無法發送結果", zap.Error(err))
				}
			}

			// 確認訊息
			d.Ack(false)
		}
	}()

	logger.Info("⏳ 等待任務...")
	<-forever
}
