// AIVA SSRF Detector - Go Implementation
// 日期: 2025-10-13
// 功能: 高性能 SSRF 漏洞檢測

package main

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/kyle0527/aiva/services/function/function_ssrf_go/internal/detector"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

const (
	rabbitmqURL = "amqp://aiva:dev_password@localhost:5672/"
	taskQueue   = "task.function.ssrf"
	resultQueue = "results.function.finding"
)

var logger *zap.Logger

func main() {
	// 初始化 Logger
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		panic(err)
	}
	defer logger.Sync()

	logger.Info("🚀 AIVA SSRF Detector 啟動中...")

	// 連接 RabbitMQ
	logger.Info("📡 連接 RabbitMQ...")
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
				d.Nack(false, true) // 拒絕並重新排隊
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
