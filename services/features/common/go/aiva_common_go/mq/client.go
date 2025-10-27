package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// MQClient 提供統一的 RabbitMQ 操作介面
type MQClient struct {
	conn    *amqp.Connection
	channel *amqp.Channel
	logger  *zap.Logger
	url     string
}

// Config RabbitMQ 配置
type Config struct {
	URL            string
	ReconnectDelay time.Duration
	PrefetchCount  int
	AutoAck        bool
}

// NewMQClient 建立新的 MQ 客戶端
func NewMQClient(url string, logger *zap.Logger) (*MQClient, error) {
	client := &MQClient{
		url:    url,
		logger: logger,
	}

	if err := client.connect(); err != nil {
		return nil, err
	}

	return client, nil
}

// connect 建立連接
func (c *MQClient) connect() error {
	c.logger.Info("連接 RabbitMQ...", zap.String("url", maskPassword(c.url)))

	conn, err := amqp.Dial(c.url)
	if err != nil {
		return fmt.Errorf("無法連接 RabbitMQ: %w", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return fmt.Errorf("無法創建 Channel: %w", err)
	}

	c.conn = conn
	c.channel = ch

	c.logger.Info("✅ RabbitMQ 連接成功")
	return nil
}

// DeclareQueue 聲明隊列
func (c *MQClient) DeclareQueue(name string) error {
	_, err := c.channel.QueueDeclare(
		name,  // name
		true,  // durable
		false, // delete when unused
		false, // exclusive
		false, // no-wait
		nil,   // arguments
	)
	return err
}

// Consume 消費訊息
func (c *MQClient) Consume(queueName string, handler func([]byte) error) error {
	if err := c.DeclareQueue(queueName); err != nil {
		return fmt.Errorf("聲明隊列失敗: %w", err)
	}

	// 設置 Qos
	if err := c.channel.Qos(1, 0, false); err != nil {
		return fmt.Errorf("設置 Qos 失敗: %w", err)
	}

	msgs, err := c.channel.Consume(
		queueName,
		"",    // consumer
		false, // auto-ack (設為 false,手動確認)
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // args
	)
	if err != nil {
		return fmt.Errorf("註冊消費者失敗: %w", err)
	}

	c.logger.Info("開始消費訊息", zap.String("queue", queueName))

	for msg := range msgs {
		c.logger.Debug("收到訊息",
			zap.String("queue", queueName),
			zap.Int("size", len(msg.Body)))

		if err := handler(msg.Body); err != nil {
			c.logger.Error("處理訊息失敗",
				zap.Error(err),
				zap.String("queue", queueName))

			// 實施重試邏輯，防止 poison pill 消息無限循環
			shouldRequeue := c.shouldRetryMessage(msg, err)

			if shouldRequeue {
				c.logger.Warn("重新入隊消息進行重試")
				msg.Nack(false, true) // Nack 訊息,重新放回隊列
			} else {
				c.logger.Error("達到最大重試次數，發送到死信隊列")
				msg.Nack(false, false) // Nack 訊息，發送到死信隊列
			}
		} else {
			// Ack 訊息
			msg.Ack(false)
		}
	}

	return nil
}

// Publish 發布訊息
func (c *MQClient) Publish(queueName string, body interface{}) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("序列化失敗: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return c.channel.PublishWithContext(
		ctx,
		"",        // exchange
		queueName, // routing key
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			Body:         data,
			DeliveryMode: amqp.Persistent,
			Timestamp:    time.Now(),
		},
	)
}

// Close 關閉連接
func (c *MQClient) Close() error {
	if c.channel != nil {
		c.channel.Close()
	}
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// maskPassword 隱藏密碼用於日誌
func maskPassword(rawURL string) string {
	if rawURL == "" {
		return ""
	}

	// 使用標準 net/url 套件解析 URL
	u, err := url.Parse(rawURL)
	if err != nil {
		// 如果解析失敗,返回通用遮蔽字串
		return "amqp://***:***@..."
	}

	// 如果沒有使用者資訊,直接返回原 URL
	if u.User == nil {
		return rawURL
	}

	// 遮蔽密碼
	username := u.User.Username()
	if username == "" {
		return rawURL
	}

	// 重建 URL,保留使用者名稱但遮蔽密碼
	u.User = url.UserPassword(username, "***")
	return u.String()
}

// shouldRetryMessage 檢查消息是否應該重試
// 實施統一的重試策略，防止 poison pill 消息無限循環
func (c *MQClient) shouldRetryMessage(delivery amqp.Delivery, err error) bool {
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
		c.logger.Error("消息已達到最大重試次數，發送到死信隊列",
			zap.Int("retry_count", retryCount),
			zap.Int("max_attempts", maxRetryAttempts),
			zap.Error(err))
		return false
	}

	c.logger.Warn("消息重試",
		zap.Int("attempt", retryCount+1),
		zap.Int("max_attempts", maxRetryAttempts),
		zap.Error(err))
	return true
}
