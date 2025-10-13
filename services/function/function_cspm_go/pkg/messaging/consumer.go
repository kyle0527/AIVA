package messaging

import (
	"context"
	"encoding/json"

	"github.com/kyle0527/aiva/services/function/function_cspm_go/pkg/models"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// Consumer RabbitMQ 消費者
type Consumer struct {
	conn      *amqp.Connection
	channel   *amqp.Channel
	queueName string
	logger    *zap.Logger
}

// NewConsumer 建立消費者
func NewConsumer(url, queueName string, logger *zap.Logger) (*Consumer, error) {
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, err
	}

	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, err
	}

	// 宣告佇列
	_, err = channel.QueueDeclare(
		queueName,
		true,  // durable
		false, // delete when unused
		false, // exclusive
		false, // no-wait
		nil,   // arguments
	)
	if err != nil {
		channel.Close()
		conn.Close()
		return nil, err
	}

	return &Consumer{
		conn:      conn,
		channel:   channel,
		queueName: queueName,
		logger:    logger,
	}, nil
}

// Consume 開始消費訊息
func (c *Consumer) Consume(ctx context.Context, handler func([]byte) error) error {
	msgs, err := c.channel.Consume(
		c.queueName,
		"",    // consumer
		false, // auto-ack
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // args
	)
	if err != nil {
		return err
	}

	c.logger.Info("Started consuming messages", zap.String("queue", c.queueName))

	for {
		select {
		case <-ctx.Done():
			c.logger.Info("Context cancelled, stopping consumer")
			return nil

		case msg, ok := <-msgs:
			if !ok {
				c.logger.Warn("Message channel closed")
				return nil
			}

			// 處理訊息
			if err := handler(msg.Body); err != nil {
				c.logger.Error("Handler error", zap.Error(err))
				msg.Nack(false, true) // requeue
			} else {
				msg.Ack(false)
			}
		}
	}
}

// ParseTask 解析任務
func ParseTask(data []byte) (models.FunctionTaskPayload, error) {
	var task models.FunctionTaskPayload
	err := json.Unmarshal(data, &task)
	return task, err
}

// Close 關閉連接
func (c *Consumer) Close() error {
	if c.channel != nil {
		c.channel.Close()
	}
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
