package messaging

import (
	"encoding/json"

	"github.com/kyle0527/aiva/services/function/function_cspm_go/pkg/models"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

const FindingQueue = "findings"

// Publisher RabbitMQ 發布者
type Publisher struct {
	conn    *amqp.Connection
	channel *amqp.Channel
	logger  *zap.Logger
}

// NewPublisher 建立發布者
func NewPublisher(url string, logger *zap.Logger) (*Publisher, error) {
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, err
	}

	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, err
	}

	// 宣告 findings 佇列
	_, err = channel.QueueDeclare(
		FindingQueue,
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

	return &Publisher{
		conn:    conn,
		channel: channel,
		logger:  logger,
	}, nil
}

// PublishFinding 發布 Finding
func (p *Publisher) PublishFinding(finding models.FindingPayload) error {
	body, err := json.Marshal(finding)
	if err != nil {
		return err
	}

	err = p.channel.Publish(
		"",           // exchange
		FindingQueue, // routing key
		false,        // mandatory
		false,        // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        body,
		},
	)

	if err != nil {
		p.logger.Error("Failed to publish finding", zap.Error(err))
		return err
	}

	p.logger.Debug("Published finding", zap.String("finding_id", finding.FindingID))
	return nil
}

// Close 關閉連接
func (p *Publisher) Close() error {
	if p.channel != nil {
		p.channel.Close()
	}
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}
