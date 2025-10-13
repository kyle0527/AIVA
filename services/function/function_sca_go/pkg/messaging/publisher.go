package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"

	"github.com/kyle0527/aiva/services/function/function_sca_go/pkg/models"
)

// Publisher 訊息發布器
type Publisher struct {
	channel *amqp.Channel
	logger  *zap.Logger
}

// NewPublisher 建立發布器
func NewPublisher(channel *amqp.Channel, logger *zap.Logger) *Publisher {
	return &Publisher{
		channel: channel,
		logger:  logger,
	}
}

// PublishFinding 發布 Finding
func (p *Publisher) PublishFinding(ctx context.Context, finding models.FindingPayload) error {
	// 序列化
	body, err := json.Marshal(finding)
	if err != nil {
		return fmt.Errorf("failed to marshal finding: %w", err)
	}

	// 發布到 RabbitMQ
	err = p.channel.PublishWithContext(
		ctx,
		"",                // exchange
		"results.finding", // routing key (topic)
		false,             // mandatory
		false,             // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			Body:         body,
			DeliveryMode: amqp.Persistent,
			Timestamp:    time.Now(),
			MessageId:    finding.FindingID,
		},
	)

	if err != nil {
		return fmt.Errorf("failed to publish: %w", err)
	}

	p.logger.Info("Published finding",
		zap.String("finding_id", finding.FindingID),
		zap.String("severity", finding.Severity),
	)

	return nil
}
