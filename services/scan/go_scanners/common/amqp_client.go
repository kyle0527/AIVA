package common

import (
    "os"
    "time"

    amqp "github.com/rabbitmq/amqp091-go"
    "go.uber.org/zap"
)

type ScannerAMQPClient struct {
    Conn   *amqp.Connection
    Chan   *amqp.Channel
    Logger *zap.Logger
}

func NewScannerAMQPClient() (*ScannerAMQPClient, error) {
    url := os.Getenv("AIVA_AMQP_URL")
    if url == "" {
        url = "amqp://guest:guest@rabbitmq:5672/"
    }
    conn, err := amqp.DialConfig(url, amqp.Config{Properties: amqp.Table{
        "connection_name": "go_scanner",
    }})
    if err != nil {
        return nil, err
    }
    ch, err := conn.Channel()
    if err != nil {
        return nil, err
    }
    _ = ch.Qos(50, 0, false)
    logger, _ := zap.NewProduction(zap.AddCallerSkip(1))
    return &ScannerAMQPClient{Conn: conn, Chan: ch, Logger: logger}, nil
}

func (c *ScannerAMQPClient) DeclareQueue(name string) error {
    _, err := c.Chan.QueueDeclare(name, true, false, false, false, nil)
    return err
}

func (c *ScannerAMQPClient) Consume(name string) (<-chan amqp.Delivery, error) {
    return c.Chan.Consume(name, "", false, false, false, false, nil)
}

func (c *ScannerAMQPClient) Publish(name string, body []byte) error {
    return c.Chan.Publish("", name, false, false, amqp.Publishing{
        ContentType: "application/json",
        Body:        body,
        Timestamp:   time.Now(),
    })
}

func (c *ScannerAMQPClient) Close() error {
    if c.Chan != nil {
        _ = c.Chan.Close()
    }
    if c.Conn != nil {
        return c.Conn.Close()
    }
    return nil
}
