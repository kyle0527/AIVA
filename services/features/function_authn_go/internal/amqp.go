package internal

import (
    "encoding/json"
    "fmt"
    "os"

    amqp "github.com/rabbitmq/amqp091-go"
)

type Broker struct {
    ch *amqp.Channel
}

func DialBroker() (*Broker, error) {
    url := os.Getenv("AMQP_URL")
    if url == "" {
        url = os.Getenv("AIVA_AMQP_URL")
    }
    if url == "" {
        url = "amqp://guest:guest@localhost:5672/"
    }
    conn, err := amqp.Dial(url)
    if err != nil {
        return nil, err
    }
    ch, err := conn.Channel()
    if err != nil {
        return nil, err
    }
    return &Broker{ch: ch}, nil
}

func (b *Broker) Subscribe(queue string) (<-chan amqp.Delivery, error) {
    _, err := b.ch.QueueDeclare(queue, true, false, false, false, nil)
    if err != nil {
        return nil, err
    }
    return b.ch.Consume(queue, "", true, false, false, false, nil)
}

func (b *Broker) Publish(queue string, payload any) error {
    _, err := b.ch.QueueDeclare(queue, true, false, false, false, nil)
    if err != nil {
        return err
    }
    data, _ := json.Marshal(payload)
    return b.ch.Publish("", queue, false, false, amqp.Publishing{
        ContentType: "application/json",
        Body:        data,
    })
}

func (b *Broker) Close() error {
    return b.ch.Close()
}

type FindingMessage struct {
    Topic   string      `json:"topic"`
    Payload interface{} `json:"payload"`
}

type StatusPayload struct {
    Status string `json:"status"`
    Note   string `json:"note,omitempty"`
}

func (b *Broker) PublishFinding(topic string, payload any) error {
    return b.Publish(topic, FindingMessage{Topic: topic, Payload: payload})
}

func (b *Broker) PublishStatus(topic, status, note string) error {
    return b.Publish(topic, FindingMessage{Topic: topic, Payload: StatusPayload{Status: status, Note: note}})
}

func TopicFromEnv(envName, def string) string {
    v := os.Getenv(envName)
    if v == "" {
        return def
    }
    return v
}

func MustEnv(k, def string) string {
    v := os.Getenv(k)
    if v == "" {
        return def
    }
    return v
}

func Debugf(msg string, args ...interface{}) {
    fmt.Fprintf(os.Stdout, msg+"\n", args...)
}
