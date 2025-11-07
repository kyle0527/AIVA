package common

import (
	"context"
	"encoding/json"
	"os"

	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

type BaseScanner interface {
    Scan(ctx context.Context, task ScanTask) ScanResult
    GetName() string
    GetVersion() string
    GetCapabilities() []string
    HealthCheck() error
}

type ScannerWorker struct {
    Scanner BaseScanner
    Client  *ScannerAMQPClient
    Logger  *zap.Logger
    TaskQueue   string
    ResultQueue string
}

func NewScannerWorker(scanner BaseScanner) (*ScannerWorker, error) {
    client, err := NewScannerAMQPClient()
    if err != nil {
        return nil, err
    }
    logger, _ := zap.NewProduction()
    taskQ := os.Getenv("SCAN_TASKS_QUEUE")
    if taskQ == "" {
        taskQ = "SCAN_TASKS_GENERIC"
    }
    resultQ := os.Getenv("SCAN_RESULTS_QUEUE")
    if resultQ == "" {
        resultQ = "SCAN_RESULTS"
    }
    return &ScannerWorker{
        Scanner: scanner,
        Client: client,
        Logger: logger,
        TaskQueue: taskQ,
        ResultQueue: resultQ,
    }, nil
}

func (w *ScannerWorker) Start(ctx context.Context) error {
    if err := w.Client.DeclareQueue(w.TaskQueue); err != nil {
        return err
    }
    if err := w.Client.DeclareQueue(w.ResultQueue); err != nil {
        return err
    }
    msgs, err := w.Client.Consume(w.TaskQueue)
    if err != nil {
        return err
    }
    w.Logger.Info("Scanner worker started", zap.String("queue", w.TaskQueue), zap.String("scanner", w.Scanner.GetName()))
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case d := <-msgs:
            if err := w.handleMessage(d); err != nil {
                w.Logger.Error("handleMessage error", zap.Error(err))
                _ = d.Nack(false, false)
            } else {
                _ = d.Ack(false)
            }
        }
    }
}

func (w *ScannerWorker) handleMessage(d amqp.Delivery) error {
    var task ScanTask
    if err := json.Unmarshal(d.Body, &task); err != nil {
        return err
    }
    res := w.Scanner.Scan(context.Background(), task)
    payload := ToJSON(res)
    return w.Client.Publish(w.ResultQueue, payload)
}
