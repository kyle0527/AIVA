package main

import (
    "encoding/json"
    "log"
    "os"

    "aiva/function_authn_go/internal"
)

func main() {
    b, err := internal.DialBroker()
    if err != nil {
        log.Fatalf("broker connect failed: %v", err)
    }
    defer b.Close()

    taskQueue := internal.TopicFromEnv("TOPIC_TASK_AUTHN", "TASK_FUNCTION_AUTHN")
    findingQueue := internal.TopicFromEnv("TOPIC_FINDING", "FINDING_DETECTED")
    statusQueue := internal.TopicFromEnv("TOPIC_STATUS", "TASK_STATUS")

    msgs, err := b.Subscribe(taskQueue)
    if err != nil {
        log.Fatalf("subscribe failed: %v", err)
    }

    cfg := internal.DefaultConfig()
    eng := internal.NewAuthnEngine(cfg)

    log.Printf("AUTHN_GO Worker started. TaskQueue=%s", taskQueue)

    for d := range msgs {
        var task internal.AuthnTask
        if err := json.Unmarshal(d.Body, &task); err != nil {
            _ = b.PublishStatus(statusQueue, "ERROR", "invalid task format")
            continue
        }
        _ = b.PublishStatus(statusQueue, "IN_PROGRESS", "authn task started")
        findings := eng.RunTests(task)
        for _, f := range findings {
            _ = b.PublishFinding(findingQueue, f)
        }
        _ = b.PublishStatus(statusQueue, "COMPLETED", "authn task done")
    }
}
