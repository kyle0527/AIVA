package main

import (
    "context"
    "os"
    "aiva/scan/go_scanners/common"
    )


func main() {
    os.Setenv("SCAN_TASKS_QUEUE", envOr("SCAN_TASKS_SSRF_GO", "SCAN_TASKS_SSRF_GO"))
    if os.Getenv("SCAN_RESULTS_QUEUE") == "" {
        os.Setenv("SCAN_RESULTS_QUEUE", "SCAN_RESULTS")
    }
    s := NewSSRFScanner()
    worker, err := common.NewScannerWorker(s)
    if err != nil { panic(err) }
    if err := worker.Start(context.Background()); err != nil {
        panic(err)
    }
}

func envOr(k, d string) string {
    if v := os.Getenv(k); v != "" { return v }
    return d
}
