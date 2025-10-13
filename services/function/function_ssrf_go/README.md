# AIVA SSRF Detector (Go)

使用 Go 實現的高性能 SSRF (Server-Side Request Forgery) 漏洞檢測器。

## 特性

- ✅ 高並發檢測 (Go goroutines)
- ✅ 內網 IP 範圍阻擋
- ✅ 支援 AWS/GCP Metadata 檢測
- ✅ 自動重試與超時控制
- ✅ 通過 RabbitMQ 接收任務

## 安裝

```powershell
# 初始化 Go 模組
go mod download

# 編譯
go build -o ssrf_worker.exe cmd/worker/main.go
```

## 運行

```powershell
# 直接運行
go run cmd/worker/main.go

# 或執行編譯後的程式
.\ssrf_worker.exe
```

## 環境變數

```env
RABBITMQ_URL=amqp://aiva:dev_password@localhost:5672/
```

## 檢測 Payloads

- AWS IMDS: `http://169.254.169.254/latest/meta-data/`
- GCP Metadata: `http://metadata.google.internal/computeMetadata/v1/`
- Localhost: `http://127.0.0.1/`, `http://localhost/`
- Private IPs: `http://192.168.1.1/`, `http://10.0.0.1/`

## 任務格式

```json
{
  "task_id": "task_xxx",
  "module": "ssrf",
  "target": "https://vulnerable-site.com/fetch",
  "metadata": {}
}
```

## 結果格式

```json
{
  "task_id": "task_xxx",
  "module": "ssrf",
  "severity": "HIGH",
  "title": "SSRF Vulnerability Detected",
  "summary": "成功訪問內網資源: http://169.254.169.254/...",
  "evidence": "URL: ...\nStatus: 200\n...",
  "cwe_ids": ["CWE-918"]
}
```

## 性能

- 單次檢測: <1 秒
- 並發能力: 1000+ 任務/秒
- 記憶體佔用: ~10 MB
