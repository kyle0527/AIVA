# Function-SCA (Software Composition Analysis)

Go 實現的軟體組成分析模組，使用 Google OSV-Scanner 檢測第三方依賴套件的已知漏洞。

## 功能

- 支援多種套件管理系統（Node.js, Python, Go, Rust, Java, PHP, Ruby）
- 整合 Google OSV-Scanner
- 自動偵測專案中的套件管理檔案
- 與 AIVA RabbitMQ 訊息系統整合
- 高併發掃描

## 架構

```
function_sca_go/
├── cmd/
│   └── worker/
│       └── main.go              # 主程式入口
├── internal/
│   └── scanner/
│       └── sca_scanner.go       # SCA 掃描邏輯
├── pkg/
│   ├── messaging/
│   │   └── publisher.go         # RabbitMQ 發布器
│   └── models/
│       └── models.go            # 數據模型
├── go.mod
└── README.md
```

## 依賴

- Go 1.25.0+
- Google OSV-Scanner (需預先安裝)
- RabbitMQ

## 安裝 OSV-Scanner

```bash
# 使用 Go 安裝
go install github.com/google/osv-scanner/cmd/osv-scanner@latest

# 或使用預編譯二進制檔
# https://github.com/google/osv-scanner/releases
```

## 建置

```bash
cd services/function/function_sca_go

# 下載依賴
go mod download

# 建置
go build -o bin/sca-worker cmd/worker/main.go
```

## 執行

```bash
# 設定環境變數
export RABBITMQ_URL="amqp://guest:guest@localhost:5672/"

# 執行
./bin/sca-worker
```

## 環境變數

| 變數名稱 | 說明 | 預設值 |
|---------|------|--------|
| `RABBITMQ_URL` | RabbitMQ 連線 URL | `amqp://guest:guest@localhost:5672/` |

## 支援的套件管理檔案

- **Node.js**: package.json, package-lock.json, yarn.lock, pnpm-lock.yaml
- **Python**: pyproject.toml, requirements.txt, Pipfile.lock, poetry.lock
- **Go**: go.mod, go.sum
- **Rust**: Cargo.toml, Cargo.lock
- **Java**: pom.xml, build.gradle
- **PHP**: composer.json, composer.lock
- **Ruby**: Gemfile.lock

## 訊息格式

### 輸入 (Topic: `tasks.function.sca`)

```json
{
  "task_id": "task_123",
  "function_type": "SCA",
  "target": {
    "url": "/path/to/project or https://github.com/user/repo"
  }
}
```

### 輸出 (Topic: `results.finding`)

```json
{
  "finding_id": "finding_sca_123",
  "task_id": "task_123",
  "vulnerability": {
    "type": "SCA",
    "name": "CVE-2024-1234 in lodash@4.17.20",
    "description": "Prototype Pollution vulnerability",
    "cve_id": "CVE-2024-1234",
    "ghsa_id": "GHSA-xxxx-yyyy-zzzz"
  },
  "severity": "HIGH",
  "confidence": "FIRM",
  "target": {
    "url": "/path/to/package.json",
    "parameter": "lodash@4.17.20"
  },
  "evidence": {
    "request": "/path/to/package.json",
    "response": "Package: lodash@4.17.20\nEcosystem: npm",
    "payload": "CVE-2024-1234",
    "proof_of_concept": "詳細的漏洞說明..."
  },
  "recommendation": {
    "remediation": "更新 lodash 到 4.17.21 或更高版本",
    "references": [
      "https://nvd.nist.gov/vuln/detail/CVE-2024-1234"
    ]
  },
  "tags": ["SCA", "Dependency", "npm", "CVE-2024-1234"]
}
```

## Docker 支援

```dockerfile
FROM golang:1.25-alpine AS builder

WORKDIR /app
COPY go.* ./
RUN go mod download

COPY . .
RUN go build -o bin/sca-worker cmd/worker/main.go

FROM alpine:latest
RUN apk add --no-cache git ca-certificates

# 安裝 OSV-Scanner
COPY --from=ghcr.io/google/osv-scanner:latest /osv-scanner /usr/local/bin/

COPY --from=builder /app/bin/sca-worker /usr/local/bin/

CMD ["sca-worker"]
```

## 測試

```bash
# 單元測試
go test ./...

# 整合測試
go test -tags=integration ./...
```

## 效能

- 掃描速度：約 100-500 個套件/秒（取決於網路和 OSV API）
- 記憶體使用：約 50-100 MB
- 併發支援：預設 QoS=1，可調整
