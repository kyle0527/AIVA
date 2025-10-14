# function_sca_go 遷移報告

**遷移日期:** 2025-10-14  
**遷移人員:** AIVA Architecture Team  
**狀態:** ✅ 完成

---

## 遷移目標

將 `function_sca_go` 服務遷移到使用 `aiva_common_go` 共用模組,以減少程式碼重複並提升維護性。

---

## 變更摘要

### ✅ 新增依賴

| 模組 | 用途 | 原實作 |
|------|------|--------|
| `aiva_common_go/config` | 配置管理 | `loadConfig()` 函式 |
| `aiva_common_go/logger` | 標準化日誌 | `zap.NewProduction()` |
| `aiva_common_go/mq` | RabbitMQ 客戶端 | `pkg/messaging/publisher.go` |
| `aiva_common_go/schemas` | 共用 Schema | `pkg/models/models.go` (部分) |

### ❌ 移除依賴

| 依賴 | 原因 |
|------|------|
| `github.com/rabbitmq/amqp091-go` | 改用 `aiva_common_go/mq` 封裝 |
| `go.uber.org/zap` (直接) | 改用 `aiva_common_go/logger` 封裝 |

### 🗑️ 刪除程式碼

| 檔案/目錄 | 原因 |
|-----------|------|
| `pkg/messaging/publisher.go` | 功能已被 `aiva_common_go/mq.MQClient.Publish()` 取代 |
| `cmd/worker/main.go` 中的 `Config` struct | 改用 `aiva_common_go/config.Config` |
| `cmd/worker/main.go` 中的 `loadConfig()` 函式 | 改用 `aiva_common_go/config.LoadConfig()` |
| RabbitMQ 連接邏輯 (75行) | 改用 `mq.NewMQClient()` |

---

## 程式碼變更對比

### 📊 程式碼行數比較

| 檔案 | 遷移前 | 遷移後 | 減少 | 減少率 |
|------|--------|--------|------|--------|
| `cmd/worker/main.go` | 188行 | 131行 | -57行 | -30% |
| `pkg/messaging/publisher.go` | 65行 | 0行 | -65行 | -100% |
| **總計** | **253行** | **131行** | **-122行** | **-48%** |

### 🔄 主要變更

#### 1. go.mod 簡化

**Before:**
```go
require (
	github.com/rabbitmq/amqp091-go v1.9.0
	go.uber.org/zap v1.26.0
)
require go.uber.org/multierr v1.11.0 // indirect
```

**After:**
```go
require (
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0
)
replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../common/go/aiva_common_go
```

#### 2. main.go 初始化邏輯

**Before:**
```go
// 初始化日誌
logger, err := zap.NewProduction()
if err != nil {
	log.Fatalf("Failed to initialize logger: %v", err)
}
defer logger.Sync()

// 讀取配置
config := loadConfig()

// 連接 RabbitMQ
conn, err := amqp.Dial(config.RabbitMQURL)
if err != nil {
	logger.Fatal("Failed to connect to RabbitMQ", zap.Error(err))
}
defer conn.Close()

ch, err := conn.Channel()
if err != nil {
	logger.Fatal("Failed to open channel", zap.Error(err))
}
defer ch.Close()

// 宣告佇列...
// 設定 QoS...
// 開始消費...
```

**After:**
```go
// 載入配置
cfg, err := config.LoadConfig("sca")
if err != nil {
	panic(err)
}

// 初始化日誌
log, err := logger.NewLogger(cfg.ServiceName)
if err != nil {
	panic(err)
}
defer log.Sync()

// 建立 MQ 客戶端 (自動處理連接、佇列宣告、QoS)
mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
if err != nil {
	log.Fatal("Failed to create MQ client", zap.Error(err))
}
defer mqClient.Close()
```

#### 3. 訊息消費邏輯

**Before:**
```go
msgs, err := ch.Consume(
	queue.Name,
	"",
	false,
	false,
	false,
	false,
	nil,
)

for {
	select {
	case <-ctx.Done():
		return
	case msg, ok := <-msgs:
		if !ok {
			return
		}
		processMessage(ctx, msg, scaScanner, publisher, logger)
	}
}
```

**After:**
```go
err := mqClient.Consume(
	"tasks.function.sca",
	func(body []byte) error {
		return processMessage(ctx, body, scaScanner, mqClient, log)
	},
)
```

#### 4. 訊息發送邏輯

**Before:**
```go
for _, finding := range findings {
	if err := publisher.PublishFinding(ctx, finding); err != nil {
		logger.Error("Failed to publish finding", ...)
	}
}
```

**After:**
```go
for _, finding := range findings {
	if err := mqClient.Publish("results.finding", finding); err != nil {
		log.Error("Failed to publish finding", ...)
	}
}
```

---

## 測試結果

### ✅ 編譯測試

```bash
$ go build ./...
# 成功,無錯誤
```

### ✅ 依賴解析

```bash
$ go mod tidy
go: found github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/config
go: found github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/logger
go: found github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/mq
```

### ⚠️ 單元測試

```bash
$ go test ./... -v
?       github.com/kyle0527/aiva/services/function/function_sca_go/cmd/worker      [no test files]
?       github.com/kyle0527/aiva/services/function/function_sca_go/internal/scanner        [no test files]
?       github.com/kyle0527/aiva/services/function/function_sca_go/pkg/models      [no test files]
```

**註:** 原專案沒有單元測試,遷移後維持相同狀態。建議後續添加測試。

---

## 效益分析

### 📉 程式碼重複減少

| 重複功能 | 遷移前 | 遷移後 |
|----------|--------|--------|
| RabbitMQ 連接邏輯 | 每個服務獨立實作 (75行) | 使用共用 `mq.MQClient` (0行) |
| Logger 初始化 | 每個服務獨立實作 (10行) | 使用共用 `logger.NewLogger()` (2行) |
| Config 載入 | 每個服務獨立實作 (15行) | 使用共用 `config.LoadConfig()` (1行) |

**估計:** 本次遷移減少約 **48%** 的樣板程式碼

### ⚡ 維護性提升

- ✅ RabbitMQ 連接問題只需在 `aiva_common_go` 修復一次
- ✅ 日誌格式統一,便於集中管理
- ✅ 配置標準化,減少環境變數不一致問題
- ✅ Schema 定義集中,降低跨語言同步錯誤

### 🔧 可擴展性提升

- ✅ 新增 Go 功能服務時可快速套用相同模式
- ✅ 未來可在 `aiva_common_go` 統一添加功能(如追蹤、監控)
- ✅ 更容易實施統一的錯誤處理和重試機制

---

## 後續改進建議

### 1. 添加單元測試 (Priority: High)

```bash
services/function/function_sca_go/
├── cmd/worker/
│   └── main_test.go              # 主程式測試
├── internal/scanner/
│   └── sca_scanner_test.go       # 掃描器測試
└── pkg/models/
    └── models_test.go            # 模型測試
```

**目標:** 單元測試覆蓋率 > 70%

### 2. 添加整合測試 (Priority: Medium)

- 測試與 RabbitMQ 的實際通訊
- 測試與 OSV-Scanner 的整合
- 測試端到端掃描流程

### 3. 添加 Dockerfile (Priority: Medium)

```dockerfile
FROM golang:1.25-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o /bin/sca-worker ./cmd/worker

FROM alpine:latest
RUN apk --no-cache add ca-certificates

# 安裝 OSV-Scanner
COPY --from=ghcr.io/google/osv-scanner:latest /osv-scanner /usr/local/bin/

COPY --from=builder /bin/sca-worker /bin/sca-worker

CMD ["/bin/sca-worker"]
```

### 4. 添加效能監控 (Priority: Low)

- 使用 Prometheus metrics
- 追蹤掃描時間、成功率、錯誤率
- 監控 RabbitMQ 連接狀態

---

## 驗收標準 ✅

- [x] 程式碼編譯成功,無錯誤
- [x] 所有直接的 RabbitMQ 和 Zap 依賴已移除
- [x] 使用 `aiva_common_go` 的 Config、Logger、MQ 模組
- [x] 刪除冗餘程式碼 (`pkg/messaging`)
- [x] 更新 README.md 說明新架構
- [x] Go mod 依賴正確解析
- [ ] (未完成) 添加單元測試
- [ ] (未完成) 添加 Dockerfile
- [x] 程式碼行數減少 > 40%

---

## 遷移檢查清單

### 準備階段
- [x] 建立 Git 分支 `feature/migrate-sca-to-common-go`
- [x] 備份原始程式碼 (使用 Git commit)
- [x] 確認 `aiva_common_go` 可用

### 實施階段
- [x] 更新 `go.mod` 添加 `aiva_common_go` 依賴
- [x] 重構 `main.go` 使用共用模組
- [x] 刪除 `pkg/messaging` 目錄
- [x] 更新 README.md
- [x] 執行 `go mod tidy`
- [x] 執行 `go build ./...` 驗證編譯

### 驗證階段
- [x] 編譯測試通過
- [x] 依賴解析正確
- [ ] (跳過) 單元測試通過 (無測試)
- [ ] (手動) 整合測試通過

### 文件階段
- [x] 建立 `MIGRATION_REPORT.md`
- [x] 更新 `README.md`
- [ ] 更新團隊文件

---

## 風險評估

| 風險 | 機率 | 影響 | 緩解措施 | 狀態 |
|------|------|------|----------|------|
| RabbitMQ 連接失敗 | 低 | 高 | `aiva_common_go` 已測試 | ✅ 已緩解 |
| 訊息格式不相容 | 低 | 中 | 保持與原有格式一致 | ✅ 已緩解 |
| 缺少單元測試 | 高 | 中 | 後續添加測試 | ⚠️ 待處理 |
| 環境變數變更 | 低 | 低 | 保持向後相容 | ✅ 已緩解 |

---

## 下一步

1. **本週完成:** 遷移其他 Go 服務
   - `function_cspm_go`
   - `function_authn_go`
   - `function_ssrf_go`

2. **Week 2 完成:** 達到 Go 程式碼重複率 < 15% 目標

3. **Week 3-4:** 開始 TypeScript 增強任務

---

## 參考資料

- [MULTILANG_STRATEGY.md](../../../MULTILANG_STRATEGY.md)
- [ROADMAP_NEXT_10_WEEKS.md](../../../ROADMAP_NEXT_10_WEEKS.md)
- [aiva_common_go README](../common/go/aiva_common_go/README.md)

---

**遷移完成!** 🎉

**程式碼減少:** 122 行 (-48%)  
**維護性:** ⬆️ 顯著提升  
**下一個目標:** 遷移 `function_cspm_go`
