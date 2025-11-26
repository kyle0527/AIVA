# Go 服務遷移工具使用指南

## 📑 目錄

- [� 目錄](#目錄)
- [�📁 文件說明](#文件說明)
  - [遷移腳本位置](#遷移腳本位置)
- [🚀 使用方法](#使用方法)
  - [方法 1: 遷移單個服務](#方法-1-遷移單個服務)
  - [方法 2: 批量遷移所有服務](#方法-2-批量遷移所有服務)
  - [方法 3: 驗證編譯狀態](#方法-3-驗證編譯狀態)
- [⚠️ 常見錯誤](#常見錯誤)
  - [錯誤 1: 找不到腳本](#錯誤-1-找不到腳本)
  - [錯誤 2: 服務目錄不存在](#錯誤-2-服務目錄不存在)
- [📋 完整遷移流程](#完整遷移流程)
  - [Step 1: 批量自動遷移](#step-1-批量自動遷移)
  - [Step 2: 手動修正 main.go](#step-2-手動修正-maingo)
  - [Step 3: 更新 Scanner 使用 schemas](#step-3-更新-scanner-使用-schemas)
  - [Step 4: 驗證編譯](#step-4-驗證編譯)
  - [Step 5: 批量驗證](#step-5-批量驗證)
- [✅ 遷移檢查清單](#遷移檢查清單)
- [🎯 快速參考](#快速參考)
  - [關鍵 API 變更](#關鍵-api-變更)
- [🆘 需要幫助？](#需要幫助)
- [🔗 相關資源](#相關資源)
  - [模組開發指南](#模組開發指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)
  - [服務文檔](#服務文檔)

---
## � 目錄

- [📁 文件說明](#-文件說明)
- [🚀 使用方式](#-使用方式)
- [🔍 驗證與測試](#-驗證與測試)
- [⚠️ 注意事項](#-注意事項)
- [🛠️ 故障排除](#-故障排除)
- [📊 遷移狀態](#-遷移狀態)
- [🔗 相關資源](#-相關資源)

## �📁 文件說明

### 遷移腳本位置

所有腳本都在 `services/function/` 目錄下：

```
services/function/
├── migrate_go_service.ps1          # 單個服務遷移
├── migrate_all_go_services.ps1     # 批量遷移所有服務
├── verify_go_builds.ps1            # 驗證編譯狀態
├── function_sca_go/                # ✅ 已遷移
├── function_cspm_go/               # ✅ 已遷移
├── function_authn_go/              # ⏳ 待遷移
└── function_ssrf_go/               # ⏳ 待遷移
```

---

## 🚀 使用方法

### 方法 1: 遷移單個服務

```powershell
# 進入 services/function 目錄
cd c:\AMD\AIVA\services\function

# 遷移指定服務
.\migrate_go_service.ps1 -ServiceName function_authn_go
```

### 方法 2: 批量遷移所有服務

```powershell
# 進入 services/function 目錄
cd c:\AMD\AIVA\services\function

# 批量遷移
.\migrate_all_go_services.ps1
```

### 方法 3: 驗證編譯狀態

```powershell
# 進入 services/function 目錄
cd c:\AMD\AIVA\services\function

# 驗證所有服務
.\verify_go_builds.ps1
```

---

## ⚠️ 常見錯誤

### 錯誤 1: 找不到腳本

**症狀:**
```
.\migrate_go_service.ps1: The term '.\migrate_go_service.ps1' is not recognized
```

**原因:** 在錯誤的目錄執行腳本

**解決方法:**
```powershell
# 確保在正確的目錄
cd c:\AMD\AIVA\services\function

# 再次執行
.\migrate_go_service.ps1 -ServiceName function_authn_go
```

### 錯誤 2: 服務目錄不存在

**症狀:**
```
❌ 服務目錄不存在: xxx
```

**解決方法:** 檢查服務名稱拼寫是否正確

---

## 📋 完整遷移流程

### Step 1: 批量自動遷移

```powershell
cd c:\AMD\AIVA\services\function
.\migrate_all_go_services.ps1
```

**自動完成:**
- ✅ 更新所有服務的 go.mod
- ✅ 添加 aiva_common_go 依賴
- ✅ 刪除 pkg/messaging 和 pkg/models
- ✅ 運行 go mod tidy

### Step 2: 手動修正 main.go

對於每個需要修正的服務，按照以下模板修改 `cmd/worker/main.go`:

```go
package main

import (
    "context"
    "encoding/json"
    "os"
    "os/signal"
    "syscall"

    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/config"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/logger"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/mq"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
    "go.uber.org/zap"
)

func main() {
    // 1. 載入配置（需要服務名參數）
    cfg, err := config.LoadConfig("service-name")
    if err != nil {
        panic(err)
    }
    
    // 2. 初始化日誌（需要服務名參數）
    log, err := logger.NewLogger(cfg.ServiceName)
    if err != nil {
        panic(err)
    }
    defer log.Sync()
    
    log.Info("Starting service",
        zap.String("service", cfg.ServiceName),
        zap.String("version", "2.0.0"))
    
    // 3. 初始化 MQ 客戶端
    // v2.0 架構已移除 RabbitMQ，使用直接數據合約通信
    // mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
    if err != nil {
        log.Fatal("MQ connection failed", zap.Error(err))
    }
    defer mqClient.Close()
    
    // 4. 優雅關閉
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    
    go func() {
        <-sigChan
        log.Info("Shutting down gracefully...")
        cancel()
    }()
    
    // 5. 開始消費（無需 ctx 參數）
    queueName := "tasks.function.xxx"
    err = mqClient.Consume(queueName, func(body []byte) error {
        return handleTask(ctx, body, log, mqClient)
    })
    
    if err != nil {
        log.Fatal("Consumer error", zap.Error(err))
    }
}

func handleTask(
    ctx context.Context,
    taskData []byte,
    log *zap.Logger,
    mqClient *mq.MQClient,
) error {
    // 解析任務
    var task schemas.FunctionTaskPayload
    if err := json.Unmarshal(taskData, &task); err != nil {
        log.Error("Failed to parse task", zap.Error(err))
        return err
    }
    
    log.Info("Processing task", zap.String("task_id", task.TaskID))
    
    // 執行業務邏輯
    findings := performScan(&task)
    
    // 發布結果
    for _, finding := range findings {
        if err := mqClient.Publish("findings.new", finding); err != nil {
            log.Error("Failed to publish finding", zap.Error(err))
            return err
        }
    }
    
    return nil
}
```

### Step 3: 更新 Scanner 使用 schemas

確保 scanner 文件使用正確的類型：

```go
import (
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
)

// 函數簽名使用指標
func (s *Scanner) Scan(ctx context.Context, task *schemas.FunctionTaskPayload) ([]*schemas.FindingPayload, error) {
    var findings []*schemas.FindingPayload
    
    // 業務邏輯...
    
    return findings, nil
}
```

### Step 4: 驗證編譯

```powershell
# 進入服務目錄
cd function_authn_go

# 編譯
go build ./...

# 測試
go test ./...

# 返回上層
cd ..
```

### Step 5: 批量驗證

```powershell
# 驗證所有服務
.\verify_go_builds.ps1
```

---

## ✅ 遷移檢查清單

對於每個服務，確保：

```markdown
- [ ] go.mod 包含 aiva_common_go 依賴（直接依賴）
- [ ] go.mod 有 replace 指令指向共享庫
- [ ] main.go 使用 config.LoadConfig(serviceName)
- [ ] main.go 使用 logger.NewLogger(serviceName)
- [ ] main.go 使用 mqClient.Consume(queue, handler) （無 ctx）
- [ ] scanner 使用 *schemas.FunctionTaskPayload（指標）
- [ ] scanner 返回 []*schemas.FindingPayload（指標切片）
- [ ] 刪除了 pkg/messaging
- [ ] 刪除了 pkg/models
- [ ] go build ./... 成功
- [ ] go test ./... 通過
- [ ] 無編譯警告
```

---

## 🎯 快速參考

### 關鍵 API 變更

| 功能 | 舊方式 | 新方式 |
|------|--------|--------|
| 配置載入 | 手動解析環境變量 | `config.LoadConfig("service-name")` |
| 日誌初始化 | 手動配置 zap | `logger.NewLogger(cfg.ServiceName)` |
| MQ 連接 | 手動創建連接 | `mq.NewMQClient(url, log)` |
| 消費消息 | `Consume(ctx, queue, handler)` | `Consume(queue, handler)` ← 無 ctx |
| Task 類型 | `schemas.TaskPayload` | `*schemas.FunctionTaskPayload` ← 指標 |
| 返回類型 | `[]schemas.FindingPayload` | `[]*schemas.FindingPayload` ← 指標切片 |

---

## 🆘 需要幫助？

如果遇到問題：

1. 查看已遷移服務的實例：
   - `function_sca_go/cmd/worker/main.go`
   - `function_cspm_go/cmd/worker/main.go`

2. 檢查共享庫文檔：
   - `services/function/common/go/aiva_common_go/README.md`

3. 運行驗證腳本診斷：
   ```powershell
   .\verify_go_builds.ps1
   ```

---

**最後更新:** 2025-10-14  
**維護者:** AIVA 架構團隊


---

## 🔗 相關資源

### 模組開發指南
- 📖 [Python 開發指南](./PYTHON_DEVELOPMENT_GUIDE.md)
- 📖 [Go 開發指南](./GO_DEVELOPMENT_GUIDE.md)
- 📖 [Rust 開發指南](./RUST_DEVELOPMENT_GUIDE.md)
- 📖 [AI 引擎指南](./AI_ENGINE_GUIDE.md)
- 📖 [功能模組開發指南](./FEATURE_MODULES_DEVELOPMENT_GUIDE.md)
- 📖 [模組遷移指南](./MODULE_MIGRATION_GUIDE.md)

### 架構指南
- 📖 [跨語言 Schema 指南](../architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md)
- 📖 [兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 開發指南
- 📖 [開發快速指南](../development/DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

### 服務文檔
- 🔧 [Features 模組](../../services/features/README.md)
- 🔧 [Scan 引擎文檔](../../services/scan/README.md)

