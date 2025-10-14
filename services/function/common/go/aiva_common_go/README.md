# AIVA Common Go Library

## 目的
為所有 Go 微服務提供統一的基礎設施程式碼,消除重複實作。

## 提供的功能

### 1. RabbitMQ 連接管理
- 自動重連機制
- 統一的消費者/生產者介面
- 連接池管理

### 2. 日誌系統
- 基於 zap 的結構化日誌
- 統一的日誌格式和等級
- 與 trace_id 整合

### 3. 配置管理
- 從環境變數載入配置
- 驗證配置的完整性

### 4. Schema 定義
- 與 Python aiva_common.schemas 對應的 Go struct
- JSON 序列化/反序列化

## 使用方式

```go
import "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go"

// 初始化
config := aiva_common_go.LoadConfig()
logger := aiva_common_go.NewLogger("function_sca")
mqClient := aiva_common_go.NewMQClient(config.RabbitMQURL, logger)

// 消費訊息
mqClient.Consume("task.function.sca", func(msg []byte) error {
    // 處理邏輯
    return nil
})
```

## 遷移指南

參考 `docs/MIGRATION_GUIDE.md` 了解如何將現有服務遷移到使用共用函式庫。
