# Function CSPM Go 遷移報告

**遷移日期**: 2025年10月14日  
**遷移人員**: AI Assistant  
**遷移狀態**: ✅ 完成

## 📋 遷移概述

將 `function_cspm_go` 服務遷移至使用統一的 `aiva_common_go` 共享庫，消除重複代碼，提升代碼一致性和可維護性。

## 🎯 遷移目標

- ✅ 使用 `aiva_common_go/config` 替代本地配置
- ✅ 使用 `aiva_common_go/logger` 替代本地 zap logger 設置
- ✅ 使用 `aiva_common_go/mq` 替代本地 RabbitMQ 客戶端
- ✅ 使用 `aiva_common_go/schemas` 統一數據結構
- ✅ 刪除重複的 `pkg/messaging` 和 `pkg/models` 代碼

## 📊 代碼變更統計

### 文件修改

| 文件 | 修改類型 | 變更說明 |
|------|---------|---------|
| `go.mod` | 更新 | 添加 aiva_common_go 直接依賴 |
| `cmd/worker/main.go` | 重構 | 使用共享模組，修復函數簽名 |
| `internal/scanner/cspm_scanner.go` | 重構 | 使用統一 schemas，定義本地 CSPMMisconfig |
| `pkg/messaging/` | 刪除 | 已被 aiva_common_go/mq 替代 |
| `pkg/models/` | 刪除 | 已被 aiva_common_go/schemas 替代 |

### 代碼行數變化

```
刪除文件:
- pkg/messaging/consumer.go
- pkg/messaging/publisher.go
- pkg/models/models.go

總計減少: ~150+ 行重複代碼
```

## 🔧 技術細節

### 1. go.mod 更新

```go
require (
    github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0
    github.com/rabbitmq/amqp091-go v1.10.0
    go.uber.org/zap v1.26.0
)

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../common/go/aiva_common_go
```

### 2. main.go 重構

**修復函數簽名**:
- `config.LoadConfig()` → `config.LoadConfig("function-cspm")` ✅
- `logger.NewLogger()` → `logger.NewLogger(cfg.ServiceName)` ✅
- `mqClient.Consume(ctx, queueName, handler)` → `mqClient.Consume(queueName, handler)` ✅

**錯誤處理**:
```go
cfg, err := config.LoadConfig("function-cspm")
if err != nil {
    panic(err)
}

log, err := logger.NewLogger(cfg.ServiceName)
if err != nil {
    panic(err)
}
```

### 3. Scanner 結構調整

**定義本地 CSPMMisconfig 類型**:
```go
type CSPMMisconfig struct {
    ID          string
    Title       string
    Description string
    Severity    string
    Resolution  string
    FilePath    string
    ResourceID  string
}
```

**映射到統一 FindingPayload**:
- 使用 `Metadata` 字段存儲 CSPM 特定數據（provider, rule_id, file_path）
- 正確使用指標類型（`*FindingEvidence`, `*FindingImpact`, `*FindingRecommendation`）
- 使用 `stringPtr()` 輔助函數處理可選字段

### 4. 編譯錯誤修復

修復了 25 個編譯錯誤:
- ✅ 13 個 `schemas.CSPMMisconfig undefined` → 定義本地類型
- ✅ 3 個 函數簽名錯誤 → 修正參數
- ✅ 7 個 字段不匹配 → 使用正確的 schema 字段
- ✅ 2 個 類型錯誤 → 使用指標類型

## ✅ 驗證結果

### 編譯驗證
```bash
PS> go build ./...
✅ 編譯成功，無錯誤
```

### 依賴驗證
```bash
PS> go mod tidy
✅ 依賴整理成功
```

### 靜態檢查
```
✅ 無編譯錯誤
✅ 無警告
✅ go.mod 格式正確
```

## 📈 改進效果

### 代碼質量
- ✅ **消除重複代碼**: 刪除 ~150+ 行重複的 RabbitMQ 和 models 代碼
- ✅ **提升一致性**: 使用統一的配置、日誌、MQ 模式
- ✅ **改善可維護性**: 共享庫集中管理，修改一處即可影響所有服務

### 架構優化
- ✅ **模組化**: 清晰的依賴關係
- ✅ **可擴展性**: 新服務可直接使用 aiva_common_go
- ✅ **類型安全**: 統一的 schemas 避免類型不匹配

## 🚀 後續工作

### 下一步任務
1. **遷移 function_authn_go** (Week 1 Task 1.3)
2. **遷移 function_ssrf_go** (Week 1 Task 1.4)
3. **整合測試**: 驗證 CSPM 服務在完整系統中的運行

### 建議
- 考慮將 `CSPMMisconfig` 添加到 `aiva_common_go/schemas` 作為共享類型
- 為 CSPM 特定字段考慮擴展統一 schema（如果其他服務也需要）

## 📝 經驗總結

### 成功經驗
1. **先分析再行動**: 完整分析 25 個錯誤後一次性修復
2. **查閱官方文檔**: 確認 Go module 最佳實踐
3. **分步驟執行**: go.mod → main.go → scanner → cleanup
4. **充分驗證**: 每步都運行編譯檢查

### 遇到的挑戰
1. **文件損壞**: Scanner 文件曾出現重複內容，由用戶手動還原
2. **字段映射**: 統一 schema 缺少 CSPM 特定字段，使用 Metadata 解決
3. **函數簽名**: API 變化需要仔細對照源代碼

## 🎉 結論

`function_cspm_go` 成功遷移至 `aiva_common_go`，達成以下目標:
- ✅ 消除重複代碼
- ✅ 使用統一架構
- ✅ 提升可維護性
- ✅ 編譯和驗證通過

**遷移狀態**: 完成 ✅  
**下一步**: 繼續 Week 1 Task 1.3 - 遷移 function_authn_go
