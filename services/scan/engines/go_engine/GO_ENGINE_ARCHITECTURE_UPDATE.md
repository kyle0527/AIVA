# Go 引擎架構更新說明

## 背景

Go 引擎的各個掃描器（SSRF、SCA、CSPM）中存在 RabbitMQ Worker 相關的 TODO，這些是研發初期的計劃。

## 新架構方案

根據 AIVA 系統的架構升級（移除 RabbitMQ，採用 AI 直接指揮），Go 引擎不再需要實現獨立的 Worker 模式。

### 調用方式

Go 引擎通過以下方式被調用：

1. **Python 調度器** (`services/scan/engines/go_engine/dispatcher/python_bridge.py`)
   - 提供統一的 Python 接口
   - 負責編譯 Go 掃描器
   - 通過命令行參數調用編譯後的二進制
   - 解析 JSON 輸出並返回結果

2. **命令行模式**
   ```bash
   # SSRF 掃描器
   ./ssrf-scanner --target https://example.com --output json
   
   # SCA 掃描器
   ./sca-scanner --project-path /path/to/project --output json
   
   # CSPM 掃描器
   ./cspm-scanner --cloud-provider aws --region us-east-1 --output json
   ```

### 需要實現的命令行參數

為了配合新架構，Go 掃描器需要實現以下功能：

#### SSRF Scanner (`cmd/ssrf-scanner/main.go`)

```go
package main

import (
    "flag"
    "fmt"
    "os"
    "encoding/json"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/ssrf/detector"
    "go.uber.org/zap"
)

func main() {
    // 解析命令行參數
    target := flag.String("target", "", "Target URL to scan")
    outputFormat := flag.String("output", "json", "Output format (json/text)")
    timeout := flag.Int("timeout", 30, "Scan timeout in seconds")
    flag.Parse()
    
    if *target == "" {
        fmt.Fprintf(os.Stderr, "Error: --target is required\n")
        flag.Usage()
        os.Exit(1)
    }
    
    // 初始化 Logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // 創建掃描器並執行
    detector := detector.NewSSRFDetector(logger)
    results, err := detector.Scan(*target, *timeout)
    
    if err != nil {
        fmt.Fprintf(os.Stderr, "Scan failed: %v\n", err)
        os.Exit(1)
    }
    
    // 輸出結果
    if *outputFormat == "json" {
        json.NewEncoder(os.Stdout).Encode(results)
    } else {
        fmt.Println(results)
    }
}
```

#### SCA Scanner (`cmd/sca-scanner/main.go`)

```go
package main

import (
    "flag"
    "encoding/json"
    "os"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca/analyzer"
    "go.uber.org/zap"
)

func main() {
    projectPath := flag.String("project-path", "", "Path to project directory")
    outputFormat := flag.String("output", "json", "Output format")
    flag.Parse()
    
    // ... 實現邏輯
}
```

#### CSPM Scanner (`cmd/cspm-scanner/main.go`)

```go
package main

import (
    "flag"
    "encoding/json"
    "os"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/cspm/audit"
    "go.uber.org/zap"
)

func main() {
    cloudProvider := flag.String("cloud-provider", "aws", "Cloud provider (aws/azure/gcp)")
    region := flag.String("region", "", "Cloud region")
    outputFormat := flag.String("output", "json", "Output format")
    flag.Parse()
    
    // ... 實現邏輯
}
```

## 輸出格式規範

所有 Go 掃描器應輸出統一的 JSON 格式：

```json
{
    "scanner": "ssrf",
    "version": "1.0.0",
    "scan_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z",
    "target": "https://example.com",
    "status": "completed",
    "assets": [
        {
            "type": "vulnerability",
            "value": "SSRF detected at /api/proxy",
            "severity": "high",
            "parameters": {
                "url": "/api/proxy",
                "method": "POST",
                "payload": "url=http://169.254.169.254/latest/meta-data/"
            }
        }
    ],
    "metadata": {
        "scan_duration_ms": 1500,
        "requests_made": 25,
        "vulnerabilities_found": 1
    },
    "error": null
}
```

## 待辦事項

### 高優先級
- [ ] 實現 SSRF Scanner 的命令行參數處理
- [ ] 實現 SCA Scanner 的命令行參數處理
- [ ] 實現 CSPM Scanner 的命令行參數處理
- [ ] 統一 JSON 輸出格式
- [ ] 添加錯誤處理和超時控制

### 中優先級
- [ ] 添加詳細的使用說明和示例
- [ ] 實現進度回報機制
- [ ] 添加配置文件支持

### 低優先級
- [ ] 性能優化和並行處理
- [ ] 添加更多輸出格式（XML、YAML）
- [ ] 實現持久化和恢復機制

## 移除的功能

以下功能在新架構中**不再需要**：

- ❌ RabbitMQ Worker 模式
- ❌ 消息訂閱/發布
- ❌ 異步 Worker 循環
- ❌ 消息隊列配置
- ❌ Worker 註冊和心跳

## 依賴清理

可以從 `go.mod` 中移除以下依賴：

```go
// 不再需要
github.com/rabbitmq/amqp091-go v1.10.0
```

## 集成測試

Python 調度器會負責：
1. 編譯 Go 掃描器
2. 準備命令行參數
3. 執行掃描器二進制
4. 解析 JSON 輸出
5. 錯誤處理和重試

Go 掃描器只需要：
1. 接收命令行參數
2. 執行掃描邏輯
3. 輸出標準 JSON 格式
4. 返回正確的退出代碼

## 參考

- Python 調度器: `services/scan/engines/go_engine/dispatcher/python_bridge.py`
- 新架構文檔: `SCAN_MODULE_RABBITMQ_REMOVAL.md`
- 命令中心: `services/aiva_common/command_center.py`
