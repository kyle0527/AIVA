# Go 引擎掃描器

AIVA Go 引擎包含三個專業掃描器，專注於高性能並發掃描。

## 掃描器列表

### 1. SSRF Scanner (Server-Side Request Forgery)
- **路徑**: `ssrf_scanner/`
- **功能**: 檢測 SSRF 漏洞，包括內部網路存取和雲端元數據洩漏
- **掃描時間**: 約 30-60 秒/目標

### 2. CSPM Scanner (Cloud Security Posture Management)
- **路徑**: `cspm_scanner/`
- **功能**: 檢測雲端安全配置錯誤（S3 公開、安全組、IAM 策略等）
- **掃描時間**: 約 1-2 分鐘/目標

### 3. SCA Scanner (Software Composition Analysis)
- **路徑**: `sca_scanner/`
- **功能**: 分析第三方依賴漏洞，使用 OSV 數據庫
- **掃描時間**: 約 2-3 分鐘/目標

## 構建掃描器

### Windows (PowerShell)
```powershell
.\build_scanners.ps1
```

### Linux/macOS (Bash)
```bash
chmod +x build_scanners.sh
./build_scanners.sh
```

## 手動構建單個掃描器

```bash
# 進入掃描器目錄
cd ssrf_scanner  # 或 cspm_scanner, sca_scanner

# 下載依賴
go mod download

# 構建
go build -o worker.exe -ldflags="-s -w" .
```

## 使用方式

### 1. 通過 Python Worker (推薦)
Python Worker 會自動調用 Go 掃描器：

```python
# 啟動 Go Worker
python worker.py
```

### 2. 直接調用掃描器
```bash
# 準備任務文件 task.json
{
  "task_id": "test_001",
  "scan_id": "scan_001",
  "target": {"url": "https://example.com?param=value"},
  "config": {"timeout": 30}
}

# 執行掃描
./ssrf_scanner/worker.exe --task-file task.json
```

## 環境變數

- `AIVA_AMQP_URL`: RabbitMQ 連線 URL (預設: `amqp://guest:guest@rabbitmq:5672/`)
- `SCAN_TASKS_QUEUE`: 任務隊列名稱
- `SCAN_RESULTS_QUEUE`: 結果隊列名稱 (預設: `SCAN_RESULTS`)

## 依賴需求

- **Go**: 1.21 或更高版本
- **RabbitMQ**: 消息隊列服務
- **網路**: 掃描器需要外部網路存取

## 架構

```
go_engine/
├── worker.py              # Python Worker (協調器)
├── build_scanners.ps1     # Windows 構建腳本
├── build_scanners.sh      # Linux/macOS 構建腳本
├── common/                # 共用程式碼
│   ├── amqp_client.go    # RabbitMQ 客戶端
│   ├── scanner_base.go   # 掃描器基礎介面
│   └── sarif_converter.go # SARIF 格式轉換
├── ssrf_scanner/         # SSRF 掃描器
│   ├── main.go
│   ├── ssrf_detector.go
│   └── worker.exe        # 編譯後的二進制
├── cspm_scanner/         # CSPM 掃描器
│   ├── main.go
│   ├── cloud_detector.go
│   └── worker.exe
└── sca_scanner/          # SCA 掃描器
    ├── main.go
    ├── dependency_detector.go
    └── worker.exe
```

## 故障排除

### 問題: 掃描器未找到
**解決**: 確保已運行構建腳本：
```bash
.\build_scanners.ps1  # Windows
./build_scanners.sh   # Linux/macOS
```

### 問題: Go 版本過舊
**解決**: 升級到 Go 1.21+：
```bash
go version  # 檢查版本
```

### 問題: 編譯錯誤 "module not found"
**解決**: 下載依賴：
```bash
cd <scanner_dir>
go mod download
go mod tidy
```

### 問題: RabbitMQ 連線失敗
**解決**: 檢查 AMQP URL 和 RabbitMQ 服務狀態：
```bash
# 檢查 RabbitMQ 是否運行
docker ps | grep rabbitmq

# 設定正確的 AMQP URL
## 配置說明

**研發階段**：無需設置環境變數，自動使用預設值。

預設配置：
```go
AMQP_URL = "amqp://guest:guest@localhost:5672/"
```
```

## 開發指南

### 添加新掃描器

1. 創建新目錄：`mkdir my_scanner`
2. 實作 `BaseScanner` 介面
3. 創建 `main.go` 和掃描邏輯
4. 更新 `build_scanners.ps1` 和 `build_scanners.sh`
5. 在 `worker.py` 中添加調用邏輯

### 測試掃描器

```bash
# 單元測試
cd <scanner_dir>
go test ./...

# 整合測試
python -m pytest tests/test_go_engine.py
```

## 性能優化

- **並發掃描**: Go 掃描器使用 goroutine 實現高並發
- **資源限制**: 通過環境變數控制並發數
- **超時設定**: 合理設定掃描超時避免卡死

## 相關文件

- [SCAN_FLOW_DIAGRAMS.md](../../SCAN_FLOW_DIAGRAMS.md) - 完整掃描流程
- [aiva_common README](../../../aiva_common/README.md) - Schema 規範
- [Go 官方文檔](https://go.dev/doc/) - Go 語言文檔
