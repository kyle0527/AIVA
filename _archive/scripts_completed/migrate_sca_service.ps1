# SCA 服務遷移腳本 - 使用 aiva_common_go

Write-Host "🔄 開始遷移 function_sca_go 到使用共用模組..." -ForegroundColor Green

$scaPath = "c:\AMD\AIVA\services\function\function_sca_go"

# 備份原始檔案
Write-Host "📦 備份原始檔案..." -ForegroundColor Cyan
Copy-Item "$scaPath\cmd\worker\main.go" "$scaPath\cmd\worker\main.go.backup" -Force

Write-Host ""
Write-Host "⚠️  手動遷移步驟:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. 更新 go.mod 添加依賴:"
Write-Host "   cd $scaPath"
Write-Host "   go mod edit -require github.com/kyle0527/aiva/services/function/common/go/aiva_common_go@latest"
Write-Host "   go mod edit -replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go=../common/go/aiva_common_go"
Write-Host ""
Write-Host "2. 重構 cmd/worker/main.go:"
Write-Host "   - 移除手動的 RabbitMQ 連接程式碼"
Write-Host "   - 使用 config.LoadConfig() 載入配置"
Write-Host "   - 使用 logger.NewLogger() 建立日誌"
Write-Host "   - 使用 mq.NewMQClient() 建立 MQ 客戶端"
Write-Host ""
Write-Host "3. 參考範例程式碼:"
Write-Host "   code c:\AMD\AIVA\MULTILANG_STRATEGY.md"
Write-Host "   # 搜尋 'function_sca_go (重構後)'"
Write-Host ""

# 建立遷移後的範例程式碼
$newMainGo = @'
// cmd/worker/main.go (使用 aiva_common_go 重構)
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

var (
	log      *zap.Logger
	mqClient *mq.MQClient
	cfg      *config.Config
)

func main() {
	var err error

	// 1. 載入配置
	cfg, err = config.LoadConfig("sca")
	if err != nil {
		panic(err)
	}

	// 2. 初始化日誌
	log, err = logger.NewLogger(cfg.ServiceName)
	if err != nil {
		panic(err)
	}
	defer log.Sync()

	log.Info("🚀 AIVA SCA Service 啟動中...",
		zap.String("environment", cfg.Environment))

	// 3. 初始化 MQ 客戶端
	mqClient, err = mq.NewMQClient(cfg.RabbitMQURL, log)
	if err != nil {
		log.Fatal("MQ 連接失敗", zap.Error(err))
	}
	defer mqClient.Close()

	// 4. 開始消費任務
	go startConsuming()

	// 5. 等待中斷信號
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Info("👋 服務正常關閉")
}

func startConsuming() {
	log.Info("👂 開始監聽任務...", zap.String("queue", cfg.TaskQueue))

	err := mqClient.Consume(cfg.TaskQueue, handleSCATask)
	if err != nil {
		log.Fatal("消費任務失敗", zap.Error(err))
	}
}

func handleSCATask(body []byte) error {
	log.Debug("收到 SCA 任務", zap.Int("size", len(body)))

	var task schemas.TaskPayload
	if err := json.Unmarshal(body, &task); err != nil {
		log.Error("解析任務失敗", zap.Error(err))
		return err
	}

	log.Info("執行 SCA 掃描",
		zap.String("task_id", task.TaskID),
		zap.String("scan_id", task.ScanID))

	// 執行 SCA 業務邏輯
	findings, err := performSCA(task)
	if err != nil {
		log.Error("SCA 掃描失敗", zap.Error(err))
		return err
	}

	// 發布結果
	for _, finding := range findings {
		if err := mqClient.Publish(cfg.ResultQueue, finding); err != nil {
			log.Error("發布結果失敗", zap.Error(err))
			return err
		}
	}

	log.Info("SCA 掃描完成",
		zap.String("task_id", task.TaskID),
		zap.Int("findings", len(findings)))

	return nil
}

func performSCA(task schemas.TaskPayload) ([]schemas.FindingPayload, error) {
	// TODO: 實際的 SCA 掃描邏輯
	// 1. 解析 task.Input 獲取目標 (repository_url, file_path 等)
	// 2. 執行依賴掃描 (可整合 Trivy, Snyk 等工具)
	// 3. 轉換為 AIVA 標準格式

	findings := []schemas.FindingPayload{}

	// 範例: 整合 Trivy
	// findings = scanWithTrivy(task.Input["image_name"])

	return findings, nil
}
'@

$examplePath = "$scaPath\cmd\worker\main.go.example"
$newMainGo | Out-File -FilePath $examplePath -Encoding UTF8

Write-Host "✅ 已生成範例程式碼: $examplePath" -ForegroundColor Green
Write-Host ""
Write-Host "💡 比較原始與重構後的差異:" -ForegroundColor Cyan
Write-Host "   code --diff $scaPath\cmd\worker\main.go $examplePath"
Write-Host ""
Write-Host "預期效果:" -ForegroundColor Green
Write-Host "  ✅ 程式碼行數減少 ~40%"
Write-Host "  ✅ 統一錯誤處理和日誌格式"
Write-Host "  ✅ 更易於單元測試"
Write-Host "  ✅ 配置管理更靈活"
Write-Host ""
