# Go 服務遷移到 aiva_common_go 自動化腳本
# 用途: 自動化遷移 Go 服務使用共享庫
# 使用: .\migrate_go_service.ps1 -ServiceName function_authn_go

param(
    [Parameter(Mandatory=$true)]
    [string]$ServiceName
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 開始遷移 $ServiceName 到 aiva_common_go" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# 確定腳本所在目錄（services/function）
$scriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($scriptDir)) {
    $scriptDir = Get-Location
}

Write-Host "📂 腳本目錄: $scriptDir" -ForegroundColor Gray

# 檢查服務目錄是否存在
$servicePath = Join-Path $scriptDir $ServiceName
if (-not (Test-Path $servicePath)) {
    Write-Host "❌ 服務目錄不存在: $servicePath" -ForegroundColor Red
    Write-Host "提示: 請在 services/function 目錄下執行此腳本" -ForegroundColor Yellow
    Write-Host "或使用完整路徑: cd c:\AMD\AIVA\services\function" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ 找到服務目錄: $servicePath" -ForegroundColor Green

# 保存當前目錄
Push-Location

# 進入服務目錄
Set-Location $servicePath
Write-Host "📂 當前目錄: $(Get-Location)" -ForegroundColor Gray

# 步驟 1: 更新 go.mod
Write-Host "`n📝 步驟 1: 更新 go.mod..." -ForegroundColor Yellow

$goModContent = @"
module github.com/kyle0527/aiva/services/function/$ServiceName

go 1.21

require (
	github.com/kyle0527/aiva/services/function/common/go/aiva_common_go v0.0.0-00010101000000-000000000000
	github.com/rabbitmq/amqp091-go v1.10.0
	go.uber.org/zap v1.26.0
)

require (
	github.com/stretchr/testify v1.8.4 // indirect
	go.uber.org/multierr v1.11.0 // indirect
)

replace github.com/kyle0527/aiva/services/function/common/go/aiva_common_go => ../common/go/aiva_common_go
"@

$goModContent | Set-Content -Path "go.mod" -Encoding UTF8
Write-Host "✅ go.mod 已更新" -ForegroundColor Green

# 步驟 2: 運行 go mod tidy
Write-Host "`n📦 步驟 2: 整理依賴..." -ForegroundColor Yellow
go mod tidy
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ go mod tidy 失敗" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 依賴整理完成" -ForegroundColor Green

# 步驟 3: 檢查編譯
Write-Host "`n🔨 步驟 3: 檢查編譯..." -ForegroundColor Yellow
go build ./...
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  編譯失敗，需要手動修復 main.go" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "請確保 main.go 使用以下模式:" -ForegroundColor Cyan
    Write-Host @"
import (
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/config"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/logger"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/mq"
    "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas"
)

func main() {
    cfg, err := config.LoadConfig("$ServiceName")
    if err != nil {
        panic(err)
    }
    
    log, err := logger.NewLogger(cfg.ServiceName)
    if err != nil {
        panic(err)
    }
    defer log.Sync()
    
    mqClient, err := mq.NewMQClient(cfg.RabbitMQURL, log)
    if err != nil {
        log.Fatal("MQ 連接失敗", zap.Error(err))
    }
    defer mqClient.Close()
    
    err = mqClient.Consume(queueName, handleTask)
    // ...
}
"@ -ForegroundColor White
} else {
    Write-Host "✅ 編譯成功" -ForegroundColor Green
}

# 步驟 4: 刪除舊的重複代碼
Write-Host "`n🗑️  步驟 4: 清理重複代碼..." -ForegroundColor Yellow

$toDelete = @("pkg\messaging", "pkg\models")
foreach ($dir in $toDelete) {
    $fullPath = Join-Path $servicePath $dir
    if (Test-Path $fullPath) {
        Remove-Item -Recurse -Force $fullPath
        Write-Host "  ✅ 已刪除: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️  跳過 (不存在): $dir" -ForegroundColor Gray
    }
}

# 步驟 5: 最終驗證
Write-Host "`n✅ 步驟 5: 最終驗證..." -ForegroundColor Yellow
go build ./...
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 最終編譯成功" -ForegroundColor Green
} else {
    Write-Host "⚠️  需要手動調整代碼" -ForegroundColor Yellow
}

# 總結
Write-Host "`n" + ("=" * 60) -ForegroundColor Gray
Write-Host "🎉 遷移腳本執行完成!" -ForegroundColor Green
Write-Host ""
Write-Host "遷移檢查清單:" -ForegroundColor Cyan
Write-Host "  [✓] go.mod 已更新" -ForegroundColor Green
Write-Host "  [✓] 依賴已整理" -ForegroundColor Green
Write-Host "  [?] main.go 需要手動驗證" -ForegroundColor Yellow
Write-Host "  [✓] 重複代碼已清理" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:" -ForegroundColor Cyan
Write-Host "  1. cd $servicePath" -ForegroundColor White
Write-Host "  2. 檢查並修正 cmd/worker/main.go" -ForegroundColor White
Write-Host "  3. 更新 internal/scanner 使用 schemas" -ForegroundColor White
Write-Host "  4. 運行: go build ./..." -ForegroundColor White
Write-Host "  5. 運行: go test ./..." -ForegroundColor White
Write-Host ""

# 返回原始目錄
Pop-Location
Write-Host "✅ 已返回原始目錄" -ForegroundColor Gray
