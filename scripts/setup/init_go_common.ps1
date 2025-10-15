# AIVA Go 共用模組初始化腳本

Write-Host "🚀 初始化 aiva_common_go 模組..." -ForegroundColor Green

$commonGoPath = "c:\AMD\AIVA\services\function\common\go\aiva_common_go"

# 切換到模組目錄
Set-Location $commonGoPath

Write-Host "📦 執行 go mod tidy..." -ForegroundColor Cyan
go mod tidy

Write-Host "⬇️  下載依賴..." -ForegroundColor Cyan
go mod download

Write-Host "🔍 驗證模組..." -ForegroundColor Cyan
go mod verify

Write-Host "🧪 執行測試..." -ForegroundColor Cyan
go test ./... -v

Write-Host "" 
Write-Host "✅ aiva_common_go 初始化完成!" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "1. 查看 README.md 了解如何使用"
Write-Host "2. 參考 MULTILANG_STRATEGY.md 遷移現有服務"
Write-Host "3. 執行 .\migrate_sca_service.ps1 遷移第一個服務"
Write-Host ""
