# Go 模組依賴初始化腳本
# 用途: 下載並安裝 Go 依賴

Write-Host "🔵 初始化 Go SSRF 模組依賴..." -ForegroundColor Cyan

$goPath = "services\function\function_ssrf_go"

if (-not (Test-Path $goPath)) {
    Write-Host "❌ 路徑不存在: $goPath" -ForegroundColor Red
    exit 1
}

# 檢查 Go 是否安裝
$goInstalled = Get-Command go -ErrorAction SilentlyContinue
if (-not $goInstalled) {
    Write-Host "❌ Go 未安裝!" -ForegroundColor Red
    Write-Host "請下載安裝: https://go.dev/dl/" -ForegroundColor Yellow
    exit 1
}

Set-Location $goPath

Write-Host "📦 清理模組快取..." -ForegroundColor Yellow
go clean -modcache

Write-Host "📦 下載依賴..." -ForegroundColor Yellow
go mod download

Write-Host "📦 整理依賴..." -ForegroundColor Yellow
go mod tidy

Write-Host "✅ Go 依賴初始化完成!" -ForegroundColor Green
Write-Host ""
Write-Host "可以執行以下命令測試:" -ForegroundColor Yellow
Write-Host "  go run cmd/worker/main.go" -ForegroundColor White
Write-Host "  或" -ForegroundColor Gray
Write-Host "  go build -o ssrf_worker.exe cmd/worker/main.go" -ForegroundColor White
Write-Host ""

Set-Location $PSScriptRoot
