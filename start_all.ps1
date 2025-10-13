# AIVA 完整系統啟動腳本
# 日期: 2025-10-13
# 用途: 一鍵啟動所有 Python 模組 (MVP)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 AIVA 智慧漏洞評估系統 - 啟動中..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 設定錯誤處理
$ErrorActionPreference = "Continue"

# 檢查虛擬環境
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "❌ 找不到虛擬環境! 請先執行: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# 啟動虛擬環境
Write-Host "📦 啟動 Python 虛擬環境..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 檢查 Docker 服務
Write-Host "`n📦 檢查 Docker 服務..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker 未運行! 請先啟動 Docker Desktop" -ForegroundColor Red
    exit 1
}

# 啟動基礎設施
Write-Host "`n🐳 啟動 RabbitMQ + PostgreSQL..." -ForegroundColor Yellow
docker-compose -f docker\docker-compose.yml up -d

# 等待服務就緒
Write-Host "⏳ 等待服務初始化 (15秒)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# 檢查 RabbitMQ 是否就緒
$maxRetries = 5
$retryCount = 0
while ($retryCount -lt $maxRetries) {
    $rabbitTest = Test-NetConnection localhost -Port 5672 -WarningAction SilentlyContinue -InformationLevel Quiet
    if ($rabbitTest) {
        Write-Host "✅ RabbitMQ 已就緒" -ForegroundColor Green
        break
    }
    $retryCount++
    Write-Host "⏳ 等待 RabbitMQ... ($retryCount/$maxRetries)" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

# 檢查 PostgreSQL 是否就緒
$postgresTest = Test-NetConnection localhost -Port 5432 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($postgresTest) {
    Write-Host "✅ PostgreSQL 已就緒" -ForegroundColor Green
} else {
    Write-Host "⚠️  PostgreSQL 可能尚未就緒" -ForegroundColor Yellow
}

Write-Host "`n" -NoNewline

# 啟動各模組
$modules = @(
    @{
        Name = "Core (智慧分析引擎)"
        Path = "services\core\aiva_core"
        Command = "python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload"
        Icon = "🧠"
    },
    @{
        Name = "Scan (爬蟲引擎)"
        Path = "services\scan\aiva_scan"
        Command = "python worker.py"
        Icon = "🕷️ "
    },
    @{
        Name = "Function XSS"
        Path = "services\function\function_xss\aiva_func_xss"
        Command = "python worker.py"
        Icon = "🔍"
    },
    @{
        Name = "Function SQLi"
        Path = "services\function\function_sqli\aiva_func_sqli"
        Command = "python worker.py"
        Icon = "💉"
    },
    @{
        Name = "Function SSRF"
        Path = "services\function\function_ssrf\aiva_func_ssrf"
        Command = "python worker.py"
        Icon = "🌐"
    },
    @{
        Name = "Function IDOR"
        Path = "services\function\function_idor\aiva_func_idor"
        Command = "python worker.py"
        Icon = "🔑"
    },
    @{
        Name = "Integration (報告整合)"
        Path = "services\integration\aiva_integration"
        Command = "python -m uvicorn app:app --host 0.0.0.0 --port 8003 --reload"
        Icon = "📊"
    }
)

foreach ($module in $modules) {
    Write-Host "$($module.Icon) 啟動 $($module.Name)..." -ForegroundColor Cyan
    
    # 檢查路徑是否存在
    if (-not (Test-Path $module.Path)) {
        Write-Host "   ⚠️  路徑不存在: $($module.Path)" -ForegroundColor Yellow
        continue
    }
    
    # 在新視窗中啟動
    $scriptBlock = @"
Set-Location '$PWD'
& .\.venv\Scripts\Activate.ps1
Set-Location $($module.Path)
Write-Host '========================================' -ForegroundColor Green
Write-Host '$($module.Icon) $($module.Name) 運行中...' -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
Write-Host ''
$($module.Command)
"@
    
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", $scriptBlock
    Start-Sleep -Seconds 2
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ 所有模組啟動完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 服務資訊:" -ForegroundColor Yellow
Write-Host "   • Core API:        http://localhost:8001/docs" -ForegroundColor White
Write-Host "   • Integration API: http://localhost:8003/docs" -ForegroundColor White
Write-Host "   • RabbitMQ 管理:    http://localhost:15672" -ForegroundColor White
Write-Host "     帳號: aiva / 密碼: dev_password" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 使用 stop_all.ps1 停止所有服務" -ForegroundColor Yellow
Write-Host ""
