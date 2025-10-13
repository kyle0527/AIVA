# AIVA 系統狀態檢查腳本
# 日期: 2025-10-13
# 用途: 檢查所有服務運行狀態

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📊 AIVA 系統狀態檢查" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 檢查 Docker 容器
Write-Host "🐳 Docker 容器狀態:" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray
$containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String -Pattern "aiva|rabbitmq|postgres"
if ($containers) {
    $containers | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
} else {
    Write-Host "   ❌ 沒有運行中的容器" -ForegroundColor Red
}

# 檢查埠號
Write-Host "`n🔌 服務埠號檢查:" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

$ports = @(
    @{ Port = 8001; Service = "Core API" },
    @{ Port = 8003; Service = "Integration API" },
    @{ Port = 5672; Service = "RabbitMQ AMQP" },
    @{ Port = 15672; Service = "RabbitMQ 管理介面" },
    @{ Port = 5432; Service = "PostgreSQL" }
)

foreach ($item in $ports) {
    $test = Test-NetConnection localhost -Port $item.Port -WarningAction SilentlyContinue -InformationLevel Quiet
    if ($test) {
        Write-Host "   ✅ Port $($item.Port) - $($item.Service)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Port $($item.Port) - $($item.Service)" -ForegroundColor Red
    }
}

# 檢查 Python 進程
Write-Host "`n🐍 Python 進程:" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "   ✅ 運行中: $($pythonProcesses.Count) 個 Python 進程" -ForegroundColor Green
    
    # 檢查是否有 uvicorn (API 服務)
    $uvicornCount = ($pythonProcesses | Where-Object { $_.CommandLine -like "*uvicorn*" }).Count
    if ($uvicornCount -gt 0) {
        Write-Host "   ✅ API 服務: $uvicornCount 個" -ForegroundColor Green
    }
    
    # 檢查是否有 worker
    $workerCount = ($pythonProcesses | Where-Object { $_.CommandLine -like "*worker.py*" }).Count
    if ($workerCount -gt 0) {
        Write-Host "   ✅ Worker: $workerCount 個" -ForegroundColor Green
    }
} else {
    Write-Host "   ❌ 沒有運行中的 Python 進程" -ForegroundColor Red
}

# 測試 API 端點
Write-Host "`n🌐 API 健康檢查:" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

# Core API
try {
    $coreResponse = Invoke-WebRequest -Uri "http://localhost:8001/docs" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    if ($coreResponse.StatusCode -eq 200) {
        Write-Host "   ✅ Core API (http://localhost:8001)" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ Core API 無回應" -ForegroundColor Red
}

# Integration API
try {
    $integrationResponse = Invoke-WebRequest -Uri "http://localhost:8003/docs" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    if ($integrationResponse.StatusCode -eq 200) {
        Write-Host "   ✅ Integration API (http://localhost:8003)" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ Integration API 無回應" -ForegroundColor Red
}

# RabbitMQ 管理介面
try {
    $rabbitResponse = Invoke-WebRequest -Uri "http://localhost:15672" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    if ($rabbitResponse.StatusCode -eq 200) {
        Write-Host "   ✅ RabbitMQ 管理介面 (http://localhost:15672)" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ RabbitMQ 管理介面無回應" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "📋 快速連結:" -ForegroundColor Yellow
Write-Host "   • Core API 文檔:     http://localhost:8001/docs" -ForegroundColor White
Write-Host "   • Integration API:   http://localhost:8003/docs" -ForegroundColor White
Write-Host "   • RabbitMQ 管理:      http://localhost:15672" -ForegroundColor White
Write-Host "     (帳號: aiva / 密碼: dev_password)" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
