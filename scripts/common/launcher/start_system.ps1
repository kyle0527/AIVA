# AIVA 系統快速啟動腳本 (現代化版本)
# 日期: 2025-11-25
# 用途: 一鍵啟動 AIVA 系統 (使用全域 Python 環境)

param(
    [ValidateSet("minimal", "standard", "full")]
    [string]$Mode = "standard",
    [switch]$SkipInfrastructure,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\D\fold7\AIVA-git"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 AIVA 智慧漏洞評估系統 - 啟動中..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "模式: $Mode" -ForegroundColor Yellow
Write-Host "日期: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

Set-Location $ProjectRoot

# ============================================
# 1. 環境檢查
# ============================================
Write-Host "🔍 檢查環境..." -ForegroundColor Yellow

# 檢查 Python
$pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonInstalled) {
    Write-Host "❌ Python 未安裝！請先安裝 Python 3.9+" -ForegroundColor Red
    exit 1
}
$pythonVersion = python --version 2>&1
Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green

# 檢查 Docker
if (-not $SkipInfrastructure) {
    $dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerInstalled) {
        Write-Host "❌ Docker 未安裝！請先安裝 Docker Desktop" -ForegroundColor Red
        exit 1
    }
    
    $dockerRunning = docker ps 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker 未運行！請先啟動 Docker Desktop" -ForegroundColor Red
        exit 1
    }
    Write-Host "   ✅ Docker 運行中" -ForegroundColor Green
}

# ============================================
# 2. 啟動基礎設施 (如果需要)
# ============================================
if (-not $SkipInfrastructure) {
    Write-Host "`n🐳 啟動基礎設施服務..." -ForegroundColor Yellow
    
    Push-Location docker
    
    switch ($Mode) {
        "minimal" {
            Write-Host "   啟動最小配置 (RabbitMQ)..." -ForegroundColor Gray
            docker-compose up -d rabbitmq
        }
        "standard" {
            Write-Host "   啟動標準配置 (RabbitMQ + PostgreSQL + Redis)..." -ForegroundColor Gray
            docker-compose up -d rabbitmq postgres redis
        }
        "full" {
            Write-Host "   啟動完整配置 (All Services)..." -ForegroundColor Gray
            docker-compose up -d
        }
    }
    
    Pop-Location
    
    # 等待服務就緒
    Write-Host "   ⏳ 等待服務初始化..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15
    
    # 驗證 RabbitMQ
    $maxRetries = 10
    $retryCount = 0
    Write-Host "   🔍 檢查 RabbitMQ..." -ForegroundColor Yellow
    while ($retryCount -lt $maxRetries) {
        $rabbitTest = Test-NetConnection localhost -Port 5672 -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($rabbitTest) {
            Write-Host "   ✅ RabbitMQ 已就緒" -ForegroundColor Green
            break
        }
        $retryCount++
        Write-Host "      等待中... ($retryCount/$maxRetries)" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
    
    if ($retryCount -eq $maxRetries) {
        Write-Host "   ⚠️  RabbitMQ 啟動超時，請檢查 Docker 日誌" -ForegroundColor Yellow
    }
}

# ============================================
# 3. 啟動 AIVA 核心服務
# ============================================
Write-Host "`n🎯 啟動 AIVA 核心服務..." -ForegroundColor Yellow

# 檢查核心模組是否可用
Write-Host "   🔍 檢查核心模組..." -ForegroundColor Gray
$coreAvailable = python -c "import services.core.aiva_core; print('OK')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ 核心模組導入失敗！請執行: pip install -e ." -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ 核心模組可用" -ForegroundColor Green

# 根據模式啟動服務
Write-Host "`n📡 啟動服務..." -ForegroundColor Yellow

switch ($Mode) {
    "minimal" {
        Write-Host "   啟動 API 服務..." -ForegroundColor Gray
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python start_ai_service.py --mode api --port 8000" `
            -WindowStyle Normal
        Start-Sleep -Seconds 3
    }
    "standard" {
        Write-Host "   啟動 Core API..." -ForegroundColor Gray
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python -m services.core.aiva_core" `
            -WindowStyle Normal
        Start-Sleep -Seconds 3
        
        Write-Host "   啟動 Integration Service..." -ForegroundColor Gray
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python -m services.integration.aiva_integration" `
            -WindowStyle Normal
        Start-Sleep -Seconds 3
    }
    "full" {
        Write-Host "   啟動所有服務..." -ForegroundColor Gray
        
        # Core Service
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python -m services.core.aiva_core" `
            -WindowStyle Normal
        Start-Sleep -Seconds 2
        
        # Integration Service
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python -m services.integration.aiva_integration" `
            -WindowStyle Normal
        Start-Sleep -Seconds 2
        
        # Scan Coordinator
        Start-Process powershell -ArgumentList "-NoExit", "-Command", `
            "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python -m services.scan.coordinators.scan_coordinator" `
            -WindowStyle Normal
        Start-Sleep -Seconds 2
    }
}

# ============================================
# 4. 驗證啟動
# ============================================
Write-Host "`n✓ 驗證服務啟動..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$endpoints = @(
    @{ Name = "Core API"; Url = "http://localhost:8000/health" }
)

foreach ($endpoint in $endpoints) {
    try {
        $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "   ✅ $($endpoint.Name): 正常" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  $($endpoint.Name): HTTP $($response.StatusCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⏳ $($endpoint.Name): 啟動中..." -ForegroundColor Gray
    }
}

# ============================================
# 輸出訪問信息
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ AIVA 系統啟動完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📡 服務端點:" -ForegroundColor Cyan
Write-Host "   • Core API: http://localhost:8000" -ForegroundColor White
Write-Host "   • API 文檔: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • Admin: http://localhost:8001" -ForegroundColor White

if (-not $SkipInfrastructure) {
    Write-Host ""
    Write-Host "🔧 基礎設施:" -ForegroundColor Cyan
    Write-Host "   • RabbitMQ UI: http://localhost:15672 (guest/guest)" -ForegroundColor White
    if ($Mode -ne "minimal") {
        Write-Host "   • PostgreSQL: localhost:5432" -ForegroundColor White
        Write-Host "   • Redis: localhost:6379" -ForegroundColor White
    }
    if ($Mode -eq "full") {
        Write-Host "   • Neo4j: http://localhost:7474 (neo4j/aiva123)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "💡 提示:" -ForegroundColor Yellow
Write-Host "   • 查看健康狀態: .\scripts\common\validation\health_check.ps1" -ForegroundColor Gray
Write-Host "   • 停止所有服務: docker-compose down" -ForegroundColor Gray
Write-Host "   • 查看日誌: docker-compose logs -f" -ForegroundColor Gray
Write-Host ""
