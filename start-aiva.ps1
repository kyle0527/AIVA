# AIVA 快速啟動腳本 (PowerShell)
# 用於本地 Docker Compose 環境

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('core', 'scanners', 'testing', 'explorers', 'validators', 'pentest', 'all', 'stop', 'status')]
    [string]$Action = 'core',
    
    [Parameter(Mandatory=$false)]
    [switch]$Build,
    
    [Parameter(Mandatory=$false)]
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 AIVA 微服務啟動器" -ForegroundColor Cyan
Write-Host "=" * 60

function Show-Status {
    Write-Host "`n📊 當前服務狀態:" -ForegroundColor Yellow
    docker-compose ps
    
    Write-Host "`n🔍 核心服務健康檢查:" -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "✅ AIVA Core: 健康 (HTTP $($response.StatusCode))" -ForegroundColor Green
    } catch {
        Write-Host "❌ AIVA Core: 不可用" -ForegroundColor Red
    }
}

function Start-Core {
    Write-Host "`n🏗️ 啟動核心服務和基礎設施..." -ForegroundColor Green
    
    if ($Build) {
        Write-Host "📦 構建 Docker 鏡像..." -ForegroundColor Yellow
        docker-compose build aiva-core
    }
    
    docker-compose up -d
    
    Write-Host "`n⏳ 等待服務啟動（60秒）..." -ForegroundColor Yellow
    Start-Sleep -Seconds 60
    
    Show-Status
    
    Write-Host "`n✅ 核心服務已啟動！" -ForegroundColor Green
    Write-Host "   API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   Admin: http://localhost:8001" -ForegroundColor Cyan
    Write-Host "   RabbitMQ UI: http://localhost:15672 (guest/guest)" -ForegroundColor Cyan
    Write-Host "   Neo4j Browser: http://localhost:7474 (neo4j/aiva123)" -ForegroundColor Cyan
}

function Start-Components {
    param([string]$Profile)
    
    Write-Host "`n🔧 啟動組件: $Profile" -ForegroundColor Green
    
    if ($Build) {
        Write-Host "📦 構建組件鏡像..." -ForegroundColor Yellow
        docker-compose build
    }
    
    docker-compose --profile $Profile up -d
    
    Write-Host "`n⏳ 等待組件啟動（30秒）..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    Show-Status
    
    Write-Host "`n✅ 組件已啟動！" -ForegroundColor Green
}

function Stop-All {
    Write-Host "`n🛑 停止所有服務..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "✅ 所有服務已停止" -ForegroundColor Green
}

function Show-Logs {
    Write-Host "`n📜 顯示實時日誌..." -ForegroundColor Yellow
    docker-compose logs -f --tail=100
}

# 主邏輯
switch ($Action) {
    'core' {
        Start-Core
    }
    'scanners' {
        Start-Components -Profile 'scanners'
    }
    'testing' {
        Start-Components -Profile 'testing'
    }
    'explorers' {
        Start-Components -Profile 'explorers'
    }
    'validators' {
        Start-Components -Profile 'validators'
    }
    'pentest' {
        Start-Components -Profile 'pentest'
    }
    'all' {
        Start-Components -Profile 'all'
    }
    'stop' {
        Stop-All
    }
    'status' {
        Show-Status
    }
}

if ($Logs) {
    Show-Logs
}

Write-Host "`n" -NoNewline
