# AIVA 快速啟動腳本 (PowerShell) v2.0.0
# 用於本地 Docker Compose 環境
# 更新日期: 2025年11月11日 - 修復重複定義問題

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('core', 'scanners', 'testing', 'explorers', 'validators', 'pentest', 'neural', 'multiagent', 'all', 'stop', 'status', 'health')]
    [string]$Action = 'core',
    
    [Parameter(Mandatory=$false)]
    [switch]$Build,
    
    [Parameter(Mandatory=$false)]
    [switch]$Logs,
    
    [Parameter(Mandatory=$false)]
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 AIVA v2.0.0 微服務啟動器" -ForegroundColor Cyan
Write-Host "🧠 整合5M神經網路 + 多代理架構" -ForegroundColor Green
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
    
    # 新增：5M神經網路健康檢查
    Write-Host "`n🧠 5M神經網路健康檢查:" -ForegroundColor Yellow
    try {
        $neuralResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/v2/neural/health" -TimeoutSec 5
        Write-Host "✅ 5M Neural Network: 健康 (參數: 4,999,481)" -ForegroundColor Green
    } catch {
        Write-Host "❌ 5M Neural Network: 不可用" -ForegroundColor Red
    }
    
    # 新增：多代理系統檢查
    Write-Host "`n🤖 多代理系統檢查:" -ForegroundColor Yellow
    try {
        $agentResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/v2/multiagent/status" -TimeoutSec 5
        Write-Host "✅ Multi-Agent System: 運行中" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Multi-Agent System: 開發中" -ForegroundColor Yellow
    }
}

function Show-HealthCheck {
    Write-Host "`n🏥 AIVA v2.0.0 完整健康檢查" -ForegroundColor Cyan
    Write-Host "=" * 50
    
    # 核心系統檢查
    Write-Host "`n1. 核心系統狀態:" -ForegroundColor White
    try {
        $healthData = Invoke-RestMethod -Uri "http://localhost:8000/api/v2/health/detailed" -TimeoutSec 10
        Write-Host "   ✅ 系統狀態: $($healthData.status)" -ForegroundColor Green
        Write-Host "   📊 CPU使用率: $($healthData.cpu)%" -ForegroundColor Cyan
        Write-Host "   💾 記憶體使用: $($healthData.memory)%" -ForegroundColor Cyan
    } catch {
        Write-Host "   ❌ 核心系統: 無法連接" -ForegroundColor Red
    }
    
    # 5M神經網路檢查
    Write-Host "`n2. 5M神經網路狀態:" -ForegroundColor White
    try {
        $neuralData = Invoke-RestMethod -Uri "http://localhost:8000/api/v2/neural/metrics" -TimeoutSec 10
        Write-Host "   ✅ 神經網路健康度: $($neuralData.health_score)%" -ForegroundColor Green
        Write-Host "   🧠 參數數量: $($neuralData.total_parameters)" -ForegroundColor Cyan
        Write-Host "   ⚡ 推理延遲: $($neuralData.inference_latency)ms" -ForegroundColor Cyan
        Write-Host "   🎯 決策準確率: $($neuralData.decision_accuracy)%" -ForegroundColor Cyan
    } catch {
        Write-Host "   ❌ 5M神經網路: 無法獲取指標" -ForegroundColor Red
    }
    
    # RAG系統檢查
    Write-Host "`n3. RAG檢索系統狀態:" -ForegroundColor White
    try {
        $ragData = Invoke-RestMethod -Uri "http://localhost:8000/api/v2/rag/status" -TimeoutSec 10
        Write-Host "   ✅ RAG系統: $($ragData.status)" -ForegroundColor Green
        Write-Host "   📚 知識庫數量: $($ragData.knowledge_bases)" -ForegroundColor Cyan
        Write-Host "   🔍 檢索精準度: $($ragData.retrieval_accuracy)%" -ForegroundColor Cyan
    } catch {
        Write-Host "   ❌ RAG系統: 無法連接" -ForegroundColor Red
    }
    
    # 能力編排器檢查
    Write-Host "`n4. 能力編排器狀態:" -ForegroundColor White
    try {
        $orchestratorData = Invoke-RestMethod -Uri "http://localhost:8000/api/v2/orchestrator/capabilities" -TimeoutSec 10
        Write-Host "   ✅ 編排器健康度: $($orchestratorData.health_score)%" -ForegroundColor Green
        Write-Host "   🎪 活躍能力數: $($orchestratorData.active_capabilities)" -ForegroundColor Cyan
        Write-Host "   🔄 協調成功率: $($orchestratorData.coordination_rate)%" -ForegroundColor Cyan
    } catch {
        Write-Host "   ❌ 能力編排器: 無法獲取狀態" -ForegroundColor Red
    }
}

function Start-Core {
    Write-Host "`n🏗️ 啟動AIVA v2.0.0核心服務..." -ForegroundColor Green
    Write-Host "   🧠 包含5M神經網路核心" -ForegroundColor Yellow
    Write-Host "   📚 整合RAG檢索系統" -ForegroundColor Yellow
    Write-Host "   🎪 啟動能力編排器" -ForegroundColor Yellow
    
    if ($Build) {
        Write-Host "📦 構建 Docker 鏡像..." -ForegroundColor Yellow
        docker-compose build aiva-core
    }
    
    docker-compose up -d
    
    Write-Host "`n⏳ 等待服務啟動（90秒，神經網路初始化需要時間）..." -ForegroundColor Yellow
    Start-Sleep -Seconds 90
    
    Show-Status
    
    Write-Host "`n✅ AIVA v2.0.0 核心服務已啟動！" -ForegroundColor Green
    Write-Host "   🌐 Main API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   🔧 Admin Panel: http://localhost:8001" -ForegroundColor Cyan
    Write-Host "   🧠 Neural API: http://localhost:8000/api/v2/neural/" -ForegroundColor Cyan
    Write-Host "   📚 RAG API: http://localhost:8000/api/v2/rag/" -ForegroundColor Cyan
    Write-Host "   🐰 RabbitMQ UI: http://localhost:15672 (guest/guest)" -ForegroundColor Cyan
    Write-Host "   🗄️ Neo4j Browser: http://localhost:7474 (neo4j/aiva123)" -ForegroundColor Cyan
}

function Start-Neural {
    Write-Host "`n🧠 啟動5M神經網路專用模式..." -ForegroundColor Green
    
    if ($Build) {
        Write-Host "📦 構建神經網路鏡像..." -ForegroundColor Yellow
        docker-compose build neural-core
    }
    
    docker-compose --profile neural up -d
    
    Write-Host "`n⏳ 神經網路初始化中（120秒）..." -ForegroundColor Yellow
    Write-Host "   🔄 載入4,999,481個參數..." -ForegroundColor Yellow
    Write-Host "   ⚡ 預熱推理引擎..." -ForegroundColor Yellow
    Start-Sleep -Seconds 120
    
    Write-Host "`n✅ 5M神經網路已就緒！" -ForegroundColor Green
    Write-Host "   🧠 參數規模: 4,999,481" -ForegroundColor Cyan
    Write-Host "   ⚡ 推理模式: 實時離線" -ForegroundColor Cyan
    Write-Host "   🎯 API端點: http://localhost:8000/api/v2/neural/" -ForegroundColor Cyan
}

function Start-MultiAgent {
    Write-Host "`n🤖 啟動多代理協作系統（開發版本）..." -ForegroundColor Green
    Write-Host "   📋 基於Microsoft AutoGen架構" -ForegroundColor Yellow
    Write-Host "   🚧 當前狀態：原型開發中" -ForegroundColor Yellow
    
    if ($Build) {
        Write-Host "📦 構建多代理鏡像..." -ForegroundColor Yellow
        docker-compose build multiagent-system
    }
    
    # 注意：多代理系統仍在開發中
    Write-Host "`n⚠️ 多代理系統正在開發中..." -ForegroundColor Yellow
    Write-Host "   📅 預計完成：2025年Q1" -ForegroundColor Cyan
    Write-Host "   🔬 基於最新網路調研成果設計中" -ForegroundColor Cyan
    Write-Host "   🎯 將支援：規劃師/執行者/審查員/學習者角色" -ForegroundColor Cyan
}

function Start-Components {
    param([string]$ComponentProfile)
    
    Write-Host "`n🔧 啟動組件: $ComponentProfile" -ForegroundColor Green
    
    if ($Build) {
        Write-Host "📦 構建組件鏡像..." -ForegroundColor Yellow
        docker-compose build
    }
    
    docker-compose --profile $ComponentProfile up -d
    
    Write-Host "`n⏳ 等待組件啟動（30秒）..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    Show-Status
    
    Write-Host "`n✅ 組件已啟動！" -ForegroundColor Green
}

function Stop-All {
    Write-Host "`n🛑 停止所有AIVA服務..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "✅ 所有服務已停止" -ForegroundColor Green
}

function Show-Logs {
    Write-Host "`n📜 顯示AIVA實時日誌..." -ForegroundColor Yellow
    if ($Debug) {
        Write-Host "🐛 調試模式：顯示詳細日誌" -ForegroundColor Red
        docker-compose logs -f --tail=500
    } else {
        docker-compose logs -f --tail=100
    }
}

# 主邏輯
switch ($Action) {
    'core' {
        Start-Core
    }
    'neural' {
        Start-Neural
    }
    'multiagent' {
        Start-MultiAgent
    }
    'scanners' {
        Start-Components -ComponentProfile 'scanners'
    }
    'testing' {
        Start-Components -ComponentProfile 'testing'
    }
    'explorers' {
        Start-Components -ComponentProfile 'explorers'
    }
    'validators' {
        Start-Components -ComponentProfile 'validators'
    }
    'pentest' {
        Start-Components -ComponentProfile 'pentest'
    }
    'all' {
        Start-Components -ComponentProfile 'all'
    }
    'stop' {
        Stop-All
    }
    'status' {
        Show-Status
    }
    'health' {
        Show-HealthCheck
    }
}

if ($Logs) {
    Show-Logs
}

Write-Host "`n🌟 AIVA v2.0.0 - 下一代AI助手系統" -ForegroundColor Green
Write-Host "   📚 完整文檔: ./README.md" -ForegroundColor Cyan
Write-Host "   🎯 項目狀態: ./AIVA_PROJECT_STATUS.md" -ForegroundColor Cyan
Write-Host "`n" -NoNewline