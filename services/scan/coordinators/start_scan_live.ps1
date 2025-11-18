#!/usr/bin/env pwsh
# AIVA 掃描模組快速啟動腳本
# 用於 Windows PowerShell

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  🚀 AIVA 多目標掃描測試 - 快速啟動" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# 檢查 Docker 是否運行
Write-Host "🔍 檢查 Docker 狀態..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker 已安裝: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker 未安裝或未運行" -ForegroundColor Red
    Write-Host "   請先安裝 Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# 檢查 Docker Compose
try {
    $composeVersion = docker-compose --version
    Write-Host "✅ Docker Compose 已安裝: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose 未安裝" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 進入目錄
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "📂 當前目錄: $scriptPath" -ForegroundColor Cyan
Write-Host ""

# 選單
Write-Host "請選擇操作:" -ForegroundColor Yellow
Write-Host "  [1] 🚀 啟動所有服務 (首次啟動會構建鏡像)" -ForegroundColor White
Write-Host "  [2] 📊 查看服務狀態" -ForegroundColor White
Write-Host "  [3] 🎯 發送測試目標 (8 個內建目標)" -ForegroundColor White
Write-Host "  [4] 📋 查看實時日誌" -ForegroundColor White
Write-Host "  [5] 🌐 打開 RabbitMQ 管理界面" -ForegroundColor White
Write-Host "  [6] 📈 查看統計指標" -ForegroundColor White
Write-Host "  [7] 🛑 停止所有服務" -ForegroundColor White
Write-Host "  [8] 🗑️  清理環境 (刪除容器和卷)" -ForegroundColor White
Write-Host "  [0] ❌ 退出" -ForegroundColor White
Write-Host ""

$choice = Read-Host "請輸入選項 (0-8)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 啟動服務..." -ForegroundColor Green
        docker-compose -f docker-compose.scan.yml up -d
        
        Write-Host ""
        Write-Host "⏳ 等待服務啟動 (30秒)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        
        Write-Host ""
        Write-Host "✅ 服務已啟動!" -ForegroundColor Green
        Write-Host ""
        docker-compose -f docker-compose.scan.yml ps
        
        Write-Host ""
        Write-Host "🌐 RabbitMQ 管理界面: http://localhost:15672" -ForegroundColor Cyan
        Write-Host "   帳號: aiva" -ForegroundColor Gray
        Write-Host "   密碼: aiva_mq_password" -ForegroundColor Gray
    }
    
    "2" {
        Write-Host ""
        Write-Host "📊 服務狀態:" -ForegroundColor Green
        docker-compose -f docker-compose.scan.yml ps
        
        Write-Host ""
        Write-Host "💾 資源使用:" -ForegroundColor Green
        docker stats --no-stream
    }
    
    "3" {
        Write-Host ""
        Write-Host "🎯 發送測試目標..." -ForegroundColor Green
        docker-compose -f docker-compose.scan.yml run --rm test-target-generator
        
        Write-Host ""
        Write-Host "✅ 測試目標已發送!" -ForegroundColor Green
        Write-Host "💡 使用選項 [4] 查看實時日誌" -ForegroundColor Yellow
    }
    
    "4" {
        Write-Host ""
        Write-Host "請選擇要查看的日誌:" -ForegroundColor Yellow
        Write-Host "  [1] Rust Mode 1 (Fast Discovery)" -ForegroundColor White
        Write-Host "  [2] Rust Mode 2 (Deep Analysis)" -ForegroundColor White
        Write-Host "  [3] Rust Mode 3 (Focused Verification)" -ForegroundColor White
        Write-Host "  [4] RabbitMQ" -ForegroundColor White
        Write-Host "  [5] 所有服務" -ForegroundColor White
        Write-Host ""
        
        $logChoice = Read-Host "請選擇 (1-5)"
        
        switch ($logChoice) {
            "1" { docker logs -f aiva-rust-fast-discovery }
            "2" { docker logs -f aiva-rust-deep-analysis }
            "3" { docker logs -f aiva-rust-focused-verification }
            "4" { docker logs -f aiva-rabbitmq }
            "5" { docker-compose -f docker-compose.scan.yml logs -f }
        }
    }
    
    "5" {
        Write-Host ""
        Write-Host "🌐 正在打開 RabbitMQ 管理界面..." -ForegroundColor Green
        Start-Process "http://localhost:15672"
        
        Write-Host ""
        Write-Host "🔑 登入資訊:" -ForegroundColor Cyan
        Write-Host "   帳號: aiva" -ForegroundColor White
        Write-Host "   密碼: aiva_mq_password" -ForegroundColor White
    }
    
    "6" {
        Write-Host ""
        Write-Host "📈 統計指標:" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "--- Mode 1: Fast Discovery ---" -ForegroundColor Cyan
        docker exec aiva-rust-fast-discovery cat /var/log/aiva/metrics/rust_fast_discovery.jsonl 2>$null | Select-Object -Last 5
        
        Write-Host ""
        Write-Host "--- Mode 2: Deep Analysis ---" -ForegroundColor Cyan
        docker exec aiva-rust-deep-analysis cat /var/log/aiva/metrics/rust_deep_analysis.jsonl 2>$null | Select-Object -Last 5
        
        Write-Host ""
        Write-Host "--- Mode 3: Focused Verification ---" -ForegroundColor Cyan
        docker exec aiva-rust-focused-verification cat /var/log/aiva/metrics/rust_focused_verification.jsonl 2>$null | Select-Object -Last 5
    }
    
    "7" {
        Write-Host ""
        Write-Host "🛑 停止所有服務..." -ForegroundColor Yellow
        docker-compose -f docker-compose.scan.yml down
        
        Write-Host ""
        Write-Host "✅ 所有服務已停止" -ForegroundColor Green
    }
    
    "8" {
        Write-Host ""
        Write-Host "⚠️  警告: 這將刪除所有容器、網絡和卷!" -ForegroundColor Red
        $confirm = Read-Host "確定要繼續嗎? (yes/no)"
        
        if ($confirm -eq "yes") {
            Write-Host ""
            Write-Host "🗑️  清理環境..." -ForegroundColor Yellow
            docker-compose -f docker-compose.scan.yml down -v --rmi all
            
            Write-Host ""
            Write-Host "✅ 環境已清理" -ForegroundColor Green
        } else {
            Write-Host "❌ 已取消" -ForegroundColor Yellow
        }
    }
    
    "0" {
        Write-Host ""
        Write-Host "👋 再見!" -ForegroundColor Green
        exit 0
    }
    
    default {
        Write-Host ""
        Write-Host "❌ 無效的選項" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# 詢問是否繼續
$continue = Read-Host "按 Enter 返回選單,或輸入 'q' 退出"
if ($continue -ne "q") {
    & $MyInvocation.MyCommand.Path
}
