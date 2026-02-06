#!/usr/bin/env pwsh
# AIVA 快速啟動腳本 v2.0
# 功能：自動啟動兩個入口點（app.py 和 main.py）

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            AIVA 系統快速啟動腳本 v2.0                       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 獲取項目根目錄
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = $SCRIPT_DIR

Write-Host "📂 項目根目錄: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host ""

# 檢查 Python 是否可用
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 錯誤: 未找到 Python" -ForegroundColor Red
    Write-Host "請安裝 Python 3.11+ 並添加到 PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "選擇啟動模式：" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [1] 完整模式（Port 8000 + Port 9000）" -ForegroundColor White
Write-Host "      - 啟動 app.py (AI 核心服務)" -ForegroundColor Gray
Write-Host "      - 啟動 main.py (安全網關)" -ForegroundColor Gray
Write-Host ""
Write-Host "  [2] AI 核心模式（僅 Port 8000）" -ForegroundColor White
Write-Host "      - 僅啟動 app.py (直接測試)" -ForegroundColor Gray
Write-Host ""
Write-Host "  [3] CLI 選單模式" -ForegroundColor White
Write-Host "      - 啟動 app.py + ai_menu.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  [0] 退出" -ForegroundColor White
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "請選擇 (0-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 啟動完整模式..." -ForegroundColor Green
        Write-Host ""
        
        # 啟動 app.py (Port 8000)
        Write-Host "📦 [1/2] 啟動 AI 核心服務 (Port 8000)..." -ForegroundColor Yellow
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$PROJECT_ROOT\services\core'; python aiva_core\service_backbone\api\app.py" -WindowStyle Normal
        Start-Sleep -Seconds 3
        
        # 啟動 main.py (Port 9000)
        Write-Host "🛡️  [2/2] 啟動安全網關 (Port 9000)..." -ForegroundColor Yellow
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$PROJECT_ROOT\services\core'; python main.py" -WindowStyle Normal
        
        Write-Host ""
        Write-Host "✅ 兩個服務已在新視窗中啟動" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 訪問方式：" -ForegroundColor Cyan
        Write-Host "   - HTTP API (外部): http://localhost:9000" -ForegroundColor White
        Write-Host "   - AI 核心 (內部): http://localhost:8000" -ForegroundColor White
        Write-Host "   - API 文檔: http://localhost:9000/docs" -ForegroundColor White
        Write-Host ""
    }
    
    "2" {
        Write-Host ""
        Write-Host "🚀 啟動 AI 核心模式..." -ForegroundColor Green
        Write-Host ""
        
        Write-Host "📦 啟動 AI 核心服務 (Port 8000)..." -ForegroundColor Yellow
        Set-Location "$PROJECT_ROOT\services\core"
        python aiva_core\service_backbone\api\app.py
    }
    
    "3" {
        Write-Host ""
        Write-Host "🚀 啟動 CLI 選單模式..." -ForegroundColor Green
        Write-Host ""
        
        # 啟動 app.py (Port 8000)
        Write-Host "📦 [1/2] 啟動 AI 核心服務 (Port 8000)..." -ForegroundColor Yellow
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$PROJECT_ROOT\services\core'; python aiva_core\service_backbone\api\app.py" -WindowStyle Normal
        Start-Sleep -Seconds 3
        
        # 啟動 CLI 選單
        Write-Host "🎮 [2/2] 啟動 CLI 選單..." -ForegroundColor Yellow
        Set-Location "$PROJECT_ROOT"
        python services\core\aiva_core\core_capabilities\dialog\ai_menu.py
    }
    
    "0" {
        Write-Host ""
        Write-Host "👋 退出" -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host ""
        Write-Host "❌ 無效選擇" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "按任意鍵關閉此視窗..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
