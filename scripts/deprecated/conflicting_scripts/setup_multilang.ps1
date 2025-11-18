# AIVA 多語言環境設置腳本
# 日期: 2025-10-13
# 用途: 一次性設置所有語言的開發環境

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🔧 AIVA 多語言環境設置" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# ============================================
# 1. Python 環境
# ============================================
Write-Host "🐍 設置 Python 環境..." -ForegroundColor Cyan

if (-not (Test-Path ".\.venv")) {
    Write-Host "   📦 建立虛擬環境..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "   📦 啟動虛擬環境並安裝依賴..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -e .

Write-Host "   ✅ Python 環境設置完成`n" -ForegroundColor Green

# ============================================
# 2. Node.js 環境
# ============================================
Write-Host "🟢 設置 Node.js 環境..." -ForegroundColor Cyan

$nodePath = "services\scan\aiva_scan_node"
if (Test-Path $nodePath) {
    # 檢查 Node.js 是否安裝
    $nodeInstalled = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeInstalled) {
        Write-Host "   📦 安裝 Node.js 依賴..." -ForegroundColor Yellow
        Set-Location $nodePath
        npm install
        
        Write-Host "   🌐 安裝 Playwright 瀏覽器..." -ForegroundColor Yellow
        npx playwright install --with-deps chromium
        
        Set-Location $PSScriptRoot
        Write-Host "   ✅ Node.js 環境設置完成`n" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Node.js 未安裝" -ForegroundColor Yellow
        Write-Host "   請下載安裝: https://nodejs.org/" -ForegroundColor Yellow
        Write-Host "   建議版本: 20.x LTS`n" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  Node.js 模組不存在,已跳過`n" -ForegroundColor Gray
}

# ============================================
# 3. Go 環境
# ============================================
Write-Host "🔵 設置 Go 環境..." -ForegroundColor Cyan

$goPath = "services\function\function_ssrf_go"
if (Test-Path $goPath) {
    # 檢查 Go 是否安裝
    $goInstalled = Get-Command go -ErrorAction SilentlyContinue
    if ($goInstalled) {
        Write-Host "   📦 下載 Go 依賴..." -ForegroundColor Yellow
        Set-Location $goPath
        go mod download
        go mod tidy
        
        Set-Location $PSScriptRoot
        Write-Host "   ✅ Go 環境設置完成`n" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Go 未安裝" -ForegroundColor Yellow
        Write-Host "   請下載安裝: https://go.dev/dl/" -ForegroundColor Yellow
        Write-Host "   建議版本: 1.21+`n" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  Go 模組不存在,已跳過`n" -ForegroundColor Gray
}

# ============================================
# 4. Rust 環境
# ============================================
Write-Host "🦀 設置 Rust 環境..." -ForegroundColor Cyan

$rustPath = "services\scan\info_gatherer_rust"
if (Test-Path $rustPath) {
    # 檢查 Cargo 是否安裝
    $cargoInstalled = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cargoInstalled) {
        Write-Host "   📦 編譯 Rust 專案 (釋出版本)..." -ForegroundColor Yellow
        Set-Location $rustPath
        cargo build --release
        
        Set-Location $PSScriptRoot
        Write-Host "   ✅ Rust 環境設置完成`n" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Rust 未安裝" -ForegroundColor Yellow
        Write-Host "   請執行安裝命令:" -ForegroundColor Yellow
        Write-Host "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`n" -ForegroundColor Gray
    }
} else {
    Write-Host "   ℹ️  Rust 模組不存在,已跳過`n" -ForegroundColor Gray
}

# ============================================
# 5. Docker 環境
# ============================================
Write-Host "🐳 檢查 Docker 環境..." -ForegroundColor Cyan

$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerInstalled) {
    Write-Host "   ✅ Docker 已安裝" -ForegroundColor Green
    
    # 檢查 Docker 是否運行
    $dockerRunning = docker ps 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Docker 正在運行`n" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Docker 未運行,請啟動 Docker Desktop`n" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ⚠️  Docker 未安裝" -ForegroundColor Yellow
    Write-Host "   請下載安裝: https://www.docker.com/products/docker-desktop`n" -ForegroundColor Yellow
}

# ============================================
# 總結
# ============================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📋 環境設置總結" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 檢查各語言
$pythonOk = Test-Path ".\.venv"
$nodeOk = (Test-Path "$nodePath\node_modules") -or -not (Test-Path $nodePath)
$goOk = (Get-Command go -ErrorAction SilentlyContinue) -ne $null
$rustOk = (Get-Command cargo -ErrorAction SilentlyContinue) -ne $null
$dockerOk = (Get-Command docker -ErrorAction SilentlyContinue) -ne $null

Write-Host "🐍 Python:  $(if ($pythonOk) { '✅ 已就緒' } else { '❌ 未就緒' })" -ForegroundColor $(if ($pythonOk) { 'Green' } else { 'Red' })
Write-Host "🟢 Node.js: $(if ($nodeOk) { '✅ 已就緒' } else { '❌ 未就緒' })" -ForegroundColor $(if ($nodeOk) { 'Green' } else { 'Red' })
Write-Host "🔵 Go:      $(if ($goOk) { '✅ 已就緒' } else { '❌ 未就緒' })" -ForegroundColor $(if ($goOk) { 'Green' } else { 'Red' })
Write-Host "🦀 Rust:    $(if ($rustOk) { '✅ 已就緒' } else { '❌ 未就緒' })" -ForegroundColor $(if ($rustOk) { 'Green' } else { 'Red' })
Write-Host "🐳 Docker:  $(if ($dockerOk) { '✅ 已就緒' } else { '❌ 未就緒' })" -ForegroundColor $(if ($dockerOk) { 'Green' } else { 'Red' })
Write-Host ""

if ($pythonOk -and $dockerOk) {
    Write-Host "✅ 基礎環境已就緒,可以啟動系統!" -ForegroundColor Green
    Write-Host "💡 執行: .\start_all_multilang.ps1" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  部分環境未就緒,請檢查上述錯誤訊息" -ForegroundColor Yellow
}

Write-Host ""
