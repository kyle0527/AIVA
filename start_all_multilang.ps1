# AIVA 多語言模組啟動腳本
# 日期: 2025-10-13
# 用途: 啟動 Python + Node.js + Go + Rust 所有模組

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🌐 AIVA 多語言系統 - 啟動中..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# ============================================
# 1. 基礎設施檢查
# ============================================
Write-Host "📦 檢查基礎設施..." -ForegroundColor Yellow

# 檢查 Docker
$dockerRunning = docker ps 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker 未運行! 請先啟動 Docker Desktop" -ForegroundColor Red
    exit 1
}

# 啟動 RabbitMQ + PostgreSQL
Write-Host "🐳 啟動 RabbitMQ + PostgreSQL..." -ForegroundColor Yellow
docker-compose -f docker\docker-compose.yml up -d
Start-Sleep -Seconds 15

Write-Host "✅ 基礎設施已就緒`n" -ForegroundColor Green

# ============================================
# 2. Python 模組
# ============================================
Write-Host "🐍 啟動 Python 模組..." -ForegroundColor Cyan
Write-Host "-----------------------------------" -ForegroundColor Gray

# 檢查虛擬環境
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "⚠️  虛擬環境不存在,跳過 Python 模組" -ForegroundColor Yellow
} else {
    $pythonModules = @(
        @{ Name = "Core"; Path = "services\core\aiva_core"; Cmd = "python -m uvicorn app:app --port 8001 --reload" },
        @{ Name = "Scan"; Path = "services\scan\aiva_scan"; Cmd = "python worker.py" },
        @{ Name = "XSS"; Path = "services\function\function_xss\aiva_func_xss"; Cmd = "python worker.py" },
        @{ Name = "SQLi"; Path = "services\function\function_sqli\aiva_func_sqli"; Cmd = "python worker.py" },
        @{ Name = "SSRF"; Path = "services\function\function_ssrf\aiva_func_ssrf"; Cmd = "python worker.py" },
        @{ Name = "IDOR"; Path = "services\function\function_idor\aiva_func_idor"; Cmd = "python worker.py" },
        @{ Name = "Integration"; Path = "services\integration\aiva_integration"; Cmd = "python -m uvicorn app:app --port 8003 --reload" }
    )

    foreach ($module in $pythonModules) {
        if (Test-Path $module.Path) {
            Write-Host "   🐍 $($module.Name)" -ForegroundColor Green
            $script = @"
Set-Location '$PWD'
& .\.venv\Scripts\Activate.ps1
Set-Location $($module.Path)
`$host.UI.RawUI.WindowTitle = 'AIVA - Python - $($module.Name)'
$($module.Cmd)
"@
            Start-Process pwsh -ArgumentList "-NoExit", "-Command", $script
            Start-Sleep -Seconds 1
        }
    }
}

# ============================================
# 3. Node.js 模組
# ============================================
Write-Host "`n🟢 啟動 Node.js 模組..." -ForegroundColor Cyan
Write-Host "-----------------------------------" -ForegroundColor Gray

$nodePath = "services\scan\aiva_scan_node"
if (Test-Path $nodePath) {
    if (Test-Path "$nodePath\node_modules") {
        Write-Host "   🟢 Scan (Playwright)" -ForegroundColor Green
        $script = @"
Set-Location '$PWD\$nodePath'
`$host.UI.RawUI.WindowTitle = 'AIVA - Node.js - Scan'
npm run dev
"@
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", $script
        Start-Sleep -Seconds 2
    } else {
        Write-Host "   ⚠️  Node.js 模組未安裝依賴,執行: cd $nodePath && npm install" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  Node.js 模組不存在,已跳過" -ForegroundColor Gray
}

# ============================================
# 4. Go 模組
# ============================================
Write-Host "`n🔵 啟動 Go 模組..." -ForegroundColor Cyan
Write-Host "-----------------------------------" -ForegroundColor Gray

$goPath = "services\function\function_ssrf_go"
if (Test-Path $goPath) {
    # 檢查 Go 是否安裝
    $goInstalled = Get-Command go -ErrorAction SilentlyContinue
    if ($goInstalled) {
        Write-Host "   🔵 SSRF Detector" -ForegroundColor Green
        $script = @"
Set-Location '$PWD\$goPath'
`$host.UI.RawUI.WindowTitle = 'AIVA - Go - SSRF'
go run cmd/worker/main.go
"@
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", $script
        Start-Sleep -Seconds 2
    } else {
        Write-Host "   ⚠️  Go 未安裝,請先安裝: https://go.dev/dl/" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  Go 模組不存在,已跳過" -ForegroundColor Gray
}

# ============================================
# 5. Rust 模組
# ============================================
Write-Host "`n🦀 啟動 Rust 模組..." -ForegroundColor Cyan
Write-Host "-----------------------------------" -ForegroundColor Gray

$rustPath = "services\scan\info_gatherer_rust"
if (Test-Path $rustPath) {
    # 檢查 Cargo 是否安裝
    $cargoInstalled = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cargoInstalled) {
        Write-Host "   🦀 Sensitive Info Gatherer" -ForegroundColor Green
        $script = @"
Set-Location '$PWD\$rustPath'
`$host.UI.RawUI.WindowTitle = 'AIVA - Rust - Info Gatherer'
cargo run --release
"@
        Start-Process pwsh -ArgumentList "-NoExit", "-Command", $script
        Start-Sleep -Seconds 3
    } else {
        Write-Host "   ⚠️  Rust 未安裝,請先安裝: https://www.rust-lang.org/tools/install" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ℹ️  Rust 模組不存在,已跳過" -ForegroundColor Gray
}

# ============================================
# 總結
# ============================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ 多語言系統啟動完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 模組統計:" -ForegroundColor Yellow
Write-Host "   🐍 Python:  7 個模組" -ForegroundColor White
Write-Host "   🟢 Node.js: 1 個模組" -ForegroundColor White
Write-Host "   🔵 Go:      1 個模組" -ForegroundColor White
Write-Host "   🦀 Rust:    1 個模組" -ForegroundColor White
Write-Host ""
Write-Host "📍 服務端點:" -ForegroundColor Yellow
Write-Host "   • Core API:        http://localhost:8001/docs" -ForegroundColor White
Write-Host "   • Integration API: http://localhost:8003/docs" -ForegroundColor White
Write-Host "   • RabbitMQ:        http://localhost:15672" -ForegroundColor White
Write-Host ""
Write-Host "💡 使用 stop_all_multilang.ps1 停止所有服務" -ForegroundColor Yellow
Write-Host ""
