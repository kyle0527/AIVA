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
docker ps 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Docker 未運行，嘗試啟動 Docker Desktop..." -ForegroundColor Yellow

    # 嘗試啟動 Docker Desktop
    $dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktopPath) {
        Write-Host "🔄 正在啟動 Docker Desktop..." -ForegroundColor Cyan
        Start-Process $dockerDesktopPath

        # 等待 Docker 啟動 (最多等待 60 秒)
        $timeout = 60
        $elapsed = 0
        do {
            Start-Sleep -Seconds 5
            $elapsed += 5
            Write-Host "   ⏳ 等待 Docker 啟動... ($elapsed/$timeout 秒)" -ForegroundColor Gray
            docker ps 2>$null | Out-Null
        } while ($LASTEXITCODE -ne 0 -and $elapsed -lt $timeout)

        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Docker 啟動超時! 請手動啟動 Docker Desktop 後重新執行此腳本" -ForegroundColor Red
            Write-Host "💡 或者執行: Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'" -ForegroundColor Yellow
            exit 1
        } else {
            Write-Host "✅ Docker 已成功啟動!" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ 找不到 Docker Desktop! 請確認已安裝 Docker Desktop" -ForegroundColor Red
        Write-Host "💡 下載地址: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
        exit 1
    }
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
    Write-Host "⚠️  虛擬環境不存在，嘗試創建..." -ForegroundColor Yellow

    # 檢查 Python 是否安裝
    $pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonInstalled) {
        Write-Host "🔄 創建 Python 虛擬環境..." -ForegroundColor Cyan
        python -m venv .venv

        if (Test-Path ".\.venv\Scripts\Activate.ps1") {
            Write-Host "✅ 虛擬環境創建成功!" -ForegroundColor Green
            Write-Host "🔄 安裝依賴..." -ForegroundColor Cyan
            & .\.venv\Scripts\Activate.ps1
            if (Test-Path "requirements.txt") {
                pip install -r requirements.txt
            }
            if (Test-Path "pyproject.toml") {
                pip install -e .
            }
        } else {
            Write-Host "❌ 虛擬環境創建失敗，跳過 Python 模組" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Python 未安裝，跳過 Python 模組" -ForegroundColor Red
        Write-Host "💡 請先安裝 Python: https://www.python.org/downloads/" -ForegroundColor Yellow
    }
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
Write-Host "💡 使用 health_check_multilang.ps1 檢查系統狀態" -ForegroundColor Yellow
Write-Host ""

# 可選：等待 10 秒後自動進行健康檢查
Write-Host "⏳ 10 秒後將進行健康檢查..." -ForegroundColor Gray
Start-Sleep -Seconds 10
Write-Host ""

# 執行健康檢查
if (Test-Path "health_check_multilang.ps1") {
    & .\health_check_multilang.ps1
} else {
    Write-Host "⚠️  健康檢查腳本不存在" -ForegroundColor Yellow
}
