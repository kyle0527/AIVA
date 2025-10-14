# AIVA UI 面板自動啟動腳本
# 自動選擇可用端口啟動 Web UI

Write-Host "🚀 正在啟動 AIVA UI 面板..." -ForegroundColor Green
Write-Host "📍 專案位置: $PSScriptRoot" -ForegroundColor Cyan

# 檢查 Python 是否可用
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python 版本: $pythonVersion" -ForegroundColor Yellow
} catch {
    Write-Host "❌ 錯誤: 找不到 Python，請確保已安裝 Python 3.8+" -ForegroundColor Red
    Read-Host "按 Enter 結束"
    exit 1
}

# 檢查必要套件
Write-Host "📦 檢查必要套件..." -ForegroundColor Cyan
$packages = @("fastapi", "uvicorn", "pydantic")
foreach ($package in $packages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ $package" -ForegroundColor Green
        } else {
            Write-Host "  ❌ $package (未安裝)" -ForegroundColor Red
            Write-Host "正在安裝 $package..." -ForegroundColor Yellow
            pip install $package
        }
    } catch {
        Write-Host "  ❌ $package (檢查失敗)" -ForegroundColor Red
    }
}

Write-Host "`n🌐 啟動 UI 伺服器 (自動端口選擇)..." -ForegroundColor Green
Write-Host "💡 提示: 按 Ctrl+C 停止伺服器" -ForegroundColor Yellow
Write-Host "=" * 60

try {
    # 切換到專案根目錄並執行
    Set-Location $PSScriptRoot
    python start_ui_auto.py
} catch {
    Write-Host "`n❌ 啟動失敗: $_" -ForegroundColor Red
    Read-Host "按 Enter 結束"
    exit 1
}

Write-Host "`n👋 伺服器已關閉" -ForegroundColor Yellow