# AIVA 多語言模組停止腳本
# 日期: 2025-10-13

Write-Host "========================================" -ForegroundColor Red
Write-Host "🛑 停止 AIVA 多語言系統..." -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# 停止 Python
Write-Host "🐍 停止 Python 進程..." -ForegroundColor Yellow
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    $pythonProcs | Stop-Process -Force
    Write-Host "   ✅ 已停止 $($pythonProcs.Count) 個 Python 進程" -ForegroundColor Green
}

# 停止 Node.js
Write-Host "`n🟢 停止 Node.js 進程..." -ForegroundColor Yellow
$nodeProcs = Get-Process node -ErrorAction SilentlyContinue
if ($nodeProcs) {
    $nodeProcs | Stop-Process -Force
    Write-Host "   ✅ 已停止 $($nodeProcs.Count) 個 Node.js 進程" -ForegroundColor Green
}

# 停止 Go
Write-Host "`n🔵 停止 Go 進程..." -ForegroundColor Yellow
$goProcs = Get-Process | Where-Object { $_.ProcessName -like "*ssrf*" -or $_.ProcessName -like "*worker*" }
if ($goProcs) {
    $goProcs | Stop-Process -Force
    Write-Host "   ✅ 已停止 Go 進程" -ForegroundColor Green
}

# 停止 Rust
Write-Host "`n🦀 停止 Rust 進程..." -ForegroundColor Yellow
$rustProcs = Get-Process | Where-Object { $_.ProcessName -like "*aiva-info*" -or $_.ProcessName -like "*info_gatherer*" }
if ($rustProcs) {
    $rustProcs | Stop-Process -Force
    Write-Host "   ✅ 已停止 Rust 進程" -ForegroundColor Green
}

# 停止 Docker
Write-Host "`n🐳 停止 Docker 容器..." -ForegroundColor Yellow
docker-compose -f docker\docker-compose.yml down
Write-Host "   ✅ Docker 容器已停止" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ 多語言系統已完全停止" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
