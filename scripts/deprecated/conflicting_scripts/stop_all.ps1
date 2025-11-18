# AIVA 系統停止腳本
# 日期: 2025-10-13
# 用途: 停止所有服務

Write-Host "========================================" -ForegroundColor Red
Write-Host "🛑 停止 AIVA 系統..." -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# 停止所有 Python 進程
Write-Host "🔴 停止 Python 進程..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $pythonProcesses | Stop-Process -Force
    Write-Host "   ✅ 已停止 $($pythonProcesses.Count) 個 Python 進程" -ForegroundColor Green
} else {
    Write-Host "   ℹ️  沒有運行中的 Python 進程" -ForegroundColor Gray
}

# 停止 Uvicorn 進程 (如果有殘留)
Write-Host "`n🔴 檢查 Uvicorn 進程..." -ForegroundColor Yellow
$uvicornProcesses = Get-Process | Where-Object { $_.CommandLine -like "*uvicorn*" } -ErrorAction SilentlyContinue
if ($uvicornProcesses) {
    $uvicornProcesses | Stop-Process -Force
    Write-Host "   ✅ 已停止 Uvicorn 進程" -ForegroundColor Green
}

# 停止 Docker 容器
Write-Host "`n🔴 停止 Docker 容器..." -ForegroundColor Yellow
docker-compose -f docker\docker-compose.yml down

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Docker 容器已停止" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Docker 容器停止時發生錯誤" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ AIVA 系統已完全停止" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
