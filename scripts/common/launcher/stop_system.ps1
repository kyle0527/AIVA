# AIVA 系統停止腳本 (現代化版本)
# 日期: 2025-11-25
# 用途: 安全停止所有 AIVA 服務

param(
    [switch]$Force,
    [switch]$KeepInfrastructure,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Red
Write-Host "🛑 AIVA 系統停止中..." -ForegroundColor White
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# ============================================
# 1. 停止 Python 服務
# ============================================
Write-Host "🐍 停止 Python 服務..." -ForegroundColor Yellow

$pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "   發現 $($pythonProcesses.Count) 個 Python 進程" -ForegroundColor Gray
    
    foreach ($proc in $pythonProcesses) {
        $procName = $proc.Name
        $procId = $proc.Id
        
        try {
            if ($Force) {
                Stop-Process -Id $procId -Force
                Write-Host "   ✓ 強制停止 PID $procId" -ForegroundColor Yellow
            } else {
                $proc.CloseMainWindow() | Out-Null
                Start-Sleep -Seconds 2
                if (-not $proc.HasExited) {
                    Stop-Process -Id $procId
                }
                Write-Host "   ✓ 已停止 PID $procId" -ForegroundColor Green
            }
        } catch {
            Write-Host "   ⚠️  無法停止 PID $procId" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "   ℹ️  未發現運行中的 Python 服務" -ForegroundColor Gray
}

# ============================================
# 2. 停止 Docker 容器
# ============================================
if (-not $KeepInfrastructure) {
    Write-Host "`n🐳 停止 Docker 容器..." -ForegroundColor Yellow
    
    $ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
    Push-Location "$ProjectRoot\docker"
    
    if ($Force) {
        Write-Host "   強制停止並移除容器..." -ForegroundColor Yellow
        docker-compose down -v 2>$null
    } else {
        Write-Host "   正常停止容器..." -ForegroundColor Gray
        docker-compose down 2>$null
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ Docker 容器已停止" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️  Docker 容器未運行或已停止" -ForegroundColor Gray
    }
    
    Pop-Location
} else {
    Write-Host "`n🐳 保留基礎設施服務..." -ForegroundColor Yellow
}

# ============================================
# 3. 清理臨時文件 (可選)
# ============================================
Write-Host "`n🧹 清理臨時文件..." -ForegroundColor Yellow

$tempDirs = @(
    "$ProjectRoot\logs\*.log",
    "$ProjectRoot\*.tmp"
)

foreach ($pattern in $tempDirs) {
    if (Test-Path $pattern) {
        Remove-Item $pattern -Force -ErrorAction SilentlyContinue
        Write-Host "   ✓ 清理: $pattern" -ForegroundColor Gray
    }
}

# ============================================
# 4. 驗證停止
# ============================================
Write-Host "`n✓ 驗證服務狀態..." -ForegroundColor Yellow

$pythonStillRunning = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonStillRunning) {
    Write-Host "   ⚠️  仍有 $($pythonStillRunning.Count) 個 Python 進程運行中" -ForegroundColor Yellow
    if ($Verbose) {
        foreach ($proc in $pythonStillRunning) {
            Write-Host "      PID $($proc.Id)" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "   ✓ 所有 Python 服務已停止" -ForegroundColor Green
}

if (-not $KeepInfrastructure) {
    $dockerStillRunning = docker ps -q 2>$null
    if ($dockerStillRunning) {
        Write-Host "   ℹ️  仍有其他 Docker 容器運行中" -ForegroundColor Gray
    } else {
        Write-Host "   ✓ 所有 Docker 容器已停止" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ AIVA 系統已停止" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($KeepInfrastructure) {
    Write-Host "💡 基礎設施服務仍在運行" -ForegroundColor Yellow
    Write-Host "   如需停止，請執行: docker-compose down" -ForegroundColor Gray
    Write-Host ""
}
