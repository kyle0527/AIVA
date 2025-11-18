# AIVA 多語言系統診斷腳本
# 日期: 2025-10-15
# 用途: 診斷系統問題並提供修復建議

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "🔧 AIVA 系統診斷工具" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

$issues = @()
$suggestions = @()

# ============================================
# 1. 基本環境檢查
# ============================================
Write-Host "🔍 檢查基本環境..." -ForegroundColor Yellow

# 檢查 Python
$pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
if ($pythonInstalled) {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python 未安裝" -ForegroundColor Red
    $issues += "Python 未安裝"
    $suggestions += "請安裝 Python: https://www.python.org/downloads/"
}

# 檢查 Node.js
$nodeInstalled = Get-Command node -ErrorAction SilentlyContinue
if ($nodeInstalled) {
    $nodeVersion = node --version 2>&1
    Write-Host "   ✅ Node.js: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Node.js 未安裝" -ForegroundColor Yellow
    $suggestions += "建議安裝 Node.js: https://nodejs.org/"
}

# 檢查 Go
$goInstalled = Get-Command go -ErrorAction SilentlyContinue
if ($goInstalled) {
    $goVersion = go version 2>&1
    Write-Host "   ✅ Go: $goVersion" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Go 未安裝" -ForegroundColor Yellow
    $suggestions += "建議安裝 Go: https://go.dev/dl/"
}

# 檢查 Rust
$cargoInstalled = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargoInstalled) {
    $rustVersion = cargo --version 2>&1
    Write-Host "   ✅ Rust: $rustVersion" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Rust 未安裝" -ForegroundColor Yellow
    $suggestions += "建議安裝 Rust: https://www.rust-lang.org/tools/install"
}

# 檢查 Docker
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerInstalled) {
    $dockerVersion = docker --version 2>&1
    Write-Host "   ✅ Docker: $dockerVersion" -ForegroundColor Green

    # 檢查 Docker 是否運行
    docker ps 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Docker 服務正在運行" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Docker 服務未運行" -ForegroundColor Red
        $issues += "Docker 服務未運行"
        $suggestions += "啟動 Docker Desktop 或執行: Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"
    }
} else {
    Write-Host "   ❌ Docker 未安裝" -ForegroundColor Red
    $issues += "Docker 未安裝"
    $suggestions += "請安裝 Docker Desktop: https://www.docker.com/products/docker-desktop/"
}

# ============================================
# 2. 項目結構檢查
# ============================================
Write-Host "`n📁 檢查項目結構..." -ForegroundColor Yellow

$requiredPaths = @(
    "services",
    "services\core",
    "services\scan",
    "services\function",
    "services\integration",
    "docker\docker-compose.yml"
)

foreach ($path in $requiredPaths) {
    if (Test-Path $path) {
        Write-Host "   ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $path 不存在" -ForegroundColor Red
        $issues += "$path 目錄/文件不存在"
    }
}

# 檢查虛擬環境
if (Test-Path ".\.venv") {
    Write-Host "   ✅ Python 虛擬環境存在" -ForegroundColor Green
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "   ✅ 虛擬環境腳本可用" -ForegroundColor Green
    } else {
        Write-Host "   ❌ 虛擬環境腳本不可用" -ForegroundColor Red
        $issues += "虛擬環境腳本損壞"
        $suggestions += "重建虛擬環境: Remove-Item -Recurse .venv; python -m venv .venv"
    }
} else {
    Write-Host "   ⚠️  Python 虛擬環境不存在" -ForegroundColor Yellow
    $suggestions += "創建虛擬環境: python -m venv .venv"
}

# ============================================
# 3. 配置文件檢查
# ============================================
Write-Host "`n⚙️  檢查配置文件..." -ForegroundColor Yellow

$configFiles = @(
    "pyproject.toml",
    "requirements.txt",
    "docker\docker-compose.yml"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $file 不存在" -ForegroundColor Yellow
    }
}

# ============================================
# 4. 端口檢查
# ============================================
Write-Host "`n🔌 檢查端口占用..." -ForegroundColor Yellow

$ports = @(8001, 8003, 5432, 5672, 15672, 6379, 7474, 7687)
foreach ($port in $ports) {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("localhost", $port)
        $tcpClient.Close()
        Write-Host "   ⚠️  端口 $port 已被占用" -ForegroundColor Yellow
    } catch {
        Write-Host "   ✅ 端口 $port 可用" -ForegroundColor Green
    }
}

# ============================================
# 總結和建議
# ============================================
Write-Host "`n========================================" -ForegroundColor Magenta
if ($issues.Count -eq 0) {
    Write-Host "🎉 診斷完成 - 未發現嚴重問題!" -ForegroundColor Green
} else {
    Write-Host "⚠️  診斷完成 - 發現 $($issues.Count) 個問題" -ForegroundColor Yellow
    Write-Host "`n🔧 發現的問題:" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "   • $issue" -ForegroundColor White
    }
}

if ($suggestions.Count -gt 0) {
    Write-Host "`n💡 修復建議:" -ForegroundColor Yellow
    foreach ($suggestion in $suggestions) {
        Write-Host "   • $suggestion" -ForegroundColor White
    }
}

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host ""
