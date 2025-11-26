# AIVA 系統診斷工具 (現代化版本)
# 日期: 2025-11-25
# 用途: 診斷系統問題並提供修復建議

param(
    [switch]$Verbose,
    [switch]$ExportReport
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "🔧 AIVA 系統診斷工具" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "版本: 2025-11-25 (全域環境版)" -ForegroundColor Gray
Write-Host ""

$issues = @()
$suggestions = @()
$diagnosticData = @{}

# ============================================
# 1. 基本環境檢查
# ============================================
Write-Host "🔍 檢查基本環境..." -ForegroundColor Yellow

# 檢查 Python
$pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
if ($pythonInstalled) {
    $pythonVersion = python --version 2>&1
    $pythonPath = (Get-Command python).Source
    Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green
    Write-Host "      路徑: $pythonPath" -ForegroundColor Gray
    
    # 檢查 Python 版本
    $versionMatch = $pythonVersion -match '(\d+)\.(\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
            $issues += "Python 版本過舊 ($pythonVersion)"
            $suggestions += "請升級到 Python 3.9 或更高版本"
        }
    }
    
    $diagnosticData['Python'] = @{
        Version = $pythonVersion
        Path = $pythonPath
        Status = "OK"
    }
} else {
    Write-Host "   ❌ Python 未安裝" -ForegroundColor Red
    $issues += "Python 未安裝"
    $suggestions += "請安裝 Python 3.9+: https://www.python.org/downloads/"
    $diagnosticData['Python'] = @{ Status = "NOT_INSTALLED" }
}

# 檢查 Node.js
$nodeInstalled = Get-Command node -ErrorAction SilentlyContinue
if ($nodeInstalled) {
    $nodeVersion = node --version 2>&1
    Write-Host "   ✅ Node.js: $nodeVersion" -ForegroundColor Green
    $diagnosticData['NodeJS'] = @{ Version = $nodeVersion; Status = "OK" }
} else {
    Write-Host "   ⚠️  Node.js 未安裝 (可選)" -ForegroundColor Yellow
    $diagnosticData['NodeJS'] = @{ Status = "NOT_INSTALLED" }
}

# 檢查 Go
$goInstalled = Get-Command go -ErrorAction SilentlyContinue
if ($goInstalled) {
    $goVersion = go version 2>&1
    Write-Host "   ✅ Go: $goVersion" -ForegroundColor Green
    $diagnosticData['Go'] = @{ Version = $goVersion; Status = "OK" }
} else {
    Write-Host "   ⚠️  Go 未安裝 (可選)" -ForegroundColor Yellow
    $diagnosticData['Go'] = @{ Status = "NOT_INSTALLED" }
}

# 檢查 Rust
$cargoInstalled = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargoInstalled) {
    $rustVersion = rustc --version 2>&1
    Write-Host "   ✅ Rust: $rustVersion" -ForegroundColor Green
    $diagnosticData['Rust'] = @{ Version = $rustVersion; Status = "OK" }
} else {
    Write-Host "   ⚠️  Rust 未安裝 (可選)" -ForegroundColor Yellow
    $diagnosticData['Rust'] = @{ Status = "NOT_INSTALLED" }
}

# 檢查 Docker
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerInstalled) {
    $dockerVersion = docker --version 2>&1
    Write-Host "   ✅ Docker: $dockerVersion" -ForegroundColor Green
    
    # 檢查 Docker 服務狀態
    $dockerRunning = docker ps 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Docker 服務運行中" -ForegroundColor Green
        $diagnosticData['Docker'] = @{ Version = $dockerVersion; Status = "RUNNING" }
    } else {
        Write-Host "      ⚠️  Docker 服務未運行" -ForegroundColor Yellow
        $issues += "Docker 服務未運行"
        $suggestions += "請啟動 Docker Desktop"
        $diagnosticData['Docker'] = @{ Version = $dockerVersion; Status = "STOPPED" }
    }
} else {
    Write-Host "   ⚠️  Docker 未安裝" -ForegroundColor Yellow
    $suggestions += "建議安裝 Docker Desktop: https://www.docker.com/products/docker-desktop"
    $diagnosticData['Docker'] = @{ Status = "NOT_INSTALLED" }
}

# ============================================
# 2. Python 套件檢查
# ============================================
Write-Host "`n🐍 檢查 Python 套件..." -ForegroundColor Yellow

if ($pythonInstalled) {
    $requiredPackages = @(
        "fastapi",
        "pydantic",
        "aiohttp",
        "sqlalchemy",
        "redis",
        "playwright"
    )
    
    $missingPackages = @()
    foreach ($package in $requiredPackages) {
        $installed = python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $package" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $package 未安裝" -ForegroundColor Red
            $missingPackages += $package
        }
    }
    
    if ($missingPackages.Count -gt 0) {
        $issues += "缺少 Python 套件: $($missingPackages -join ', ')"
        $suggestions += "執行: pip install $($missingPackages -join ' ')"
    }
    
    $diagnosticData['PythonPackages'] = @{
        Required = $requiredPackages
        Missing = $missingPackages
    }
}

# ============================================
# 3. 專案結構檢查
# ============================================
Write-Host "`n📁 檢查專案結構..." -ForegroundColor Yellow

$projectRoot = "C:\D\fold7\AIVA-git"
$requiredDirs = @(
    "services",
    "services\core",
    "services\scan",
    "services\integration",
    "docker",
    "config"
)

foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $projectRoot $dir
    if (Test-Path $fullPath) {
        Write-Host "   ✅ $dir" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $dir 不存在" -ForegroundColor Red
        $issues += "目錄缺失: $dir"
    }
}

# ============================================
# 4. 配置文件檢查
# ============================================
Write-Host "`n⚙️  檢查配置文件..." -ForegroundColor Yellow

$configFiles = @(
    "pyproject.toml",
    "requirements.txt",
    "docker\docker-compose.yml"
)

foreach ($file in $configFiles) {
    $fullPath = Join-Path $projectRoot $file
    if (Test-Path $fullPath) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $file 不存在" -ForegroundColor Yellow
    }
}

# ============================================
# 5. 端口占用檢查
# ============================================
Write-Host "`n🔌 檢查端口占用..." -ForegroundColor Yellow

$ports = @(
    @{ Port = 8000; Service = "AIVA Core API" },
    @{ Port = 8001; Service = "AIVA Admin" },
    @{ Port = 8003; Service = "Integration API" },
    @{ Port = 5672; Service = "RabbitMQ" },
    @{ Port = 15672; Service = "RabbitMQ Management" },
    @{ Port = 5432; Service = "PostgreSQL" },
    @{ Port = 6379; Service = "Redis" },
    @{ Port = 7474; Service = "Neo4j Browser" },
    @{ Port = 7687; Service = "Neo4j Bolt" }
)

foreach ($portInfo in $ports) {
    $connection = Test-NetConnection localhost -Port $portInfo.Port -WarningAction SilentlyContinue -InformationLevel Quiet
    if ($connection) {
        Write-Host "   🟢 端口 $($portInfo.Port) - $($portInfo.Service)" -ForegroundColor Green
    } else {
        Write-Host "   ⚪ 端口 $($portInfo.Port) - $($portInfo.Service) (未使用)" -ForegroundColor Gray
    }
}

# ============================================
# 6. 記憶體和磁碟檢查
# ============================================
Write-Host "`n💾 檢查系統資源..." -ForegroundColor Yellow

$os = Get-CimInstance Win32_OperatingSystem
$totalRAM = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$freeRAM = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedRAM = $totalRAM - $freeRAM

Write-Host "   記憶體: $usedRAM GB / $totalRAM GB (使用 $([math]::Round($usedRAM/$totalRAM*100, 1))%)" -ForegroundColor $(if ($usedRAM/$totalRAM -gt 0.9) { "Red" } else { "Green" })

$drive = Get-PSDrive C
$totalSpace = [math]::Round($drive.Used / 1GB + $drive.Free / 1GB, 2)
$freeSpace = [math]::Round($drive.Free / 1GB, 2)
Write-Host "   磁碟 C: $freeSpace GB 可用 / $totalSpace GB 總計" -ForegroundColor $(if ($freeSpace -lt 10) { "Red" } else { "Green" })

if ($freeSpace -lt 10) {
    $issues += "磁碟空間不足"
    $suggestions += "建議清理磁碟空間，至少保留 10GB"
}

# ============================================
# 輸出診斷報告
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📊 診斷報告" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

if ($issues.Count -eq 0) {
    Write-Host "`n✅ 系統狀態良好，未發現問題！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  發現 $($issues.Count) 個問題" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔧 發現的問題:" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "   • $issue" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "💡 修復建議:" -ForegroundColor Yellow
    foreach ($suggestion in $suggestions | Select-Object -Unique) {
        Write-Host "   • $suggestion" -ForegroundColor Yellow
    }
}

# 匯出報告
if ($ExportReport) {
    $reportPath = "aiva_diagnostic_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $diagnosticData | ConvertTo-Json -Depth 10 | Out-File $reportPath -Encoding UTF8
    Write-Host "`n📄 診斷報告已匯出: $reportPath" -ForegroundColor Cyan
}

Write-Host ""
