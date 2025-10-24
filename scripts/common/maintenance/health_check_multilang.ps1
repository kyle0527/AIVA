# AIVA 多語言系統健康檢查腳本
# 日期: 2025-10-15
# 用途: 檢查所有啟動的服務是否正常運行

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🔍 AIVA 多語言系統 - 健康檢查" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "SilentlyContinue"

# ============================================
# 1. 基礎設施檢查
# ============================================
Write-Host "🏗️  基礎設施狀態" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

# 檢查 Docker 容器
$dockerContainers = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker 容器狀態:" -ForegroundColor Green
    Write-Host $dockerContainers
} else {
    Write-Host "❌ Docker 未運行" -ForegroundColor Red
}

# ============================================
# 2. Web 服務檢查
# ============================================
Write-Host "`n🌐 Web 服務端點檢查" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

$endpoints = @(
    @{ Name = "Core API"; Url = "http://localhost:8001/health"; Expected = 200 },
    @{ Name = "Integration API"; Url = "http://localhost:8003/health"; Expected = 200 },
    @{ Name = "RabbitMQ Management"; Url = "http://localhost:15672"; Expected = 200 },
    @{ Name = "PostgreSQL"; Host = "localhost"; Port = 5432 },
    @{ Name = "Redis"; Host = "localhost"; Port = 6379 },
    @{ Name = "Neo4j"; Url = "http://localhost:7474"; Expected = 200 }
)

foreach ($endpoint in $endpoints) {
    if ($endpoint.Url) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq $endpoint.Expected) {
                Write-Host "   ✅ $($endpoint.Name): 正常" -ForegroundColor Green
            } else {
                Write-Host "   ⚠️  $($endpoint.Name): 狀態碼 $($response.StatusCode)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "   ❌ $($endpoint.Name): 無法連接" -ForegroundColor Red
        }
    } elseif ($endpoint.Host -and $endpoint.Port) {
        try {
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect($endpoint.Host, $endpoint.Port)
            $tcpClient.Close()
            Write-Host "   ✅ $($endpoint.Name): 正常" -ForegroundColor Green
        } catch {
            Write-Host "   ❌ $($endpoint.Name): 無法連接 $($endpoint.Host):$($endpoint.Port)" -ForegroundColor Red
        }
    }
}

# ============================================
# 3. 進程檢查
# ============================================
Write-Host "`n⚡ 運行進程檢查" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

$processChecks = @(
    @{ Name = "Python 進程"; Pattern = "python" },
    @{ Name = "Node.js 進程"; Pattern = "node" },
    @{ Name = "Go 進程"; Pattern = "go" },
    @{ Name = "Rust 進程"; Pattern = "cargo" }
)

foreach ($check in $processChecks) {
    $processes = Get-Process | Where-Object { $_.ProcessName -like "*$($check.Pattern)*" }
    if ($processes) {
        Write-Host "   ✅ $($check.Name): $($processes.Count) 個運行中" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️  $($check.Name): 未檢測到" -ForegroundColor Gray
    }
}

# ============================================
# 4. 資源使用情況
# ============================================
Write-Host "`n📊 系統資源使用" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Gray

# CPU 使用率
$cpu = Get-Counter -Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1
$cpuUsage = [math]::Round(100 - $cpu.CounterSamples.CookedValue, 2)
Write-Host "   💻 CPU 使用率: $cpuUsage%" -ForegroundColor White

# 記憶體使用
$memory = Get-CimInstance -ClassName Win32_OperatingSystem
$totalMemory = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
$freeMemory = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
$usedMemory = $totalMemory - $freeMemory
$memoryUsagePercent = [math]::Round(($usedMemory / $totalMemory) * 100, 2)
Write-Host "   🧠 記憶體使用: $usedMemory GB / $totalMemory GB ($memoryUsagePercent%)" -ForegroundColor White

# 磁碟使用 (C: 磁碟機)
$disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
$totalDisk = [math]::Round($disk.Size / 1GB, 2)
$freeDisk = [math]::Round($disk.FreeSpace / 1GB, 2)
$usedDisk = $totalDisk - $freeDisk
$diskUsagePercent = [math]::Round(($usedDisk / $totalDisk) * 100, 2)
Write-Host "   💽 磁碟使用 (C:): $usedDisk GB / $totalDisk GB ($diskUsagePercent%)" -ForegroundColor White

# ============================================
# 總結
# ============================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "📋 健康檢查完成" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 提示:" -ForegroundColor Yellow
Write-Host "   • 如果發現問題，請使用 start_all_multilang.ps1 重新啟動" -ForegroundColor White
Write-Host "   • 檢查日誌檔案以獲取詳細錯誤信息" -ForegroundColor White
Write-Host "   • 確保所有依賴項都已正確安裝" -ForegroundColor White
Write-Host ""
