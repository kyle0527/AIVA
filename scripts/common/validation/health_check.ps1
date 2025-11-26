# AIVA 系統健康檢查工具 (現代化版本)
# 日期: 2025-11-25
# 用途: 檢查所有服務的運行狀態

param(
    [switch]$Continuous,
    [int]$Interval = 30,
    [switch]$ExportReport
)

$ErrorActionPreference = "SilentlyContinue"

function Test-ServiceHealth {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Host,
        [int]$Port,
        [int]$ExpectedStatus = 200
    )
    
    $result = @{
        Name = $Name
        Status = "Unknown"
        ResponseTime = $null
        Details = ""
    }
    
    if ($Url) {
        try {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
            $stopwatch.Stop()
            
            $result.ResponseTime = $stopwatch.ElapsedMilliseconds
            
            if ($response.StatusCode -eq $ExpectedStatus) {
                $result.Status = "Healthy"
                $result.Details = "HTTP $($response.StatusCode) - $($result.ResponseTime)ms"
            } else {
                $result.Status = "Warning"
                $result.Details = "HTTP $($response.StatusCode)"
            }
        } catch {
            $result.Status = "Down"
            $result.Details = "無法連接"
        }
    } elseif ($Host -and $Port) {
        try {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $tcpClient.Connect($Host, $Port)
            $stopwatch.Stop()
            $tcpClient.Close()
            
            $result.Status = "Healthy"
            $result.ResponseTime = $stopwatch.ElapsedMilliseconds
            $result.Details = "TCP $Host`:$Port - $($result.ResponseTime)ms"
        } catch {
            $result.Status = "Down"
            $result.Details = "無法連接 $Host`:$Port"
        }
    }
    
    return $result
}

function Show-HealthCheck {
    Clear-Host
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "🔍 AIVA 系統健康檢查" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "檢查時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # ============================================
    # 1. 基礎設施檢查
    # ============================================
    Write-Host "🏗️  基礎設施狀態" -ForegroundColor Yellow
    Write-Host "-----------------------------------" -ForegroundColor Gray
    
    # 檢查 Docker
    $dockerRunning = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker 容器運行中:" -ForegroundColor Green
        $containers = docker ps --format "{{.Names}}" 2>$null
        foreach ($container in $containers) {
            $status = docker inspect --format='{{.State.Status}}' $container 2>$null
            $uptime = docker inspect --format='{{.State.StartedAt}}' $container 2>$null
            Write-Host "   🐳 $container - $status" -ForegroundColor $(if ($status -eq "running") { "Green" } else { "Red" })
        }
    } else {
        Write-Host "❌ Docker 未運行或無容器" -ForegroundColor Red
    }
    
    # ============================================
    # 2. Python 服務檢查
    # ============================================
    Write-Host "`n🐍 Python 服務" -ForegroundColor Yellow
    Write-Host "-----------------------------------" -ForegroundColor Gray
    
    $pythonServices = @(
        @{ Name = "AIVA Core"; Url = "http://localhost:8000/health" },
        @{ Name = "AIVA Core API Docs"; Url = "http://localhost:8000/docs" },
        @{ Name = "Integration Service"; Url = "http://localhost:8003/health" }
    )
    
    foreach ($service in $pythonServices) {
        $health = Test-ServiceHealth -Name $service.Name -Url $service.Url
        $icon = switch ($health.Status) {
            "Healthy" { "✅" }
            "Warning" { "⚠️" }
            "Down" { "❌" }
            default { "❓" }
        }
        $color = switch ($health.Status) {
            "Healthy" { "Green" }
            "Warning" { "Yellow" }
            "Down" { "Red" }
            default { "Gray" }
        }
        Write-Host "   $icon $($health.Name): $($health.Details)" -ForegroundColor $color
    }
    
    # ============================================
    # 3. 基礎設施服務檢查
    # ============================================
    Write-Host "`n🔧 基礎設施服務" -ForegroundColor Yellow
    Write-Host "-----------------------------------" -ForegroundColor Gray
    
    $infrastructureServices = @(
        @{ Name = "RabbitMQ Management"; Url = "http://localhost:15672" },
        @{ Name = "RabbitMQ AMQP"; Host = "localhost"; Port = 5672 },
        @{ Name = "PostgreSQL"; Host = "localhost"; Port = 5432 },
        @{ Name = "Redis"; Host = "localhost"; Port = 6379 },
        @{ Name = "Neo4j Browser"; Url = "http://localhost:7474" },
        @{ Name = "Neo4j Bolt"; Host = "localhost"; Port = 7687 }
    )
    
    foreach ($service in $infrastructureServices) {
        $health = Test-ServiceHealth @service
        $icon = switch ($health.Status) {
            "Healthy" { "✅" }
            "Warning" { "⚠️" }
            "Down" { "❌" }
            default { "❓" }
        }
        $color = switch ($health.Status) {
            "Healthy" { "Green" }
            "Warning" { "Yellow" }
            "Down" { "Red" }
            default { "Gray" }
        }
        Write-Host "   $icon $($health.Name): $($health.Details)" -ForegroundColor $color
    }
    
    # ============================================
    # 4. 系統資源檢查
    # ============================================
    Write-Host "`n💻 系統資源" -ForegroundColor Yellow
    Write-Host "-----------------------------------" -ForegroundColor Gray
    
    # CPU 使用率
    $cpu = Get-WmiObject Win32_Processor | Measure-Object -Property LoadPercentage -Average
    $cpuUsage = $cpu.Average
    $cpuColor = if ($cpuUsage -gt 80) { "Red" } elseif ($cpuUsage -gt 60) { "Yellow" } else { "Green" }
    Write-Host "   CPU: $cpuUsage%" -ForegroundColor $cpuColor
    
    # 記憶體使用率
    $os = Get-CimInstance Win32_OperatingSystem
    $totalRAM = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freeRAM = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $usedRAM = $totalRAM - $freeRAM
    $ramPercent = [math]::Round($usedRAM / $totalRAM * 100, 1)
    $ramColor = if ($ramPercent -gt 90) { "Red" } elseif ($ramPercent -gt 75) { "Yellow" } else { "Green" }
    Write-Host "   記憶體: $usedRAM GB / $totalRAM GB ($ramPercent%)" -ForegroundColor $ramColor
    
    # 磁碟空間
    $drive = Get-PSDrive C
    $freeSpace = [math]::Round($drive.Free / 1GB, 2)
    $diskColor = if ($freeSpace -lt 10) { "Red" } elseif ($freeSpace -lt 50) { "Yellow" } else { "Green" }
    Write-Host "   磁碟 C: $freeSpace GB 可用" -ForegroundColor $diskColor
    
    # ============================================
    # 5. Python 進程檢查
    # ============================================
    Write-Host "`n🔍 Python 進程" -ForegroundColor Yellow
    Write-Host "-----------------------------------" -ForegroundColor Gray
    
    $pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        Write-Host "   發現 $($pythonProcesses.Count) 個 Python 進程" -ForegroundColor Green
        foreach ($proc in $pythonProcesses | Select-Object -First 5) {
            $memMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
            Write-Host "   • PID $($proc.Id): $memMB MB" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ⚠️  未檢測到 Python 進程" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    if ($Continuous) {
        Write-Host "下次檢查: $Interval 秒後 (按 Ctrl+C 停止)" -ForegroundColor Gray
    }
}

# 主循環
if ($Continuous) {
    Write-Host "持續監控模式啟動 (每 $Interval 秒檢查一次)" -ForegroundColor Cyan
    Write-Host "按 Ctrl+C 停止監控" -ForegroundColor Yellow
    Write-Host ""
    
    while ($true) {
        Show-HealthCheck
        Start-Sleep -Seconds $Interval
    }
} else {
    Show-HealthCheck
}

Write-Host "========================================" -ForegroundColor Cyan
