# AIVA Features Supplement v2 - Workers Test Script
# 測試所有補充功能模組Worker的運行狀態

param(
    [switch]$Detailed,
    [switch]$HealthCheck,
    [string]$ComposeFile = "docker-compose.features.yml"
)

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    $colorMap = @{ "Red" = "Red"; "Green" = "Green"; "Yellow" = "Yellow"; "Blue" = "Blue"; "White" = "White" }
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

function Test-ServiceStatus {
    param([string]$ServiceName)
    
    try {
        $status = docker-compose -f $ComposeFile ps $ServiceName | Select-String $ServiceName
        if ($status -match "Up") {
            Write-ColorOutput "✅ $ServiceName : Running" "Green"
            return $true
        } else {
            Write-ColorOutput "❌ $ServiceName : Not Running" "Red"
            return $false
        }
    } catch {
        Write-ColorOutput "❌ $ServiceName : Error checking status" "Red"
        return $false
    }
}

function Get-ServiceLogs {
    param([string]$ServiceName, [int]$Lines = 10)
    
    Write-ColorOutput "📋 Recent logs for $ServiceName (last $Lines lines):" "Blue"
    try {
        docker-compose -f $ComposeFile logs --tail=$Lines $ServiceName
    } catch {
        Write-ColorOutput "Unable to get logs for $ServiceName" "Red"
    }
}

Write-ColorOutput "🧪 Testing AIVA Features Supplement v2 Workers..." "Yellow"
Write-ColorOutput "===============================================" "Yellow"

# 切換到專案根目錄
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path "$scriptPath\..\.."
Set-Location $projectRoot

Write-ColorOutput "📍 Testing from: $(Get-Location)" "Blue"
Write-ColorOutput "📄 Compose file: $ComposeFile" "Blue"
Write-Host ""

# 檢查Docker Compose文件是否存在
if (-not (Test-Path $ComposeFile)) {
    Write-ColorOutput "❌ Docker Compose file not found: $ComposeFile" "Red"
    exit 1
}

# 測試服務列表
$services = @("ssrf_worker", "idor_worker", "authn_go_worker")
$results = @{}

Write-ColorOutput "🔍 Checking service status..." "Yellow"
Write-Host ""

foreach ($service in $services) {
    $results[$service] = Test-ServiceStatus -ServiceName $service
}

Write-Host ""

# 如果需要詳細輸出
if ($Detailed) {
    Write-ColorOutput "📊 Detailed Status Report:" "Yellow"
    Write-ColorOutput "=========================" "Yellow"
    
    foreach ($service in $services) {
        Write-Host ""
        Write-ColorOutput "--- $service ---" "Blue"
        
        try {
            $containerInfo = docker-compose -f $ComposeFile ps $service
            Write-Host $containerInfo
        } catch {
            Write-ColorOutput "Unable to get container info for $service" "Red"
        }
        
        if ($results[$service]) {
            Get-ServiceLogs -ServiceName $service -Lines 5
        }
    }
}

# 健康檢查
if ($HealthCheck) {
    Write-Host ""
    Write-ColorOutput "🏥 Health Check:" "Yellow"
    Write-ColorOutput "===============" "Yellow"
    
    # 檢查RabbitMQ連接
    Write-ColorOutput "Checking RabbitMQ connectivity..." "Blue"
    try {
        $rabbitStatus = docker-compose -f $ComposeFile exec -T rabbitmq rabbitmqctl status 2>$null
        if ($rabbitStatus) {
            Write-ColorOutput "✅ RabbitMQ is accessible" "Green"
        } else {
            Write-ColorOutput "⚠️  RabbitMQ status check failed" "Yellow"
        }
    } catch {
        Write-ColorOutput "⚠️  Unable to check RabbitMQ status" "Yellow"
    }
    
    # 檢查網絡連接
    Write-ColorOutput "Checking Docker network..." "Blue"
    try {
        $networkInfo = docker network ls | Select-String "aiva"
        if ($networkInfo) {
            Write-ColorOutput "✅ AIVA network exists" "Green"
        } else {
            Write-ColorOutput "⚠️  AIVA network may not exist" "Yellow"
        }
    } catch {
        Write-ColorOutput "⚠️  Unable to check network status" "Yellow"
    }
}

# 結果總結
Write-Host ""
Write-ColorOutput "📊 Test Results Summary:" "Yellow"
Write-ColorOutput "======================" "Yellow"

$successCount = ($results.Values | Where-Object { $_ -eq $true }).Count
$totalCount = $results.Count

foreach ($service in $results.Keys) {
    $status = if ($results[$service]) { "✅ RUNNING" } else { "❌ STOPPED" }
    $color = if ($results[$service]) { "Green" } else { "Red" }
    Write-ColorOutput "$service : $status" $color
}

Write-Host ""

if ($successCount -eq $totalCount) {
    Write-ColorOutput "🎉 All workers are running! ($successCount/$totalCount)" "Green"
    $exitCode = 0
} else {
    Write-ColorOutput "⚠️  Some workers are not running. ($successCount/$totalCount active)" "Yellow"
    $exitCode = 1
}

Write-Host ""
Write-ColorOutput "🔧 Useful Commands:" "Blue"
Write-ColorOutput "Start all: docker-compose -f $ComposeFile up -d" "White"
Write-ColorOutput "Stop all: docker-compose -f $ComposeFile down" "White"  
Write-ColorOutput "View logs: docker-compose -f $ComposeFile logs -f [service_name]" "White"
Write-ColorOutput "Restart: docker-compose -f $ComposeFile restart [service_name]" "White"

exit $exitCode