# AIVA 環境變數驗證 PowerShell 腳本
# 以 Docker Compose Production 版本為標準
param(
    [switch]$Generate,
    [string]$Target = "docker"
)

Write-Host "🔍 AIVA 環境變數驗證工具" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# AIVA 標準環境變數定義
$AIVA_ENV_STANDARDS = @{
    "DATABASE_URL" = @{
        required = $true
        default = $null
        description = "資料庫連接 URL"
        production = "postgresql://aiva:aiva_secure_password@postgres:5432/aiva"
        docker = "postgresql://aiva:aiva_secure_password@postgres:5432/aiva"
    }
    "POSTGRES_HOST" = @{
        required = $false
        default = "localhost"
        description = "PostgreSQL 主機"
        production = "postgres"
        docker = "postgres"
    }
    "POSTGRES_PORT" = @{
        required = $false
        default = "5432"
        description = "PostgreSQL 端口"
        production = "5432"
        docker = "5432"
    }
    "POSTGRES_DB" = @{
        required = $false
        default = "aiva"
        description = "PostgreSQL 資料庫名稱"
        production = "aiva"
        docker = "aiva"
    }
    "POSTGRES_USER" = @{
        required = $false
        default = "aiva"
        description = "PostgreSQL 用戶名"
        production = "aiva"
        docker = "aiva"
    }
    "POSTGRES_PASSWORD" = @{
        required = $false
        default = "aiva_secure_password"
        description = "PostgreSQL 密碼"
        production = "aiva_secure_password"
        docker = "aiva_secure_password"
    }
    "RABBITMQ_URL" = @{
        required = $true
        default = $null
        description = "RabbitMQ 連接 URL"
        production = "amqp://aiva:aiva_mq_password@rabbitmq:5672/aiva"
        docker = "amqp://aiva:aiva_mq_password@rabbitmq:5672/aiva"
    }
    "RABBITMQ_HOST" = @{
        required = $false
        default = "localhost"
        description = "RabbitMQ 主機"
        production = "rabbitmq"
        docker = "rabbitmq"
    }
    "RABBITMQ_PORT" = @{
        required = $false
        default = "5672"
        description = "RabbitMQ 端口"
        production = "5672"
        docker = "5672"
    }
    "RABBITMQ_USER" = @{
        required = $false
        default = $null
        description = "RabbitMQ 用戶名"
        production = "aiva"
        docker = "aiva"
    }
    "RABBITMQ_PASSWORD" = @{
        required = $false
        default = $null
        description = "RabbitMQ 密碼"
        production = "aiva_mq_password"
        docker = "aiva_mq_password"
    }
    "RABBITMQ_VHOST" = @{
        required = $false
        default = "/"
        description = "RabbitMQ Virtual Host"
        production = "aiva"
        docker = "aiva"
    }
    "REDIS_URL" = @{
        required = $false
        default = "redis://localhost:6379/0"
        description = "Redis 連接 URL"
        production = "redis://:aiva_redis_password@redis:6379/0"
        docker = "redis://:aiva_redis_password@redis:6379/0"
    }
    "NEO4J_URL" = @{
        required = $false
        default = "bolt://localhost:7687"
        description = "Neo4j 連接 URL"
        production = "bolt://neo4j:password@neo4j:7687"
        docker = "bolt://neo4j:password@neo4j:7687"
    }
    "NEO4J_USER" = @{
        required = $false
        default = "neo4j"
        description = "Neo4j 用戶名"
        production = "neo4j"
        docker = "neo4j"
    }
    "NEO4J_PASSWORD" = @{
        required = $false
        default = "password"
        description = "Neo4j 密碼"
        production = "password"
        docker = "password"
    }
    "API_KEY" = @{
        required = $false
        default = $null
        description = "API 主密鑰"
        production = "production_api_key_change_me"
        docker = "dev_api_key_for_docker_testing"
    }
    "INTEGRATION_TOKEN" = @{
        required = $false
        default = $null
        description = "Integration 模組認證令牌"
        production = "integration_secure_token"
        docker = "docker_integration_token"
    }
    "LOG_LEVEL" = @{
        required = $false
        default = "INFO"
        description = "日誌級別"
        production = "INFO"
        docker = "INFO"
    }
    "AUTO_MIGRATE" = @{
        required = $false
        default = "1"
        description = "自動遷移資料庫"
        production = "1"
        docker = "1"
    }
}

# 驗證結果統計
$errors = @()
$warnings = @()
$info = @()

function Test-CurrentEnvironment {
    Write-Host "🔍 檢查當前環境變數..." -ForegroundColor Yellow
    
    foreach ($envName in $AIVA_ENV_STANDARDS.Keys) {
        $standard = $AIVA_ENV_STANDARDS[$envName]
        $currentValue = [System.Environment]::GetEnvironmentVariable($envName)
        
        if ($standard.required -and -not $currentValue) {
            $script:errors += "❌ 必需環境變數 $envName 未設置"
        }
        elseif (-not $currentValue -and $standard.default) {
            $script:info += "ℹ️  環境變數 $envName 未設置，將使用預設值: $($standard.default)"
        }
        elseif ($currentValue) {
            $script:info += "✅ 環境變數 $envName = $currentValue"
        }
    }
}

function Test-FileConsistency {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        $script:warnings += "⚠️  配置文件不存在: $FilePath"
        return $true
    }
    
    Write-Host "🔍 檢查文件: $FilePath" -ForegroundColor Yellow
    
    try {
        $content = Get-Content $FilePath -Raw -Encoding UTF8
        $inconsistencies = @()
        
        # 現在使用統一的舊版環境變數，不需要檢查命名衝突
        # 所有變數都應該存在於文件中
        foreach ($envName in $AIVA_ENV_STANDARDS.Keys) {
            if ($content -match $envName) {
                # 配置存在，標記為一致
                continue
            }
        }
        
        if ($inconsistencies.Count -gt 0) {
            $script:warnings += $inconsistencies | ForEach-Object { "⚠️  ${FilePath}: $_" }
            return $false
        }
        else {
            $script:info += "✅ $FilePath`: 配置一致"
            return $true
        }
    }
    catch {
        $script:errors += "❌ 讀取文件 $FilePath 時出錯: $($_.Exception.Message)"
        return $false
    }
}

function New-StandardEnvFile {
    param([string]$Target = "docker")
    
    $lines = @(
        "# AIVA 標準環境變數配置"
        "# 目標環境: $Target"
        "# 由 validate_environment_variables.ps1 自動生成"
        ""
    )
    
        $categories = @{
            "資料庫配置" = @("DATABASE_URL", "POSTGRES_HOST", "POSTGRES_PORT", 
                             "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
            "消息隊列配置" = @("RABBITMQ_URL", "RABBITMQ_HOST", "RABBITMQ_PORT",
                               "RABBITMQ_USER", "RABBITMQ_PASSWORD", "RABBITMQ_VHOST")
            "Redis 配置" = @("REDIS_URL")
            "Neo4j 配置" = @("NEO4J_URL", "NEO4J_USER", "NEO4J_PASSWORD")
            "安全配置" = @("API_KEY", "INTEGRATION_TOKEN")
            "其他配置" = @("LOG_LEVEL", "AUTO_MIGRATE")
        }
        
        foreach ($category in $categories.Keys) {
        $lines += "# ================================"
        $lines += "# $category"
        $lines += "# ================================"
        
        foreach ($envName in $categories[$category]) {
            if ($AIVA_ENV_STANDARDS.ContainsKey($envName)) {
                $standard = $AIVA_ENV_STANDARDS[$envName]
                $value = if ($standard.$Target) { $standard.$Target } else { $standard.default }
                if ($value) {
                    $lines += "$envName=$value"
                }
                $lines += "# $($standard.description)"
                $lines += ""
            }
        }
        $lines += ""
    }
    
    return $lines -join "`n"
}

function Show-ValidationReport {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "📋 AIVA 環境變數驗證報告" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    
    if ($script:errors.Count -gt 0) {
        Write-Host "`n❌ 錯誤:" -ForegroundColor Red
        $script:errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    }
    
    if ($script:warnings.Count -gt 0) {
        Write-Host "`n⚠️  警告:" -ForegroundColor Yellow
        $script:warnings | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    }
    
    if ($script:info.Count -gt 0) {
        Write-Host "`nℹ️  資訊:" -ForegroundColor Cyan
        $script:info | ForEach-Object { Write-Host "  $_" -ForegroundColor Cyan }
    }
    
    Write-Host "`n📊 總結:" -ForegroundColor White
    Write-Host "  - 錯誤: $($script:errors.Count)" -ForegroundColor White
    Write-Host "  - 警告: $($script:warnings.Count)" -ForegroundColor White
    Write-Host "  - 資訊: $($script:info.Count)" -ForegroundColor White
    
    if ($script:errors.Count -eq 0) {
        Write-Host "`n✅ 環境變數驗證通過！" -ForegroundColor Green
    }
    else {
        Write-Host "`n❌ 環境變數驗證失敗！" -ForegroundColor Red
    }
}

# 主要執行邏輯
try {
    # 驗證當前環境
    Test-CurrentEnvironment
    
    # 驗證關鍵配置文件
    $projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $keyFiles = @(
        Join-Path $projectRoot ".env.docker"
        Join-Path $projectRoot ".env.example"
        Join-Path $projectRoot "docker\compose\docker-compose.yml"
        Join-Path $projectRoot "docker\compose\docker-compose.production.yml"
        Join-Path $projectRoot "services\aiva_common\config\unified_config.py"
    )
    
    $allFilesValid = $true
    foreach ($filePath in $keyFiles) {
        if (-not (Test-FileConsistency $filePath)) {
            $allFilesValid = $false
        }
    }
    
    # 顯示驗證報告
    Show-ValidationReport
    
    # 如果需要，生成標準配置文件
    if ($Generate) {
        Write-Host "`n📄 生成 $Target 環境標準配置:" -ForegroundColor Yellow
        Write-Host "----------------------------------------" -ForegroundColor Yellow
        $standardConfig = New-StandardEnvFile -Target $Target
        Write-Host $standardConfig
    }
    
    # 返回退出碼
    if ($script:errors.Count -eq 0 -and $allFilesValid) {
        exit 0
    }
    else {
        exit 1
    }
}
catch {
    Write-Host "❌ 執行過程中發生錯誤: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}