# AIVA Docker 映像檔快速建立腳本
# 建立日期: 2025-10-30
# 作者: AIVA Team
# 功能: 自動建立所有 AIVA Docker 映像檔

param(
    [Parameter(HelpMessage="指定要建立的映像檔類型 (all, core, component, minimal, integration)")]
    [ValidateSet("all", "core", "component", "minimal", "integration")]
    [string]$Type = "all",
    
    [Parameter(HelpMessage="映像檔標籤版本")]
    [string]$Tag = "latest",
    
    [Parameter(HelpMessage="是否使用建立緩存")]
    [switch]$NoCache = $false,
    
    [Parameter(HelpMessage="是否顯示詳細建立過程")]
    [switch]$Verbose = $false,
    
    [Parameter(HelpMessage="建立後是否自動清理")]
    [switch]$CleanUp = $false,
    
    [Parameter(HelpMessage="建立環境 (development, staging, production)")]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development"
)

# 設定工作目錄
$WorkingDir = Get-Location
$ProjectRoot = Split-Path $PSScriptRoot -Parent

# 顏色輸出函數
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️  $Message" "Cyan" }
function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" "Green" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️  $Message" "Yellow" }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" "Red" }

# 檢查 Docker 是否可用
function Test-Docker {
    Write-Info "檢查 Docker 環境..."
    try {
        $dockerVersion = docker --version
        Write-Success "Docker 可用: $dockerVersion"
        return $true
    }
    catch {
        Write-Error "Docker 未安裝或無法連接到 Docker daemon"
        return $false
    }
}

# 建立單個映像檔
function Build-DockerImage {
    param(
        [string]$ImageName,
        [string]$DockerfilePath,
        [string]$Context = ".",
        [hashtable]$BuildArgs = @{}
    )
    
    Write-Info "開始建立映像檔: $ImageName"
    
    # 建立命令
    $buildCmd = @("docker", "build")
    
    # 添加參數
    if ($NoCache) { $buildCmd += "--no-cache" }
    if ($Verbose) { $buildCmd += "--progress=plain" }
    
    # 添加建立參數
    foreach ($arg in $BuildArgs.GetEnumerator()) {
        $buildCmd += "--build-arg"
        $buildCmd += "$($arg.Key)=$($arg.Value)"
    }
    
    # 添加 Dockerfile 和標籤
    $buildCmd += "-f", $DockerfilePath
    $buildCmd += "-t", "$ImageName`:$Tag"
    $buildCmd += $Context
    
    Write-Info "執行命令: $($buildCmd -join ' ')"
    
    # 執行建立
    $startTime = Get-Date
    try {
        & $buildCmd[0] $buildCmd[1..$buildCmd.Count]
        if ($LASTEXITCODE -eq 0) {
            $duration = (Get-Date) - $startTime
            Write-Success "映像檔 $ImageName 建立成功 (耗時: $($duration.TotalSeconds.ToString('F1'))s)"
            return $true
        } else {
            Write-Error "映像檔 $ImageName 建立失敗"
            return $false
        }
    }
    catch {
        Write-Error "建立過程中發生錯誤: $($_.Exception.Message)"
        return $false
    }
}

# 驗證映像檔
function Test-DockerImage {
    param([string]$ImageName)
    
    Write-Info "驗證映像檔: $ImageName"
    try {
        $result = docker run --rm "$ImageName`:$Tag" python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "映像檔 $ImageName 驗證通過: $result"
            return $true
        } else {
            Write-Warning "映像檔 $ImageName 驗證失敗"
            return $false
        }
    }
    catch {
        Write-Warning "無法驗證映像檔 $ImageName`: $($_.Exception.Message)"
        return $false
    }
}

# 顯示映像檔資訊
function Show-ImageInfo {
    param([string]$ImageName)
    
    Write-Info "映像檔資訊: $ImageName"
    try {
        $imageInfo = docker images "$ImageName`:$Tag" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        Write-Host $imageInfo
    }
    catch {
        Write-Warning "無法獲取映像檔資訊: $($_.Exception.Message)"
    }
}

# 清理建立緩存
function Clear-BuildCache {
    Write-Info "清理 Docker 建立緩存..."
    try {
        docker builder prune -f
        Write-Success "建立緩存清理完成"
    }
    catch {
        Write-Warning "清理建立緩存失敗: $($_.Exception.Message)"
    }
}

# 主要建立流程
function Start-Build {
    Write-ColorOutput "🚀 AIVA Docker 映像檔建立器" "Magenta"
    Write-ColorOutput "=====================================`n" "Magenta"
    
    # 檢查環境
    if (-not (Test-Docker)) {
        exit 1
    }
    
    # 切換到專案根目錄
    Set-Location $ProjectRoot
    Write-Info "工作目錄: $(Get-Location)"
    
    # 建立參數
    $buildArgs = @{
        "ENVIRONMENT" = $Environment
        "BUILD_DATE" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        "LOG_LEVEL" = if ($Environment -eq "production") { "INFO" } else { "DEBUG" }
    }
    
    # 定義映像檔配置
    $imageConfigs = @{
        "core" = @{
            Name = "aiva-core"
            Dockerfile = "docker/core/Dockerfile.core"
            Description = "AIVA 核心 AI 服務"
        }
        "component" = @{
            Name = "aiva-component"
            Dockerfile = "docker/components/Dockerfile.component"
            Description = "AIVA 功能組件服務"
        }
        "minimal" = @{
            Name = "aiva-core-minimal"
            Dockerfile = "docker/core/Dockerfile.core.minimal"
            Description = "AIVA 最小化核心服務"
        }
        "integration" = @{
            Name = "aiva-integration"
            Dockerfile = "docker/infrastructure/Dockerfile.integration"
            Description = "AIVA 整合服務"
        }
    }
    
    # 決定要建立的映像檔
    $buildTargets = if ($Type -eq "all") { $imageConfigs.Keys } else { @($Type) }
    
    Write-Info "將建立以下映像檔: $($buildTargets -join ', ')"
    Write-Info "映像檔標籤: $Tag"
    Write-Info "建立環境: $Environment"
    Write-Host ""
    
    # 開始建立
    $totalStartTime = Get-Date
    $successCount = 0
    $totalCount = $buildTargets.Count
    
    foreach ($target in $buildTargets) {
        if ($imageConfigs.ContainsKey($target)) {
            $config = $imageConfigs[$target]
            Write-ColorOutput "`n📦 建立 $($config.Description)" "Yellow"
            Write-ColorOutput "─────────────────────────────────────" "Yellow"
            
            # 檢查 Dockerfile 是否存在
            if (-not (Test-Path $config.Dockerfile)) {
                Write-Error "Dockerfile 不存在: $($config.Dockerfile)"
                continue
            }
            
            # 建立映像檔
            if (Build-DockerImage -ImageName $config.Name -DockerfilePath $config.Dockerfile -BuildArgs $buildArgs) {
                # 驗證映像檔
                if (Test-DockerImage -ImageName $config.Name) {
                    Show-ImageInfo -ImageName $config.Name
                    $successCount++
                }
            }
        } else {
            Write-Error "未知的映像檔類型: $target"
        }
    }
    
    # 建立總結
    $totalDuration = (Get-Date) - $totalStartTime
    Write-ColorOutput "`n🎯 建立總結" "Magenta"
    Write-ColorOutput "=================" "Magenta"
    Write-Info "成功建立: $successCount/$totalCount 個映像檔"
    Write-Info "總耗時: $($totalDuration.TotalMinutes.ToString('F1')) 分鐘"
    
    # 顯示所有 AIVA 映像檔
    Write-Info "`n所有 AIVA 映像檔:"
    try {
        docker images | Select-String "aiva"
    }
    catch {
        Write-Warning "無法列出映像檔"
    }
    
    # 清理
    if ($CleanUp) {
        Clear-BuildCache
    }
    
    # 回到原始目錄
    Set-Location $WorkingDir
    
    if ($successCount -eq $totalCount) {
        Write-Success "`n🎉 所有映像檔建立完成！"
        exit 0
    } else {
        Write-Error "`n⚠️  部分映像檔建立失敗"
        exit 1
    }
}

# 顯示幫助資訊
function Show-Help {
    Write-ColorOutput "AIVA Docker 映像檔建立器" "Magenta"
    Write-ColorOutput "======================" "Magenta"
    Write-Host ""
    Write-Host "用法："
    Write-Host "  .\build-docker-images.ps1 [參數]"
    Write-Host ""
    Write-Host "參數："
    Write-Host "  -Type <類型>       指定建立類型 (all, core, component, minimal, integration)"
    Write-Host "  -Tag <標籤>        映像檔標籤版本 (預設: latest)"
    Write-Host "  -NoCache          不使用建立緩存"
    Write-Host "  -Verbose          顯示詳細建立過程"
    Write-Host "  -CleanUp          建立後清理緩存"
    Write-Host "  -Environment <環境> 建立環境 (development, staging, production)"
    Write-Host ""
    Write-Host "範例："
    Write-Host "  .\build-docker-images.ps1 -Type all -Tag v1.0.0"
    Write-Host "  .\build-docker-images.ps1 -Type core -Environment production"
    Write-Host "  .\build-docker-images.ps1 -Type component -NoCache -Verbose"
    Write-Host ""
}

# 檢查是否請求幫助
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Show-Help
    exit 0
}

# 執行主要建立流程
try {
    Start-Build
}
catch {
    Write-Error "腳本執行失敗: $($_.Exception.Message)"
    Write-Error $_.ScriptStackTrace
    exit 1
}