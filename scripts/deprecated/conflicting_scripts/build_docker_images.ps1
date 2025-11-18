# AIVA Features Supplement v2 - Docker Build Script (Windows PowerShell)
# 構建所有補充功能模組的Docker映像

param(
    [switch]$SkipSSRF,
    [switch]$SkipIDOR, 
    [switch]$SkipAUTHN,
    [switch]$Verbose
)

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = "Red"
        "Green" = "Green" 
        "Yellow" = "Yellow"
        "Blue" = "Blue"
        "White" = "White"
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

function Build-DockerImage {
    param(
        [string]$ServiceName,
        [string]$DockerfilePath,
        [string]$ImageTag
    )
    
    Write-ColorOutput "Building $ServiceName..." "Yellow"
    
    $buildCmd = "docker build -f `"$DockerfilePath`" -t `"$ImageTag`" ."
    
    if ($Verbose) {
        Write-ColorOutput "Command: $buildCmd" "Blue"
    }
    
    try {
        Invoke-Expression $buildCmd
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Successfully built $ImageTag" "Green"
            return $true
        } else {
            Write-ColorOutput "❌ Failed to build $ImageTag (Exit Code: $LASTEXITCODE)" "Red"
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Exception building $ImageTag : $_" "Red"
        return $false
    }
}

Write-ColorOutput "🚀 Building AIVA Features Supplement v2 Docker Images..." "Yellow"
Write-ColorOutput "==================================================" "Yellow"

# 切換到專案根目錄
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path "$scriptPath\..\.."
Set-Location $projectRoot

Write-ColorOutput "📍 Current directory: $(Get-Location)" "Blue"
Write-Host ""

$buildResults = @{}

# 構建SSRF Worker
if (-not $SkipSSRF) {
    Write-ColorOutput "1. Building SSRF Worker..." "Yellow"
    $buildResults["SSRF"] = Build-DockerImage -ServiceName "SSRF Worker" -DockerfilePath "services\features\function_ssrf\Dockerfile" -ImageTag "aiva/ssrf_worker:latest"
    Write-Host ""
}

# 構建IDOR Worker
if (-not $SkipIDOR) {
    Write-ColorOutput "2. Building IDOR Worker..." "Yellow"  
    $buildResults["IDOR"] = Build-DockerImage -ServiceName "IDOR Worker" -DockerfilePath "services\features\function_idor\Dockerfile" -ImageTag "aiva/idor_worker:latest"
    Write-Host ""
}

# 構建AUTHN GO Worker
if (-not $SkipAUTHN) {
    Write-ColorOutput "3. Building AUTHN GO Worker..." "Yellow"
    $buildResults["AUTHN_GO"] = Build-DockerImage -ServiceName "AUTHN GO Worker" -DockerfilePath "services\features\function_authn_go\Dockerfile" -ImageTag "aiva/authn_go_worker:latest"
    Write-Host ""
}

# 結果總結
Write-ColorOutput "📊 Build Summary:" "Yellow"
Write-ColorOutput "================" "Yellow"

$successCount = 0
$totalCount = $buildResults.Count

foreach ($module in $buildResults.Keys) {
    $status = if ($buildResults[$module]) { "✅ SUCCESS" } else { "❌ FAILED" }
    $color = if ($buildResults[$module]) { "Green" } else { "Red" }
    
    Write-ColorOutput "$module : $status" $color
    
    if ($buildResults[$module]) {
        $successCount++
    }
}

Write-Host ""

if ($successCount -eq $totalCount) {
    Write-ColorOutput "🎉 All images built successfully! ($successCount/$totalCount)" "Green"
} else {
    Write-ColorOutput "⚠️  Some builds failed. ($successCount/$totalCount successful)" "Yellow"
}

Write-Host ""

# 顯示構建的映像
Write-ColorOutput "📦 Built Images:" "Blue"
try {
    docker images | Select-String "aiva/" | Select-String "ssrf_worker|idor_worker|authn_go_worker"
} catch {
    Write-ColorOutput "Unable to list images. Please run 'docker images' manually." "Yellow"
}

Write-Host ""
Write-ColorOutput "🔧 Next Steps:" "Blue"
Write-ColorOutput "1. Run: docker-compose -f docker-compose.features.yml up -d" "White"
Write-ColorOutput "2. Check logs: docker-compose -f docker-compose.features.yml logs -f" "White"
Write-ColorOutput "3. Validate: .\scripts\features\test_workers.ps1" "White"