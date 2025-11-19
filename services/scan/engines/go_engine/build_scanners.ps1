# Go 掃描器構建腳本
# 用於構建所有 Go 掃描器的二進制文件

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Go Scanners for AIVA" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 獲取當前腳本目錄
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 檢查 Go 是否安裝
try {
    $goVersion = go version
    Write-Host "✓ Go is installed: $goVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Go is not installed. Please install Go 1.21 or later." -ForegroundColor Red
    exit 1
}

# 掃描器列表
$scanners = @(
    @{Name="SSRF Scanner"; Dir="ssrf_scanner"; Output="worker.exe"},
    @{Name="CSPM Scanner"; Dir="cspm_scanner"; Output="worker.exe"},
    @{Name="SCA Scanner"; Dir="sca_scanner"; Output="worker.exe"}
)

$successCount = 0
$failCount = 0

foreach ($scanner in $scanners) {
    Write-Host ""
    Write-Host "Building $($scanner.Name)..." -ForegroundColor Yellow
    Write-Host "  Directory: $($scanner.Dir)" -ForegroundColor Gray
    
    $scannerPath = Join-Path $scriptDir $scanner.Dir
    
    if (-Not (Test-Path $scannerPath)) {
        Write-Host "  ✗ Directory not found: $scannerPath" -ForegroundColor Red
        $failCount++
        continue
    }
    
    Push-Location $scannerPath
    
    try {
        # 檢查 main.go 是否存在
        if (-Not (Test-Path "main.go")) {
            Write-Host "  ✗ main.go not found" -ForegroundColor Red
            $failCount++
            Pop-Location
            continue
        }
        
        # 下載依賴
        Write-Host "  → Downloading dependencies..." -ForegroundColor Gray
        go mod download 2>&1 | Out-Null
        
        # 構建
        Write-Host "  → Compiling..." -ForegroundColor Gray
        $output = $scanner.Output
        go build -o $output -ldflags="-s -w" . 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            if (Test-Path $output) {
                $fileSize = (Get-Item $output).Length / 1MB
                Write-Host "  ✓ Built successfully: $output ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
                $successCount++
            } else {
                Write-Host "  ✗ Build failed: output file not found" -ForegroundColor Red
                $failCount++
            }
        } else {
            Write-Host "  ✗ Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host "  ✗ Error: $_" -ForegroundColor Red
        $failCount++
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Successful: $successCount" -ForegroundColor Green
Write-Host "  Failed: $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($failCount -gt 0) {
    Write-Host "Some builds failed. Please check the output above for details." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "All scanners built successfully! ✓" -ForegroundColor Green
    exit 0
}
