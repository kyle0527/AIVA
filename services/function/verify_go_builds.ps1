# Go 服務編譯驗證腳本
# 用途: 驗證所有 Go 服務是否編譯成功

$ErrorActionPreference = "Continue"

Write-Host "🔍 開始驗證所有 Go 服務編譯狀態" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

$services = @(
    "function_sca_go",
    "function_cspm_go",
    "function_authn_go",
    "function_ssrf_go"
)

$results = @()

foreach ($service in $services) {
    Write-Host "`n📦 檢查: $service" -ForegroundColor Yellow
    
    $servicePath = Join-Path $PSScriptRoot $service
    
    if (-not (Test-Path $servicePath)) {
        Write-Host "  ⏭️  跳過 (目錄不存在)" -ForegroundColor Gray
        $results += @{Service=$service; Status="跳過"; Icon="⏭️"; Color="Gray"}
        continue
    }
    
    Push-Location $servicePath
    
    # 檢查 go.mod
    if (Test-Path "go.mod") {
        $goModContent = Get-Content "go.mod" -Raw
        if ($goModContent -match "aiva_common_go") {
            Write-Host "  ✅ 使用 aiva_common_go" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  未使用 aiva_common_go" -ForegroundColor Yellow
        }
    }
    
    # 嘗試編譯
    Write-Host "  🔨 編譯中..." -ForegroundColor Cyan
    $output = go build ./... 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ 編譯成功" -ForegroundColor Green
        $results += @{Service=$service; Status="編譯成功"; Icon="✅"; Color="Green"}
    } else {
        Write-Host "  ❌ 編譯失敗" -ForegroundColor Red
        Write-Host "  錯誤: $output" -ForegroundColor Red
        $results += @{Service=$service; Status="編譯失敗"; Icon="❌"; Color="Red"}
    }
    
    Pop-Location
}

# 總結報告
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "📊 編譯驗證報告" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

Write-Host "`n服務狀態:" -ForegroundColor White
foreach ($result in $results) {
    Write-Host ("{0} {1,-25} {2}" -f $result.Icon, $result.Service, $result.Status) -ForegroundColor $result.Color
}

$successCount = ($results | Where-Object {$_.Status -eq "編譯成功"}).Count
$failCount = ($results | Where-Object {$_.Status -eq "編譯失敗"}).Count
$skipCount = ($results | Where-Object {$_.Status -eq "跳過"}).Count

Write-Host "`n統計:" -ForegroundColor White
Write-Host "  ✅ 成功: $successCount" -ForegroundColor Green
Write-Host "  ❌ 失敗: $failCount" -ForegroundColor Red
Write-Host "  ⏭️  跳過: $skipCount" -ForegroundColor Gray
Write-Host "  📊 總計: $($results.Count)" -ForegroundColor Cyan

if ($failCount -eq 0) {
    Write-Host "`n🎉 所有服務編譯通過!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠️  有 $failCount 個服務編譯失敗，請檢查" -ForegroundColor Yellow
    exit 1
}
