# 批量遷移所有 Go 服務腳本
# 用途: 一次性遷移所有 Go 服務到 aiva_common_go

$ErrorActionPreference = "Stop"

Write-Host "🚀 開始批量遷移所有 Go 服務" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# 定義所有需要遷移的服務
$services = @(
    @{Name="function_authn_go"; Status="待遷移"},
    @{Name="function_ssrf_go"; Status="待遷移"}
)

# 已遷移的服務（跳過）
$migratedServices = @("function_sca_go", "function_cspm_go")

Write-Host "`n已遷移服務 (跳過):" -ForegroundColor Green
foreach ($s in $migratedServices) {
    Write-Host "  ✅ $s" -ForegroundColor Green
}

Write-Host "`n待遷移服務:" -ForegroundColor Yellow
foreach ($s in $services) {
    Write-Host "  ⏳ $($s.Name)" -ForegroundColor Yellow
}

Write-Host "`n" + ("=" * 60) -ForegroundColor Gray
$confirm = Read-Host "是否繼續批量遷移? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "❌ 已取消" -ForegroundColor Red
    exit 0
}

$successCount = 0
$failCount = 0
$results = @()

foreach ($service in $services) {
    $serviceName = $service.Name
    Write-Host "`n`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "📦 正在遷移: $serviceName" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    try {
        # 調用單個服務遷移腳本
        $scriptPath = Join-Path $PSScriptRoot "migrate_go_service.ps1"
        & $scriptPath -ServiceName $serviceName
        
        if ($LASTEXITCODE -eq 0) {
            $successCount++
            $results += @{Service=$serviceName; Status="✅ 成功"; Color="Green"}
            Write-Host "`n✅ $serviceName 遷移成功" -ForegroundColor Green
        } else {
            $failCount++
            $results += @{Service=$serviceName; Status="⚠️  部分成功"; Color="Yellow"}
            Write-Host "`n⚠️  $serviceName 需要手動調整" -ForegroundColor Yellow
        }
    } catch {
        $failCount++
        $results += @{Service=$serviceName; Status="❌ 失敗"; Color="Red"}
        Write-Host "`n❌ $serviceName 遷移失敗: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 1
}

# 最終報告
Write-Host "`n`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "📊 批量遷移完成報告" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

Write-Host "`n遷移結果:" -ForegroundColor White
foreach ($result in $results) {
    Write-Host ("  {0,-25} {1}" -f $result.Service, $result.Status) -ForegroundColor $result.Color
}

Write-Host "`n統計:" -ForegroundColor White
Write-Host "  成功: $successCount" -ForegroundColor Green
Write-Host "  失敗: $failCount" -ForegroundColor Red
Write-Host "  總計: $($services.Count)" -ForegroundColor Cyan

Write-Host "`n所有服務狀態:" -ForegroundColor White
Write-Host "  ✅ function_sca_go (已遷移)" -ForegroundColor Green
Write-Host "  ✅ function_cspm_go (已遷移)" -ForegroundColor Green
foreach ($result in $results) {
    Write-Host ("  {0} {1}" -f $result.Status, $result.Service) -ForegroundColor $result.Color
}

$totalServices = $migratedServices.Count + $services.Count
$completedServices = $migratedServices.Count + $successCount
$progressPercent = [math]::Round(($completedServices / $totalServices) * 100, 1)

Write-Host "`n進度: $completedServices/$totalServices ($progressPercent%)" -ForegroundColor Cyan

if ($failCount -eq 0) {
    Write-Host "`n🎉 所有服務遷移成功!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  有 $failCount 個服務需要手動調整" -ForegroundColor Yellow
}

Write-Host "`n下一步:" -ForegroundColor Cyan
Write-Host "  1. 檢查失敗/警告的服務" -ForegroundColor White
Write-Host "  2. 手動修復 main.go 和 scanner 文件" -ForegroundColor White
Write-Host "  3. 在每個服務目錄運行: go build ./..." -ForegroundColor White
Write-Host "  4. 創建遷移報告" -ForegroundColor White
Write-Host ""
