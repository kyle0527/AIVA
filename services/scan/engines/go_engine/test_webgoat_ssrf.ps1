$input = @"
{
  "scan_id": "webgoat_ssrf_test",
  "targets": ["http://localhost:8080/WebGoat/SSRF/task1"],
  "concurrency": 5,
  "timeout": 15
}
"@

Write-Host "=== WebGoat SSRF 掃描測試 ===" -ForegroundColor Cyan
$input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json | ForEach-Object {
    Write-Host "`n狀態: $($_.status)" -ForegroundColor Green
    Write-Host "掃描耗時: $($_.execution_time)s"
    Write-Host "目標數: $($_.targets_scanned)"
    Write-Host "請求數: $($_.requests_made)"
    Write-Host "發現資產: $($_.assets.Count)" -ForegroundColor Yellow
    
    if ($_.assets.Count -gt 0) {
        Write-Host "`n🚨 檢測到漏洞:" -ForegroundColor Red
        $_.assets | Select-Object -First 3 | ForEach-Object {
            Write-Host "  [$($_.severity)] $($_.name)"
            Write-Host "  置信度: $($_.confidence)"
            Write-Host "  參數: $($_.details.vulnerable_param)"
            Write-Host "  證據: $($_.details.evidence.indicators_found -join ', ')" -ForegroundColor Gray
            Write-Host ""
        }
    } else {
        Write-Host "`n✅ 未發現 SSRF 漏洞（或目標未登錄）" -ForegroundColor Green
    }
}
