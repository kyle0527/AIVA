#!/usr/bin/env pwsh
# 真實 SSRF 檢測測試腳本

Write-Host "🎯 SSRF Scanner 真實靶場測試" -ForegroundColor Cyan
Write-Host "=" * 60

# 測試 1: WebGoat SSRF 課程
Write-Host "`n📌 測試 1: WebGoat SSRF 課程端點" -ForegroundColor Yellow
$input1 = @{
    scan_id = "webgoat_ssrf_test"
    targets = @(
        "http://localhost:8080/WebGoat/SSRF/task1",
        "http://localhost:8080/WebGoat/SSRF/task2"
    )
    concurrency = 2
    timeout = 20
} | ConvertTo-Json -Compress

Write-Host "輸入: $input1`n" -ForegroundColor Gray

$result1 = $input1 | .\bin\ssrf-scanner.exe 2>&1 | Out-String
$json1 = $result1 | ConvertFrom-Json -ErrorAction SilentlyContinue

if ($json1) {
    Write-Host "✅ 掃描完成!" -ForegroundColor Green
    Write-Host "   - 目標數: $($json1.targets_scanned)" -ForegroundColor White
    Write-Host "   - 發現資產: $($json1.assets.Count)" -ForegroundColor White
    Write-Host "   - 執行時間: $([math]::Round($json1.execution_time, 2))s" -ForegroundColor White
    Write-Host "   - 成功率: $($json1.success_count)/$($json1.targets_scanned)" -ForegroundColor White
    
    # 顯示前 3 個發現
    if ($json1.assets.Count -gt 0) {
        Write-Host "`n🔍 檢測到的漏洞類型:" -ForegroundColor Cyan
        $json1.assets | Select-Object -First 3 | ForEach-Object {
            Write-Host "   - $($_.name) [$($_.severity)]" -ForegroundColor Yellow
            Write-Host "     URL: $($_.details.affected_url)" -ForegroundColor Gray
            Write-Host "     Payload: $($_.details.payload_type)" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "❌ 解析失敗" -ForegroundColor Red
}

# 測試 2: 自定義 SSRF 測試目標
Write-Host "`n`n📌 測試 2: 簡化測試（僅測試 AWS IMDS）" -ForegroundColor Yellow
$input2 = @{
    scan_id = "simple_test"
    targets = @("http://localhost:8080/WebGoat/SSRF/task1")
    concurrency = 1
    timeout = 10
} | ConvertTo-Json -Compress

$result2 = $input2 | .\bin\ssrf-scanner.exe 2>&1 | Out-String
$json2 = $result2 | ConvertFrom-Json -ErrorAction SilentlyContinue

if ($json2) {
    Write-Host "✅ 掃描完成: $($json2.assets.Count) 個資產" -ForegroundColor Green
    
    # 按嚴重性分組
    $bySeverity = $json2.assets | Group-Object -Property severity
    Write-Host "`n📊 按嚴重性統計:" -ForegroundColor Cyan
    $bySeverity | ForEach-Object {
        Write-Host "   - $($_.Name): $($_.Count)" -ForegroundColor White
    }
}

Write-Host "`n" + ("=" * 60)
Write-Host "✅ 測試完成!" -ForegroundColor Green
