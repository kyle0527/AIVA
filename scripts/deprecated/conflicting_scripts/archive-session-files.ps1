# AIVA Session 檔案自動歸檔腳本
# 用途：定期清理和歸檔 AI session 相關檔案

param(
    [int]$RetentionDays = 7,
    [string]$LogsPath = "C:\D\fold7\AIVA-git\logs",
    [string]$ArchivePath = "C:\D\fold7\AIVA-git\logs\archive",
    [string]$ReportsPath = "C:\D\fold7\AIVA-git\reports\ai_sessions"
)

Write-Host "🤖 AIVA Session 檔案歸檔腳本 v1.0" -ForegroundColor Cyan
Write-Host "📅 保留天數: $RetentionDays 天" -ForegroundColor Yellow
Write-Host "📂 日誌目錄: $LogsPath" -ForegroundColor Yellow
Write-Host "📦 歸檔目錄: $ArchivePath" -ForegroundColor Yellow

# 確保目錄存在
if (-not (Test-Path $ArchivePath)) {
    New-Item -ItemType Directory -Path $ArchivePath -Force | Out-Null
    Write-Host "✅ 建立歸檔目錄: $ArchivePath" -ForegroundColor Green
}

if (-not (Test-Path $ReportsPath)) {
    New-Item -ItemType Directory -Path $ReportsPath -Force | Out-Null
    Write-Host "✅ 建立報告目錄: $ReportsPath" -ForegroundColor Green
}

$cutoffDate = (Get-Date).AddDays(-$RetentionDays)
Write-Host "⏰ 歸檔截止日期: $($cutoffDate.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Yellow

# 1. 歸檔舊的 session log 檔案
Write-Host "`n🔄 處理 Session Log 檔案..." -ForegroundColor Cyan
$sessionLogs = Get-ChildItem "$LogsPath\aiva_session_*.log" | Where-Object {$_.LastWriteTime -lt $cutoffDate}
$archivedCount = 0

foreach ($log in $sessionLogs) {
    $yearMonth = $log.LastWriteTime.ToString("yyyy-MM")
    $monthArchivePath = Join-Path $ArchivePath $yearMonth
    
    if (-not (Test-Path $monthArchivePath)) {
        New-Item -ItemType Directory -Path $monthArchivePath -Force | Out-Null
    }
    
    Move-Item $log.FullName $monthArchivePath -Force
    $archivedCount++
    Write-Host "  📦 歸檔: $($log.Name) → archive\$yearMonth\" -ForegroundColor Gray
}

# 2. 遷移 session 報告檔案
Write-Host "`n🔄 處理 Session 報告檔案..." -ForegroundColor Cyan
$sessionReports = Get-ChildItem "$LogsPath\aiva_session_*_report.json" -ErrorAction SilentlyContinue
$movedReports = 0

foreach ($report in $sessionReports) {
    $destination = Join-Path $ReportsPath $report.Name
    Move-Item $report.FullName $destination -Force
    $movedReports++
    Write-Host "  📊 遷移: $($report.Name) → reports\ai_sessions\" -ForegroundColor Gray
}

# 3. 清理舊的 p0_validation 日誌（可選）
Write-Host "`n🔄 處理驗證日誌檔案..." -ForegroundColor Cyan
$validationLogs = Get-ChildItem "$LogsPath\p0_validation_*.log" | Where-Object {$_.LastWriteTime -lt $cutoffDate}
$validationArchived = 0

foreach ($log in $validationLogs) {
    $yearMonth = $log.LastWriteTime.ToString("yyyy-MM")
    $monthArchivePath = Join-Path $ArchivePath "validation\$yearMonth"
    
    if (-not (Test-Path $monthArchivePath)) {
        New-Item -ItemType Directory -Path $monthArchivePath -Force | Out-Null
    }
    
    Move-Item $log.FullName $monthArchivePath -Force
    $validationArchived++
}

# 4. 統計報告
Write-Host "`n📊 歸檔統計報告:" -ForegroundColor Green
Write-Host "  📦 Session Logs 歸檔: $archivedCount 個檔案" -ForegroundColor White
Write-Host "  📊 Session 報告遷移: $movedReports 個檔案" -ForegroundColor White
Write-Host "  🔍 驗證日誌歸檔: $validationArchived 個檔案" -ForegroundColor White

# 5. 磁碟空間統計
$logsSize = (Get-ChildItem $LogsPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$archiveSize = (Get-ChildItem $ArchivePath -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "`n💾 磁碟使用統計:" -ForegroundColor Green
Write-Host "  📂 logs/ 目錄大小: $([math]::Round($logsSize, 2)) MB" -ForegroundColor White
Write-Host "  📦 archive/ 目錄大小: $([math]::Round($archiveSize, 2)) MB" -ForegroundColor White

Write-Host "`n✅ 歸檔作業完成！" -ForegroundColor Green
Write-Host "🕒 執行時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow