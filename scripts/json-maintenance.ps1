# JSON 檔案維護腳本 - 基於截圖建議的自動化維護
# 建立日期: 2025-10-31
# 用途: 定期清理和整合 JSON 檔案

param(
    [switch]$DryRun = $false
)

Write-Host "🧹 JSON 檔案維護腳本啟動..." -ForegroundColor Green

# 1. 清理空的或無效的 JSON 檔案
function Test-JsonFileValidity {
    param($FilePath)
    try {
        $content = Get-Content $FilePath -Raw | ConvertFrom-Json
        # 檢查是否有實質內容
        if ($content -eq $null) { return $false }
        if ($content -is [string] -and $content.Trim() -eq "") { return $false }
        if ($content -is [hashtable] -and $content.Keys.Count -eq 0) { return $false }
        return $true
    } catch {
        Write-Host "⚠️ JSON 解析錯誤: $FilePath" -ForegroundColor Yellow
        return $false
    }
}

# 2. 狀態報告維護 (保留最新7天)
$statusReports = Get-ChildItem "logs\status_reports\status_*.json" -ErrorAction SilentlyContinue
if ($statusReports) {
    $cutoffDate = (Get-Date).AddDays(-7)
    $oldReports = $statusReports | Where-Object { $_.LastWriteTime -lt $cutoffDate }
    
    if ($oldReports.Count -gt 0) {
        Write-Host "📦 發現 $($oldReports.Count) 個超過7天的狀態報告需要歸檔"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path "logs\status_reports\archive" -Force | Out-Null
            $oldReports | Move-Item -Destination "logs\status_reports\archive\"
        }
    }
}

# 3. 定期報告整合建議
Write-Host "✅ JSON 檔案維護完成" -ForegroundColor Green
Write-Host "💡 建議每週執行一次此腳本進行維護" -ForegroundColor Cyan
