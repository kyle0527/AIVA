# Project Structure 維護腳本
# 用途：自動歸檔舊版本檔案並更新主版本

param(
    [int]$KeepVersions = 3,
    [string]$ProjectPath = "C:\D\fold7\AIVA-git\_out\project_structure"
)

Write-Host "🔧 Project Structure 維護腳本 v1.0" -ForegroundColor Cyan
Write-Host "📂 工作目錄: $ProjectPath" -ForegroundColor Yellow
Write-Host "📋 保留版本數: $KeepVersions" -ForegroundColor Yellow

Set-Location $ProjectPath

# 1. 更新主版本
Write-Host "`n🔄 更新主版本檔案..." -ForegroundColor Cyan
$latestFile = Get-ChildItem "..\tree_ultimate_chinese_*.txt" -ErrorAction SilentlyContinue | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1

if ($latestFile) {
    Copy-Item $latestFile.FullName "tree_ultimate_chinese_MAIN.txt" -Force
    Write-Host "  ✅ 主版本已更新: $($latestFile.Name)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ 未找到新的版本檔案" -ForegroundColor Yellow
}

# 2. 歸檔舊版本
Write-Host "`n📦 歸檔舊版本檔案..." -ForegroundColor Cyan
$allTreeFiles = Get-ChildItem "..\tree_ultimate_chinese_*.txt" -ErrorAction SilentlyContinue | 
    Sort-Object LastWriteTime -Descending

if ($allTreeFiles.Count -gt $KeepVersions) {
    $filesToArchive = $allTreeFiles | Select-Object -Skip $KeepVersions
    
    foreach ($file in $filesToArchive) {
        $yearMonth = $file.LastWriteTime.ToString("yyyy-MM")
        $archivePath = "versions\$yearMonth"
        
        if (-not (Test-Path $archivePath)) {
            New-Item -ItemType Directory -Path $archivePath -Force | Out-Null
            Write-Host "  📁 建立歸檔目錄: $archivePath" -ForegroundColor Gray
        }
        
        $destination = Join-Path $archivePath $file.Name
        Move-Item $file.FullName $destination -Force
        Write-Host "  📦 歸檔: $($file.Name) → $archivePath\" -ForegroundColor Gray
    }
    
    Write-Host "  ✅ 歸檔完成: $($filesToArchive.Count) 個檔案" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ 檔案數量未超過保留限制，無需歸檔" -ForegroundColor Yellow
}

# 3. 統計報告
Write-Host "`n📊 維護統計報告:" -ForegroundColor Green
$mainFiles = (Get-ChildItem "*.txt", "*.html", "*.md" -File).Count
$archivedFiles = (Get-ChildItem "versions\" -Recurse -File -ErrorAction SilentlyContinue).Count
$totalSize = (Get-ChildItem "." -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "  📄 主目錄檔案: $mainFiles 個" -ForegroundColor White
Write-Host "  📦 歷史歸檔檔案: $archivedFiles 個" -ForegroundColor White
Write-Host "  💾 總大小: $([math]::Round($totalSize, 2)) MB" -ForegroundColor White

# 4. 檢查檔案完整性
Write-Host "`n🔍 檔案完整性檢查:" -ForegroundColor Green
$requiredFiles = @(
    "tree_ultimate_chinese_MAIN.txt",
    "tree_ultimate_chinese_FINAL.txt", 
    "tree.html",
    "README.md"
)

foreach ($requiredFile in $requiredFiles) {
    if (Test-Path $requiredFile) {
        $size = (Get-Item $requiredFile).Length
        Write-Host "  ✅ $requiredFile ($size bytes)" -ForegroundColor White
    } else {
        Write-Host "  ❌ $requiredFile - 檔案缺失！" -ForegroundColor Red
    }
}

Write-Host "`n✅ 維護作業完成！" -ForegroundColor Green
Write-Host "🕒 執行時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow