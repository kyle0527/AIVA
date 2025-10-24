#!/usr/bin/env pwsh
# AIVA 廢棄檔案清理腳本
# 清理備份檔案、過時檔案和臨時檔案

param(
    [switch]$DryRun,
    [switch]$Force,
    [string]$BackupDir = "_cleanup_backup"
)

# 設定顏色輸出
function Write-StepHeader($message) {
    Write-Host ""
    Write-Host "🔧 $message" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Gray
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "⚠️ $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

# 檢查當前目錄
if (!(Test-Path "pyproject.toml") -or !(Test-Path "services")) {
    Write-Error "請在 AIVA 專案根目錄執行此腳本"
    exit 1
}

Write-StepHeader "AIVA 廢棄檔案清理開始"

# 創建備份目錄
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = Join-Path $BackupDir $timestamp
if (!$DryRun) {
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    Write-Success "創建備份目錄: $backupPath"
}

# 定義要清理的檔案模式
$filesToClean = @(
    # 備份檔案
    "services/scan/aiva_scan/dynamic_engine/example_usage.py.backup",
    "services/core/aiva_core/ui_panel/server.py.backup",
    "services/core/aiva_core/ui_panel/dashboard.py.backup", 
    "services/core/aiva_core/ai_engine/bio_neuron_core.py.backup",
    "services/core/aiva_core/ai_engine/knowledge_base.py.backup",
    "services/core/aiva_core/ai_engine_backup/knowledge_base.py.backup",
    "services/core/aiva_core/ai_engine_backup/bio_neuron_core.py.backup",
    "services/function/function_sca_go/internal/analyzer/enhanced_analyzer.go.backup"
)

# 定義要清理的目錄
$dirsToClean = @(
    "services/core/aiva_core/ai_engine_backup"
)

Write-StepHeader "掃描廢棄檔案"

$totalFiles = 0
$totalSize = 0
$existingFiles = @()

# 檢查檔案
foreach ($file in $filesToClean) {
    $fullPath = Join-Path (Get-Location) $file
    if (Test-Path $fullPath) {
        $size = (Get-Item $fullPath).Length
        $totalSize += $size
        $totalFiles++
        $existingFiles += @{Path = $fullPath; RelativePath = $file; Size = $size}
        Write-Host "🔍 發現: $file ($([math]::Round($size/1024, 1)) KB)"
    }
}

# 檢查目錄
foreach ($dir in $dirsToClean) {
    $fullPath = Join-Path (Get-Location) $dir
    if (Test-Path $fullPath) {
        $files = Get-ChildItem -Path $fullPath -Recurse -File
        $dirSize = ($files | Measure-Object -Property Length -Sum).Sum
        if ($null -eq $dirSize) { $dirSize = 0 }
        $totalSize += $dirSize
        $totalFiles += $files.Count
        Write-Host "🔍 發現目錄: $dir ($($files.Count) 檔案, $([math]::Round($dirSize/1024, 1)) KB)"
        $existingFiles += @{Path = $fullPath; RelativePath = $dir; Size = $dirSize; IsDirectory = $true}
    }
}

Write-Host ""
Write-Host "📊 掃描結果:"
Write-Host "   檔案數量: $totalFiles"
Write-Host "   總大小: $([math]::Round($totalSize/1024/1024, 2)) MB"

if ($totalFiles -eq 0) {
    Write-Success "沒有發現廢棄檔案，專案已經很整潔！"
    exit 0
}

# 詢問確認（除非使用 -Force）
if (!$Force -and !$DryRun) {
    Write-Host ""
    $confirm = Read-Host "確定要刪除這些檔案嗎？(y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Warning "取消操作"
        exit 0
    }
}

Write-StepHeader "執行清理操作"

$cleanedFiles = 0
$cleanedSize = 0

foreach ($item in $existingFiles) {
    $sourcePath = $item.Path
    $relativePath = $item.RelativePath
    $size = $item.Size
    
    try {
        if ($DryRun) {
            Write-Host "🔮 [模擬] 將刪除: $relativePath"
        } else {
            # 備份
            if ($item.IsDirectory) {
                $backupItemPath = Join-Path $backupPath $relativePath
                Copy-Item -Path $sourcePath -Destination $backupItemPath -Recurse -Force
                Write-Host "📦 已備份目錄: $relativePath"
                
                # 刪除目錄
                Remove-Item -Path $sourcePath -Recurse -Force
                Write-Success "🗑️ 已刪除目錄: $relativePath"
            } else {
                $backupItemPath = Join-Path $backupPath $relativePath
                $backupDir = Split-Path $backupItemPath -Parent
                New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
                Copy-Item -Path $sourcePath -Destination $backupItemPath -Force
                Write-Host "📦 已備份: $relativePath"
                
                # 刪除檔案
                Remove-Item -Path $sourcePath -Force
                Write-Success "🗑️ 已刪除: $relativePath"
            }
        }
        
        $cleanedFiles++
        $cleanedSize += $size
        
    } catch {
        Write-Error "處理失敗 $relativePath : $($_.Exception.Message)"
    }
}

Write-StepHeader "清理完成報告"

if ($DryRun) {
    Write-Host "🔮 模擬模式 - 沒有實際刪除檔案"
    Write-Host "   將清理檔案: $cleanedFiles"
    Write-Host "   將節省空間: $([math]::Round($cleanedSize/1024/1024, 2)) MB"
} else {
    Write-Success "清理完成！"
    Write-Host "   已清理檔案: $cleanedFiles"
    Write-Host "   節省空間: $([math]::Round($cleanedSize/1024/1024, 2)) MB"
    Write-Host "   備份位置: $backupPath"
}

# 驗證關鍵檔案存在
Write-StepHeader "驗證系統完整性"

$criticalFiles = @(
    "services/core/aiva_core/ai_engine/bio_neuron_core.py",
    "services/core/aiva_core/ai_engine/knowledge_base.py",
    "services/core/aiva_core/ui_panel/server.py",
    "services/core/aiva_core/ui_panel/dashboard.py"
)

$allGood = $true
foreach ($file in $criticalFiles) {
    if (Test-Path $file) {
        Write-Success "✓ $file 存在"
    } else {
        Write-Error "✗ $file 遺失！"
        $allGood = $false
    }
}

if ($allGood) {
    Write-Success "🎉 系統完整性驗證通過！"
} else {
    Write-Error "🚨 發現系統檔案遺失，請檢查備份：$backupPath"
}

Write-Host ""
Write-Host "🎯 清理操作完成" -ForegroundColor Green