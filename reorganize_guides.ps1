# AIVA 指南目錄重組腳本
# 日期: 2026-01-28
# 目的: 統一 guides/ 目錄結構，確保所有文檔符合「指南」命名規範

param(
    [switch]$DryRun = $false,    # 乾運行模式，只顯示將要執行的操作
    [switch]$Force = $false,      # 強制覆蓋已存在的文件
    [switch]$SkipBackup = $false  # 跳過備份步驟
)

$ErrorActionPreference = "Stop"
$rootPath = "C:\D\fold7\AIVA-git"
$guidesPath = Join-Path $rootPath "guides"

# 顏色定義
$colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Detail = "Gray"
}

Write-Host "=========================================" -ForegroundColor $colors.Info
Write-Host "  AIVA 指南目錄重組腳本 v1.0" -ForegroundColor $colors.Info
Write-Host "=========================================" -ForegroundColor $colors.Info
Write-Host ""

if ($DryRun) {
    Write-Host "⚠️  乾運行模式 - 不會實際移動或修改文件" -ForegroundColor $colors.Warning
    Write-Host ""
}

# 記錄操作
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $rootPath "_out\guides_reorganization_$timestamp.log"
$backupDir = Join-Path $rootPath "_out\backup_guides_$timestamp"

New-Item -ItemType Directory -Path (Split-Path $logFile) -Force -ErrorAction SilentlyContinue | Out-Null

function Write-Log {
    param(
        [string]$Message,
        [string]$Color = "White",
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $logFile -Value $logMessage -Encoding UTF8
}

function Test-FileExists {
    param([string]$Path)
    $fullPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $rootPath $Path }
    return Test-Path $fullPath
}

function Get-RelativePath {
    param([string]$Path)
    if ($Path.StartsWith($rootPath)) {
        return $Path.Substring($rootPath.Length).TrimStart('\', '/')
    }
    return $Path
}

function Backup-File {
    param([string]$SourcePath)
    
    if ($SkipBackup -or $DryRun) { return $true }
    
    try {
        $relativePath = Get-RelativePath $SourcePath
        $backupPath = Join-Path $backupDir $relativePath
        $backupParent = Split-Path $backupPath -Parent
        
        if (-not (Test-Path $backupParent)) {
            New-Item -ItemType Directory -Path $backupParent -Force | Out-Null
        }
        
        Copy-Item -Path $SourcePath -Destination $backupPath -Force
        Write-Log "    💾 已備份: $relativePath" $colors.Detail "BACKUP"
        return $true
    }
    catch {
        Write-Log "    ❌ 備份失敗: $relativePath - $($_.Exception.Message)" $colors.Error "ERROR"
        return $false
    }
}

function Move-FileWithValidation {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Description,
        [bool]$IsRename = $false
    )
    
    $sourcePath = if ([System.IO.Path]::IsPathRooted($Source)) { $Source } else { Join-Path $rootPath $Source }
    $destPath = if ([System.IO.Path]::IsPathRooted($Destination)) { $Destination } else { Join-Path $rootPath $Destination }
    
    $relSource = Get-RelativePath $sourcePath
    $relDest = Get-RelativePath $destPath
    
    # 檢查來源文件
    if (-not (Test-Path $sourcePath)) {
        Write-Log "  ⚠️  來源文件不存在: $relSource" $colors.Warning "WARN"
        return $false
    }
    
    # 乾運行模式
    if ($DryRun) {
        $operation = if ($IsRename) { "重命名" } else { "移動" }
        Write-Log "  [DRY-RUN] $operation : $relSource → $relDest" $colors.Info "DRY-RUN"
        return $true
    }
    
    try {
        # 備份原文件
        if (-not (Backup-File $sourcePath)) {
            Write-Log "  ⚠️  備份失敗，跳過此操作: $relSource" $colors.Warning "WARN"
            return $false
        }
        
        # 確保目標目錄存在
        $destDir = Split-Path $destPath -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # 檢查目標文件是否已存在
        if ((Test-Path $destPath) -and -not $Force) {
            Write-Log "  ⚠️  目標文件已存在，跳過: $relDest" $colors.Warning "WARN"
            return $false
        }
        
        # 執行移動
        Move-Item -Path $sourcePath -Destination $destPath -Force:$Force
        
        # 驗證
        if (Test-Path $destPath) {
            $icon = if ($IsRename) { "✏️" } else { "✅" }
            Write-Log "  $icon $Description" $colors.Success "SUCCESS"
            Write-Log "    來源: $relSource" $colors.Detail "DETAIL"
            Write-Log "    目標: $relDest" $colors.Detail "DETAIL"
            return $true
        }
        else {
            Write-Log "  ❌ 操作失敗（驗證未通過）: $relSource" $colors.Error "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "  ❌ 操作失敗: $relSource - $($_.Exception.Message)" $colors.Error "ERROR"
        return $false
    }
}

# 統計變量
$stats = @{
    Phase1_Success = 0
    Phase1_Failed = 0
    Phase2_Success = 0
    Phase2_Failed = 0
    Phase3_Success = 0
    Phase3_Failed = 0
}

Write-Log "`n📋 操作前檢查" $colors.Info "CHECK"

# 檢查關鍵路徑
$requiredPaths = @(
    "guides",
    "guides\general",
    "guides\development",
    "guides\technical",
    "guides\architecture",
    "_archive\03_historical_reports",
    "_archive\06_documentation_archive"
)

foreach ($path in $requiredPaths) {
    $fullPath = Join-Path $rootPath $path
    if (Test-Path $fullPath) {
        Write-Log "  ✓ 路徑存在: $path" $colors.Detail "CHECK"
    }
    else {
        Write-Log "  ⚠️  路徑不存在，將自動創建: $path" $colors.Warning "CHECK"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
}

# ============================================
# Phase 1: 從根目錄移動指南文檔到 guides/
# ============================================
Write-Log "`n📥 Phase 1: 從根目錄移動指南文檔到 guides/" $colors.Info "PHASE1"

$phase1Files = @(
    @{
        Source = "EXTERNAL_CAPABILITIES_USAGE_GUIDE.md"
        Dest = "guides\general\EXTERNAL_CAPABILITIES_USAGE_GUIDE.md"
        Desc = "移動「外部能力使用指南」到 guides/general/"
    },
    @{
        Source = "FUNCTION_CALLABLE_JUDGMENT_GUIDE.md"
        Dest = "guides\development\FUNCTION_CALLABLE_JUDGMENT_GUIDE.md"
        Desc = "移動「功能可調用判斷指南」到 guides/development/"
    },
    @{
        Source = "RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md"
        Dest = "guides\technical\RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md"
        Desc = "移動「RAG觸發與通知指南」到 guides/technical/"
    },
    @{
        Source = "QUICK_REFERENCE.md"
        Dest = "guides\general\QUICK_REFERENCE_GUIDE.md"
        Desc = "移動並重命名「快速參考」到 guides/general/"
    }
)

foreach ($file in $phase1Files) {
    if (Move-FileWithValidation -Source $file.Source -Destination $file.Dest -Description $file.Desc) {
        $stats.Phase1_Success++
    }
    else {
        $stats.Phase1_Failed++
    }
}

Write-Log "`n  📊 Phase 1 統計: 成功 $($stats.Phase1_Success) / 失敗 $($stats.Phase1_Failed)" $colors.Info "STATS"

# ============================================
# Phase 2: 重命名 guides/ 內部文檔
# ============================================
Write-Log "`n✏️  Phase 2: 重命名 guides/ 內部文檔（統一為指南格式）" $colors.Info "PHASE2"

$phase2Files = @(
    @{
        Source = "guides\architecture\服務架構演進規劃.md"
        Dest = "guides\architecture\服務架構演進指南.md"
        Desc = "重命名「服務架構演進規劃」→「服務架構演進指南」"
    },
    @{
        Source = "guides\architecture\AIVA_AI核心架構實施路線圖.md"
        Dest = "guides\architecture\AIVA_AI核心架構實施指南.md"
        Desc = "重命名「AI核心架構實施路線圖」→「AI核心架構實施指南」"
    },
    @{
        Source = "guides\development\PLUGINS_AND_TOOLS_INVENTORY.md"
        Dest = "guides\development\PLUGINS_AND_TOOLS_GUIDE.md"
        Desc = "重命名「工具清單」→「工具指南」"
    },
    @{
        Source = "guides\development\指令系統優化_使用示例.md"
        Dest = "guides\development\指令系統優化使用指南.md"
        Desc = "重命名「指令系統優化使用示例」→「指令系統優化使用指南」"
    }
)

foreach ($file in $phase2Files) {
    if (Move-FileWithValidation -Source $file.Source -Destination $file.Dest -Description $file.Desc -IsRename $true) {
        $stats.Phase2_Success++
    }
    else {
        $stats.Phase2_Failed++
    }
}

Write-Log "`n  📊 Phase 2 統計: 成功 $($stats.Phase2_Success) / 失敗 $($stats.Phase2_Failed)" $colors.Info "STATS"

# ============================================
# Phase 3: 歸檔 guides/ 內的分析報告
# ============================================
Write-Log "`n🗄️  Phase 3: 歸檔 guides/ 內的分析報告" $colors.Info "PHASE3"

$phase3Files = @(
    @{
        Source = "guides\architecture\AIVA雙閉環架構可行性評估報告_v2.md"
        Dest = "_archive\06_documentation_archive\2026-01\AIVA雙閉環架構可行性評估報告_v2_20260128.md"
        Desc = "歸檔「雙閉環架構可行性評估報告」（評估報告，非指南）"
    },
    @{
        Source = "guides\architecture\SERVICES_AI_提升分析報告.md"
        Dest = "_archive\06_documentation_archive\2026-01\SERVICES_AI_提升分析報告_20260128.md"
        Desc = "歸檔「Services AI提升分析報告」（分析報告，非指南）"
    },
    @{
        Source = "guides\stage8_analysis_learning_integration.md"
        Dest = "_archive\03_historical_reports\2026-01\stage8_analysis_learning_integration_20260128.md"
        Desc = "歸檔「stage8學習整合分析」（階段性報告，非指南）"
    }
)

foreach ($file in $phase3Files) {
    if (Move-FileWithValidation -Source $file.Source -Destination $file.Dest -Description $file.Desc) {
        $stats.Phase3_Success++
    }
    else {
        $stats.Phase3_Failed++
    }
}

Write-Log "`n  📊 Phase 3 統計: 成功 $($stats.Phase3_Success) / 失敗 $($stats.Phase3_Failed)" $colors.Info "STATS"

# ============================================
# 驗證與總結
# ============================================
Write-Log "`n🔍 執行後驗證" $colors.Info "VERIFY"

if (-not $DryRun) {
    # 驗證關鍵文件
    $verifyFiles = @(
        "guides\general\EXTERNAL_CAPABILITIES_USAGE_GUIDE.md",
        "guides\development\FUNCTION_CALLABLE_JUDGMENT_GUIDE.md",
        "guides\technical\RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md",
        "guides\general\QUICK_REFERENCE_GUIDE.md",
        "guides\architecture\服務架構演進指南.md",
        "guides\architecture\AIVA_AI核心架構實施指南.md",
        "guides\development\PLUGINS_AND_TOOLS_GUIDE.md",
        "guides\development\指令系統優化使用指南.md"
    )
    
    $verifySuccess = 0
    $verifyFailed = 0
    
    foreach ($file in $verifyFiles) {
        $fullPath = Join-Path $rootPath $file
        if (Test-Path $fullPath) {
            Write-Log "  ✓ 驗證通過: $file" $colors.Detail "VERIFY"
            $verifySuccess++
        }
        else {
            Write-Log "  ✗ 驗證失敗: $file" $colors.Warning "VERIFY"
            $verifyFailed++
        }
    }
    
    Write-Log "`n  📊 驗證統計: 通過 $verifySuccess / 失敗 $verifyFailed" $colors.Info "STATS"
}

# ============================================
# 最終統計
# ============================================
Write-Log "`n=========================================" $colors.Info "SUMMARY"
Write-Log "  執行完成" $colors.Success "SUMMARY"
Write-Log "=========================================" $colors.Info "SUMMARY"

$totalSuccess = $stats.Phase1_Success + $stats.Phase2_Success + $stats.Phase3_Success
$totalFailed = $stats.Phase1_Failed + $stats.Phase2_Failed + $stats.Phase3_Failed
$totalOperations = $totalSuccess + $totalFailed

Write-Log "`n📊 總體統計:" $colors.Info "SUMMARY"
Write-Log "  總操作數: $totalOperations" $colors.Info "SUMMARY"
Write-Log "  成功: $totalSuccess ✅" $colors.Success "SUMMARY"
Write-Log "  失敗: $totalFailed ❌" $(if ($totalFailed -eq 0) { $colors.Success } else { $colors.Warning }) "SUMMARY"
Write-Log "" $colors.Info "SUMMARY"

Write-Log "📝 詳細日誌: $logFile" $colors.Detail "SUMMARY"

if (-not $DryRun -and -not $SkipBackup) {
    Write-Log "💾 備份目錄: $backupDir" $colors.Detail "SUMMARY"
}

Write-Log "" $colors.Info "SUMMARY"

if ($DryRun) {
    Write-Log "⚠️  這是乾運行模式，沒有實際移動文件" $colors.Warning "SUMMARY"
    Write-Log "   要執行實際操作，請運行: .\reorganize_guides.ps1" $colors.Warning "SUMMARY"
    Write-Log "   強制覆蓋模式: .\reorganize_guides.ps1 -Force" $colors.Warning "SUMMARY"
}
else {
    Write-Log "✅ 下一步:" $colors.Success "SUMMARY"
    Write-Log "   1. 檢查 guides/ 目錄結構是否正確" $colors.Detail "SUMMARY"
    Write-Log "   2. 更新 guides/README.md 索引" $colors.Detail "SUMMARY"
    Write-Log "   3. 檢查 _archive/ARCHIVE_INDEX.md" $colors.Detail "SUMMARY"
    Write-Log "   4. 執行 reorganize_docs.ps1 處理根目錄其他文檔" $colors.Detail "SUMMARY"
    
    if ($totalFailed -gt 0) {
        Write-Log "`n⚠️  注意: 有 $totalFailed 個操作失敗，請檢查日誌文件" $colors.Warning "SUMMARY"
    }
}

Write-Host ""

# 返回統計結果
return @{
    TotalOperations = $totalOperations
    Success = $totalSuccess
    Failed = $totalFailed
    LogFile = $logFile
    BackupDir = if ($SkipBackup -or $DryRun) { $null } else { $backupDir }
}
