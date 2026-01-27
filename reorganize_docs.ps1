# AIVA 文檔重組自動化腳本
# 日期: 2026-01-28
# 目的: 自動移動文檔到正確位置並歸檔過時文檔

param(
    [switch]$DryRun = $false,  # 乾運行模式，只顯示將要執行的操作
    [switch]$Force = $false     # 強制覆蓋已存在的文件
)

$ErrorActionPreference = "Stop"
$rootPath = "C:\D\fold7\AIVA-git"

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  AIVA 文檔重組自動化腳本" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "⚠️  乾運行模式 - 不會實際移動文件" -ForegroundColor Yellow
    Write-Host ""
}

# 記錄操作
$logFile = Join-Path $rootPath "_out\doc_reorganization_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Path (Split-Path $logFile) -Force -ErrorAction SilentlyContinue | Out-Null

function Write-Log {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $logFile -Value $logMessage
}

function Move-FileWithLog {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Description
    )
    
    $sourcePath = Join-Path $rootPath $Source
    $destPath = Join-Path $rootPath $Destination
    
    if (-not (Test-Path $sourcePath)) {
        Write-Log "  ⚠️  來源文件不存在: $Source" "Yellow"
        return $false
    }
    
    if ($DryRun) {
        Write-Log "  [DRY-RUN] $Source → $Destination" "Cyan"
        return $true
    }
    
    try {
        # 確保目標目錄存在
        $destDir = Split-Path $destPath -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # 檢查目標文件是否已存在
        if ((Test-Path $destPath) -and -not $Force) {
            Write-Log "  ⚠️  目標文件已存在，跳過: $Destination" "Yellow"
            return $false
        }
        
        Move-Item -Path $sourcePath -Destination $destPath -Force:$Force
        Write-Log "  ✅ $Description" "Green"
        return $true
    }
    catch {
        Write-Log "  ❌ 移動失敗: $Source - $($_.Exception.Message)" "Red"
        return $false
    }
}

# ============================================
# Phase 1: 創建目錄結構
# ============================================
Write-Log "`n📁 Phase 1: 創建目錄結構" "Cyan"

$directories = @(
    "docs\01_architecture",
    "docs\02_design_decisions",
    "docs\04_user_guides",
    "docs\05_api_reference",
    "docs\06_deployment",
    "docs\rag_system",
    "docs\learning_system",
    "_archive\03_historical_reports\2026-01",
    "_archive\06_documentation_archive\2026-01"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $rootPath $dir
    if (-not (Test-Path $fullPath)) {
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Log "  ✅ 創建目錄: $dir" "Green"
        } else {
            Write-Log "  [DRY-RUN] 將創建目錄: $dir" "Cyan"
        }
    } else {
        Write-Log "  ✓ 目錄已存在: $dir" "Gray"
    }
}

# ============================================
# Phase 2: 移動最新架構文檔
# ============================================
Write-Log "`n🏗️  Phase 2: 移動最新架構文檔" "Cyan"

$archDocs = @(
    @{
        Source = "AI_CAPABILITY_SELECTION_MECHANISM_REPORT.md"
        Dest = "docs\01_architecture\AI_CAPABILITY_SELECTION_MECHANISM.md"
        Desc = "AI能力選擇機制報告"
    },
    @{
        Source = "SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md"
        Dest = "docs\01_architecture\SERVICES_ARCHITECTURE_ANALYSIS.md"
        Desc = "Services架構分析報告"
    },
    @{
        Source = "COMMANDER_CLI_ARCHITECTURE_UPDATE.md"
        Dest = "docs\01_architecture\COMMANDER_CLI_ARCHITECTURE.md"
        Desc = "Commander CLI架構更新"
    },
    @{
        Source = "UNIFIED_NAMING_CONVENTION.md"
        Dest = "docs\01_architecture\UNIFIED_NAMING_CONVENTION.md"
        Desc = "統一命名規範"
    }
)

foreach ($doc in $archDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 3: 移動設計決策文檔
# ============================================
Write-Log "`n🎯 Phase 3: 處理設計決策文檔" "Cyan"

# CLI設計決策需要重寫為ADR格式，所以先複製
$cliDecisionSource = Join-Path $rootPath "CLI_vs_DirectImport_對比.md"
$cliDecisionDest = Join-Path $rootPath "docs\02_design_decisions\ADR_001_CLI_vs_DirectImport.md"

if (Test-Path $cliDecisionSource) {
    if (-not $DryRun) {
        Copy-Item -Path $cliDecisionSource -Destination $cliDecisionDest -Force:$Force
        Write-Log "  ✅ 複製 CLI設計決策 (待重寫為ADR格式)" "Green"
        Write-Log "    → 原文件暫時保留，等重寫完成後再歸檔" "Gray"
    } else {
        Write-Log "  [DRY-RUN] 將複製 CLI_vs_DirectImport_對比.md" "Cyan"
    }
}

# ============================================
# Phase 4: 移動使用指南
# ============================================
Write-Log "`n📖 Phase 4: 移動使用指南" "Cyan"

$guideDocs = @(
    @{
        Source = "EXTERNAL_CAPABILITIES_USAGE_GUIDE.md"
        Dest = "docs\04_user_guides\EXTERNAL_CAPABILITIES_USAGE_GUIDE.md"
        Desc = "外部能力使用指南"
    },
    @{
        Source = "FUNCTION_CALLABLE_JUDGMENT_GUIDE.md"
        Dest = "docs\04_user_guides\FUNCTION_CALLABLE_JUDGMENT_GUIDE.md"
        Desc = "功能可調用判斷指南"
    },
    @{
        Source = "USAGE_GUIDE_VALIDATION_REPORT.md"
        Dest = "docs\04_user_guides\USAGE_GUIDE_VALIDATION_REPORT.md"
        Desc = "使用指南驗證報告"
    },
    @{
        Source = "QUICK_REFERENCE.md"
        Dest = "docs\04_user_guides\QUICK_REFERENCE.md"
        Desc = "快速參考"
    }
)

foreach ($doc in $guideDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 5: 移動 RAG 系統文檔
# ============================================
Write-Log "`n🔍 Phase 5: 移動 RAG 系統文檔" "Cyan"

$ragDocs = @(
    @{
        Source = "RAG_CLI_COMMAND_DECISION_SYSTEM.md"
        Dest = "docs\rag_system\RAG_CLI_COMMAND_DECISION_SYSTEM.md"
        Desc = "RAG CLI命令決策系統"
    },
    @{
        Source = "RAG_INTERNAL_EXPLORATION_INTEGRATION.md"
        Dest = "docs\rag_system\RAG_INTERNAL_EXPLORATION_INTEGRATION.md"
        Desc = "RAG內部探索整合"
    },
    @{
        Source = "RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md"
        Dest = "docs\rag_system\RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md"
        Desc = "RAG觸發與通知指南"
    },
    @{
        Source = "VECTOR_STORE_AND_RAG_ARCHITECTURE.md"
        Dest = "docs\rag_system\VECTOR_STORE_AND_RAG_ARCHITECTURE.md"
        Desc = "向量庫與RAG架構"
    }
)

foreach ($doc in $ragDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 6: 移動學習系統文檔
# ============================================
Write-Log "`n🧠 Phase 6: 移動學習系統文檔" "Cyan"

$learningDocs = @(
    @{
        Source = "LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md"
        Dest = "docs\learning_system\LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md"
        Desc = "學習系統完整架構"
    },
    @{
        Source = "AI_LEARNING_DATA_FLOW.md"
        Dest = "docs\learning_system\AI_LEARNING_DATA_FLOW.md"
        Desc = "AI學習數據流"
    },
    @{
        Source = "DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md"
        Dest = "docs\learning_system\DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md"
        Desc = "雙循環實現差距與計劃"
    }
)

foreach ($doc in $learningDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 7: 移動整合層文檔到分析報告
# ============================================
Write-Log "`n🔗 Phase 7: 移動整合層文檔到分析報告" "Cyan"

$integrationDocs = @(
    @{
        Source = "AI_EXECUTOR_INTEGRATION_COMPLETE.md"
        Dest = "docs\03_analysis_reports\AI_EXECUTOR_INTEGRATION_COMPLETE.md"
        Desc = "AI執行器整合完成報告"
    },
    @{
        Source = "CLI_AI_INTEGRATION_IMPLEMENTATION.md"
        Dest = "docs\03_analysis_reports\CLI_AI_INTEGRATION_IMPLEMENTATION.md"
        Desc = "CLI AI整合實現報告"
    },
    @{
        Source = "CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md"
        Dest = "docs\03_analysis_reports\CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md"
        Desc = "分類器與執行器架構對比"
    },
    @{
        Source = "AI_MODULES_CAPABILITY_CHECK.md"
        Dest = "docs\03_analysis_reports\AI_MODULES_CAPABILITY_CHECK.md"
        Desc = "AI模組能力檢查"
    }
)

foreach ($doc in $integrationDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 8: 歸檔過時的問題診斷報告
# ============================================
Write-Log "`n🗄️  Phase 8: 歸檔過時的問題診斷報告" "Cyan"

$archiveHistoricalDocs = @(
    @{
        Source = "CLI問題診斷報告.md"
        Dest = "_archive\03_historical_reports\2026-01\CLI問題診斷報告_20260119.md"
        Desc = "CLI問題診斷報告（已解決）"
    },
    @{
        Source = "CLI導入路徑錯誤分析.md"
        Dest = "_archive\03_historical_reports\2026-01\CLI導入路徑錯誤分析_20260119.md"
        Desc = "CLI導入路徑錯誤分析（已修復）"
    },
    @{
        Source = "BACKUP_ANALYSIS_REPORT.md"
        Dest = "_archive\03_historical_reports\2026-01\BACKUP_ANALYSIS_REPORT_20260121.md"
        Desc = "備份分析報告（已完成）"
    },
    @{
        Source = "BACKUP_CLEANUP_EXECUTION_REPORT.md"
        Dest = "_archive\03_historical_reports\2026-01\BACKUP_CLEANUP_EXECUTION_REPORT_20260121.md"
        Desc = "備份清理執行報告（已完成）"
    },
    @{
        Source = "DATA_DIFFERENCE_ANALYSIS.md"
        Dest = "_archive\03_historical_reports\2026-01\DATA_DIFFERENCE_ANALYSIS_20260120.md"
        Desc = "數據差異分析（已完成）"
    },
    @{
        Source = "22_flows_test_report.md"
        Dest = "_archive\03_historical_reports\2026-01\22_flows_test_report_20260121.md"
        Desc = "22個Flows測試報告（歷史記錄）"
    }
)

foreach ($doc in $archiveHistoricalDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 9: 歸檔被新報告取代的文檔
# ============================================
Write-Log "`n🗄️  Phase 9: 歸檔被新報告取代的文檔" "Cyan"

$archiveReplacedDocs = @(
    @{
        Source = "CLI架構實現總結.md"
        Dest = "_archive\06_documentation_archive\2026-01\CLI架構實現總結_20260119.md"
        Desc = "CLI架構實現總結（被COMMANDER_CLI_ARCHITECTURE取代）"
    },
    @{
        Source = "services_directory_structure_analysis.md"
        Dest = "_archive\06_documentation_archive\2026-01\services_directory_structure_analysis_20260121.md"
        Desc = "Services目錄結構分析（被SERVICES_ARCHITECTURE_ANALYSIS取代）"
    },
    @{
        Source = "XSS_174_FLOWS_ANALYSIS_REPORT.md"
        Dest = "_archive\06_documentation_archive\2026-01\XSS_174_FLOWS_ANALYSIS_REPORT_20260121.md"
        Desc = "XSS 174 Flows分析報告（已整合）"
    },
    @{
        Source = "WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE.md"
        Dest = "_archive\06_documentation_archive\2026-01\WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE_20260121.md"
        Desc = "171個內部函數不可調用說明（已解決）"
    }
)

foreach ($doc in $archiveReplacedDocs) {
    Move-FileWithLog -Source $doc.Source -Destination $doc.Dest -Description $doc.Desc
}

# ============================================
# Phase 10: 移動功能分類到正確位置
# ============================================
Write-Log "`n📂 Phase 10: 移動功能分類文檔" "Cyan"

$classificationDoc = @{
    Source = "function_xss_operable_classification.md"
    Dest = "features_classification\function_xss_operable_classification.md"
    Desc = "XSS功能可操作分類"
}

Move-FileWithLog -Source $classificationDoc.Source -Destination $classificationDoc.Dest -Description $classificationDoc.Desc

# ============================================
# 總結
# ============================================
Write-Log "`n=======================================" "Cyan"
Write-Log "  文檔重組完成" "Green"
Write-Log "=======================================" "Cyan"
Write-Log ""
Write-Log "📝 日誌文件: $logFile" "Gray"
Write-Log ""

if ($DryRun) {
    Write-Log "⚠️  這是乾運行模式，沒有實際移動文件" "Yellow"
    Write-Log "   要執行實際操作，請運行: .\reorganize_docs.ps1" "Yellow"
} else {
    Write-Log "✅ 下一步:" "Green"
    Write-Log "   1. 檢查根目錄，確認只保留 README.md 和 CHANGELOG.md" "Gray"
    Write-Log "   2. 更新 _archive\ARCHIVE_INDEX.md" "Gray"
    Write-Log "   3. 更新 README.md 中的文檔連結" "Gray"
    Write-Log "   4. 開始建立優先級 P0 的新文檔" "Gray"
}

Write-Host ""
