# AIVA Reports 整併腳本
# 將老舊重複文件合併為統一的綜合報告，並刪除原始文件

param(
    [switch]$DryRun = $false,
    [string]$ReportsPath = "C:\D\fold7\AIVA-git\reports"
)

Write-Host "🔄 AIVA Reports 整併工具" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "⚠️  DRY RUN 模式 - 不會實際修改文件" -ForegroundColor Yellow
}

# 整併計劃定義
$ConsolidationPlan = @{
    # 1. 編碼分析報告整併 (3→1)
    "ENCODING_CONSOLIDATED_REPORT" = @{
        TargetFile = "ENCODING_ANALYSIS_CONSOLIDATED_REPORT.md"
        SourceFiles = @(
            "ENCODING_ANALYSIS_REPORT.md",
            "ENCODING_DETECTION_FINAL_REPORT.md", 
            "ENCODING_DETECTION_TECHNICAL_DOCUMENTATION.md"
        )
        Category = "技術分析"
    }
    
    # 2. 語言轉換分析整併 (4→1) 
    "LANGUAGE_ANALYSIS_CONSOLIDATED" = @{
        TargetFile = "LANGUAGE_CONVERSION_CONSOLIDATED_REPORT.md"
        SourceFiles = @(
            "language_conversion_guide_validation_20251101_003144.md",
            "LANGUAGE_CONVERSION_GUIDE_VALIDATION_SUMMARY.md",
            "javascript_analysis_standardization_plan.md",
            "javascript_analysis_standardization_success_report.md"
        )
        Category = "語言標準化"
    }
    
    # 3. 覆蓋率分析整併 (5→1)
    "COVERAGE_ANALYSIS_CONSOLIDATED" = @{
        TargetFile = "COVERAGE_ANALYSIS_CONSOLIDATED_REPORT.md"
        SourceFiles = @(
            "contract_coverage_analysis.md",
            "contract_coverage_health_analysis_20251101.md",
            "coverage_analysis_20251101.md",
            "coverage_verification_20251101.md",
            "expansion_plan_20251101.md"
        )
        Category = "覆蓋率分析"
    }
    
    # 4. 安全事件統一化整併 (3→1)
    "SECURITY_UNIFICATION_CONSOLIDATED" = @{
        TargetFile = "SECURITY_EVENTS_CONSOLIDATED_REPORT.md"
        SourceFiles = @(
            "security_events_unification_analysis.md",
            "security_events_unification_success_report.md",
            "import_path_check_report.md"
        )
        Category = "安全分析"
    }
    
    # 5. 階段執行報告整併 (3→1)
    "PHASE_EXECUTION_CONSOLIDATED" = @{
        TargetFile = "PHASE_EXECUTION_CONSOLIDATED_REPORT.md"
        SourceFiles = @(
            "phase_2_execution_report_20251101.md",
            "contract_health_report_20251101_152743.md",
            "queue_naming_simplified.md",
            "queue_naming_validation.md"
        )
        Category = "執行階段"
    }
}

# 整併模板函數
function New-ConsolidatedReport {
    param(
        [string]$Title,
        [string]$Category,
        [string[]]$SourceFiles,
        [string]$TargetPath
    )
    
    $Content = @"
# 📊 $Title

**整併日期**: $(Get-Date -Format 'yyyy年MM月dd日')  
**文檔分類**: $Category  
**原始文件數**: $($SourceFiles.Count) 個文件  

---

## 📑 目錄

- [整併概述](#整併概述)
- [原始文件列表](#原始文件列表)
- [整併內容](#整併內容)
- [總結與建議](#總結與建議)

---

## 🔄 整併概述

本文檔將以下 $($SourceFiles.Count) 個相關報告進行整併，避免重複內容並提供統一的分析視角：

$($SourceFiles | ForEach-Object { "- ``$_``" } | Out-String)

---

## 📋 原始文件列表

| 文件名稱 | 文件大小 | 整併狀態 |
|----------|----------|----------|

"@

    # 添加源文件信息表格
    foreach ($sourceFile in $SourceFiles) {
        $fullPath = Join-Path $ReportsPath $sourceFile
        if (Test-Path $fullPath) {
            $fileInfo = Get-Item $fullPath
            $size = "{0:N0} bytes" -f $fileInfo.Length
            $Content += "| $sourceFile | $size | ✅ 已整併 |`n"
        } else {
            $Content += "| $sourceFile | 文件不存在 | ❌ 未找到 |`n"
        }
    }

    $Content += @"

---

## 🔍 整併內容

"@

    # 整併每個源文件的內容
    $sectionNumber = 1
    foreach ($sourceFile in $SourceFiles) {
        $fullPath = Join-Path $ReportsPath $sourceFile
        if (Test-Path $fullPath) {
            Write-Host "   📄 整併文件: $sourceFile" -ForegroundColor Gray
            
            $Content += @"

### $sectionNumber. $sourceFile

"@
            
            try {
                $fileContent = Get-Content $fullPath -Raw -Encoding UTF8
                
                # 移除原文件的標題和前言
                $cleanContent = $fileContent -replace '^#[^#].*?\n', '' -replace '^---.*?---\s*\n', ''
                
                # 添加處理後的內容
                if ($cleanContent.Trim()) {
                    $Content += $cleanContent.Trim() + "`n`n"
                } else {
                    $Content += "*此文件內容為空或無法解析*`n`n"
                }
                
            } catch {
                $Content += "*讀取文件時發生錯誤: $($_.Exception.Message)*`n`n"
            }
            
            $sectionNumber++
        }
    }

    $Content += @"

---

## 📈 總結與建議

### ✅ 整併完成項目
- 成功整併 $($SourceFiles.Count) 個相關文件
- 統一了文檔格式和結構
- 消除了內容重複和版本混亂

### 🎯 後續維護建議
1. **統一更新**: 相關內容變更時，統一在此文檔中維護
2. **版本控制**: 重大變更時更新文檔版本號
3. **定期檢查**: 確保整併內容與實際狀態一致

### 📋 已刪除的原始文件
$($SourceFiles | ForEach-Object { "- ``$_`` (已刪除)" } | Out-String)

---

*整併工具自動生成 | $(Get-Date -Format 'yyyy年MM月dd日 HH:mm:ss')*
"@

    return $Content
}

# 執行整併
$TotalFiles = 0
$ProcessedPlans = 0
$CreatedFiles = @()
$DeletedFiles = @()

Write-Host "`n🔍 分析整併計劃..." -ForegroundColor Green

foreach ($planName in $ConsolidationPlan.Keys) {
    $plan = $ConsolidationPlan[$planName]
    $targetFile = Join-Path $ReportsPath $plan.TargetFile
    
    Write-Host "`n📋 處理計劃: $planName" -ForegroundColor Yellow
    Write-Host "   目標文件: $($plan.TargetFile)" -ForegroundColor Gray
    Write-Host "   源文件數: $($plan.SourceFiles.Count)" -ForegroundColor Gray
    
    # 檢查源文件是否存在
    $existingFiles = @()
    foreach ($sourceFile in $plan.SourceFiles) {
        $sourcePath = Join-Path $ReportsPath $sourceFile
        if (Test-Path $sourcePath) {
            $existingFiles += $sourceFile
            Write-Host "   ✅ 找到: $sourceFile" -ForegroundColor Green
        } else {
            Write-Host "   ❌ 未找到: $sourceFile" -ForegroundColor Red
        }
    }
    
    if ($existingFiles.Count -eq 0) {
        Write-Host "   ⚠️  跳過 - 沒有找到源文件" -ForegroundColor Yellow
        continue
    }
    
    # 生成整併報告
    if (-not $DryRun) {
        try {
            $consolidatedContent = New-ConsolidatedReport -Title $planName -Category $plan.Category -SourceFiles $existingFiles -TargetPath $targetFile
            
            # 寫入整併文件
            Set-Content -Path $targetFile -Value $consolidatedContent -Encoding UTF8
            $CreatedFiles += $plan.TargetFile
            Write-Host "   ✅ 創建整併文件: $($plan.TargetFile)" -ForegroundColor Green
            
            # 刪除原始文件
            foreach ($sourceFile in $existingFiles) {
                $sourcePath = Join-Path $ReportsPath $sourceFile
                Remove-Item $sourcePath -Force
                $DeletedFiles += $sourceFile
                Write-Host "   🗑️  刪除原始文件: $sourceFile" -ForegroundColor Gray
            }
            
        } catch {
            Write-Host "   ❌ 錯誤: $($_.Exception.Message)" -ForegroundColor Red
            continue
        }
    } else {
        Write-Host "   📝 DRY RUN: 將創建 $($plan.TargetFile)" -ForegroundColor Cyan
        Write-Host "   📝 DRY RUN: 將刪除 $($existingFiles.Count) 個文件" -ForegroundColor Cyan
    }
    
    $TotalFiles += $existingFiles.Count
    $ProcessedPlans++
}

# 清理臨時和JSON文件
Write-Host "`n🧹 清理臨時文件..." -ForegroundColor Green

$TempPatterns = @(
    "*.json",
    "*_20251101_*.md",
    "*_20251103_*.md", 
    "scan_report_*.json",
    "contract_health_check_*.json",
    "*.csv",
    "pentest_report_*.json"
)

$CleanupFiles = @()
foreach ($pattern in $TempPatterns) {
    $files = Get-ChildItem -Path $ReportsPath -Name $pattern -File
    foreach ($file in $files) {
        $fullPath = Join-Path $ReportsPath $file
        $CleanupFiles += $file
        
        if (-not $DryRun) {
            Remove-Item $fullPath -Force
            Write-Host "   🗑️  清理: $file" -ForegroundColor Gray
        } else {
            Write-Host "   📝 DRY RUN: 將清理 $file" -ForegroundColor Cyan
        }
    }
}

# 總結報告
Write-Host "`n📊 整併完成總結" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "處理的整併計劃: $ProcessedPlans 個" -ForegroundColor Green
Write-Host "整併的原始文件: $TotalFiles 個" -ForegroundColor Green  
Write-Host "創建的整併文件: $($CreatedFiles.Count) 個" -ForegroundColor Green
Write-Host "清理的臨時文件: $($CleanupFiles.Count) 個" -ForegroundColor Green
Write-Host "總刪除文件數: $($DeletedFiles.Count + $CleanupFiles.Count) 個" -ForegroundColor Yellow

if ($CreatedFiles.Count -gt 0) {
    Write-Host "`n✅ 創建的整併文件:" -ForegroundColor Green
    $CreatedFiles | ForEach-Object { Write-Host "   - $_" -ForegroundColor Gray }
}

if (-not $DryRun) {
    Write-Host "`n🎉 整併完成！reports 目錄已優化。" -ForegroundColor Green
} else {
    Write-Host "`n📋 DRY RUN 完成。使用 -DryRun:`$false 執行實際整併。" -ForegroundColor Yellow
}