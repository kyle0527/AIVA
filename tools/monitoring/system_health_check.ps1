# ===================================================================
# AIVA 系統完整性檢查腳本
# 版本: 1.0
# 建立日期: 2025-10-18
# 用途: 全面檢查 AIVA 系統健康度和組件狀態
# ===================================================================

param(
    [switch]$Detailed,    # 詳細模式
    [switch]$SaveReport,  # 儲存報告
    [switch]$QuickCheck   # 快速檢查模式
)

# 設定輸出顏色
$script:Colors = @{
    Header = 'Cyan'
    Success = 'Green'
    Warning = 'Yellow'
    Error = 'Red'
    Info = 'White'
    Separator = 'DarkGray'
}

# 全域變數
$script:CheckResults = @()
$script:OverallScore = 0
$script:MaxScore = 0
$script:StartTime = Get-Date

# ===================================================================
# 輔助函數
# ===================================================================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White',
        [switch]$NoNewline
    )
    
    if ($NoNewline) {
        Write-Host $Message -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Add-CheckResult {
    param(
        [string]$Category,
        [string]$Item,
        [bool]$Passed,
        [string]$Details = "",
        [int]$Points = 1
    )
    
    $script:CheckResults += [PSCustomObject]@{
        Category = $Category
        Item = $Item
        Status = if ($Passed) { "✅" } else { "❌" }
        Passed = $Passed
        Details = $Details
        Points = $Points
        Timestamp = Get-Date
    }
    
    $script:MaxScore += $Points
    if ($Passed) {
        $script:OverallScore += $Points
    }
}

function Write-SectionHeader {
    param([string]$Title)
    
    Write-ColorOutput "`n$('=' * 60)" $Colors.Separator
    Write-ColorOutput "🔍 $Title" $Colors.Header
    Write-ColorOutput $('=' * 60) $Colors.Separator
}

# ===================================================================
# 檢查函數
# ===================================================================

function Test-CoreModules {
    Write-SectionHeader "核心模組檢查"
    
    # 檢查 aiva_common
    $aivaCommonExists = Test-Path "services\aiva_common"
    Add-CheckResult "核心模組" "aiva_common 存在" $aivaCommonExists "核心共用模組" 2
    
    if ($aivaCommonExists) {
        # 檢查 aiva_common 內部結構
        $enumsCount = (Get-ChildItem -Path "services\aiva_common\enums" -Filter "*.py" -ErrorAction SilentlyContinue | Measure-Object).Count
        $schemasCount = (Get-ChildItem -Path "services\aiva_common\schemas" -Filter "*.py" -ErrorAction SilentlyContinue | Measure-Object).Count
        $utilsCount = (Get-ChildItem -Path "services\aiva_common\utils" -Filter "*.py" -ErrorAction SilentlyContinue | Measure-Object).Count
        
        Add-CheckResult "核心模組" "Enums 模組" ($enumsCount -gt 5) "$enumsCount 個 enum 文件" 1
        Add-CheckResult "核心模組" "Schemas 模組" ($schemasCount -gt 10) "$schemasCount 個 schema 文件" 1
        Add-CheckResult "核心模組" "Utils 模組" ($utilsCount -gt 5) "$utilsCount 個 util 文件" 1
        
        Write-ColorOutput "   📊 aiva_common 統計: Enums($enumsCount), Schemas($schemasCount), Utils($utilsCount)" $Colors.Info
    }
    
    # 檢查 AI 核心
    $aiCoreExists = Test-Path "ai_core"
    Add-CheckResult "核心模組" "AI 核心" $aiCoreExists "AI 功能核心模組" 2
    
    if (-not $aiCoreExists) {
        Write-ColorOutput "   ⚠️ AI 核心模組不存在，建議重建" $Colors.Warning
    }
}

function Test-ServiceIntegration {
    Write-SectionHeader "服務整合檢查"
    
    $services = @("core", "scan", "integration", "function")
    
    foreach ($service in $services) {
        $servicePath = "services\$service"
        $serviceExists = Test-Path $servicePath
        
        if ($serviceExists) {
            # 檢查是否使用 aiva_common
            $usesAivaCommon = $false
            try {
                $pythonFiles = Get-ChildItem -Path $servicePath -Filter "*.py" -Recurse -ErrorAction SilentlyContinue
                foreach ($file in $pythonFiles) {
                    $content = Get-Content $file.FullName -ErrorAction SilentlyContinue
                    if ($content -match "from aiva_common|import aiva_common") {
                        $usesAivaCommon = $true
                        break
                    }
                }
            } catch {
                Write-ColorOutput "   ⚠️ 無法檢查 $service 服務的 aiva_common 使用情況" $Colors.Warning
            }
            
            Add-CheckResult "服務整合" "$service 服務" $serviceExists "$service 服務存在" 1
            Add-CheckResult "服務整合" "$service 使用 aiva_common" $usesAivaCommon "整合狀態" 1
            
            $status = if ($usesAivaCommon) { "✅ 已整合" } else { "⚠️ 未整合" }
            Write-ColorOutput "   📁 $service`: $status" $(if ($usesAivaCommon) { $Colors.Success } else { $Colors.Warning })
        } else {
            Add-CheckResult "服務整合" "$service 服務" $false "$service 服務不存在" 1
            Write-ColorOutput "   ❌ $service`: 不存在" $Colors.Error
        }
    }
}

function Test-Configuration {
    Write-SectionHeader "配置檔案檢查"
    
    $configFiles = @{
        "pyproject.toml" = "專案配置"
        "requirements.txt" = "Python 依賴"
        "pyrightconfig.json" = "TypeScript 檢查配置"
        "mypy.ini" = "靜態型別檢查"
        "ruff.toml" = "程式碼品質檢查"
        "README.md" = "專案文檔"
    }
    
    foreach ($file in $configFiles.Keys) {
        $exists = Test-Path $file
        $lines = 0
        if ($exists) {
            try {
                $lines = (Get-Content $file | Measure-Object -Line).Lines
            } catch {
                $lines = 0
            }
        }
        
        Add-CheckResult "配置檔案" $configFiles[$file] $exists "$lines 行" 1
        
        $status = if ($exists) { "✅ ($lines 行)" } else { "❌ 缺失" }
        Write-ColorOutput "   📄 $file`: $status" $(if ($exists) { $Colors.Success } else { $Colors.Error })
    }
}

function Test-TestCoverage {
    Write-SectionHeader "測試覆蓋檢查"
    
    $testFiles = Get-ChildItem -Path "." -Filter "test_*.py" -Recurse -ErrorAction SilentlyContinue
    $testCount = ($testFiles | Measure-Object).Count
    
    $hasSufficientTests = $testCount -gt 50
    Add-CheckResult "測試覆蓋" "測試文件數量" $hasSufficientTests "$testCount 個測試文件" 2
    
    Write-ColorOutput "   📊 發現 $testCount 個測試文件" $Colors.Info
    
    if ($Detailed -and $testCount -gt 0) {
        Write-ColorOutput "   🔍 測試文件樣本:" $Colors.Info
        $testFiles | Select-Object -First 5 | ForEach-Object {
            Write-ColorOutput "     - $($_.Name)" $Colors.Info
        }
        if ($testCount -gt 5) {
            Write-ColorOutput "     ... 以及其他 $($testCount - 5) 個" $Colors.Info
        }
    }
}

function Test-Tools {
    Write-SectionHeader "工具和腳本檢查"
    
    $toolDirs = @{
        "scripts" = "開發腳本"
        "tools" = "工具集"
        "examples" = "範例程式"
    }
    
    foreach ($dir in $toolDirs.Keys) {
        $exists = Test-Path $dir
        $fileCount = 0
        
        if ($exists) {
            try {
                $fileCount = (Get-ChildItem -Path $dir -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
            } catch {
                $fileCount = 0
            }
        }
        
        Add-CheckResult "工具生態" $toolDirs[$dir] $exists "$fileCount 個文件" 1
        
        $status = if ($exists) { "✅ ($fileCount 個文件)" } else { "❌ 不存在" }
        Write-ColorOutput "   📁 $dir`: $status" $(if ($exists) { $Colors.Success } else { $Colors.Error })
    }
    
    # 特別檢查重要工具
    $criticalTools = @{
        "tools\schema_manager.py" = "Schema 管理器"
        "tools\cleanup_deprecated_files.ps1" = "自動化清理工具"
    }
    
    foreach ($tool in $criticalTools.Keys) {
        $exists = Test-Path $tool
        Add-CheckResult "關鍵工具" $criticalTools[$tool] $exists "" 1
        
        $status = if ($exists) { "✅" } else { "❌" }
        Write-ColorOutput "   🛠️ $($criticalTools[$tool])`: $status" $(if ($exists) { $Colors.Success } else { $Colors.Error })
    }
}

function Test-Dependencies {
    Write-SectionHeader "依賴檢查"
    
    # Python 依賴檢查
    $pythonDeps = @("pydantic", "asyncio", "pathlib", "typing", "dataclasses", "enum")
    
    foreach ($dep in $pythonDeps) {
        try {
            $result = python -c "import $dep; print('OK')" 2>$null
            $available = $result -eq "OK"
        } catch {
            $available = $false
        }
        
        Add-CheckResult "Python 依賴" $dep $available "" 1
        
        $status = if ($available) { "✅" } else { "❌" }
        Write-ColorOutput "   🐍 $dep`: $status" $(if ($available) { $Colors.Success } else { $Colors.Error })
    }
}

function Test-SystemStatistics {
    Write-SectionHeader "系統統計"
    
    try {
        $totalPy = (Get-ChildItem -Path "." -Filter "*.py" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        $totalMd = (Get-ChildItem -Path "." -Filter "*.md" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        $totalJson = (Get-ChildItem -Path "." -Filter "*.json" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
        
        Write-ColorOutput "   📊 Python 文件: $totalPy" $Colors.Info
        Write-ColorOutput "   📊 Markdown 文件: $totalMd" $Colors.Info
        Write-ColorOutput "   📊 JSON 配置文件: $totalJson" $Colors.Info
        
        # 評估檔案數量合理性
        $pyFilesAdequate = $totalPy -gt 100
        $docsAdequate = $totalMd -gt 10
        
        Add-CheckResult "系統規模" "Python 文件數量" $pyFilesAdequate "$totalPy 個文件" 1
        Add-CheckResult "系統規模" "文檔數量" $docsAdequate "$totalMd 個 Markdown 文件" 1
        
    } catch {
        Write-ColorOutput "   ⚠️ 無法統計系統文件" $Colors.Warning
    }
}

# ===================================================================
# 報告生成
# ===================================================================

function Generate-HealthReport {
    Write-SectionHeader "系統健康度報告"
    
    $healthPercentage = if ($script:MaxScore -gt 0) { 
        [math]::Round(($script:OverallScore / $script:MaxScore) * 100, 1) 
    } else { 0 }
    
    $healthColor = switch ($healthPercentage) {
        { $_ -ge 90 } { $Colors.Success }
        { $_ -ge 70 } { $Colors.Warning }
        default { $Colors.Error }
    }
    
    Write-ColorOutput "💯 總體健康度: $script:OverallScore/$script:MaxScore ($healthPercentage%)" $healthColor
    
    # 按類別統計
    $categories = $script:CheckResults | Group-Object Category
    
    Write-ColorOutput "`n📊 各類別詳細狀況:" $Colors.Info
    foreach ($category in $categories) {
        $passed = ($category.Group | Where-Object { $_.Passed }).Count
        $total = $category.Group.Count
        $percentage = [math]::Round(($passed / $total) * 100, 1)
        
        $categoryColor = if ($percentage -ge 80) { $Colors.Success } 
                        elseif ($percentage -ge 60) { $Colors.Warning } 
                        else { $Colors.Error }
        
        Write-ColorOutput "   $($category.Name): $passed/$total ($percentage%)" $categoryColor
        
        if ($Detailed) {
            foreach ($item in $category.Group) {
                Write-ColorOutput "     $($item.Status) $($item.Item)" $Colors.Info
                if ($item.Details) {
                    Write-ColorOutput "       └─ $($item.Details)" $Colors.Info
                }
            }
        }
    }
    
    # 建議
    Write-ColorOutput "`n🎯 改善建議:" $Colors.Header
    
    $failedChecks = $script:CheckResults | Where-Object { -not $_.Passed }
    if ($failedChecks.Count -eq 0) {
        Write-ColorOutput "   🎉 所有檢查都通過了！系統狀態良好。" $Colors.Success
    } else {
        $priorityItems = $failedChecks | Sort-Object Points -Descending | Select-Object -First 5
        foreach ($item in $priorityItems) {
            Write-ColorOutput "   ⚡ 優先處理: $($item.Category) - $($item.Item)" $Colors.Warning
        }
    }
}

function Save-Report {
    if (-not $SaveReport) { return }
    
    $reportPath = "system_health_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    
    $report = @{
        Timestamp = $script:StartTime
        Duration = (Get-Date) - $script:StartTime
        OverallScore = $script:OverallScore
        MaxScore = $script:MaxScore
        HealthPercentage = if ($script:MaxScore -gt 0) { ($script:OverallScore / $script:MaxScore) * 100 } else { 0 }
        Results = $script:CheckResults
        Summary = @{
            TotalChecks = $script:CheckResults.Count
            PassedChecks = ($script:CheckResults | Where-Object { $_.Passed }).Count
            FailedChecks = ($script:CheckResults | Where-Object { -not $_.Passed }).Count
        }
    }
    
    try {
        $report | ConvertTo-Json -Depth 5 | Out-File -FilePath $reportPath -Encoding UTF8
        Write-ColorOutput "`n💾 報告已儲存至: $reportPath" $Colors.Success
    } catch {
        Write-ColorOutput "`n❌ 無法儲存報告: $($_.Exception.Message)" $Colors.Error
    }
}

# ===================================================================
# 主要執行流程
# ===================================================================

function Main {
    Clear-Host
    
    Write-ColorOutput @"
╔══════════════════════════════════════════════════════════════╗
║                    AIVA 系統健康度檢查                        ║
║                     版本 1.0 - 2025-10-18                   ║
╚══════════════════════════════════════════════════════════════╝
"@ $Colors.Header

    Write-ColorOutput "🚀 開始執行系統檢查..." $Colors.Info
    Write-ColorOutput "📍 當前目錄: $(Get-Location)" $Colors.Info
    Write-ColorOutput "⏰ 開始時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" $Colors.Info
    
    if ($QuickCheck) {
        Write-ColorOutput "⚡ 執行快速檢查模式" $Colors.Warning
    } elseif ($Detailed) {
        Write-ColorOutput "🔍 執行詳細檢查模式" $Colors.Info
    }
    
    # 執行各項檢查
    Test-CoreModules
    Test-ServiceIntegration
    Test-Configuration
    Test-TestCoverage
    Test-Tools
    Test-Dependencies
    Test-SystemStatistics
    
    # 生成報告
    Generate-HealthReport
    Save-Report
    
    $duration = (Get-Date) - $script:StartTime
    Write-ColorOutput "`n⏱️ 檢查完成，耗時: $($duration.TotalSeconds.ToString('F1')) 秒" $Colors.Info
    
    # 返回狀態碼
    $healthPercentage = if ($script:MaxScore -gt 0) { ($script:OverallScore / $script:MaxScore) * 100 } else { 0 }
    if ($healthPercentage -ge 80) {
        exit 0  # 健康
    } elseif ($healthPercentage -ge 60) {
        exit 1  # 警告
    } else {
        exit 2  # 需要注意
    }
}

# 執行主函數
Main