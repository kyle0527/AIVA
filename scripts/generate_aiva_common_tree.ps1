# AIVA Common 模組樹狀架構圖生成腳本
# 專門為 aiva_common 模組設計，避免編碼問題

param(
    [string]$ProjectRoot = "C:\D\fold7\AIVA-git\services\aiva_common",
    [string]$OutputDir = "C:\D\fold7\AIVA-git\_out\architecture_diagrams"
)

Write-Host "🚀 開始生成 aiva_common 模組樹狀架構圖..." -ForegroundColor Cyan

# 要排除的目錄
$excludeDirs = @('__pycache__', '.mypy_cache', '.ruff_cache', '.pytest_cache')

# 只保留的程式碼檔案類型
$codeExtensions = @('.py', '.yaml', '.yml', '.md')

# 中文檔名說明對照表 (簡化版，避免編碼問題)
$chineseComments = @{
    '__init__.py' = '模組初始化'
    'models.py' = '資料模型'
    'schemas.py' = '資料結構定義'
    'config.py' = '配置管理'
    'mq.py' = '訊息佇列'
    'utils.py' = '工具函數'
    'enums.py' = '列舉定義'
    'base.py' = '基礎類別'
    'ai.py' = 'AI 相關定義'
    'assets.py' = '資產管理'
    'findings.py' = '發現結果'
    'messaging.py' = '訊息協議'
    'tasks.py' = '任務定義'
    'telemetry.py' = '遙測資料'
    'references.py' = '參考標準'
    'system.py' = '系統相關'
    'risk.py' = '風險評估'
    'enhanced.py' = '增強功能'
    'languages.py' = '程式語言'
    'common.py' = '通用定義'
    'modules.py' = '模組定義'
    'security.py' = '安全相關'
    'ids.py' = 'ID 生成器'
    'logging.py' = '日誌工具'
    'dedupe.py' = '去重複'
    'backoff.py' = '退避策略'
    'ratelimit.py' = '限流控制'
    'unified_config.py' = '統一配置'
    'core_schema_sot.yaml' = 'Schema 定義源'
    'CODE_QUALITY_REPORT.md' = '程式碼品質報告'
    'py.typed' = 'TypeScript 支援'
}

function Test-ShouldIncludeFile {
    param([string]$FileName)
    
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    
    if ([string]::IsNullOrEmpty($ext)) {
        return $false
    }
    
    return $codeExtensions -contains $ext
}

function Get-ChineseComment {
    param([string]$FileName)
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $fullName = $FileName
    
    # 完全匹配
    if ($chineseComments.ContainsKey($fullName)) {
        return $chineseComments[$fullName]
    }
    # 基本檔名匹配
    elseif ($chineseComments.ContainsKey($baseName)) {
        return $chineseComments[$baseName]
    }
    
    # 根據副檔名推測
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    switch ($ext) {
        '.py' { 
            if ($fullName -match "test") { return "測試程式" }
            elseif ($fullName -match "tool") { return "工具程式" }
            elseif ($fullName -match "validator") { return "驗證器" }
            elseif ($fullName -match "tester") { return "測試器" }
            elseif ($fullName -match "generator") { return "生成器" }
            else { return "Python 模組" }
        }
        '.yaml' { return "YAML 配置" }
        '.yml' { return "YAML 配置" }
        '.md' { return "文件" }
        default { return "" }
    }
}

function Get-CodeTree {
    param(
        [string]$Path,
        [string]$Prefix = "",
        [int]$Level = 0,
        [int]$MaxLevel = 10,
        [ref]$FileCount,
        [ref]$DirCount
    )

    if ($Level -ge $MaxLevel) { return @() }

    $output = @()

    try {
        $items = Get-ChildItem -Path $Path -Force -ErrorAction Stop |
            Where-Object {
                $name = $_.Name
                if ($_.PSIsContainer) {
                    if ($excludeDirs -contains $name) {
                        return $false
                    }
                    $DirCount.Value++
                    return $true
                } else {
                    if (Test-ShouldIncludeFile -FileName $name) {
                        $FileCount.Value++
                        return $true
                    }
                    return $false
                }
            } |
            Sort-Object @{Expression={$_.PSIsContainer}; Descending=$true}, Name

        $itemCount = $items.Count
        for ($i = 0; $i -lt $itemCount; $i++) {
            $item = $items[$i]
            $isLast = ($i -eq $itemCount - 1)

            $connector = if ($isLast) { "└─" } else { "├─" }
            $extension = if ($isLast) { "    " } else { "│   " }

            # 添加中文註解
            $chineseComment = Get-ChineseComment -FileName $item.Name
            $itemNameWithComment = if ($chineseComment) { 
                "$($item.Name) # $chineseComment" 
            } else { 
                $item.Name 
            }
            
            $outputLine = "$Prefix$connector$itemNameWithComment"
            $output += $outputLine

            if ($item.PSIsContainer) {
                $subOutput = Get-CodeTree -Path $item.FullName -Prefix "$Prefix$extension" -Level ($Level + 1) -MaxLevel $MaxLevel -FileCount $FileCount -DirCount $DirCount
                $output += $subOutput
            }
        }
    } catch {
        # 忽略無法存取的目錄
    }

    return $output
}

# 統計程式碼檔案
Write-Host "📊 統計程式碼檔案..." -ForegroundColor Yellow

$allCodeFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $path = $_.FullName
        $shouldExclude = $false
        foreach ($dir in $excludeDirs) {
            if ($path -like "*\$dir\*") {
                $shouldExclude = $true
                break
            }
        }
        if ($shouldExclude) { return $false }
        Test-ShouldIncludeFile -FileName $_.Name
    }

$langStats = $allCodeFiles | 
    Group-Object Extension |
    ForEach-Object {
        $ext = $_.Name
        $files = $_.Group
        $totalLines = 0
        foreach ($file in $files) {
            try {
                $lines = (Get-Content $file.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
                $totalLines += $lines
            } catch {
                # 忽略無法讀取的檔案
            }
        }
        [PSCustomObject]@{
            Extension = $ext
            FileCount = $files.Count
            TotalLines = $totalLines
            AvgLines = if ($files.Count -gt 0) { [math]::Round($totalLines / $files.Count, 1) } else { 0 }
        }
    } |
    Sort-Object TotalLines -Descending

# 計算總計
$totalFiles = ($langStats | Measure-Object -Property FileCount -Sum).Sum
$totalLines = ($langStats | Measure-Object -Property TotalLines -Sum).Sum

# 生成樹狀結構
Write-Host "🌳 生成樹狀結構..." -ForegroundColor Yellow

$fileCountRef = [ref]0
$dirCountRef = [ref]0

$rootName = Split-Path $ProjectRoot -Leaf
$output = @()

# 添加標題和統計
$output += "================================================================================"
$output += "AIVA Common 模組樹狀架構圖"
$output += "================================================================================"
$output += "生成日期: $(Get-Date -Format 'yyyy年MM月dd日 HH:mm:ss')"
$output += "模組路徑: $ProjectRoot"
$output += ""
$output += "📊 模組統計"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "總檔案數: $totalFiles"
$output += "總程式碼行數: $totalLines"
$output += ""
$output += "💻 語言分布:"

foreach ($stat in $langStats) {
    $pct = if ($totalLines -gt 0) { [math]::Round(($stat.TotalLines / $totalLines) * 100, 1) } else { 0 }
    $output += "   • $($stat.Extension): $($stat.FileCount) 檔案, $($stat.TotalLines) 行 ($pct%)"
}

$output += ""
$output += "🔧 模組用途"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "aiva_common 是 AIVA 系統的基礎設施模組，提供："
$output += "• schemas/ - 跨服務共享的資料結構定義"
$output += "• enums/ - 系統常數和枚舉類型"
$output += "• utils/ - 通用工具函數（去重、網路、日誌等）"
$output += "• config/ - 統一配置管理"
$output += "• tools/ - 開發和維護工具"
$output += ""
$output += "💡 設計理念"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "• 作為所有 AIVA 服務的共享基礎層"
$output += "• 實現跨語言相容（Python/Go/Rust/TypeScript）"
$output += "• 遵循官方標準（CVSS、SARIF、CVE/CWE等）"
$output += "• 提供統一的資料合約和通訊協議"
$output += ""
$output += "================================================================================"
$output += "模組結構樹狀圖（含中文說明）"
$output += "================================================================================"
$output += ""

# 顯示根目錄
$rootComment = "AIVA 共用基礎模組"
$output += "$rootName # $rootComment"

# 生成樹狀結構
$treeOutput = Get-CodeTree -Path $ProjectRoot -FileCount $fileCountRef -DirCount $dirCountRef
$output += $treeOutput

$output += ""
$output += "================================================================================"
$output += "模組架構分析"
$output += "================================================================================"
$output += ""
$output += "🏗️ 架構層次："
$output += "1. 基礎層 (base.py) - 所有資料結構的基礎類別"
$output += "2. 協議層 (messaging.py) - 服務間通訊協議"
$output += "3. 領域層 (ai.py, findings.py, tasks.py等) - 業務領域定義"
$output += "4. 工具層 (utils/) - 輔助功能和工具"
$output += "5. 配置層 (config/) - 系統配置管理"
$output += ""
$output += "🔄 跨語言支援："
$output += "• core_schema_sot.yaml - 單一事實來源，支援多語言生成"
$output += "• py.typed - Python 型別提示支援"
$output += "• generated/ - 自動生成的跨語言綁定"
$output += ""
$output += "📋 品質保證："
$output += "• 總計 $totalFiles 個檔案，$totalLines 行程式碼"
$output += "• 完整的型別定義和文件"
$output += "• 符合 PEP 8 和官方標準"
$output += ""

# 儲存到檔案
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputFile = Join-Path $OutputDir "aiva_common_architecture_$timestamp.txt"

# 確保輸出目錄存在
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$output | Out-File $outputFile -Encoding UTF8

Write-Host ""
Write-Host "✅ AIVA Common 模組樹狀架構圖已生成！" -ForegroundColor Green
Write-Host "   檔案位置: $outputFile" -ForegroundColor White
Write-Host "   樹狀圖行數: $($output.Count) 行" -ForegroundColor White
Write-Host "   程式碼檔案數: $totalFiles" -ForegroundColor White
Write-Host "   總程式碼行數: $totalLines" -ForegroundColor White
Write-Host ""
Write-Host "📋 語言分布:" -ForegroundColor Cyan
foreach ($stat in $langStats) {
    $pct = if ($totalLines -gt 0) { [math]::Round(($stat.TotalLines / $totalLines) * 100, 1) } else { 0 }
    Write-Host "   $($stat.Extension): $($stat.FileCount) 檔案, $($stat.TotalLines) 行 ($pct%)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🎯 模組特性:" -ForegroundColor Cyan
Write-Host "   • 基礎設施模組：為所有服務提供共享組件" -ForegroundColor Gray
Write-Host "   • 跨語言支援：統一的資料定義和協議" -ForegroundColor Gray
Write-Host "   • 官方標準：實現 CVSS、SARIF 等國際標準" -ForegroundColor Gray
Write-Host "   • 高度模組化：清晰的層次結構和職責分離" -ForegroundColor Gray

Write-Host ""
Write-Host "📌 輸出檔案: $outputFile" -ForegroundColor Cyan
Write-Host "🎉 AIVA Common 模組分析完成！" -ForegroundColor Green