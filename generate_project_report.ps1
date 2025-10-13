# AIVA 專案完整報告生成器
# 整合樹狀圖、統計數據、程式碼分析

param(
    [string]$ProjectRoot = "c:\D\E\AIVA\AIVA-main",
    [string]$OutputDir = "c:\D\E\AIVA\AIVA-main\_out"
)

Write-Host "🚀 開始生成專案完整報告..." -ForegroundColor Cyan

# 要排除的目錄
$excludeDirs = @(
    '.git', '__pycache__', '.mypy_cache', '.ruff_cache',
    'node_modules', '.venv', 'venv', '.pytest_cache', 
    '.tox', 'dist', 'build', '.egg-info', '.eggs',
    'htmlcov', '.coverage', '.hypothesis', '.idea', '.vscode'
)

# ==================== 1. 收集統計數據 ====================
Write-Host "`n📊 收集專案統計數據..." -ForegroundColor Yellow

# 副檔名統計
$allFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { 
        $path = $_.FullName
        -not ($excludeDirs | Where-Object { $path -like "*\$_\*" })
    }

$extStats = $allFiles | Group-Object Extension | 
    Select-Object @{Name='Extension';Expression={if($_.Name -eq ''){'.no_ext'}else{$_.Name}}}, Count | 
    Sort-Object Count -Descending

# 程式碼行數統計
$codeExts = @('.py', '.js', '.ts', '.sql', '.sh', '.bat', '.ps1', '.md', '.yaml', '.yml', '.toml', '.json', '.html', '.css')

$locStats = $allFiles | 
    Where-Object { $codeExts -contains $_.Extension } | 
    ForEach-Object {
        $lineCount = (Get-Content $_.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        [PSCustomObject]@{
            Extension = $_.Extension
            Lines = $lineCount
            File = $_.Name
        }
    } | 
    Group-Object Extension | 
    Select-Object @{Name='Extension';Expression={$_.Name}}, 
                  @{Name='TotalLines';Expression={($_.Group | Measure-Object -Property Lines -Sum).Sum}}, 
                  @{Name='FileCount';Expression={$_.Count}},
                  @{Name='AvgLines';Expression={[math]::Round(($_.Group | Measure-Object -Property Lines -Average).Average, 1)}} | 
    Sort-Object TotalLines -Descending

# ==================== 2. 生成樹狀圖 ====================
Write-Host "🌳 生成專案樹狀結構..." -ForegroundColor Yellow

function Get-CleanTree {
    param(
        [string]$Path,
        [string]$Prefix = "",
        [int]$Level = 0,
        [int]$MaxLevel = 10
    )
    
    if ($Level -ge $MaxLevel) { return }
    
    try {
        $items = Get-ChildItem -Path $Path -Force -ErrorAction Stop | 
            Where-Object { 
                $name = $_.Name
                if ($_.PSIsContainer) {
                    $excludeDirs -notcontains $name
                } else {
                    $name -notlike '*.pyc' -and $name -notlike '*.pyo'
                }
            } | 
            Sort-Object @{Expression={$_.PSIsContainer}; Descending=$true}, Name
        
        $itemCount = $items.Count
        for ($i = 0; $i -lt $itemCount; $i++) {
            $item = $items[$i]
            $isLast = ($i -eq $itemCount - 1)
            
            $connector = if ($isLast) { "└─" } else { "├─" }
            $extension = if ($isLast) { "    " } else { "│   " }
            
            if ($item.PSIsContainer) {
                $output = "$Prefix$connector📁 $($item.Name)/"
                Write-Output $output
                Get-CleanTree -Path $item.FullName -Prefix "$Prefix$extension" -Level ($Level + 1) -MaxLevel $MaxLevel
            } else {
                $icon = switch -Wildcard ($item.Extension) {
                    '.py' { '🐍' }
                    '.js' { '📜' }
                    '.ts' { '📘' }
                    '.md' { '📝' }
                    '.json' { '⚙️' }
                    '.yml' { '🔧' }
                    '.yaml' { '🔧' }
                    '.sql' { '🗄️' }
                    '.sh' { '⚡' }
                    '.bat' { '⚡' }
                    '.ps1' { '⚡' }
                    '.txt' { '📄' }
                    '.html' { '🌐' }
                    '.css' { '🎨' }
                    default { '📄' }
                }
                Write-Output "$Prefix$connector$icon $($item.Name)"
            }
        }
    } catch {
        # 忽略無法存取的目錄
    }
}

$rootName = Split-Path $ProjectRoot -Leaf
$treeOutput = @()
$treeOutput += "📦 $rootName"
$treeOutput += Get-CleanTree -Path $ProjectRoot

# ==================== 3. 生成整合報告 ====================
Write-Host "📝 生成整合報告..." -ForegroundColor Yellow

$reportContent = @"
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AIVA 專案完整分析報告                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

生成時間: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
專案路徑: $ProjectRoot

═══════════════════════════════════════════════════════════════════════════════
📊 專案統計摘要
═══════════════════════════════════════════════════════════════════════════════

總文件數量: $($extStats | Measure-Object -Property Count -Sum | Select-Object -ExpandProperty Sum)
總程式碼行數: $($locStats | Measure-Object -Property TotalLines -Sum | Select-Object -ExpandProperty Sum)
程式碼檔案數: $($locStats | Measure-Object -Property FileCount -Sum | Select-Object -ExpandProperty Sum)

───────────────────────────────────────────────────────────────────────────────
🎯 檔案類型統計 (Top 10)
───────────────────────────────────────────────────────────────────────────────

$($extStats | Select-Object -First 10 | ForEach-Object {
    $ext = $_.Extension.PadRight(15)
    $count = $_.Count.ToString().PadLeft(6)
    "  $ext $count 個檔案"
} | Out-String)

───────────────────────────────────────────────────────────────────────────────
💻 程式碼行數統計 (依副檔名)
───────────────────────────────────────────────────────────────────────────────

$($locStats | ForEach-Object {
    $ext = $_.Extension.PadRight(10)
    $lines = $_.TotalLines.ToString().PadLeft(8)
    $files = $_.FileCount.ToString().PadLeft(5)
    $avg = $_.AvgLines.ToString().PadLeft(7)
    "  $ext $lines 行  ($files 個檔案, 平均 $avg 行/檔案)"
} | Out-String)

───────────────────────────────────────────────────────────────────────────────
📈 專案規模分析
───────────────────────────────────────────────────────────────────────────────

Python 程式碼行數: $($locStats | Where-Object Extension -eq '.py' | Select-Object -ExpandProperty TotalLines) 行
Python 檔案數量: $($locStats | Where-Object Extension -eq '.py' | Select-Object -ExpandProperty FileCount) 個
平均每個 Python 檔案: $($locStats | Where-Object Extension -eq '.py' | Select-Object -ExpandProperty AvgLines) 行

文檔 (Markdown) 行數: $($locStats | Where-Object Extension -eq '.md' | Select-Object -ExpandProperty TotalLines) 行
配置檔案數量: $(($extStats | Where-Object {$_.Extension -in @('.json', '.yml', '.yaml', '.toml', '.ini')}) | Measure-Object -Property Count -Sum | Select-Object -ExpandProperty Sum) 個

───────────────────────────────────────────────────────────────────────────────
🚫 已排除的目錄類型
───────────────────────────────────────────────────────────────────────────────

$($excludeDirs | ForEach-Object { "  • $_" } | Out-String)

═══════════════════════════════════════════════════════════════════════════════
🌳 專案目錄結構
═══════════════════════════════════════════════════════════════════════════════

$($treeOutput | Out-String)

═══════════════════════════════════════════════════════════════════════════════
� 專案架構說明 (AI 補充區)
═══════════════════════════════════════════════════════════════════════════════

⚠️ 此區域由 AI 分析補充,執行腳本後請提供此報告給 AI 進行深度分析

【待補充內容】
• 核心模組功能說明
• 漏洞檢測引擎架構
• 掃描引擎工作原理
• 整合層設計模式
• 工作流程說明
• 技術棧詳細列表

請將此報告提供給 GitHub Copilot,它會分析程式碼並補充詳細的中文說明。


═══════════════════════════════════════════════════════════════════════════════
�📌 報告說明
═══════════════════════════════════════════════════════════════════════════════

• 本報告整合了專案的檔案統計、程式碼行數分析和目錄結構
• 已自動排除虛擬環境、快取檔案、IDE 配置等非程式碼目錄
• 圖示說明:
  🐍 Python   📜 JavaScript   📘 TypeScript   📝 Markdown
  ⚙️ JSON      🔧 YAML         🗄️ SQL          ⚡ Shell/Batch
  🌐 HTML      🎨 CSS          📁 目錄         📄 其他檔案


═══════════════════════════════════════════════════════════════════════════════
✨ 報告結束
═══════════════════════════════════════════════════════════════════════════════
"@

# 儲存報告
$reportFile = Join-Path $OutputDir "PROJECT_REPORT.txt"
$reportContent | Out-File $reportFile -Encoding utf8

Write-Host "✅ 報告已生成: PROJECT_REPORT.txt" -ForegroundColor Green

# ==================== 4. 清理其他檔案 ====================
Write-Host "`n🧹 清理舊的檔案..." -ForegroundColor Yellow

$filesToKeep = @('PROJECT_REPORT.txt')
$filesToDelete = Get-ChildItem $OutputDir -File | Where-Object { $filesToKeep -notcontains $_.Name }

foreach ($file in $filesToDelete) {
    Remove-Item $file.FullName -Force
    Write-Host "  🗑️  已刪除: $($file.Name)" -ForegroundColor Gray
}

# ==================== 5. 完成 ====================
Write-Host "`n" -NoNewline
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          ✨ 報告生成完成！                    ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📄 報告位置: $reportFile" -ForegroundColor Cyan
Write-Host "📊 統計資料: 已整合" -ForegroundColor Cyan
Write-Host "🌳 目錄結構: 已整合" -ForegroundColor Cyan
Write-Host "🗑️  舊檔案: 已清理" -ForegroundColor Cyan
Write-Host ""

# 打開報告
code $reportFile
