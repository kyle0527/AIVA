# AIVA 專案完整報告生成器
# 整合樹狀圖、統計數據、程式碼分析

param(
    [string]$ProjectRoot = "C:\D\fold7\AIVA-git",
    [string]$OutputDir = "C:\D\fold7\AIVA-git\_out"
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
$codeExts = @('.py', '.js', '.ts', '.go', '.rs', '.sql', '.sh', '.bat', '.ps1', '.md', '.yaml', '.yml', '.toml', '.json', '.html', '.css')

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

# 多語言統計
$pythonStats = $locStats | Where-Object Extension -eq '.py'
$goStats = $locStats | Where-Object Extension -eq '.go'
$rustStats = $locStats | Where-Object Extension -eq '.rs'
$tsStats = $locStats | Where-Object Extension -in @('.ts', '.js')

$totalCodeLines = ($locStats | Measure-Object -Property TotalLines -Sum).Sum
$pythonPct = if ($pythonStats) { [math]::Round(($pythonStats.TotalLines / $totalCodeLines) * 100, 1) } else { 0 }
$goPct = if ($goStats) { [math]::Round(($goStats.TotalLines / $totalCodeLines) * 100, 1) } else { 0 }
$rustPct = if ($rustStats) { [math]::Round(($rustStats.TotalLines / $totalCodeLines) * 100, 1) } else { 0 }
$tsPct = if ($tsStats) { [math]::Round((($tsStats | Measure-Object -Property TotalLines -Sum).Sum / $totalCodeLines) * 100, 1) } else { 0 }

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
                    '.go' { '🔷' }
                    '.rs' { '🦀' }
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

🐍 Python 程式碼: $($pythonStats.TotalLines) 行 ($($pythonStats.FileCount) 個檔案, $pythonPct% 佔比)
   平均每個檔案: $($pythonStats.AvgLines) 行

🔷 Go 程式碼: $(if ($goStats) { $goStats.TotalLines } else { 0 }) 行 ($(if ($goStats) { $goStats.FileCount } else { 0 }) 個檔案, $goPct% 佔比)
   $(if ($goStats) { "平均每個檔案: $($goStats.AvgLines) 行" } else { "無 Go 檔案" })

🦀 Rust 程式碼: $(if ($rustStats) { $rustStats.TotalLines } else { 0 }) 行 ($(if ($rustStats) { $rustStats.FileCount } else { 0 }) 個檔案, $rustPct% 佔比)
   $(if ($rustStats) { "平均每個檔案: $($rustStats.AvgLines) 行" } else { "無 Rust 檔案" })

📘 TypeScript/JavaScript: $(if ($tsStats) { ($tsStats | Measure-Object -Property TotalLines -Sum).Sum } else { 0 }) 行 ($(if ($tsStats) { ($tsStats | Measure-Object -Property FileCount -Sum).Sum } else { 0 }) 個檔案, $tsPct% 佔比)
   $(if ($tsStats) { "平均每個檔案: " + [math]::Round((($tsStats | Measure-Object -Property AvgLines -Average).Average), 1) + " 行" } else { "無 TS/JS 檔案" })

📝 文檔 (Markdown) 行數: $($locStats | Where-Object Extension -eq '.md' | Select-Object -ExpandProperty TotalLines) 行
⚙️  配置檔案數量: $(($extStats | Where-Object {$_.Extension -in @('.json', '.yml', '.yaml', '.toml', '.ini')}) | Measure-Object -Property Count -Sum | Select-Object -ExpandProperty Sum) 個

───────────────────────────────────────────────────────────────────────────────
🌐 多語言架構概覽
───────────────────────────────────────────────────────────────────────────────

總程式碼檔案數: $($locStats | Measure-Object -Property FileCount -Sum | Select-Object -ExpandProperty Sum)
總程式碼行數: $totalCodeLines

語言分布:
  Python:     $pythonPct% ████████████████████
  Go:         $goPct%
  Rust:       $rustPct%
  TS/JS:      $tsPct%
  其他:       $([math]::Round(100 - $pythonPct - $goPct - $rustPct - $tsPct, 1))%

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
📌 報告說明
═══════════════════════════════════════════════════════════════════════════════

• 本報告整合了專案的檔案統計、程式碼行數分析和目錄結構
• 已自動排除虛擬環境、快取檔案、IDE 配置等非程式碼目錄
• 圖示說明:
  🐍 Python   📜 JavaScript   📘 TypeScript   📝 Markdown
  ⚙️ JSON      🔧 YAML         🗄️ SQL          ⚡ Shell/Batch
  🔷 Go        🦀 Rust         🌐 HTML         🎨 CSS
  📁 目錄      📄 其他檔案

• 多語言架構:
  - Python: 主要業務邏輯、Web API、核心引擎
  - Go: 高效能模組 (身份驗證、雲端安全、組成分析)
  - Rust: 靜態分析、資訊收集 (記憶體安全、高效能)
  - TypeScript: 動態掃描引擎 (Playwright 瀏覽器自動化)


═══════════════════════════════════════════════════════════════════════════════
✨ 報告結束
═══════════════════════════════════════════════════════════════════════════════
"@

# 儲存報告
$reportFile = Join-Path $OutputDir "PROJECT_REPORT.txt"
$reportContent | Out-File $reportFile -Encoding utf8

Write-Host "✅ 報告已生成: PROJECT_REPORT.txt" -ForegroundColor Green

# ==================== 4. 生成 Mermaid 圖表 ====================
Write-Host "`n📊 生成 Mermaid 架構圖..." -ForegroundColor Yellow

$mermaidContent = @"
# AIVA 專案架構圖

## 1. 多語言架構概覽

``````mermaid
graph TB
    subgraph "🐍 Python Layer"
        PY_API[FastAPI Web API]
        PY_CORE[核心引擎]
        PY_SCAN[掃描服務]
        PY_INTG[整合層]
    end

    subgraph "🔷 Go Layer"
        GO_AUTH[身份驗證檢測]
        GO_CSPM[雲端安全]
        GO_SCA[軟體組成分析]
        GO_SSRF[SSRF 檢測]
    end

    subgraph "🦀 Rust Layer"
        RS_SAST[靜態分析引擎]
        RS_INFO[資訊收集器]
    end

    subgraph "📘 TypeScript Layer"
        TS_SCAN[Playwright 掃描]
    end

    subgraph "🗄️ Data Layer"
        DB[(PostgreSQL)]
        MQ[RabbitMQ]
    end

    PY_API --> PY_CORE
    PY_CORE --> PY_SCAN
    PY_SCAN --> PY_INTG

    PY_INTG -->|RPC| GO_AUTH
    PY_INTG -->|RPC| GO_CSPM
    PY_INTG -->|RPC| GO_SCA
    PY_INTG -->|RPC| GO_SSRF
    PY_INTG -->|RPC| RS_SAST
    PY_INTG -->|RPC| RS_INFO
    PY_INTG -->|RPC| TS_SCAN

    GO_AUTH --> MQ
    GO_CSPM --> MQ
    GO_SCA --> MQ
    GO_SSRF --> MQ
    RS_SAST --> MQ
    RS_INFO --> MQ
    TS_SCAN --> MQ

    MQ --> DB
    PY_CORE --> DB

    style PY_API fill:#3776ab
    style GO_AUTH fill:#00ADD8
    style RS_SAST fill:#CE422B
    style TS_SCAN fill:#3178C6
``````

## 2. 程式碼分布統計

``````mermaid
pie title 程式碼行數分布
    "Python ($pythonPct%)" : $($pythonStats.TotalLines)
    "Go ($goPct%)" : $(if ($goStats) { $goStats.TotalLines } else { 0 })
    "Rust ($rustPct%)" : $(if ($rustStats) { $rustStats.TotalLines } else { 0 })
    "TypeScript/JS ($tsPct%)" : $(if ($tsStats) { ($tsStats | Measure-Object -Property TotalLines -Sum).Sum } else { 0 })
    "其他" : $(($locStats | Where-Object { $_.Extension -notin @('.py', '.go', '.rs', '.ts', '.js') } | Measure-Object -Property TotalLines -Sum).Sum)
``````

## 3. 模組關係圖

``````mermaid
graph LR
    subgraph "services"
        aiva_common[aiva_common<br/>共用模組]
        core[core<br/>核心引擎]
        function[function<br/>功能模組]
        integration[integration<br/>整合層]
        scan[scan<br/>掃描引擎]
    end

    subgraph "function 子模組"
        func_py[Python 模組]
        func_go[Go 模組]
        func_rs[Rust 模組]
    end

    subgraph "scan 子模組"
        scan_py[Python 掃描]
        scan_ts[Node.js 掃描]
    end

    core --> aiva_common
    scan --> aiva_common
    function --> aiva_common
    integration --> aiva_common

    integration --> function
    integration --> scan

    function --> func_py
    function --> func_go
    function --> func_rs

    scan --> scan_py
    scan --> scan_ts

    style aiva_common fill:#90EE90
    style core fill:#FFD700
    style function fill:#87CEEB
    style integration fill:#FFA07A
    style scan fill:#DDA0DD
``````

## 4. 技術棧選擇流程圖

``````mermaid
flowchart TD
    Start([新功能需求]) --> Perf{需要高效能?}
    Perf -->|是| Memory{需要記憶體安全?}
    Perf -->|否| Web{是 Web API?}

    Memory -->|是| Rust[使用 Rust<br/>靜態分析/資訊收集]
    Memory -->|否| Go[使用 Go<br/>認證/雲端安全/SCA]

    Web -->|是| Python[使用 Python<br/>FastAPI/核心邏輯]
    Web -->|否| Browser{需要瀏覽器?}

    Browser -->|是| TS[使用 TypeScript<br/>Playwright 掃描]
    Browser -->|否| Python

    Rust --> MQ[Message Queue]
    Go --> MQ
    Python --> MQ
    TS --> MQ

    MQ --> Deploy([部署模組])

    style Rust fill:#CE422B
    style Go fill:#00ADD8
    style Python fill:#3776ab
    style TS fill:#3178C6
``````

生成時間: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

$mermaidFile = Join-Path $OutputDir "tree.mmd"
$mermaidContent | Out-File $mermaidFile -Encoding utf8

Write-Host "✅ Mermaid 圖表已生成: tree.mmd" -ForegroundColor Green

# ==================== 5. 清理其他檔案 ====================
Write-Host "`n🧹 清理舊的檔案..." -ForegroundColor Yellow

$filesToKeep = @('PROJECT_REPORT.txt', 'tree.mmd')
$filesToDelete = Get-ChildItem $OutputDir -File | Where-Object { $filesToKeep -notcontains $_.Name -and $_.Extension -ne '.csv' }

foreach ($file in $filesToDelete) {
    Remove-Item $file.FullName -Force
    Write-Host "  🗑️  已刪除: $($file.Name)" -ForegroundColor Gray
}

# ==================== 6. 完成 ====================
Write-Host "`n" -NoNewline
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          ✨ 報告生成完成！                    ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "📄 報告位置: $reportFile" -ForegroundColor Cyan
Write-Host "📊 統計資料: 已整合" -ForegroundColor Cyan
Write-Host "🌳 目錄結構: 已整合" -ForegroundColor Cyan
Write-Host "📈 Mermaid 圖表: $mermaidFile" -ForegroundColor Cyan
Write-Host "🗑️  舊檔案: 已清理" -ForegroundColor Cyan
Write-Host ""

# 打開報告
code $reportFile
