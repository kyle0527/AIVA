# AIVA 專案統計生成腳本
# 用途: 生成專案文件統計和程式碼行數統計

$projectRoot = "c:\D\E\AIVA\AIVA-main"
$outputDir = Join-Path $projectRoot "_out"

Write-Host "🔍 開始生成專案統計..." -ForegroundColor Cyan

# 確保輸出目錄存在
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# 1. 統計副檔名出現次數
Write-Host "`n📊 生成副檔名統計..." -ForegroundColor Yellow
Get-ChildItem -Path $projectRoot -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object {$_.FullName -notlike "*\.git\*" -and $_.FullName -notlike "*\__pycache__\*"} |
    Group-Object Extension | 
    Select-Object @{Name='Extension';Expression={if($_.Name -eq ''){'.no_ext'}else{$_.Name}}}, Count | 
    Sort-Object Count -Descending | 
    Export-Csv (Join-Path $outputDir "ext_counts.csv") -NoTypeInformation -Encoding utf8

Write-Host "  ✅ ext_counts.csv 已生成" -ForegroundColor Green

# 2. 統計程式碼行數 (依副檔名)
Write-Host "`n📝 生成程式碼行數統計..." -ForegroundColor Yellow
$codeExts = @('.py', '.js', '.ts', '.sql', '.sh', '.bat', '.ps1', '.md', '.yaml', '.yml', '.toml', '.json', '.txt', '.html', '.css')

Get-ChildItem -Path $projectRoot -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object {
        $codeExts -contains $_.Extension -and 
        $_.FullName -notlike "*\.git\*" -and 
        $_.FullName -notlike "*\__pycache__\*" -and
        $_.FullName -notlike "*\.mypy_cache\*" -and
        $_.FullName -notlike "*\.ruff_cache\*"
    } | 
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
                  @{Name='AvgLinesPerFile';Expression={[math]::Round(($_.Group | Measure-Object -Property Lines -Average).Average, 1)}} | 
    Sort-Object TotalLines -Descending | 
    Export-Csv (Join-Path $outputDir "loc_by_ext.csv") -NoTypeInformation -Encoding utf8

Write-Host "  ✅ loc_by_ext.csv 已生成" -ForegroundColor Green

# 3. 生成樹狀架構圖 (5種格式)
Write-Host "`n🌳 生成專案樹狀架構圖..." -ForegroundColor Yellow

# 3.1 使用 tree 命令生成 ASCII 版本
Write-Host "  📄 生成 ASCII 樹狀圖..." -ForegroundColor Gray
tree /F /A $projectRoot | Out-File (Join-Path $outputDir "tree_ascii.txt") -Encoding utf8
Write-Host "  ✅ tree_ascii.txt 已生成" -ForegroundColor Green

# 3.2 使用 tree 命令生成 Unicode 版本
Write-Host "  📄 生成 Unicode 樹狀圖..." -ForegroundColor Gray
tree /F $projectRoot | Out-File (Join-Path $outputDir "tree_unicode.txt") -Encoding utf8
Write-Host "  ✅ tree_unicode.txt 已生成" -ForegroundColor Green

# 3.3 生成 Mermaid 格式
Write-Host "  📄 生成 Mermaid 圖表..." -ForegroundColor Gray
$mermaidTree = @"
graph TD
n1(["$($projectRoot.Split('\')[-1])"])
"@

function Get-MermaidTree {
    param(
        [string]$Path,
        [int]$ParentId = 1,
        [ref]$Counter,
        [int]$MaxDepth = 3,
        [int]$CurrentDepth = 0
    )
    
    if ($CurrentDepth -ge $MaxDepth) { return "" }
    
    $result = ""
    $items = Get-ChildItem -Path $Path -ErrorAction SilentlyContinue | 
        Where-Object { 
            $_.Name -notlike ".*" -and 
            $_.Name -ne "__pycache__" -and 
            $_.Name -ne "node_modules" -and
            $_.Name -ne ".mypy_cache" -and
            $_.Name -ne ".ruff_cache"
        } | 
        Sort-Object @{Expression={$_.PSIsContainer}; Descending=$true}, Name
    
    foreach ($item in $items) {
        $Counter.Value++
        $currentId = $Counter.Value
        
        if ($item.PSIsContainer) {
            $result += "n$currentId([`"$($item.Name)`"])`n"
            $result += "n$ParentId --> n$currentId`n"
            $result += Get-MermaidTree -Path $item.FullName -ParentId $currentId -Counter $Counter -MaxDepth $MaxDepth -CurrentDepth ($CurrentDepth + 1)
        } else {
            $result += "n$currentId[`"$($item.Name)`"]`n"
            $result += "n$ParentId --> n$currentId`n"
        }
    }
    
    return $result
}

$counter = 1
$mermaidTree += Get-MermaidTree -Path $projectRoot -ParentId 1 -Counter ([ref]$counter) -MaxDepth 3

$mermaidTree | Out-File (Join-Path $outputDir "tree.mmd") -Encoding utf8
Write-Host "  ✅ tree.mmd 已生成" -ForegroundColor Green

# 3.4 生成包含 Mermaid 的 HTML
Write-Host "  📄 生成 HTML 可視化..." -ForegroundColor Gray
$htmlContent = @"
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Project Tree</title>
<style>body{font-family:ui-monospace,Consolas,monospace;margin:24px}</style>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
</script>
</head>
<body>
<h2>Project Tree (Mermaid)</h2>
<pre class="mermaid">$mermaidTree</pre>
</body>
</html>
"@
$htmlContent | Out-File (Join-Path $outputDir "tree.html") -Encoding utf8
Write-Host "  ✅ tree.html 已生成 (可在瀏覽器中打開)" -ForegroundColor Green

# 3.5 生成簡化的 Markdown 樹狀圖
Write-Host "  📄 生成 Markdown 樹狀圖..." -ForegroundColor Gray

function Get-MarkdownTree {
    param(
        [string]$Path,
        [string]$Prefix = "",
        [int]$MaxDepth = 4,
        [int]$CurrentDepth = 0
    )
    
    if ($CurrentDepth -ge $MaxDepth) { return "" }
    
    $result = ""
    $items = Get-ChildItem -Path $Path -ErrorAction SilentlyContinue | 
        Where-Object { 
            $_.Name -notlike ".*" -and 
            $_.Name -ne "__pycache__" -and 
            $_.Name -ne "node_modules" -and
            $_.Name -ne ".mypy_cache" -and
            $_.Name -ne ".ruff_cache"
        } | 
        Sort-Object @{Expression={$_.PSIsContainer}; Descending=$true}, Name
    
    $itemCount = $items.Count
    for ($i = 0; $i -lt $itemCount; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq $itemCount - 1)
        
        if ($item.PSIsContainer) {
            $result += "$Prefix$(if($isLast){'└── '}else{'├── '})📁 **$($item.Name)**/`n"
            $newPrefix = "$Prefix$(if($isLast){'    '}else{'│   '})"
            $result += Get-MarkdownTree -Path $item.FullName -Prefix $newPrefix -MaxDepth $MaxDepth -CurrentDepth ($CurrentDepth + 1)
        } else {
            $icon = switch -Wildcard ($item.Extension) {
                '.py' { '🐍' }
                '.js' { '📜' }
                '.ts' { '📘' }
                '.md' { '📝' }
                '.json' { '⚙️' }
                '.yml' { '🔧' }
                '.yaml' { '🔧' }
                '.txt' { '📄' }
                '.sql' { '🗄️' }
                '.sh' { '⚡' }
                '.bat' { '⚡' }
                default { '📄' }
            }
            $result += "$Prefix$(if($isLast){'└── '}else{'├── '})$icon $($item.Name)`n"
        }
    }
    
    return $result
}

$markdownTree = @"
# 專案樹狀結構

生成時間: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

``````
$(Get-MarkdownTree -Path $projectRoot -MaxDepth 4)
``````
"@

$markdownTree | Out-File (Join-Path $outputDir "tree.md") -Encoding utf8
Write-Host "  ✅ tree.md 已生成 (帶有圖示的 Markdown 版本)" -ForegroundColor Green

# 4. 顯示統計摘要
Write-Host "`n📈 統計摘要:" -ForegroundColor Cyan
$extCounts = Import-Csv (Join-Path $outputDir "ext_counts.csv")
$locStats = Import-Csv (Join-Path $outputDir "loc_by_ext.csv")

Write-Host "  總文件數: $($extCounts | Measure-Object -Property Count -Sum | Select-Object -ExpandProperty Sum)" -ForegroundColor White
Write-Host "  總程式碼行數: $($locStats | Measure-Object -Property TotalLines -Sum | Select-Object -ExpandProperty Sum)" -ForegroundColor White

Write-Host "`n🎯 Top 5 副檔名 (依文件數):" -ForegroundColor Cyan
$extCounts | Select-Object -First 5 | ForEach-Object {
    Write-Host "  $($_.Extension): $($_.Count) 個文件" -ForegroundColor White
}

Write-Host "`n💻 Top 5 副檔名 (依程式碼行數):" -ForegroundColor Cyan
$locStats | Select-Object -First 5 | ForEach-Object {
    Write-Host "  $($_.Extension): $($_.TotalLines) 行 ($($_.FileCount) 個文件, 平均 $($_.AvgLinesPerFile) 行/文件)" -ForegroundColor White
}

Write-Host "`n📂 已生成的樹狀架構文件:" -ForegroundColor Cyan
Write-Host "  1. tree_ascii.txt    - ASCII 版本樹狀圖" -ForegroundColor White
Write-Host "  2. tree_unicode.txt  - Unicode 版本樹狀圖 (帶中文)" -ForegroundColor White
Write-Host "  3. tree.mmd          - Mermaid 格式 (可用於文檔)" -ForegroundColor White
Write-Host "  4. tree.html         - HTML 可視化版本 (在瀏覽器中打開)" -ForegroundColor White
Write-Host "  5. tree.md           - Markdown 版本 (帶圖示)" -ForegroundColor White

Write-Host "`n✨ 完成! 統計文件已保存至: $outputDir" -ForegroundColor Green
