<#
.SYNOPSIS
    組合 Mermaid 流程圖並生成 HTML 視覺化
    
.DESCRIPTION
    根據檔案名稱自動組合相關的 Mermaid 流程圖，生成互動式 HTML 視覺化
    參考 Mermaid.js 11.11.0 實現
    
.PARAMETER SourceDir
    Mermaid 檔案來源目錄
    
.PARAMETER OutputDir
    輸出目錄
    
.PARAMETER Category
    分類過濾器（可選）：core, scan, function, integration, all
    
.PARAMETER MermaidJsPath
    Mermaid.js 路徑（預設使用 CDN）
    
.EXAMPLE
    .\combine_mermaid_diagrams.ps1 -SourceDir "_out1101016\mermaid_details" -OutputDir "_out\combined_diagrams"
    
.EXAMPLE
    .\combine_mermaid_diagrams.ps1 -SourceDir "_out1101016\mermaid_details" -Category "core" -MermaidJsPath "C:\D\Autofix_Mermaid\autofix_mermaidV3.7\assets\mermaid-11.11.0\mermaid.js"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "_out\combined_diagrams",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("core", "scan", "function", "integration", "all", "architecture")]
    [string]$Category = "all",
    
    [Parameter(Mandatory=$false)]
    [string]$MermaidJsPath = ""
)

# ============================================================================
# 函數定義
# ============================================================================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Get-MermaidFiles {
    param(
        [string]$Path,
        [string]$Filter
    )
    
    if ($Filter -eq "all") {
        return Get-ChildItem -Path $Path -Filter "*.mmd" -Recurse
    } elseif ($Filter -eq "architecture") {
        return Get-ChildItem -Path "$Path\..\architecture_diagrams" -Filter "*.mmd" -ErrorAction SilentlyContinue
    } else {
        return Get-ChildItem -Path $Path -Filter "*$Filter*.mmd" -Recurse
    }
}

function Get-DiagramType {
    param([string]$Content)
    
    if ($Content -match '^\s*sequenceDiagram') { return "sequence" }
    elseif ($Content -match '^\s*graph\s+(TB|LR|TD|RL|BT)') { return "graph" }
    elseif ($Content -match '^\s*flowchart\s+(TB|LR|TD|RL|BT)') { return "flowchart" }
    elseif ($Content -match '^\s*classDiagram') { return "class" }
    elseif ($Content -match '^\s*stateDiagram') { return "state" }
    elseif ($Content -match '^\s*erDiagram') { return "er" }
    elseif ($Content -match '^\s*gantt') { return "gantt" }
    elseif ($Content -match '^\s*pie') { return "pie" }
    elseif ($Content -match '^\s*journey') { return "journey" }
    elseif ($Content -match '^\s*gitGraph') { return "git" }
    else { return "unknown" }
}

function Get-DiagramCategory {
    param([string]$FileName)
    
    if ($FileName -match 'core_messaging|core_module') { return "core" }
    elseif ($FileName -match 'scan') { return "scan" }
    elseif ($FileName -match 'function|sqli|xss|ssrf|idor') { return "function" }
    elseif ($FileName -match 'integration') { return "integration" }
    elseif ($FileName -match 'architecture|overall|workflow') { return "architecture" }
    else { return "other" }
}

function New-CombinedHtml {
    param(
        [array]$Diagrams,
        [string]$Title,
        [string]$OutputPath,
        [string]$MermaidJs
    )
    
    # 決定使用本地 JS 還是 CDN
    if ($MermaidJs -and (Test-Path $MermaidJs)) {
        $jsContent = Get-Content $MermaidJs -Raw
        $mermaidScript = "<script>$jsContent</script>"
        Write-ColorOutput "使用本地 Mermaid.js: $MermaidJs" "Cyan"
    } else {
        $mermaidScript = '<script type="module">
            import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.11.0/dist/mermaid.esm.min.mjs";
            mermaid.initialize({ 
                startOnLoad: true,
                theme: "default",
                securityLevel: "loose",
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: "basis"
                },
                sequence: {
                    useMaxWidth: true,
                    actorMargin: 50,
                    boxMargin: 10,
                    boxTextMargin: 5,
                    noteMargin: 10,
                    messageMargin: 35
                }
            });
        </script>'
        Write-ColorOutput "使用 CDN Mermaid.js" "Cyan"
    }
    
    $htmlContent = @"
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$Title - AIVA 流程圖組合</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        
        nav {
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .nav-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .nav-btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .content {
            padding: 30px;
        }
        
        .diagram-section {
            margin-bottom: 40px;
            padding: 25px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .diagram-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .diagram-title {
            font-size: 1.5em;
            color: #2c3e50;
            font-weight: 600;
        }
        
        .diagram-meta {
            display: flex;
            gap: 15px;
        }
        
        .badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .badge-type {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .badge-category {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .diagram-content {
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
        }
        
        .mermaid {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        }
        
        .diagram-source {
            margin-top: 15px;
            padding: 10px;
            background: #263238;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #aed581;
            overflow-x: auto;
            display: none;
        }
        
        .show-source-btn {
            margin-top: 10px;
            padding: 8px 16px;
            background: #455a64;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        
        .show-source-btn:hover {
            background: #37474f;
        }
        
        footer {
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }
        
        .toc {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .toc h2 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        
        .toc li {
            padding: 8px 0;
        }
        
        .toc a {
            color: #667eea;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .toc a:hover {
            color: #5568d3;
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            header h1 {
                font-size: 1.8em;
            }
            
            .stats {
                flex-direction: column;
                gap: 15px;
            }
            
            .nav-buttons {
                flex-direction: column;
            }
            
            .diagram-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
    </style>
    $mermaidScript
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 $Title</h1>
            <p>AIVA 跨模組流程圖組合視覺化</p>
        </header>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">$($Diagrams.Count)</div>
                <div class="stat-label">流程圖總數</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">$($Diagrams | Where-Object {$_.Type -eq 'sequence'} | Measure-Object | Select-Object -ExpandProperty Count)</div>
                <div class="stat-label">序列圖</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">$($Diagrams | Where-Object {$_.Type -eq 'graph' -or $_.Type -eq 'flowchart'} | Measure-Object | Select-Object -ExpandProperty Count)</div>
                <div class="stat-label">流程圖</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">$(Get-Date -Format 'yyyy-MM-dd')</div>
                <div class="stat-label">生成日期</div>
            </div>
        </div>
        
        <nav>
            <div class="nav-buttons">
                <button class="nav-btn" onclick="scrollToTop()">⬆️ 回到頂部</button>
                <button class="nav-btn" onclick="toggleAllSources()">📝 顯示/隱藏所有原始碼</button>
                <button class="nav-btn" onclick="exportData()">💾 匯出資料</button>
                <button class="nav-btn" onclick="printPage()">🖨️ 列印</button>
            </div>
        </nav>
        
        <div class="content">
            <div class="toc">
                <h2>📑 目錄</h2>
                <ul>
"@

    # 生成目錄
    $index = 1
    foreach ($diagram in $Diagrams) {
        $htmlContent += "                    <li><a href=`"#diagram-$index`">$index. $($diagram.Title)</a></li>`n"
        $index++
    }

    $htmlContent += @"
                </ul>
            </div>
            
"@

    # 生成每個流程圖
    $index = 1
    foreach ($diagram in $Diagrams) {
        $diagramId = "diagram-$index"
        $sourceId = "source-$index"
        
        $htmlContent += @"
            <div class="diagram-section" id="$diagramId">
                <div class="diagram-header">
                    <div class="diagram-title">$index. $($diagram.Title)</div>
                    <div class="diagram-meta">
                        <span class="badge badge-type">$($diagram.Type)</span>
                        <span class="badge badge-category">$($diagram.Category)</span>
                    </div>
                </div>
                <div class="diagram-content">
                    <div class="mermaid">
$($diagram.Content)
                    </div>
                </div>
                <button class="show-source-btn" onclick="toggleSource('$sourceId')">📝 顯示原始碼</button>
                <pre class="diagram-source" id="$sourceId">$([System.Web.HttpUtility]::HtmlEncode($diagram.Content))</pre>
            </div>
            
"@
        $index++
    }

    $htmlContent += @"
        </div>
        
        <footer>
            <p>© 2025 AIVA Platform | 生成時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')</p>
            <p>共 $($Diagrams.Count) 個流程圖 | Powered by Mermaid.js 11.11.0</p>
        </footer>
    </div>
    
    <script>
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        function toggleSource(id) {
            const source = document.getElementById(id);
            if (source.style.display === 'none' || source.style.display === '') {
                source.style.display = 'block';
            } else {
                source.style.display = 'none';
            }
        }
        
        function toggleAllSources() {
            const sources = document.querySelectorAll('.diagram-source');
            const isAnyVisible = Array.from(sources).some(s => s.style.display === 'block');
            sources.forEach(source => {
                source.style.display = isAnyVisible ? 'none' : 'block';
            });
        }
        
        function exportData() {
            const data = {
                title: '$Title',
                date: '$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')',
                diagrams: $($Diagrams.Count),
                categories: {}
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'aiva-diagrams-export.json';
            a.click();
        }
        
        function printPage() {
            window.print();
        }
        
        // 錯誤處理
        window.addEventListener('error', function(e) {
            console.error('Mermaid 渲染錯誤:', e);
        });
    </script>
</body>
</html>
"@

    $htmlContent | Out-File -FilePath $OutputPath -Encoding UTF8
    Write-ColorOutput "✅ HTML 已生成: $OutputPath" "Green"
}

# ============================================================================
# 主程式
# ============================================================================

Write-ColorOutput "`n🚀 AIVA Mermaid 流程圖組合工具" "Cyan"
Write-ColorOutput "=" * 60 "Cyan"

# 檢查來源目錄
if (-not (Test-Path $SourceDir)) {
    Write-ColorOutput "❌ 錯誤: 來源目錄不存在: $SourceDir" "Red"
    exit 1
}

# 建立輸出目錄
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-ColorOutput "📁 已建立輸出目錄: $OutputDir" "Yellow"
}

# 讀取 Mermaid 檔案
Write-ColorOutput "`n🔍 掃描 Mermaid 檔案..." "Cyan"
$files = Get-MermaidFiles -Path $SourceDir -Filter $Category

if ($files.Count -eq 0) {
    Write-ColorOutput "❌ 未找到符合條件的 Mermaid 檔案" "Red"
    exit 1
}

Write-ColorOutput "📊 找到 $($files.Count) 個檔案" "Green"

# 分析檔案
Write-ColorOutput "`n📝 分析檔案內容..." "Cyan"
$diagrams = @()

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    $type = Get-DiagramType -Content $content
    $cat = Get-DiagramCategory -FileName $file.Name
    
    $diagrams += [PSCustomObject]@{
        Title = $file.BaseName -replace '^\d+_', '' -replace '_', ' '
        Type = $type
        Category = $cat
        Content = $content.Trim()
        FileName = $file.Name
        FilePath = $file.FullName
    }
    
    Write-Host "  ✓ $($file.Name) [$type / $cat]" -ForegroundColor Gray
}

# 按分類分組
Write-ColorOutput "`n📊 統計資料:" "Cyan"
$groupedByCategory = $diagrams | Group-Object Category
foreach ($group in $groupedByCategory) {
    Write-ColorOutput "  • $($group.Name): $($group.Count) 個" "Yellow"
}

$groupedByType = $diagrams | Group-Object Type
Write-ColorOutput "`n📊 圖表類型:" "Cyan"
foreach ($group in $groupedByType) {
    Write-ColorOutput "  • $($group.Name): $($group.Count) 個" "Yellow"
}

# 生成組合 HTML
Write-ColorOutput "`n🎨 生成視覺化..." "Cyan"

if ($Category -eq "all") {
    # 生成總覽
    $outputPath = Join-Path $OutputDir "00_all_diagrams.html"
    New-CombinedHtml -Diagrams $diagrams -Title "所有流程圖" -OutputPath $outputPath -MermaidJs $MermaidJsPath
    
    # 按分類生成
    foreach ($group in $groupedByCategory) {
        $outputPath = Join-Path $OutputDir "01_$($group.Name)_diagrams.html"
        New-CombinedHtml -Diagrams $group.Group -Title "$($group.Name) 流程圖" -OutputPath $outputPath -MermaidJs $MermaidJsPath
    }
} else {
    $outputPath = Join-Path $OutputDir "$Category`_diagrams.html"
    New-CombinedHtml -Diagrams $diagrams -Title "$Category 流程圖" -OutputPath $outputPath -MermaidJs $MermaidJsPath
}

# 生成索引文件
Write-ColorOutput "`n📄 生成索引文件..." "Cyan"
$indexPath = Join-Path $OutputDir "INDEX.md"
$indexContent = @"
# AIVA Mermaid 流程圖組合索引

生成時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## 📊 統計

- **總檔案數**: $($files.Count)
- **總流程圖數**: $($diagrams.Count)
- **分類數**: $($groupedByCategory.Count)

## 📑 分類統計

"@

foreach ($group in $groupedByCategory) {
    $indexContent += "- **$($group.Name)**: $($group.Count) 個流程圖`n"
}

$indexContent += @"

## 🎨 生成的檔案

"@

$htmlFiles = Get-ChildItem -Path $OutputDir -Filter "*.html" | Sort-Object Name
foreach ($htmlFile in $htmlFiles) {
    $indexContent += "- [$($htmlFile.Name)]($($htmlFile.Name))`n"
}

$indexContent += @"

## 📋 流程圖清單

"@

foreach ($diagram in $diagrams | Sort-Object Category, Title) {
    $indexContent += "- [$($diagram.Category)] **$($diagram.Title)** [$($diagram.Type)]`n"
}

$indexContent | Out-File -FilePath $indexPath -Encoding UTF8
Write-ColorOutput "✅ 索引已生成: $indexPath" "Green"

# 完成
Write-ColorOutput "`n" "White"
Write-ColorOutput "=" * 60 "Green"
Write-ColorOutput "✅ 完成！" "Green"
Write-ColorOutput "=" * 60 "Green"
Write-ColorOutput "`n📁 輸出目錄: $OutputDir" "Cyan"
Write-ColorOutput "📊 已生成 $($htmlFiles.Count) 個 HTML 檔案" "Cyan"
Write-ColorOutput "`n🌐 使用瀏覽器開啟 HTML 檔案即可查看" "Yellow"
Write-ColorOutput "`n" "White"
