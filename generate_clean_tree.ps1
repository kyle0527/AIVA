# 生成乾淨的專案樹狀圖 (排除虛擬環境和快取)
param(
    [string]$Path = "c:\D\E\AIVA\AIVA-main",
    [string]$OutputFile = "c:\D\E\AIVA\AIVA-main\_out\tree_clean.txt"
)

# 要排除的目錄和文件模式
$excludeDirs = @(
    '.git',
    '__pycache__',
    '.mypy_cache', 
    '.ruff_cache',
    'node_modules',
    '.venv',
    'venv',
    '.pytest_cache',
    '.tox',
    'dist',
    'build',
    '.egg-info',
    '.eggs',
    'htmlcov',
    '.coverage',
    '.hypothesis',
    '.idea',
    '.vscode',
    'site-packages'
)

$excludeFiles = @(
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '.DS_Store',
    'Thumbs.db'
)

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
                # 排除特定目錄
                if ($_.PSIsContainer) {
                    $excludeDirs -notcontains $name
                } else {
                    # 排除特定檔案模式
                    $shouldInclude = $true
                    foreach ($pattern in $excludeFiles) {
                        if ($name -like $pattern) {
                            $shouldInclude = $false
                            break
                        }
                    }
                    $shouldInclude
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
                $output = "$Prefix$connector$($item.Name)"
                Write-Output $output
                Get-CleanTree -Path $item.FullName -Prefix "$Prefix$extension" -Level ($Level + 1) -MaxLevel $MaxLevel
            } else {
                Write-Output "$Prefix$connector$($item.Name)"
            }
        }
    } catch {
        # 忽略無法存取的目錄
    }
}

# 產生樹狀圖
Write-Host "🌳 正在生成乾淨的專案樹狀圖..." -ForegroundColor Cyan

$rootName = Split-Path $Path -Leaf
$output = @()
$output += $rootName
$output += Get-CleanTree -Path $Path

# 儲存到檔案
$output | Out-File $OutputFile -Encoding utf8

# 統計
$lineCount = $output.Count
$dirCount = ($output | Where-Object { $_ -notmatch '\.[a-z0-9]+$' }).Count
$fileCount = $lineCount - $dirCount

Write-Host "✅ 樹狀圖已產出!" -ForegroundColor Green
Write-Host "   檔案位置: $OutputFile" -ForegroundColor White
Write-Host "   總行數: $lineCount" -ForegroundColor White
Write-Host "   目錄數: $dirCount" -ForegroundColor White
Write-Host "   檔案數: $fileCount" -ForegroundColor White

Write-Host "`n📋 已排除的目錄類型:" -ForegroundColor Yellow
$excludeDirs | ForEach-Object { Write-Host "   - $_" -ForegroundColor Gray }
