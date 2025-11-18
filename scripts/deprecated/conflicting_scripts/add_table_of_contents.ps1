#!/usr/bin/env pwsh

# 為 reports 目錄下的 Markdown 文件自動添加目錄
# 日期: 2025-11-07
# 用途: 批量為 Markdown 文件添加目錄結構

param(
    [string]$ReportsPath = "C:\D\fold7\AIVA-git\reports",
    [switch]$DryRun,
    [switch]$Recursive
)

function Get-MarkdownFiles {
    param([string]$Path)
    
    if ($Recursive -or !$PSBoundParameters.ContainsKey('Recursive')) {
        Get-ChildItem -Path $Path -Filter "*.md" -Recurse
    } else {
        Get-ChildItem -Path $Path -Filter "*.md"
    }
}

function Get-Headings {
    param([string]$Content)
    
    $headings = @()
    $lines = $Content -split "`n"
    
    foreach ($line in $lines) {
        if ($line -match '^(#{1,6})\s+(.+)$') {
            $level = $Matches[1].Length
            $title = $Matches[2].Trim()
            
            # 跳過主標題 (H1) 和已經是目錄的部分
            if ($level -gt 1 -and $title -notmatch "目錄|目录|Table of Contents|TOC" -and $title -notmatch "📑|📋") {
                $headings += @{
                    Level = $level
                    Title = $title
                    Anchor = Get-Anchor $title
                }
            }
        }
    }
    
    return $headings
}

function Get-Anchor {
    param([string]$Title)
    
    # 清理標題，生成合適的錨點
    $anchor = $Title.ToLower()
    
    # 移除 emoji 和特殊符號
    $anchor = $anchor -replace '[🎯📁🔗📊💰🎯🚨🔍⚙️🧪📈🚀💼📋🏗️]', ''
    
    # 替換空格為連字符
    $anchor = $anchor -replace '\s+', '-'
    
    # 移除其他特殊字符
    $anchor = $anchor -replace '[^\w\-\u4e00-\u9fff]', ''
    
    # 移除多餘的連字符
    $anchor = $anchor -replace '-+', '-'
    $anchor = $anchor.Trim('-')
    
    return $anchor
}

function New-TOC {
    param([array]$Headings)
    
    $toc = @()
    $toc += "## 📑 目錄"
    $toc += ""
    
    foreach ($heading in $Headings) {
        $indent = "  " * ($heading.Level - 2)  # H2 為基準，不縮進
        $link = "[$($heading.Title)](#$($heading.Anchor))"
        $toc += "$indent- $link"
    }
    
    $toc += ""
    $toc += "---"
    $toc += ""
    
    return $toc -join "`n"
}

function Test-TOC {
    param([string]$Content)
    
    return $Content -match "(目錄|目录|Table of Contents|TOC|📑\s*目錄|📋.*目錄)"
}

function Add-TOC-To-File {
    param(
        [string]$FilePath,
        [switch]$DryRun
    )
    
    Write-Host "處理文件: $FilePath" -ForegroundColor Green
    
    try {
        $content = Get-Content -Path $FilePath -Raw -Encoding UTF8
        
        # 檢查是否已有目錄
        if (Test-TOC $content) {
            Write-Host "  ⏭️  已有目錄，跳過" -ForegroundColor Yellow
            return
        }
        
        # 提取標題
        $headings = Get-Headings $content
        
        if ($headings.Count -eq 0) {
            Write-Host "  ⚠️  未找到標題，跳過" -ForegroundColor Yellow
            return
        }
        
        # 生成目錄
        $toc = New-TOC $headings
        
        # 在第一個 H1 標題後插入目錄
        $lines = $content -split "`n"
        $insertIndex = -1
        
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^#\s+.+$') {
                # 找到第一個段落結束處插入目錄
                for ($j = $i + 1; $j -lt $lines.Count; $j++) {
                    if ($lines[$j].Trim() -eq "" -and $j + 1 -lt $lines.Count -and $lines[$j + 1].Trim() -ne "") {
                        $insertIndex = $j + 1
                        break
                    }
                }
                break
            }
        }
        
        if ($insertIndex -eq -1) {
            Write-Host "  ⚠️  無法找到合適的插入位置" -ForegroundColor Yellow
            return
        }
        
        # 插入目錄
        $newLines = @()
        $newLines += $lines[0..($insertIndex - 1)]
        $newLines += $toc -split "`n"
        $newLines += $lines[$insertIndex..($lines.Count - 1)]
        
        $newContent = $newLines -join "`n"
        
        if ($DryRun) {
            Write-Host "  🔍 [DryRun] 將添加 $($headings.Count) 個標題到目錄" -ForegroundColor Cyan
            Write-Host "     插入位置: 行 $insertIndex" -ForegroundColor Cyan
        } else {
            Set-Content -Path $FilePath -Value $newContent -Encoding UTF8 -NoNewline
            Write-Host "  ✅ 已添加目錄 ($($headings.Count) 個標題)" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "  ❌ 處理失敗: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 主程式
Write-Host "🚀 開始為 Markdown 文件添加目錄..." -ForegroundColor Cyan
Write-Host "路徑: $ReportsPath" -ForegroundColor Cyan
Write-Host "模式: $(if ($DryRun) { 'DryRun (僅預覽)' } else { '實際執行' })" -ForegroundColor Cyan
Write-Host ""

$markdownFiles = Get-MarkdownFiles -Path $ReportsPath

Write-Host "找到 $($markdownFiles.Count) 個 Markdown 文件" -ForegroundColor Cyan
Write-Host ""

$processed = 0
$added = 0
$skipped = 0

foreach ($file in $markdownFiles) {
    $processed++
    
    try {
        $beforeSize = (Get-Item $file.FullName).Length
        Add-TOC-To-File -FilePath $file.FullName -DryRun:$DryRun
        
        if (-not $DryRun) {
            $afterSize = (Get-Item $file.FullName).Length
            if ($afterSize -gt $beforeSize) {
                $added++
            } else {
                $skipped++
            }
        }
    } catch {
        Write-Host "處理文件失敗: $($file.FullName)" -ForegroundColor Red
        Write-Host "錯誤: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 處理完成!" -ForegroundColor Green
Write-Host "處理文件數: $processed" -ForegroundColor Green
if (-not $DryRun) {
    Write-Host "添加目錄數: $added" -ForegroundColor Green  
    Write-Host "跳過文件數: $skipped" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📝 使用說明:"
Write-Host "  正常執行: .\add_table_of_contents.ps1"
Write-Host "  預覽模式: .\add_table_of_contents.ps1 -DryRun"
Write-Host "  指定路徑: .\add_table_of_contents.ps1 -ReportsPath 'C:\path\to\reports'"
Write-Host "  非遞歸:  .\add_table_of_contents.ps1 -Recursive:`$false"