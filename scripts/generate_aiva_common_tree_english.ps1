# AIVA Common Module Tree Structure Generator (Ultimate Edition)
# English version to avoid encoding issues

param(
    [string]$ProjectRoot = "C:\D\fold7\AIVA-git\services\aiva_common",
    [string]$OutputDir = "C:\D\fold7\AIVA-git\_out\project_structure",
    [switch]$ShowColorInTerminal,
    [switch]$AddEnglishComments
)

# Set default values
if (-not $PSBoundParameters.ContainsKey('ShowColorInTerminal')) { $ShowColorInTerminal = $true }
if (-not $PSBoundParameters.ContainsKey('AddEnglishComments')) { $AddEnglishComments = $true }

Write-Host "🚀 Starting AIVA Common module tree generation (Ultimate Edition)..." -ForegroundColor Cyan

# Directories to exclude
$excludeDirs = @(
    '.git', '__pycache__', '.mypy_cache', '.ruff_cache',
    'node_modules', '.venv', 'venv', 'env', '.env',
    '.pytest_cache', '.tox', 'dist', 'build', 'target',
    'bin', 'obj', '.egg-info', '.eggs', 'htmlcov',
    '.coverage', '.hypothesis', '.idea', '.vscode',
    'site-packages', '_backup', '_out'
)

# Code file extensions to include
$codeExtensions = @(
    '.py', '.go', '.rs', '.ts', '.js', '.jsx', '.tsx',
    '.c', '.cpp', '.h', '.hpp', '.java', '.cs',
    '.sql', '.html', '.css', '.scss', '.vue',
    '.yaml', '.yml', '.md'
)

# File descriptions (English to avoid encoding issues)
$fileDescriptions = @{
    '__init__.py' = 'Module initialization'
    'models.py' = 'Data models'
    'schemas.py' = 'Schema definitions'
    'config.py' = 'Configuration management'
    'mq.py' = 'Message queue'
    'utils.py' = 'Utility functions'
    'enums.py' = 'Enum definitions'
    'base.py' = 'Base classes'
    'ai.py' = 'AI-related definitions'
    'assets.py' = 'Asset management'
    'findings.py' = 'Finding results'
    'messaging.py' = 'Message protocols'
    'tasks.py' = 'Task definitions'
    'telemetry.py' = 'Telemetry data'
    'references.py' = 'Reference standards'
    'system.py' = 'System-related'
    'risk.py' = 'Risk assessment'
    'enhanced.py' = 'Enhanced features'
    'languages.py' = 'Programming languages'
    'common.py' = 'Common definitions'
    'modules.py' = 'Module definitions'
    'security.py' = 'Security-related'
    'ids.py' = 'ID generator'
    'logging.py' = 'Logging utilities'
    'dedupe.py' = 'Deduplication'
    'backoff.py' = 'Backoff strategy'
    'ratelimit.py' = 'Rate limiting'
    'unified_config.py' = 'Unified configuration'
    'core_schema_sot.yaml' = 'Schema source of truth'
    'CODE_QUALITY_REPORT.md' = 'Code quality report'
    'py.typed' = 'Type hint support'
    'module_connectivity_tester.py' = 'Module connectivity tester'
    'schema_codegen_tool.py' = 'Schema code generator'
    'schema_validator.py' = 'Schema validator'
}

function Test-ShouldIncludeFile {
    param([string]$FileName)
    
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    
    if ([string]::IsNullOrEmpty($ext)) {
        return $false
    }
    
    return $codeExtensions -contains $ext
}

function Get-EnglishComment {
    param([string]$FileName, [bool]$IsDirectory = $false, [int]$AlignPosition = 60)
    
    if (-not $AddEnglishComments) {
        return ""
    }
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $fullName = $FileName
    $comment = ""
    
    # Exact match
    if ($fileDescriptions.ContainsKey($fullName)) {
        $comment = $fileDescriptions[$fullName]
    }
    # Base name match
    elseif ($fileDescriptions.ContainsKey($baseName)) {
        $comment = $fileDescriptions[$baseName]
    }
    else {
        # Pattern matching
        foreach ($pattern in $fileDescriptions.Keys) {
            if ($fullName -like "*$pattern*" -or $baseName -like "*$pattern*") {
                $comment = $fileDescriptions[$pattern]
                break
            }
        }
        
        # Infer from extension if no match found
        if (-not $comment) {
            $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
            switch ($ext) {
                '.py' { 
                    if ($fullName -match "test") { $comment = "Test module" }
                    elseif ($fullName -match "tool") { $comment = "Tool script" }
                    elseif ($fullName -match "validator") { $comment = "Validator" }
                    elseif ($fullName -match "tester") { $comment = "Tester" }
                    elseif ($fullName -match "generator") { $comment = "Generator" }
                    else { $comment = "Python module" }
                }
                '.yaml' { $comment = "YAML configuration" }
                '.yml' { $comment = "YAML configuration" }
                '.md' { $comment = "Documentation" }
                default { $comment = "" }
            }
        }
    }
    
    # Format the comment with proper spacing
    if ($comment) {
        $currentLength = $FileName.Length
        $spacesNeeded = [Math]::Max(1, $AlignPosition - $currentLength)
        $spaces = " " * $spacesNeeded
        return "$spaces# $comment"
    }
    
    return ""
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

    $treeOutput = @()

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

            # Add English comment with proper alignment
            $englishComment = Get-EnglishComment -FileName $item.Name -IsDirectory $item.PSIsContainer -AlignPosition 60
            $itemNameWithComment = "$($item.Name)$englishComment"
            
            $outputLine = "$Prefix$connector$itemNameWithComment"
            
            # Add marker for text file output (using 4 spaces for unchanged items)
            $markedLine = "    $outputLine"
            
            # Terminal output (with color if enabled)
            if ($ShowColorInTerminal) {
                Write-Host $outputLine -ForegroundColor White
            }
            
            # Add to tree output
            $treeOutput += $markedLine

            if ($item.PSIsContainer) {
                $subTree = Get-CodeTree -Path $item.FullName -Prefix "$Prefix$extension" -Level ($Level + 1) -MaxLevel $MaxLevel -FileCount $FileCount -DirCount $DirCount
                $treeOutput += $subTree
            }
        }
    } catch {
        # Ignore inaccessible directories
    }

    return $treeOutput
}

# Count code files
Write-Host "Analyzing code files..." -ForegroundColor Yellow

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
                # Ignore unreadable files
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

# Calculate totals
$totalFiles = ($langStats | Measure-Object -Property FileCount -Sum).Sum
$totalLines = ($langStats | Measure-Object -Property TotalLines -Sum).Sum

# Generate tree structure
Write-Host "Generating tree structure..." -ForegroundColor Yellow

$fileCountRef = [ref]0
$dirCountRef = [ref]0

$rootName = Split-Path $ProjectRoot -Leaf
$output = @()

# Add header and statistics (matching original format)
$output += "================================================================================"
$output += "AIVA Common Module Tree Structure (Ultimate Edition - English)"
$output += "================================================================================"
$output += "Generated: $(Get-Date -Format 'yyyy年MM月dd日 HH:mm:ss')"
$output += "Module Path: $ProjectRoot"
$output += ""
$output += "📊 Code Statistics"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "Total Files: $totalFiles"
$output += "Total Lines of Code: $totalLines"
$output += ""
$output += "💻 Language Distribution:"

foreach ($stat in $langStats) {
    $pct = if ($totalLines -gt 0) { [math]::Round(($stat.TotalLines / $totalLines) * 100, 1) } else { 0 }
    $output += "   • $($stat.Extension): $($stat.FileCount) files, $($stat.TotalLines) lines ($pct%)"
}

$output += ""
$output += "🔧 Excluded Items"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "Excluded:"
$output += "• Virtual environments: .venv, venv, env"
$output += "• Caches: __pycache__, .mypy_cache, .ruff_cache"
$output += "• Build artifacts: dist, build, target, bin, obj"
$output += "• Documentation: .md, .txt (except selected ones)"
$output += "• Configuration files: .json, .yaml, .toml, .ini (except selected ones)"
$output += "• Scripts: .ps1, .sh, .bat"
$output += ""
$output += "💡 Description"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += """Line"" = A single line of code, ending with newline character (\n)"
$output += """Character"" = A single character (including Chinese, English, symbols)"
$output += """File Count"" = Total number of qualifying code files"
$output += """Lines of Code"" = Total lines in all code files (including blank lines and comments)"
$output += ""
$output += "🎨 Marker Explanation"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "[+] = 🟢 New files or directories (shown in green in terminal)"
$output += "[-] = 🔴 Deleted files or directories (shown in red in terminal)"
$output += "    = ⚪ Unchanged (shown in white in terminal)"
$output += ""
$output += "🌏 English Filename Descriptions"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "Each filename is followed by an English description, format: filename # description"
$output += "• Intelligently inferred based on filename and directory structure"
$output += "• Covers Python, Go, Rust, TypeScript and other languages"
$output += "• Includes AIVA project-specific module and feature descriptions"
$output += ""
$output += "Note: Text file output contains [+]/[-] markers and English descriptions"
$output += "Terminal execution displays corresponding colors without [+]/[-] markers"
$output += "In next version update, [-] items will be removed, [+] items become unchanged (spaces)"
$output += ""
$output += "================================================================================"
$output += "Code Structure Tree (with English descriptions)"
$output += "================================================================================"
$output += ""

# Display title in terminal
if ($ShowColorInTerminal) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Code Structure Tree (Color Output + English Descriptions)" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "$rootName # AIVA Common Foundation Module" -ForegroundColor White
}

# Show root directory in output
$output += "$rootName"

# Generate tree structure
$treeStructure = Get-CodeTree -Path $ProjectRoot -FileCount $fileCountRef -DirCount $dirCountRef
$output += $treeStructure

$output += ""

# Save to file
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputFile = Join-Path $OutputDir "aiva_common_tree_english_$timestamp.txt"

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Save to file
$output | Out-File $outputFile -Encoding UTF8

Write-Host ""
Write-Host "🎉 AIVA Common module tree structure generated successfully!" -ForegroundColor Green
Write-Host "📁 File location: $outputFile" -ForegroundColor White
Write-Host "📊 Code statistics:" -ForegroundColor Cyan
Write-Host "   • Total files: $totalFiles" -ForegroundColor Gray
Write-Host "   • Total lines of code: $totalLines" -ForegroundColor Gray
Write-Host ""
Write-Host "💻 Language distribution:" -ForegroundColor Cyan
foreach ($stat in $langStats) {
    $pct = if ($totalLines -gt 0) { [math]::Round(($stat.TotalLines / $totalLines) * 100, 1) } else { 0 }
    Write-Host "   • $($stat.Extension): $($stat.FileCount) files, $($stat.TotalLines) lines ($pct%)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🔧 Module characteristics:" -ForegroundColor Cyan
Write-Host "   • Infrastructure module: Provides shared components for all AIVA services" -ForegroundColor Gray
Write-Host "   • Cross-language support: Unified data definitions and communication protocols" -ForegroundColor Gray
Write-Host "   • Official standards: Implements CVSS, SARIF and other international standards" -ForegroundColor Gray
Write-Host "   • Highly modular: Clear hierarchical structure and separation of concerns" -ForegroundColor Gray

Write-Host ""
Write-Host "✅ AIVA Common module tree generation completed!" -ForegroundColor Green
Write-Host "📄 Output saved to: $outputFile" -ForegroundColor Cyan