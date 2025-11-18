# AIVA Project Code Tree Structure Generator (Ultimate Edition - English)
# Features:
# 1. Generate code-only tree structure
# 2. Compare with previous version, mark new(green)/deleted(red)/unchanged(white)  
# 3. Display before/after statistics comparison
# 4. Add English descriptions after filenames
# 5. Output both plain text file and colored terminal display

param(
    [string]$ProjectRoot = "C:\D\fold7\AIVA-git",
    [string]$OutputDir = "C:\D\fold7\AIVA-git\_out\project_structure",
    [string]$PreviousTreeFile = "",  # Previous tree file path (optional)
    [switch]$ShowColorInTerminal,    # Whether to show colors in terminal
    [switch]$AddEnglishComments      # Whether to add English filename descriptions
)

# Set default values
if (-not $PSBoundParameters.ContainsKey('ShowColorInTerminal')) { $ShowColorInTerminal = $true }
if (-not $PSBoundParameters.ContainsKey('AddEnglishComments')) { $AddEnglishComments = $true }

Write-Host "🚀 Starting code tree generation (Ultimate Edition - English)..." -ForegroundColor Cyan

# Directories to exclude
$excludeDirs = @(
    '.git', '__pycache__', '.mypy_cache', '.ruff_cache',
    'node_modules', '.venv', 'venv', 'env', '.env',
    '.pytest_cache', '.tox', 'dist', 'build', 'target',
    'bin', 'obj', '.egg-info', '.eggs', 'htmlcov',
    '.coverage', '.hypothesis', '.idea', '.vscode',
    'site-packages', '_backup', '_out', 'aiva_platform_integrated.egg-info'
)

# Only keep these code file types
$codeExtensions = @(
    '.py', '.go', '.rs', '.ts', '.js', '.jsx', '.tsx',
    '.c', '.cpp', '.h', '.hpp', '.java', '.cs',
    '.sql', '.html', '.css', '.scss', '.vue'
)

# English filename descriptions lookup table
$englishComments = @{
    # Python files
    '__init__.py' = 'Module initialization'
    'models.py' = 'Data models'
    'schemas.py' = 'Schema definitions'
    'config.py' = 'Configuration management'
    'worker.py' = 'Worker executor'
    'app.py' = 'Application entry point'
    'main.py' = 'Main program'
    'server.py' = 'Server'
    'client.py' = 'Client'
    'utils.py' = 'Utility functions'
    'helper.py' = 'Helper functions'
    'manager.py' = 'Manager'
    'handler.py' = 'Handler'
    'controller.py' = 'Controller'
    'service.py' = 'Service layer'
    'api.py' = 'API interface'
    'test.py' = 'Test program'
    'demo.py' = 'Demo program'
    'example.py' = 'Example program'
    'settings.py' = 'Settings file'
    'constants.py' = 'Constants definition'
    'exceptions.py' = 'Exception handling'
    'enums.py' = 'Enum definitions'
    'types.py' = 'Type definitions'
    
    # Specific files
    'bio_neuron_core.py' = 'Bio neuron core'
    'bio_neuron_core_v2.py' = 'Bio neuron core v2'
    'bio_neuron_master.py' = 'Bio neuron master'
    'ai_commander.py' = 'AI commander'
    'ai_controller.py' = 'AI controller'
    'ai_integration_test.py' = 'AI integration test'
    'ai_schemas.py' = 'AI schemas'
    'ai_ui_schemas.py' = 'AI UI schemas'
    'multilang_coordinator.py' = 'Multi-language coordinator'
    'nlg_system.py' = 'Natural language generation system'
    'optimized_core.py' = 'Optimized core'
    'business_schemas.py' = 'Business schemas'
    
    # Feature modules
    'smart_detection_manager.py' = 'Smart detection manager'
    'smart_idor_detector.py' = 'Smart IDOR detector'
    'smart_ssrf_detector.py' = 'Smart SSRF detector'
    'enhanced_worker.py' = 'Enhanced worker'
    'detection_models.py' = 'Detection models'
    'payload_generator.py' = 'Payload generator'
    'result_publisher.py' = 'Result publisher'
    'task_queue.py' = 'Task queue'
    'telemetry.py' = 'Telemetry'
    
    # Engine classes
    'boolean_detection_engine.py' = 'Boolean detection engine'
    'error_detection_engine.py' = 'Error detection engine'
    'time_detection_engine.py' = 'Time detection engine'
    'union_detection_engine.py' = 'Union detection engine'
    'oob_detection_engine.py' = 'Out-of-band detection engine'
    
    # Go files
    'main.go' = 'Main program'
    'config.go' = 'Configuration management'
    'models.go' = 'Data models'
    'schemas.go' = 'Schema definitions'
    'client.go' = 'Client'
    'server.go' = 'Server'
    'logger.go' = 'Logger'
    'message.go' = 'Message handling'
    
    # Rust files
    'main.rs' = 'Main program'
    'lib.rs' = 'Library'
    'mod.rs' = 'Module'
    
    # TypeScript files
    'index.ts' = 'Entry file'
    'main.ts' = 'Main program'
    'app.ts' = 'Application'
    'config.ts' = 'Configuration'
    'types.ts' = 'Type definitions'
    'interfaces.ts' = 'Interface definitions'
    'services.ts' = 'Services'
    'utils.ts' = 'Utilities'
    
    # HTML/CSS
    'index.html' = 'Main page'
    'app.html' = 'Application page'
    'style.css' = 'Styles'
    'main.css' = 'Main styles'
    
    # SQL files  
    'schema.sql' = 'Database schema'
    'init.sql' = 'Database initialization'
    '001_schema.sql' = 'Database schema initialization'
    '002_enhanced_schema.sql' = 'Enhanced database schema'
    
    # Directory names (English)
    'aiva_common' = 'AIVA common module'
    'aiva_core' = 'AIVA core module'
    'aiva_scan' = 'AIVA scan module'
    'aiva_integration' = 'AIVA integration module'
    'aiva_attack' = 'AIVA attack module'
    
    'function_sqli' = 'SQL injection feature'
    'function_xss' = 'XSS feature'
    'function_idor' = 'IDOR feature'
    'function_ssrf' = 'SSRF feature'
    'function_ssrf_go' = 'Go SSRF feature'
    'function_sast_rust' = 'Rust SAST feature'
    'function_sca_go' = 'Go SCA feature'
    'function_cspm_go' = 'Go CSPM feature'
    'function_authn_go' = 'Go authentication feature'
    'function_crypto' = 'Cryptography feature'
    'function_postex' = 'Post-exploitation feature'
    
    'attack_path_analyzer' = 'Attack path analyzer'
    'config_template' = 'Configuration template'
    'middlewares' = 'Middlewares'
    'observability' = 'Observability'
    'perf_feedback' = 'Performance feedback'
    'reception' = 'Reception module'
    'remediation' = 'Remediation recommendations'
    'reporting' = 'Report generation'
    'security' = 'Security module'
    'threat_intel' = 'Threat intelligence'
    
    'core_crawling_engine' = 'Core crawling engine'
    'dynamic_engine' = 'Dynamic engine'
    'info_gatherer' = 'Information gatherer'
    'info_gatherer_rust' = 'Rust information gatherer'
    'aiva_scan_node' = 'AIVA Node.js scan module'
    
    'cmd' = 'Command line tools'
    'internal' = 'Internal modules'
    'pkg' = 'Packages'
    'src' = 'Source code'
    'config' = 'Configuration'
    'logger' = 'Logger'
    'mq' = 'Message queue'
    'schemas' = 'Schema definitions'
    'models' = 'Data models'
    'scanner' = 'Scanner'
    'analyzer' = 'Analyzer'
    'detector' = 'Detector'
    'brute_force' = 'Brute force'
    'token_test' = 'Token testing'
    
    'engines' = 'Detection engines'
    'interfaces' = 'Interface definitions'
    'services' = 'Service modules'
    'utils' = 'Utility functions'
    'examples' = 'Example programs'
    'versions' = 'Version management'
    'alembic' = 'Database migration'
    'api_gateway' = 'API gateway'
    
    'dedup' = 'Deduplication'
    'network' = 'Network module'
    'standards' = 'Standards specification'
    'types' = 'Type definitions'
    'tools' = 'Tool collection'
    'docker' = 'Docker container'
    'initdb' = 'Database initialization'
    'docs' = 'Documentation'
    'ai_engine' = 'AI engine'
    'training' = 'Training module'
    'learning' = 'Learning module'
    'ai_models' = 'AI models'
    'test_models' = 'Test models'
    'data' = 'Data'
    'knowledge' = 'Knowledge base'
    'vectors' = 'Vector storage'
    'scenarios' = 'Test scenarios'
    'logs' = 'Log files'
    'reports' = 'Reports'
    'backup' = 'Backup files'
    'web' = 'Web interface'
    'js' = 'JavaScript files'
    'wasm_modules' = 'WebAssembly modules'
    'scripts' = 'Script files'
    'deployment' = 'Deployment scripts'
    'maintenance' = 'Maintenance scripts'
    'setup' = 'Setup scripts'
    'common' = 'Common scripts'
    'tests' = 'Test files'
    'testing' = 'Testing framework'
    'utilities' = 'Utilities'
    'crosslang' = 'Cross-language support'
    'generated' = 'Auto-generated files'
    'routers' = 'API routers'
    '_archive' = 'Archived files'
    '_cleanup_backup' = 'Cleanup backup'
    'ai_commander' = 'AI commander'
    'ai_engine_backup' = 'AI engine backup'
    'scripts_completed' = 'Completed scripts'
    'historical_versions' = 'Historical versions'
    'ANALYSIS_REPORTS' = 'Analysis reports'
    'IMPLEMENTATION_REPORTS' = 'Implementation reports'  
    'MIGRATION_REPORTS' = 'Migration reports'
    'PROGRESS_REPORTS' = 'Progress reports'
    'ARCHITECTURE' = 'Architecture documentation'
    'DEVELOPMENT' = 'Development documentation'
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
    if ($englishComments.ContainsKey($fullName)) {
        $comment = $englishComments[$fullName]
    }
    # Base name match
    elseif ($englishComments.ContainsKey($baseName)) {
        $comment = $englishComments[$baseName]
    }
    else {
        # Pattern matching
        foreach ($pattern in $englishComments.Keys) {
            if ($fullName -like "*$pattern*" -or $baseName -like "*$pattern*") {
                $comment = $englishComments[$pattern]
                break
            }
        }
        
        # Infer from extension if no match found
        if (-not $comment) {
            $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
            switch ($ext) {
                '.py' { 
                    if ($fullName -match "test") { $comment = "Test program" }
                    elseif ($fullName -match "demo") { $comment = "Demo program" }
                    elseif ($fullName -match "example") { $comment = "Example program" }
                    elseif ($fullName -match "worker") { $comment = "Worker executor" }
                    elseif ($fullName -match "manager") { $comment = "Manager" }
                    elseif ($fullName -match "handler") { $comment = "Handler" }
                    elseif ($fullName -match "controller") { $comment = "Controller" }
                    elseif ($fullName -match "analyzer") { $comment = "Analyzer" }
                    elseif ($fullName -match "detector") { $comment = "Detector" }
                    elseif ($fullName -match "generator") { $comment = "Generator" }
                    elseif ($fullName -match "scanner") { $comment = "Scanner" }
                    elseif ($fullName -match "engine") { $comment = "Engine" }
                    elseif ($fullName -match "executor") { $comment = "Executor" }
                    elseif ($fullName -match "processor") { $comment = "Processor" }
                    elseif ($fullName -match "recorder") { $comment = "Recorder" }
                    elseif ($fullName -match "monitor") { $comment = "Monitor" }
                    elseif ($fullName -match "launcher") { $comment = "Launcher" }
                    elseif ($fullName -match "checker") { $comment = "Checker" }
                    elseif ($fullName -match "validator") { $comment = "Validator" }
                    elseif ($fullName -match "tester") { $comment = "Tester" }
                    elseif ($fullName -match "trainer") { $comment = "Trainer" }
                    else { $comment = "Python module" }
                }
                '.go' { $comment = "Go module" }
                '.rs' { $comment = "Rust module" }
                '.ts' { $comment = "TypeScript module" }
                '.js' { $comment = "JavaScript module" }
                '.html' { $comment = "HTML page" }
                '.css' { $comment = "Stylesheet" }
                '.sql' { $comment = "SQL script" }
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
        [string]$RelativePath = "",
        [int]$Level = 0,
        [int]$MaxLevel = 10,
        [ref]$FileCount,
        [ref]$DirCount,
        [hashtable]$PreviousTree = @{}
    )

    if ($Level -ge $MaxLevel) { return }

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

            $itemRelPath = if ($RelativePath) { "$RelativePath\$($item.Name)" } else { $item.Name }
            
            # Determine status (for future change tracking)
            $status = "unchanged"  # Default: unchanged
            
            # Add English comment with proper alignment
            $englishComment = Get-EnglishComment -FileName $item.Name -IsDirectory $item.PSIsContainer -AlignPosition 60
            $itemNameWithComment = "$($item.Name)$englishComment"
            
            $outputLine = "$Prefix$connector$itemNameWithComment"
            
            # Add marker for text file output (using 4 spaces for unchanged items)
            $markedLine = switch ($status) {
                "added" { "[+] $outputLine" }  # New
                default { "    $outputLine" }  # Unchanged
            }
            
            # Terminal output (with color if enabled)
            if ($ShowColorInTerminal) {
                switch ($status) {
                    "added" { Write-Host $outputLine -ForegroundColor Green }
                    default { Write-Host $outputLine -ForegroundColor White }
                }
            }
            
            # File output
            Write-Output $markedLine

            if ($item.PSIsContainer) {
                Get-CodeTree -Path $item.FullName -Prefix "$Prefix$extension" -RelativePath $itemRelPath -Level ($Level + 1) -MaxLevel $MaxLevel -FileCount $FileCount -DirCount $DirCount -PreviousTree $PreviousTree
            }
        }
    } catch {
        # Ignore inaccessible directories
    }
}

# Collect statistics
Write-Host "📊 Collecting statistics..." -ForegroundColor Yellow

# Count files and lines by language
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

# Read previous version statistics and tree structure
$previousStats = $null
$previousTree = @{}
if ($PreviousTreeFile -and (Test-Path $PreviousTreeFile)) {
    Write-Host "📖 Reading previous version data..." -ForegroundColor Yellow
    # Implementation for reading previous version would go here
}

# Generate tree structure
Write-Host "🌳 Generating tree structure..." -ForegroundColor Yellow
if ($ShowColorInTerminal) {
    Write-Host "   (Terminal will display color output with English descriptions)" -ForegroundColor Gray
}

$fileCountRef = [ref]0
$dirCountRef = [ref]0

$rootName = Split-Path $ProjectRoot -Leaf
$output = @()

# Add header and statistics
$output += "================================================================================"
$output += "AIVA Project Code Tree Structure (Ultimate Edition - English)"
$output += "================================================================================"
$output += "Generated: $(Get-Date -Format 'yyyy年MM月dd日 HH:mm:ss')"
$output += "Project Path: $ProjectRoot"
$output += ""
$output += "📊 Code Statistics"
$output += "────────────────────────────────────────────────────────────────────────────────"

# Display new vs old comparison
if ($previousStats) {
    $fileDiff = $totalFiles - $previousStats.TotalFiles
    $lineDiff = $totalLines - $previousStats.TotalLines
    $fileSymbol = if ($fileDiff -gt 0) { "📈" } elseif ($fileDiff -lt 0) { "📉" } else { "➡️" }
    $lineSymbol = if ($lineDiff -gt 0) { "📈" } elseif ($lineDiff -lt 0) { "📉" } else { "➡️" }
    
    # Format diff values (add + for positive, negative already has -)
    $fileDiffStr = if ($fileDiff -gt 0) { "+$fileDiff" } else { "$fileDiff" }
    $lineDiffStr = if ($lineDiff -gt 0) { "+$lineDiff" } else { "$lineDiff" }
    
    $output += "Total Files: $($previousStats.TotalFiles) → $totalFiles $fileSymbol ($fileDiffStr)"
    $output += "Total Lines of Code: $($previousStats.TotalLines) → $totalLines $lineSymbol ($lineDiffStr)"
} else {
    $output += "Total Files: $totalFiles"
    $output += "Total Lines of Code: $totalLines"
}

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
$output += "• Documentation: .md, .txt"
$output += "• Configuration files: .json, .yaml, .toml, .ini"
$output += "• Scripts: .ps1, .sh, .bat"
$output += ""
$output += "💡 Description"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += """Line"" = A single line of code, ending with newline character (\n)"
$output += """Character"" = A single character (including Chinese, English, symbols)"
$output += """File Count"" = Total number of qualifying code files"
$output += """Lines of Code"" = Total lines in all code files (including blank lines and comments)"
$output += ""
$output += "🎨 Change Marker Explanation"
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
    Write-Host "$rootName # AIVA Security Testing Platform" -ForegroundColor White
}

# Show root directory in output
$output += "$rootName"

# Generate tree structure
Get-CodeTree -Path $ProjectRoot -FileCount $fileCountRef -DirCount $dirCountRef -PreviousTree $previousTree

$output += ""

# Save to file
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputFile = Join-Path $OutputDir "tree_ultimate_english_$timestamp.txt"

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Capture all output
$allOutput = @()
$allOutput += $output

# Save to file
$allOutput | Out-File $outputFile -Encoding UTF8

Write-Host ""
Write-Host "🎉 AIVA project tree structure generated successfully!" -ForegroundColor Green
Write-Host "📁 File location: $outputFile" -ForegroundColor White
Write-Host "📊 Statistics:" -ForegroundColor Cyan
Write-Host "   • Total files: $totalFiles" -ForegroundColor Gray
Write-Host "   • Total lines of code: $totalLines" -ForegroundColor Gray
Write-Host ""
Write-Host "💻 Language distribution:" -ForegroundColor Cyan
foreach ($stat in $langStats) {
    $pct = if ($totalLines -gt 0) { [math]::Round(($stat.TotalLines / $totalLines) * 100, 1) } else { 0 }
    Write-Host "   • $($stat.Extension): $($stat.FileCount) files, $($stat.TotalLines) lines ($pct%)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🔧 Project characteristics:" -ForegroundColor Cyan
Write-Host "   • Multi-language security testing platform" -ForegroundColor Gray
Write-Host "   • Cross-language integration: Python/Go/Rust/TypeScript" -ForegroundColor Gray
Write-Host "   • AI-powered vulnerability detection and analysis" -ForegroundColor Gray
Write-Host "   • Modular architecture with clear separation of concerns" -ForegroundColor Gray

Write-Host ""
Write-Host "✅ AIVA project tree generation completed!" -ForegroundColor Green
Write-Host "📄 Output saved to: $outputFile" -ForegroundColor Cyan