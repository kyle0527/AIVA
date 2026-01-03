#!/usr/bin/env pwsh
# test_ast_tools.ps1 - 測試所有 AST 分析工具

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "AST 分析工具測試腳本" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$toolsDir = $PSScriptRoot
$projectRoot = Split-Path (Split-Path (Split-Path $toolsDir -Parent) -Parent) -Parent
$outputDir = Join-Path $projectRoot "docs\diagrams"

# 創建輸出目錄
Write-Host "📁 創建輸出目錄..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$outputDir\python" | Out-Null
New-Item -ItemType Directory -Force -Path "$outputDir\go" | Out-Null
New-Item -ItemType Directory -Force -Path "$outputDir\typescript" | Out-Null
New-Item -ItemType Directory -Force -Path "$outputDir\rust" | Out-Null
Write-Host "✅ 輸出目錄已創建" -ForegroundColor Green
Write-Host ""

# 測試 Python 工具
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試 1: Python AST 分析工具 (py2mermaid.py)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$pythonInput = Join-Path $projectRoot "services\scan\engines"
$pythonOutput = Join-Path $outputDir "python"

Write-Host "輸入路徑: $pythonInput" -ForegroundColor Gray
Write-Host "輸出路徑: $pythonOutput" -ForegroundColor Gray
Write-Host ""

try {
    python "$toolsDir\py2mermaid.py" `
        --input "$pythonInput" `
        --output "$pythonOutput" `
        --max-files 50 `
        --direction TB
    
    $pythonFiles = Get-ChildItem -Path $pythonOutput -Filter "*.mmd" | Measure-Object
    Write-Host "✅ Python 工具測試完成，生成 $($pythonFiles.Count) 個流程圖" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 工具測試失敗: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host ""

# 測試 Go 工具
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試 2: Go AST 分析工具 (go2mermaid.go)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$goInput = Join-Path $projectRoot "services\scan\engines"
$goOutput = Join-Path $outputDir "go"

Write-Host "輸入路徑: $goInput" -ForegroundColor Gray
Write-Host "輸出路徑: $goOutput" -ForegroundColor Gray
Write-Host ""

try {
    Push-Location $toolsDir
    go run go2mermaid.go `
        -i "$goInput" `
        -o "$goOutput" `
        -m 50 `
        -d TB
    Pop-Location
    
    $goFiles = Get-ChildItem -Path $goOutput -Filter "*.mmd" | Measure-Object
    Write-Host "✅ Go 工具測試完成，生成 $($goFiles.Count) 個流程圖" -ForegroundColor Green
} catch {
    Write-Host "❌ Go 工具測試失敗: $_" -ForegroundColor Red
    Pop-Location
}

Write-Host ""
Write-Host ""

# 測試 TypeScript 工具
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試 3: TypeScript AST 分析工具 (ts2mermaid.ts)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$tsInput = Join-Path $projectRoot "services\scan\engines\typescript_engine"
$tsOutput = Join-Path $outputDir "typescript"

Write-Host "輸入路徑: $tsInput" -ForegroundColor Gray
Write-Host "輸出路徑: $tsOutput" -ForegroundColor Gray
Write-Host ""

# 檢查 TypeScript 依賴
if (-not (Test-Path "$toolsDir\node_modules")) {
    Write-Host "📦 安裝 TypeScript 依賴..." -ForegroundColor Yellow
    Push-Location $toolsDir
    Copy-Item "ts2mermaid-package.json" "package.json" -Force
    npm install
    Pop-Location
}

try {
    Push-Location $toolsDir
    npx ts-node ts2mermaid.ts `
        --input "$tsInput" `
        --output "$tsOutput" `
        --max-files 50 `
        --direction TB
    Pop-Location
    
    $tsFiles = Get-ChildItem -Path $tsOutput -Filter "*.mmd" | Measure-Object
    Write-Host "✅ TypeScript 工具測試完成，生成 $($tsFiles.Count) 個流程圖" -ForegroundColor Green
} catch {
    Write-Host "❌ TypeScript 工具測試失敗: $_" -ForegroundColor Red
    Pop-Location
}

Write-Host ""
Write-Host ""

# 測試 Rust 工具
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試 4: Rust AST 分析工具 (rs2mermaid.rs)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$rustInput = Join-Path $projectRoot "services\scan\engines\rust_engine"
$rustOutput = Join-Path $outputDir "rust"

Write-Host "輸入路徑: $rustInput" -ForegroundColor Gray
Write-Host "輸出路徑: $rustOutput" -ForegroundColor Gray
Write-Host ""

# 設置 Cargo 項目
if (-not (Test-Path "$toolsDir\Cargo.toml")) {
    Write-Host "📦 設置 Rust 項目..." -ForegroundColor Yellow
    Copy-Item "$toolsDir\rs2mermaid-Cargo.toml" "$toolsDir\Cargo.toml" -Force
}

try {
    Push-Location $toolsDir
    cargo run --bin rs2mermaid -- `
        --input "$rustInput" `
        --output "$rustOutput" `
        --max-files 50 `
        --direction TB
    Pop-Location
    
    $rustFiles = Get-ChildItem -Path $rustOutput -Filter "*.mmd" | Measure-Object
    Write-Host "✅ Rust 工具測試完成，生成 $($rustFiles.Count) 個流程圖" -ForegroundColor Green
} catch {
    Write-Host "❌ Rust 工具測試失敗: $_" -ForegroundColor Red
    Pop-Location
}

Write-Host ""
Write-Host ""

# 總結
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試總結" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

$totalFiles = 0
$languages = @("python", "go", "typescript", "rust")

foreach ($lang in $languages) {
    $langDir = Join-Path $outputDir $lang
    if (Test-Path $langDir) {
        $count = (Get-ChildItem -Path $langDir -Filter "*.mmd" | Measure-Object).Count
        Write-Host "$lang : $count 個流程圖" -ForegroundColor White
        $totalFiles += $count
    }
}

Write-Host ""
Write-Host "總計生成: $totalFiles 個 Mermaid 流程圖" -ForegroundColor Green
Write-Host "輸出目錄: $outputDir" -ForegroundColor Green

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "測試完成！" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# 詢問是否要打開輸出目錄
$response = Read-Host "是否要打開輸出目錄？(Y/N)"
if ($response -eq "Y" -or $response -eq "y") {
    Invoke-Item $outputDir
}
