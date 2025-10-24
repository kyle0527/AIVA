#!/usr/bin/env pwsh
# AIVA 工具集自動化腳本
# 用於生成 JSON Schema, TypeScript 定義和枚舉

param(
    [switch]$ListModels,
    [switch]$GenerateAll,
    [switch]$GenerateJsonSchema,
    [switch]$GenerateTypeScript,
    [switch]$GenerateEnums,
    [string]$OutputDir = ".\schemas"
)

# 設置環境變數
$env:PYTHONPATH = "C:\F\AIVA\services"

# 確保輸出目錄存在
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force
    Write-Host "✅ 創建輸出目錄: $OutputDir" -ForegroundColor Green
}

function Write-StepHeader($message) {
    Write-Host ""
    Write-Host "🔧 $message" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Gray
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

if ($ListModels) {
    Write-StepHeader "列出所有可用的模型"
    try {
        aiva-contracts list-models
        Write-Success "模型列表生成完成"
    }
    catch {
        Write-Error "無法列出模型: $_"
        exit 1
    }
}

if ($GenerateJsonSchema -or $GenerateAll) {
    Write-StepHeader "生成 JSON Schema"
    $jsonPath = Join-Path $OutputDir "aiva_schemas.json"
    try {
        aiva-contracts export-jsonschema --out $jsonPath
        Write-Success "JSON Schema 已生成: $jsonPath"
        
        # 顯示檔案大小
        $size = (Get-Item $jsonPath).Length
        Write-Host "📊 檔案大小: $([math]::Round($size / 1KB, 2)) KB" -ForegroundColor Yellow
    }
    catch {
        Write-Error "JSON Schema 生成失敗: $_"
        exit 1
    }
}

if ($GenerateTypeScript -or $GenerateAll) {
    Write-StepHeader "生成 TypeScript 定義"
    $jsonPath = Join-Path $OutputDir "aiva_schemas.json"
    $tsPath = Join-Path $OutputDir "aiva_schemas.d.ts"
    
    if (!(Test-Path $jsonPath)) {
        Write-Host "⚠️  JSON Schema 不存在，先生成 JSON Schema..." -ForegroundColor Yellow
        try {
            aiva-contracts export-jsonschema --out $jsonPath
            Write-Success "JSON Schema 已生成"
        }
        catch {
            Write-Error "JSON Schema 生成失敗: $_"
            exit 1
        }
    }
    
    try {
        aiva-contracts gen-ts --json $jsonPath --out $tsPath
        Write-Success "TypeScript 定義已生成: $tsPath"
        
        # 顯示檔案大小
        $size = (Get-Item $tsPath).Length
        Write-Host "📊 檔案大小: $([math]::Round($size / 1KB, 2)) KB" -ForegroundColor Yellow
    }
    catch {
        Write-Error "TypeScript 定義生成失敗: $_"
        exit 1
    }
}

if ($GenerateEnums -or $GenerateAll) {
    Write-StepHeader "生成 TypeScript 枚舉"
    $enumsPath = Join-Path $OutputDir "enums.ts"
    try {
        $scriptPath = "C:\F\AIVA\tools\aiva-enums-plugin\aiva-enums-plugin\scripts\gen_ts_enums.py"
        python $scriptPath --out $enumsPath
        Write-Success "TypeScript 枚舉已生成: $enumsPath"
        
        # 顯示檔案大小
        $size = (Get-Item $enumsPath).Length
        Write-Host "📊 檔案大小: $([math]::Round($size / 1KB, 2)) KB" -ForegroundColor Yellow
    }
    catch {
        Write-Error "TypeScript 枚舉生成失敗: $_"
        exit 1
    }
}

if ($GenerateAll) {
    Write-StepHeader "生成摘要"
    Write-Host ""
    Write-Host "🎉 所有檔案生成完成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "生成的檔案:" -ForegroundColor Cyan
    
    $files = @("aiva_schemas.json", "aiva_schemas.d.ts", "enums.ts")
    foreach ($file in $files) {
        $path = Join-Path $OutputDir $file
        if (Test-Path $path) {
            $size = (Get-Item $path).Length
            Write-Host "  📄 $file - $([math]::Round($size / 1KB, 2)) KB" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "💡 使用建議:" -ForegroundColor Yellow
    Write-Host "  1. 將生成的檔案納入版本控制" -ForegroundColor Gray
    Write-Host "  2. 在前端專案中引用這些型別定義" -ForegroundColor Gray
    Write-Host "  3. 當 schema 或 enum 變更時重新執行此腳本" -ForegroundColor Gray
}

if (!$ListModels -and !$GenerateAll -and !$GenerateJsonSchema -and !$GenerateTypeScript -and !$GenerateEnums) {
    Write-Host ""
    Write-Host "🚀 AIVA 工具集自動化腳本" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使用方式:" -ForegroundColor White
    Write-Host "  .\generate-contracts.ps1 -ListModels          # 列出所有模型"
    Write-Host "  .\generate-contracts.ps1 -GenerateAll         # 生成所有檔案"
    Write-Host "  .\generate-contracts.ps1 -GenerateJsonSchema  # 只生成 JSON Schema"
    Write-Host "  .\generate-contracts.ps1 -GenerateTypeScript  # 只生成 TypeScript 定義"
    Write-Host "  .\generate-contracts.ps1 -GenerateEnums       # 只生成 TypeScript 枚舉"
    Write-Host ""
    Write-Host "參數:" -ForegroundColor White
    Write-Host "  -OutputDir <路徑>   # 指定輸出目錄 (預設: .\schemas)"
    Write-Host ""
    Write-Host "範例:" -ForegroundColor Green
    Write-Host "  .\generate-contracts.ps1 -GenerateAll -OutputDir .\frontend\types"
}