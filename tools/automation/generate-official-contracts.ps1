#!/usr/bin/env pwsh
# AIVA Official Schema Generation Tools
# 使用官方工具替代自製的 aiva-contracts-tooling

param(
    [switch]$GenerateAll,
    [switch]$GenerateJsonSchema,
    [switch]$GenerateTypeScript,
    [switch]$GenerateEnums,
    [switch]$GenerateGo,
    [switch]$GenerateRust,
    [switch]$ListModels,
    [string]$OutputDir = ".\schemas"
)

# 設置顏色輸出函數
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

function Write-Warning($message) {
    Write-Host "⚠️ $message" -ForegroundColor Yellow
}

# 確保輸出目錄存在
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force
    Write-Success "創建輸出目錄: $OutputDir"
}

# 檢查官方工具安裝狀態
Write-StepHeader "檢查官方工具安裝狀態"
try {
    $pydanticVersion = python -c "import pydantic; print(pydantic.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Pydantic: $pydanticVersion (官方 JSON Schema 生成器)"
    } else {
        Write-Error "Pydantic 未安裝"
    }
    
    $datamodelVersion = datamodel-codegen --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "datamodel-code-generator: $datamodelVersion (官方多語言生成器)"
    } else {
        Write-Warning "datamodel-code-generator 未安裝"
    }
    
    # 檢查 FastAPI (用於 OpenAPI 生成)
    $fastapiVersion = python -c "import fastapi; print(fastapi.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "FastAPI: $fastapiVersion (官方 OpenAPI 生成器)"
    } else {
        Write-Warning "FastAPI 未安裝"
    }
} catch {
    Write-Error "工具檢查失敗: $_"
}

# 列出所有模型
if ($ListModels) {
    Write-StepHeader "列出所有 Pydantic 模型"
    try {
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from services.aiva_common.schemas import *
import inspect
from pydantic import BaseModel

# 獲取所有 BaseModel 子類
models = []
for name in dir():
    obj = globals()[name]
    if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
        models.append(name)

print(f'發現 {len(models)} 個 Pydantic 模型:')
for model in sorted(models):
    print(f'  📦 {model}')
"
        Write-Success "模型列表生成完成"
    } catch {
        Write-Error "無法列出模型: $_"
    }
}

# 生成 JSON Schema (使用 Pydantic 官方 API)
if ($GenerateJsonSchema -or $GenerateAll) {
    Write-StepHeader "使用 Pydantic 官方 API 生成 JSON Schema"
    try {
        python tools/generate_official_schemas.py
        Write-Success "JSON Schema 已生成 (Pydantic 官方 API)"
    } catch {
        Write-Error "JSON Schema 生成失敗: $_"
        exit 1
    }
}

# 生成 TypeScript 定義
if ($GenerateTypeScript -or $GenerateAll) {
    Write-StepHeader "生成 TypeScript 介面定義"
    try {
        python tools/generate_typescript_interfaces.py
        Write-Success "TypeScript 介面已生成"
    } catch {
        Write-Error "TypeScript 生成失敗: $_"
        exit 1
    }
}

# 生成 TypeScript 枚舉
if ($GenerateEnums -or $GenerateAll) {
    Write-StepHeader "生成 TypeScript 枚舉"
    # 枚舉生成已包含在 generate_official_schemas.py 中
    if (Test-Path "$OutputDir/enums.ts") {
        Write-Success "TypeScript 枚舉已存在: $OutputDir/enums.ts"
    } else {
        python tools/generate_official_schemas.py
    }
}

# 生成 Go 結構體 (使用 datamodel-code-generator)
if ($GenerateGo -or $GenerateAll) {
    Write-StepHeader "生成 Go 結構體 (實驗性)"
    $jsonPath = Join-Path $OutputDir "aiva_schemas.json"
    $goPath = Join-Path $OutputDir "aiva_schemas.go"
    
    if (Test-Path $jsonPath) {
        try {
            # 使用 quicktype 生成 Go (如果可用)
            $quicktypeAvailable = where.exe quicktype 2>$null
            if ($quicktypeAvailable) {
                quicktype $jsonPath -o $goPath --lang go --top-level AIVASchemas
                Write-Success "Go 結構體已生成: $goPath (使用 quicktype)"
            } else {
                Write-Warning "Go 生成跳過 - quicktype 未安裝"
                Write-Warning "建議: npm install -g quicktype"
            }
        } catch {
            Write-Warning "Go 生成失敗: $_"
        }
    } else {
        Write-Error "JSON Schema 不存在，請先生成 JSON Schema"
    }
}

# 生成 Rust 結構體
if ($GenerateRust -or $GenerateAll) {
    Write-StepHeader "生成 Rust 結構體 (實驗性)"
    $jsonPath = Join-Path $OutputDir "aiva_schemas.json"
    $rustPath = Join-Path $OutputDir "aiva_schemas.rs"
    
    if (Test-Path $jsonPath) {
        try {
            $quicktypeAvailable = where.exe quicktype 2>$null
            if ($quicktypeAvailable) {
                quicktype $jsonPath -o $rustPath --lang rust --top-level AIVASchemas
                Write-Success "Rust 結構體已生成: $rustPath (使用 quicktype)"
            } else {
                Write-Warning "Rust 生成跳過 - quicktype 未安裝"
                Write-Warning "建議: npm install -g quicktype"
            }
        } catch {
            Write-Warning "Rust 生成失敗: $_"
        }
    } else {
        Write-Error "JSON Schema 不存在，請先生成 JSON Schema"
    }
}

# 顯示使用說明
if (!$ListModels -and !$GenerateAll -and !$GenerateJsonSchema -and !$GenerateTypeScript -and !$GenerateEnums -and !$GenerateGo -and !$GenerateRust) {
    Write-Host ""
    Write-Host "🔧 AIVA 官方 Schema 生成工具" -ForegroundColor Cyan
    Write-Host "使用 Pydantic 官方 API + 標準工具" -ForegroundColor Gray
    Write-Host ""
    Write-Host "用法:" -ForegroundColor Yellow
    Write-Host "  .\generate-official-contracts.ps1 -GenerateAll         # 生成所有格式"
    Write-Host "  .\generate-official-contracts.ps1 -GenerateJsonSchema  # 只生成 JSON Schema"
    Write-Host "  .\generate-official-contracts.ps1 -GenerateTypeScript  # 只生成 TypeScript"
    Write-Host "  .\generate-official-contracts.ps1 -GenerateEnums       # 只生成枚舉"
    Write-Host "  .\generate-official-contracts.ps1 -GenerateGo          # 生成 Go 結構體"
    Write-Host "  .\generate-official-contracts.ps1 -GenerateRust        # 生成 Rust 結構體"
    Write-Host "  .\generate-official-contracts.ps1 -ListModels          # 列出所有模型"
    Write-Host ""
    Write-Host "官方工具優勢:" -ForegroundColor Green
    Write-Host "  ✅ 使用 Pydantic 官方 JSON Schema API"
    Write-Host "  ✅ 標準化輸出格式，避免衝突"
    Write-Host "  ✅ 支援多語言生成 (TypeScript, Go, Rust)"
    Write-Host "  ✅ 長期維護保證"
    Write-Host ""
}

Write-Host ""
Write-Host "🎉 官方工具執行完成!" -ForegroundColor Green