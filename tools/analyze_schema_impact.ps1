#!/usr/bin/env pwsh
# AIVA Schema 變更影響分析工具
# 分析 Schema 變更對多語言生成檔案和相依服務的影響

param(
    [string]$SchemaName,
    [string]$Action = "analyze",  # analyze, preview, apply
    [switch]$ShowDetails,
    [switch]$DryRun
)

# 設定顏色輸出
function Write-StepHeader($message) {
    Write-Host ""
    Write-Host "🔍 $message" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Gray
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "⚠️ $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

function Write-Info($message) {
    Write-Host "ℹ️ $message" -ForegroundColor Blue
}

# 檢查是否在 AIVA 專案根目錄
if (!(Test-Path "pyproject.toml") -or !(Test-Path "services")) {
    Write-Error "請在 AIVA 專案根目錄執行此腳本"
    exit 1
}

Write-StepHeader "AIVA Schema 變更影響分析"

# 分析 Schema 使用情況
function Analyze-SchemaUsage {
    param([string]$schema)
    
    Write-Info "分析 Schema '$schema' 的使用情況..."
    
    $usage = @{
        PythonFiles = @()
        GoFiles = @()
        RustFiles = @()
        TypeScriptFiles = @()
        GeneratedFiles = @()
        ImportStatements = @()
    }
    
    # 搜尋 Python 檔案
    $pythonFiles = Get-ChildItem -Recurse -Filter "*.py" | Where-Object { 
        $_.FullName -notmatch "(venv|__pycache__|\.git)" 
    }
    
    foreach ($file in $pythonFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -and $content -match $schema) {
            $usage.PythonFiles += $file.FullName
            
            # 提取匯入語句
            $lines = $content -split "`n"
            foreach ($line in $lines) {
                if ($line -match "import.*$schema" -or $line -match "from.*$schema") {
                    $usage.ImportStatements += @{
                        File = $file.FullName
                        Statement = $line.Trim()
                    }
                }
            }
        }
    }
    
    # 搜尋 Go 檔案
    $goFiles = Get-ChildItem -Recurse -Filter "*.go" | Where-Object {
        $_.FullName -notmatch "\.git"
    }
    
    foreach ($file in $goFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -and $content -match $schema) {
            $usage.GoFiles += $file.FullName
        }
    }
    
    # 搜尋 Rust 檔案
    $rustFiles = Get-ChildItem -Recurse -Filter "*.rs" | Where-Object {
        $_.FullName -notmatch "\.git"
    }
    
    foreach ($file in $rustFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -and $content -match $schema) {
            $usage.RustFiles += $file.FullName
        }
    }
    
    # 搜尋 TypeScript 檔案
    $tsFiles = Get-ChildItem -Recurse -Filter "*.ts" -Include "*.d.ts" | Where-Object {
        $_.FullName -notmatch "(node_modules|\.git)"
    }
    
    foreach ($file in $tsFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -and $content -match $schema) {
            $usage.TypeScriptFiles += $file.FullName
        }
    }
    
    # 檢查生成檔案
    if (Test-Path "schemas") {
        $generatedFiles = Get-ChildItem "schemas" -File
        foreach ($file in $generatedFiles) {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if ($content -and $content -match $schema) {
                $usage.GeneratedFiles += $file.FullName
            }
        }
    }
    
    return $usage
}

# 分析變更影響
function Analyze-ChangeImpact {
    param([hashtable]$usage)
    
    $impact = @{
        High = @()
        Medium = @()
        Low = @()
        Recommendations = @()
    }
    
    # 高影響：Python 核心檔案
    $criticalPaths = @("services/aiva_common", "services/core", "services/scan")
    foreach ($file in $usage.PythonFiles) {
        foreach ($path in $criticalPaths) {
            if ($file -match $path) {
                $impact.High += "Python 核心服務: $file"
                break
            }
        }
    }
    
    # 中影響：功能模組和生成檔案
    if ($usage.GeneratedFiles.Count -gt 0) {
        $impact.Medium += "多語言生成檔案需要重新生成 ($($usage.GeneratedFiles.Count) 個檔案)"
    }
    
    foreach ($file in $usage.GoFiles) {
        $impact.Medium += "Go 服務: $file"
    }
    
    foreach ($file in $usage.RustFiles) {
        $impact.Medium += "Rust 服務: $file"
    }
    
    # 低影響：TypeScript 和測試檔案
    foreach ($file in $usage.TypeScriptFiles) {
        if ($file -match "test") {
            $impact.Low += "測試檔案: $file"
        } else {
            $impact.Medium += "TypeScript 檔案: $file"
        }
    }
    
    # 生成建議
    if ($usage.GeneratedFiles.Count -gt 0) {
        $impact.Recommendations += "執行 .\tools\generate-official-contracts.ps1 -GenerateAll 重新生成多語言檔案"
    }
    
    if ($usage.PythonFiles.Count -gt 0) {
        $impact.Recommendations += "執行單元測試驗證變更：python -m pytest tests/"
    }
    
    if ($usage.GoFiles.Count -gt 0) {
        $impact.Recommendations += "更新 Go 服務並執行測試：go test ./..."
    }
    
    if ($usage.RustFiles.Count -gt 0) {
        $impact.Recommendations += "更新 Rust 服務並執行測試：cargo test"
    }
    
    return $impact
}

# 預覽變更效果
function Preview-Changes {
    param([string]$schema)
    
    Write-Info "預覽 Schema '$schema' 變更的生成檔案差異..."
    
    # 備份當前生成檔案
    $backupDir = "_schema_preview_backup"
    if (Test-Path $backupDir) {
        Remove-Item $backupDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    if (Test-Path "schemas") {
        Copy-Item "schemas\*" $backupDir -Force
    }
    
    # 重新生成
    Write-Info "重新生成多語言檔案..."
    $result = & ".\tools\generate-official-contracts.ps1" -GenerateAll
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "生成完成，比較差異..."
        
        # 比較差異
        $hasChanges = $false
        Get-ChildItem "schemas" -File | ForEach-Object {
            $newFile = $_.FullName
            $oldFile = Join-Path $backupDir $_.Name
            
            if (Test-Path $oldFile) {
                $diff = Compare-Object (Get-Content $oldFile) (Get-Content $newFile)
                if ($diff) {
                    Write-Warning "檔案有變更: $($_.Name)"
                    if ($ShowDetails) {
                        Write-Host "差異詳情:" -ForegroundColor Yellow
                        $diff | ForEach-Object {
                            if ($_.SideIndicator -eq "<=") {
                                Write-Host "- $($_.InputObject)" -ForegroundColor Red
                            } else {
                                Write-Host "+ $($_.InputObject)" -ForegroundColor Green
                            }
                        }
                        Write-Host ""
                    }
                    $hasChanges = $true
                }
            } else {
                Write-Info "新檔案: $($_.Name)"
                $hasChanges = $true
            }
        }
        
        if (-not $hasChanges) {
            Write-Success "沒有檔案變更"
        }
    } else {
        Write-Error "生成失敗"
    }
    
    # 還原備份（如果是預覽模式）
    if ($DryRun) {
        Write-Info "還原原始檔案..."
        Remove-Item "schemas\*" -Force
        Copy-Item "$backupDir\*" "schemas\" -Force
    }
    
    # 清理備份
    Remove-Item $backupDir -Recurse -Force
}

# 主要邏輯
if ($Action -eq "analyze") {
    if (-not $SchemaName) {
        Write-Error "請指定 Schema 名稱：-SchemaName <名稱>"
        exit 1
    }
    
    $usage = Analyze-SchemaUsage -schema $SchemaName
    $impact = Analyze-ChangeImpact -usage $usage
    
    Write-StepHeader "使用情況分析結果"
    
    Write-Host "📊 發現的使用位置:" -ForegroundColor Cyan
    Write-Host "   Python 檔案: $($usage.PythonFiles.Count) 個"
    Write-Host "   Go 檔案: $($usage.GoFiles.Count) 個"
    Write-Host "   Rust 檔案: $($usage.RustFiles.Count) 個"
    Write-Host "   TypeScript 檔案: $($usage.TypeScriptFiles.Count) 個"
    Write-Host "   生成檔案: $($usage.GeneratedFiles.Count) 個"
    
    if ($ShowDetails) {
        if ($usage.PythonFiles.Count -gt 0) {
            Write-Host "`n🐍 Python 檔案:" -ForegroundColor Yellow
            $usage.PythonFiles | ForEach-Object { Write-Host "   - $_" }
        }
        
        if ($usage.GoFiles.Count -gt 0) {
            Write-Host "`n🔷 Go 檔案:" -ForegroundColor Yellow
            $usage.GoFiles | ForEach-Object { Write-Host "   - $_" }
        }
        
        if ($usage.RustFiles.Count -gt 0) {
            Write-Host "`n🦀 Rust 檔案:" -ForegroundColor Yellow
            $usage.RustFiles | ForEach-Object { Write-Host "   - $_" }
        }
        
        if ($usage.ImportStatements.Count -gt 0) {
            Write-Host "`n📥 匯入語句:" -ForegroundColor Yellow
            $usage.ImportStatements | ForEach-Object { 
                Write-Host "   - $($_.Statement) (in $($_.File))"
            }
        }
    }
    
    Write-StepHeader "變更影響評估"
    
    if ($impact.High.Count -gt 0) {
        Write-Host "🔴 高影響 ($($impact.High.Count) 項):" -ForegroundColor Red
        $impact.High | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    }
    
    if ($impact.Medium.Count -gt 0) {
        Write-Host "`n🟡 中影響 ($($impact.Medium.Count) 項):" -ForegroundColor Yellow
        $impact.Medium | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
    }
    
    if ($impact.Low.Count -gt 0) {
        Write-Host "`n🟢 低影響 ($($impact.Low.Count) 項):" -ForegroundColor Green
        $impact.Low | ForEach-Object { Write-Host "   - $_" -ForegroundColor Green }
    }
    
    if ($impact.Recommendations.Count -gt 0) {
        Write-Host "`n💡 建議操作:" -ForegroundColor Cyan
        $impact.Recommendations | ForEach-Object { Write-Host "   - $_" -ForegroundColor Cyan }
    }
    
} elseif ($Action -eq "preview") {
    Preview-Changes -schema $SchemaName
    
} elseif ($Action -eq "apply") {
    Write-Info "執行完整的 Schema 更新流程..."
    
    # 1. 驗證 Schema 定義
    Write-Info "驗證 Schema 定義..."
    python "tools\schema_manager.py" validate
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Schema 驗證失敗"
        exit 1
    }
    
    # 2. 重新生成多語言檔案
    Write-Info "重新生成多語言檔案..."
    & ".\tools\generate-official-contracts.ps1" -GenerateAll
    if ($LASTEXITCODE -ne 0) {
        Write-Error "多語言檔案生成失敗"
        exit 1
    }
    
    # 3. 執行測試
    Write-Info "執行 Python 測試..."
    python -m pytest tests/ -v
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "部分測試失敗，請檢查"
    }
    
    Write-Success "Schema 更新流程完成"
    
} else {
    Write-Error "不支援的操作: $Action"
    Write-Host "可用操作: analyze, preview, apply"
    exit 1
}

Write-Host "`n🎯 分析完成" -ForegroundColor Green