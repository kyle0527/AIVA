#!/usr/bin/env pwsh
# 跨語言編譯檢查腳本
# 使用方式: ./scripts/check_cross_language_compilation.ps1

param(
    [switch]$Verbose,
    [switch]$FailFast
)

Write-Host "🔍 AIVA 跨語言編譯檢查開始..." -ForegroundColor Cyan

$ErrorCount = 0
$WarningCount = 0

function Test-Command {
    param($Command, $WorkingDirectory, $Description)
    
    Write-Host "  測試: $Description" -ForegroundColor Yellow
    
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }
    
    try {
        $result = Invoke-Expression $Command 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "    ✅ 成功" -ForegroundColor Green
            if ($Verbose -and $result) {
                Write-Host "    輸出: $result" -ForegroundColor Gray
            }
            return $true
        } else {
            Write-Host "    ❌ 失敗 (退出碼: $exitCode)" -ForegroundColor Red
            if ($result) {
                Write-Host "    錯誤: $result" -ForegroundColor Red
            }
            $script:ErrorCount++
            return $false
        }
    }
    catch {
        Write-Host "    ❌ 異常: $($_.Exception.Message)" -ForegroundColor Red
        $script:ErrorCount++
        return $false
    }
    finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

# 1. Python 語法檢查
Write-Host "`n🐍 Python 語法檢查" -ForegroundColor Magenta

$pythonFiles = @(
    "services/core/aiva_core/ai_integration_test.py",
    "services/aiva_common/__init__.py",
    "services/features/test_schemas.py"
)

foreach ($file in $pythonFiles) {
    $result = Test-Command "python -m py_compile $file" $null "Python 語法檢查: $file"
    if (!$result -and $FailFast) {
        exit 1
    }
}

# 2. Go 服務編譯檢查
Write-Host "`n🐹 Go 服務編譯檢查" -ForegroundColor Magenta

$goServices = @(
    @{Path = "services/features/function_authn_go"; Name = "認證服務"},
    @{Path = "services/features/function_sca_go"; Name = "軟體組成分析服務"},
    @{Path = "services/features/function_cspm_go"; Name = "雲端安全態勢管理服務"},
    @{Path = "services/features/function_ssrf_go"; Name = "SSRF 檢測服務"}
)

foreach ($service in $goServices) {
    $result = Test-Command "go build ./..." $service.Path "Go 編譯: $($service.Name)"
    if (!$result -and $FailFast) {
        exit 1
    }
}

# 3. TypeScript 編譯檢查
Write-Host "`n🟦 TypeScript 編譯檢查" -ForegroundColor Magenta

$result = Test-Command "npm run build" "services/scan/aiva_scan_node" "TypeScript 編譯"
if (!$result -and $FailFast) {
    exit 1
}

# 4. Rust 編譯檢查
Write-Host "`n🦀 Rust 編譯檢查" -ForegroundColor Magenta

$result = Test-Command "cargo build" "services/scan/info_gatherer_rust" "Rust 編譯"
if (!$result -and $FailFast) {
    exit 1
}

# 5. Schema 一致性檢查
Write-Host "`n📋 Schema 一致性檢查" -ForegroundColor Magenta

# 檢查 Python 中是否有未統一的 TestResult
$pythonTestResults = Select-String -Path "services/**/*.py" -Pattern "class.*TestResult[^a-zA-Z]" -AllMatches
if ($pythonTestResults) {
    Write-Host "  ⚠️  發現可能未統一的 TestResult 類別:" -ForegroundColor Yellow
    foreach ($match in $pythonTestResults) {
        Write-Host "    $($match.Filename):$($match.LineNumber) - $($match.Line.Trim())" -ForegroundColor Yellow
    }
    $WarningCount++
}

# 檢查 Go 中是否使用了已棄用的 models 導入
$goModelsImports = Select-String -Path "services/features/*/internal/**/*.go" -Pattern "models\." -AllMatches
if ($goModelsImports) {
    Write-Host "  ❌ 發現使用已棄用的 models 導入:" -ForegroundColor Red
    foreach ($match in $goModelsImports) {
        Write-Host "    $($match.Filename):$($match.LineNumber) - $($match.Line.Trim())" -ForegroundColor Red
    }
    $ErrorCount++
}

# 總結報告
Write-Host "`n📊 檢查結果總結" -ForegroundColor Cyan
Write-Host "  錯誤: $ErrorCount" -ForegroundColor $(if ($ErrorCount -eq 0) { "Green" } else { "Red" })
Write-Host "  警告: $WarningCount" -ForegroundColor $(if ($WarningCount -eq 0) { "Green" } else { "Yellow" })

if ($ErrorCount -eq 0) {
    Write-Host "`n🎉 所有檢查通過！AIVA 跨語言編譯狀態良好。" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n💥 發現 $ErrorCount 個錯誤，請修復後重試。" -ForegroundColor Red
    exit 1
}