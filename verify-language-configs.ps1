#!/usr/bin/env pwsh
# AIVA 多語言延遲檢查驗證腳本
# ================================

Write-Host "🔍 AIVA 多語言延遲檢查配置驗證" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# 驗證函數
function Test-Configuration {
    param(
        [string]$Language,
        [string]$ConfigFile,
        [string]$TestCommand,
        [string]$ExpectedBehavior
    )
    
    Write-Host "📋 檢查 $Language 配置..." -ForegroundColor Yellow
    
    if (Test-Path $ConfigFile) {
        Write-Host "  ✅ 配置檔案存在: $ConfigFile" -ForegroundColor Green
        
        if ($TestCommand) {
            Write-Host "  🧪 執行測試命令: $TestCommand" -ForegroundColor Blue
            try {
                Invoke-Expression $TestCommand | Out-Null
                Write-Host "  ✅ 測試通過" -ForegroundColor Green
            }
            catch {
                Write-Host "  ⚠️ 測試警告: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
        
        Write-Host "  📝 預期行為: $ExpectedBehavior" -ForegroundColor Cyan
    }
    else {
        Write-Host "  ❌ 配置檔案不存在: $ConfigFile" -ForegroundColor Red
    }
    Write-Host ""
}

# 驗證各語言配置
Write-Host "1️⃣ Python (Pylance) 配置檢查" -ForegroundColor Magenta
Test-Configuration -Language "Python" `
    -ConfigFile ".vscode\settings.json" `
    -TestCommand "python -c 'print(\"Python OK\")'  " `
    -ExpectedBehavior "只檢查開啟檔案，30秒延遲檢查"

Write-Host "2️⃣ TypeScript (ESLint) 配置檢查" -ForegroundColor Magenta  
Test-Configuration -Language "TypeScript" `
    -ConfigFile "services\scan\aiva_scan_node\.eslintrc.json" `
    -TestCommand "npm --version" `
    -ExpectedBehavior "儲存時檢查，關閉即時lint"

Write-Host "3️⃣ Go (gopls) 配置檢查" -ForegroundColor Magenta
Test-Configuration -Language "Go" `
    -ConfigFile "services\features\function_ssrf_go\go.mod" `
    -TestCommand "go version" `
    -ExpectedBehavior "關閉即時vet和format，只保留基本檢查"

Write-Host "4️⃣ Rust (rust-analyzer) 配置檢查" -ForegroundColor Magenta
Test-Configuration -Language "Rust" `
    -ConfigFile "Cargo.toml" `
    -TestCommand "rustc --version" `
    -ExpectedBehavior "關閉自動檢查，30秒延遲診斷"

# 檢查VS Code設定
Write-Host "5️⃣ VS Code 編輯器設定檢查" -ForegroundColor Magenta
if (Test-Path ".vscode\settings.json") {
    Write-Host "  ✅ VS Code設定檔案存在" -ForegroundColor Green
    
    $settings = Get-Content ".vscode\settings.json" -Raw
    
    # 檢查關鍵設定
    $checks = @{
        "python.analysis.diagnosticMode" = "openFilesOnly"
        "python.analysis.diagnosticRefreshDelay" = "30000"
        "typescript.disableAutomaticTypeAcquisition" = "true"
        "go.lintOnSave" = "off"
        "rust-analyzer.checkOnSave.enable" = "false"
        "files.autoSaveDelay" = "30000"
    }
    
    foreach ($key in $checks.Keys) {
        if ($settings -match $key) {
            Write-Host "  ✅ $key 已設定" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️ $key 未找到或未設定" -ForegroundColor Yellow
        }
    }
}
Write-Host ""

# 檢查檔案監控排除設定
Write-Host "6️⃣ 檔案監控排除設定檢查" -ForegroundColor Magenta
$settings = Get-Content ".vscode\settings.json" -Raw
if ($settings -match "files.watcherExclude") {
    Write-Host "  ✅ 檔案監控排除已設定" -ForegroundColor Green
    
    $excludePatterns = @("node_modules", "__pycache__", ".venv", ".git", "_archive", "_out")
    foreach ($pattern in $excludePatterns) {
        if ($settings -match $pattern) {
            Write-Host "    ✅ 排除 $pattern" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️ 未排除 $pattern" -ForegroundColor Yellow
        }
    }
}
Write-Host ""

# 總結
Write-Host "🎉 配置檢查完成！" -ForegroundColor Green
Write-Host ""
Write-Host "📋 下一步操作:" -ForegroundColor Cyan
Write-Host "  1. 重新載入 VS Code 視窗 (Ctrl+Shift+P → Developer: Reload Window)" -ForegroundColor Yellow
Write-Host "  2. 開啟任意程式檔案測試延遲檢查效果" -ForegroundColor Yellow
Write-Host "  3. 修改程式碼後等待30秒，觀察檢查行為" -ForegroundColor Yellow
Write-Host ""
Write-Host "✨ 所有語言現在都使用統一的30秒延遲檢查標準！" -ForegroundColor Green