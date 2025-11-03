# AIVA 重複定義問題修復執行腳本
# 符合 AIVA v5.0 跨語言統一架構標準

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("1", "2", "3", "4")]
    [string]$Phase,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verify,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# 顯示使用說明
function Show-Help {
    Write-Host "🔧 AIVA 重複定義問題修復工具" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使用方式:" -ForegroundColor Yellow
    Write-Host "  .\fix-duplicates.ps1 -Phase 1           # 執行階段一修復"
    Write-Host "  .\fix-duplicates.ps1 -Verify           # 驗證修復結果"
    Write-Host "  .\fix-duplicates.ps1 -Phase 1 -DryRun  # 試運行模式"
    Write-Host "  .\fix-duplicates.ps1 -Help             # 顯示此說明"
    Write-Host ""
    Write-Host "階段說明:" -ForegroundColor Yellow
    Write-Host "  階段一: 枚舉重複定義修復 + 核心模型統一"
    Write-Host "  階段二: 跨語言合約統一 (開發中)"
    Write-Host "  階段三: 功能模組整合 (開發中)"
    Write-Host "  階段四: 完整驗證與文檔更新 (開發中)"
    Write-Host ""
    Write-Host "參數說明:" -ForegroundColor Yellow
    Write-Host "  -Phase    指定執行階段 (1-4)"
    Write-Host "  -Verify   驗證修復結果"
    Write-Host "  -DryRun   試運行模式（不實際修改檔案）"
    Write-Host "  -Verbose  詳細輸出模式"
    Write-Host "  -Help     顯示此說明"
}

# 檢查 Python 環境
function Test-PythonEnvironment {
    Write-Host "🐍 檢查 Python 環境..." -ForegroundColor Yellow
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Python 未安裝或無法訪問" -ForegroundColor Red
            return $false
        }
        
        Write-Host "✅ Python 版本: $pythonVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Python 環境檢查失敗: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 檢查 AIVA 依賴
function Test-AIVADependencies {
    Write-Host "📦 檢查 AIVA 依賴..." -ForegroundColor Yellow
    
    $requiredFiles = @(
        "services\aiva_common\schemas\base.py",
        "services\aiva_common\enums\common.py", 
        "services\aiva_common\utils\logging_utils.py"
    )
    
    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Host "❌ 缺少必要檔案:" -ForegroundColor Red
        foreach ($file in $missingFiles) {
            Write-Host "  - $file" -ForegroundColor Red
        }
        return $false
    }
    
    Write-Host "✅ AIVA 依賴檢查通過" -ForegroundColor Green
    return $true
}

# 主執行函數
function Invoke-DuplicationFix {
    param(
        [string]$ExecutePhase,
        [bool]$ExecuteVerify,
        [bool]$ExecuteDryRun,
        [bool]$ExecuteVerbose
    )
    
    # 構建命令參數
    $pythonArgs = @("scripts\analysis\duplication_fix_tool.py")
    
    if ($ExecutePhase) {
        $pythonArgs += "--phase", $ExecutePhase
    }
    
    if ($ExecuteVerify) {
        $pythonArgs += "--verify"
    }
    
    if ($ExecuteDryRun) {
        $pythonArgs += "--dry-run"
    }
    
    if ($ExecuteVerbose) {
        $pythonArgs += "--verbose"
    }
    
    # 執行 Python 腳本
    Write-Host "🚀 執行修復工具..." -ForegroundColor Cyan
    Write-Host "命令: python $($pythonArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""
    
    try {
        & python @pythonArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "🎉 執行完成！" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "❌ 執行失敗，退出碼: $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
    catch {
        Write-Host "❌ 執行過程發生錯誤: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# 主程序
try {
    # 顯示標題
    Write-Host ""
    Write-Host "🔧 =====================================" -ForegroundColor Cyan
    Write-Host "🔧   AIVA 重複定義問題修復工具 v1.0   " -ForegroundColor Cyan  
    Write-Host "🔧 =====================================" -ForegroundColor Cyan
    Write-Host ""
    
    # 檢查是否請求幫助
    if ($Help) {
        Show-Help
        exit 0
    }
    
    # 檢查參數
    if (-not $Phase -and -not $Verify) {
        Write-Host "❓ 請指定執行動作:" -ForegroundColor Yellow
        Write-Host "  -Phase 1   執行階段一修復"
        Write-Host "  -Verify    驗證修復結果"
        Write-Host "  -Help      顯示完整說明"
        Write-Host ""
        Show-Help
        exit 1
    }
    
    # 檢查工作目錄
    if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "services\aiva_common")) {
        Write-Host "❌ 請在 AIVA 專案根目錄執行此腳本" -ForegroundColor Red
        Write-Host "當前目錄: $(Get-Location)" -ForegroundColor Gray
        exit 1
    }
    
    Write-Host "📁 工作目錄: $(Get-Location)" -ForegroundColor Green
    Write-Host ""
    
    # 環境檢查
    if (-not (Test-PythonEnvironment)) {
        exit 1
    }
    
    if (-not (Test-AIVADependencies)) {
        Write-Host ""
        Write-Host "💡 建議解決方案:" -ForegroundColor Yellow
        Write-Host "  1. 確認在正確的 AIVA 專案目錄"
        Write-Host "  2. 檢查 services/aiva_common 模組是否完整"
        Write-Host "  3. 運行 pip install -e . 重新安裝依賴"
        exit 1
    }
    
    Write-Host ""
    
    # 顯示執行計劃
    if ($Phase) {
        Write-Host "📋 執行計劃:" -ForegroundColor Yellow
        Write-Host "  階段: $Phase"
        if ($DryRun) {
            Write-Host "  模式: 試運行（不實際修改檔案）" -ForegroundColor Cyan
        } else {
            Write-Host "  模式: 實際執行" -ForegroundColor Green
        }
        
        if ($Phase -eq "1") {
            Write-Host "  內容: 枚舉重複定義修復 + 核心模型統一"
        } elseif ($Phase -gt "1") {
            Write-Host "  ⚠️  注意: 階段 $Phase 尚未實現" -ForegroundColor Yellow
        }
    }
    
    if ($Verify) {
        Write-Host "📋 執行計劃: 驗證修復結果" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # 確認執行（非試運行模式）
    if (-not $DryRun -and $Phase) {
        Write-Host "⚠️  即將開始修復，這將修改專案檔案" -ForegroundColor Yellow
        $confirmation = Read-Host "是否繼續？(y/N)"
        
        if ($confirmation.ToLower() -notin @("y", "yes", "是")) {
            Write-Host "❌ 用戶取消執行" -ForegroundColor Yellow
            exit 0
        }
    }
    
    # 執行修復工具
    Invoke-DuplicationFix -ExecutePhase $Phase -ExecuteVerify $Verify -ExecuteDryRun $DryRun -ExecuteVerbose $Verbose
    
    # 顯示後續建議
    if ($Phase -and -not $DryRun) {
        Write-Host ""
        Write-Host "🎯 建議後續步驟:" -ForegroundColor Yellow
        Write-Host "  1. 執行驗證: .\fix-duplicates.ps1 -Verify"
        Write-Host "  2. 運行健康檢查: python scripts\utilities\health_check.py"
        Write-Host "  3. 執行測試: python -m pytest tests/"
        Write-Host "  4. 提交變更: git add . && git commit -m '🔧 Fix duplicate definitions Phase $Phase'"
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ 腳本執行失敗: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "堆疊追蹤: $($_.ScriptStackTrace)" -ForegroundColor Gray
    exit 1
}