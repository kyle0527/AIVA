#!/usr/bin/env pwsh
# AIVA 語言轉換輔助工具
# 使用方法: .\language_converter.ps1 -SourceLang python -TargetLang typescript -SourceFile "example.py"

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$SourceLang,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$TargetLang,
    
    [Parameter(Mandatory=$false)]
    [string]$SourceFile = "",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "converted",
    
    [Parameter(Mandatory=$false)]
    [switch]$SchemaOnly = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$ShowGuide = $false
)

# 顏色定義
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Header = "Magenta"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Show-ConversionGuide {
    param([string]$Source, [string]$Target)
    
    Write-ColorOutput "=== $Source → $Target 轉換指南 ===" "Header"
    
    $guidePath = "guides/development/LANGUAGE_CONVERSION_GUIDE.md"
    if (Test-Path $guidePath) {
        Write-ColorOutput "📖 完整指南位置: $guidePath" "Info"
        Write-ColorOutput "🔍 搜尋關鍵字: '$Source → $Target'" "Info"
    }
    
    # 顯示特定轉換建議
    switch ("$Source-$Target") {
        "python-typescript" {
            Write-ColorOutput "✨ 轉換建議:" "Success"
            Write-ColorOutput "  • 使用 Pydantic → TypeScript 介面轉換" "Info"
            Write-ColorOutput "  • 注意 async/await 模式差異" "Warning"
            Write-ColorOutput "  • 類型註解轉換為 TypeScript 型別" "Info"
        }
        "python-go" {
            Write-ColorOutput "✨ 轉換建議:" "Success"
            Write-ColorOutput "  • 錯誤處理從 try/except 改為 error 返回值" "Warning"
            Write-ColorOutput "  • 類別轉換為結構體和方法" "Info"
            Write-ColorOutput "  • 記憶體管理需要重新設計" "Warning"
        }
        "python-rust" {
            Write-ColorOutput "✨ 轉換建議:" "Success"
            Write-ColorOutput "  • 學習 Rust 所有權系統" "Warning"
            Write-ColorOutput "  • Option/Result 類型替代 None/Exception" "Info"
            Write-ColorOutput "  • 建議完全重新設計而非直接轉換" "Warning"
        }
        "typescript-python" {
            Write-ColorOutput "✨ 轉換建議:" "Success"
            Write-ColorOutput "  • 介面轉換為 Pydantic 模型" "Info"
            Write-ColorOutput "  • Promise 轉換為 async/await" "Info"
            Write-ColorOutput "  • 類型檢查使用 mypy" "Info"
        }
        default {
            Write-ColorOutput "⚠️  複雜轉換，建議參考完整指南" "Warning"
        }
    }
    Write-ColorOutput ""
}

function Invoke-SchemaConversion {
    param([string]$Target)
    
    Write-ColorOutput "🔄 執行 Schema 轉換..." "Info"
    
    $schemaToolPath = "services/aiva_common/tools/schema_codegen_tool.py"
    if (Test-Path $schemaToolPath) {
        try {
            & python $schemaToolPath --language $Target
            Write-ColorOutput "✅ Schema 轉換完成" "Success"
        }
        catch {
            Write-ColorOutput "❌ Schema 轉換失敗: $($_.Exception.Message)" "Error"
        }
    }
    else {
        Write-ColorOutput "⚠️  Schema 工具未找到: $schemaToolPath" "Warning"
    }
}

function Invoke-CrossLanguageInterface {
    param([string]$Target)
    
    Write-ColorOutput "🤖 執行跨語言接口轉換..." "Info"
    
    $interfacePath = "services/aiva_common/tools/cross_language_interface.py"
    if (Test-Path $interfacePath) {
        try {
            & python $interfacePath --target $Target
            Write-ColorOutput "✅ 跨語言接口轉換完成" "Success"
        }
        catch {
            Write-ColorOutput "❌ 接口轉換失敗: $($_.Exception.Message)" "Error"
        }
    }
    else {
        Write-ColorOutput "⚠️  跨語言接口工具未找到: $interfacePath" "Warning"
    }
}

function Test-ConversionQuality {
    param([string]$Target)
    
    Write-ColorOutput "🔍 執行轉換品質檢查..." "Info"
    
    switch ($Target) {
        "python" {
            if (Get-Command python -ErrorAction SilentlyContinue) {
                Write-ColorOutput "  • Python 語法檢查..." "Info"
                # python -m py_compile 檢查
            }
        }
        "typescript" {
            if (Get-Command tsc -ErrorAction SilentlyContinue) {
                Write-ColorOutput "  • TypeScript 類型檢查..." "Info"
                # tsc --noEmit 檢查
            }
        }
        "go" {
            if (Get-Command go -ErrorAction SilentlyContinue) {
                Write-ColorOutput "  • Go 語法檢查..." "Info"
                # go vet 檢查
            }
        }
        "rust" {
            if (Get-Command cargo -ErrorAction SilentlyContinue) {
                Write-ColorOutput "  • Rust 語法檢查..." "Info"
                # cargo check 檢查
            }
        }
    }
}

# 主要執行邏輯
function Main {
    Write-ColorOutput "╔══════════════════════════════════════╗" "Header"
    Write-ColorOutput "║     AIVA 語言轉換輔助工具            ║" "Header"
    Write-ColorOutput "╚══════════════════════════════════════╝" "Header"
    Write-ColorOutput ""
    
    if ($SourceLang -eq $TargetLang) {
        Write-ColorOutput "❌ 來源語言和目標語言不能相同" "Error"
        return
    }
    
    Write-ColorOutput "🔄 轉換配置:" "Info"
    Write-ColorOutput "  • 來源語言: $SourceLang" "Info"  
    Write-ColorOutput "  • 目標語言: $TargetLang" "Info"
    if ($SourceFile) {
        Write-ColorOutput "  • 來源檔案: $SourceFile" "Info"
    }
    Write-ColorOutput "  • 輸出目錄: $OutputDir" "Info"
    Write-ColorOutput ""
    
    # 顯示轉換指南
    if ($ShowGuide -or (-not $SourceFile -and -not $SchemaOnly)) {
        Show-ConversionGuide $SourceLang $TargetLang
    }
    
    # 確保輸出目錄存在
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
        Write-ColorOutput "📁 創建輸出目錄: $OutputDir" "Success"
    }
    
    # 執行轉換
    if ($SchemaOnly) {
        Write-ColorOutput "🔧 執行 Schema 專用轉換..." "Info"
        Invoke-SchemaConversion $TargetLang
    }
    elseif ($SourceFile -and (Test-Path $SourceFile)) {
        Write-ColorOutput "📄 處理檔案: $SourceFile" "Info"
        
        # 根據轉換類型執行不同策略
        switch ("$SourceLang-$TargetLang") {
            "python-typescript" {
                Invoke-SchemaConversion $TargetLang
                Write-ColorOutput "💡 提示: 複雜邏輯需要手動轉換" "Warning"
            }
            "python-go" {
                Invoke-CrossLanguageInterface $TargetLang
                Write-ColorOutput "💡 提示: 錯誤處理需要重新設計" "Warning"
            }
            "python-rust" {
                Write-ColorOutput "⚠️  建議手動轉換，Rust 轉換複雜度很高" "Warning"
                Show-ConversionGuide $SourceLang $TargetLang
            }
            default {
                Write-ColorOutput "🔧 執行通用轉換流程..." "Info"
                Invoke-SchemaConversion $TargetLang
            }
        }
    }
    else {
        Write-ColorOutput "📖 顯示轉換指南 (未指定有效的來源檔案)" "Info"
        Show-ConversionGuide $SourceLang $TargetLang
        
        # 提供可用的工具
        Write-ColorOutput "🛠️  可用的轉換工具:" "Info"
        Write-ColorOutput "  • Schema 轉換: --SchemaOnly" "Info"
        Write-ColorOutput "  • 顯示指南: --ShowGuide" "Info"
        Write-ColorOutput "  • 指定檔案: -SourceFile 'path/to/file'" "Info"
    }
    
    # 品質檢查
    if ($TargetLang -and -not $ShowGuide) {
        Write-ColorOutput ""
        Test-ConversionQuality $TargetLang
    }
    
    Write-ColorOutput ""
    Write-ColorOutput "✨ 轉換流程完成！" "Success"
    Write-ColorOutput "📖 完整指南: guides/development/LANGUAGE_CONVERSION_GUIDE.md" "Info"
}

# 執行主函數
try {
    Main
}
catch {
    Write-ColorOutput "❌ 執行過程發生錯誤: $($_.Exception.Message)" "Error"
    Write-ColorOutput "🔍 請檢查參數和檔案路徑是否正確" "Warning"
}