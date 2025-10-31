param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$SourceLang,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$TargetLang
)

Write-Host "╔══════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   AIVA 語言轉換輔助工具      ║" -ForegroundColor Magenta  
Write-Host "╚══════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

if ($SourceLang -eq $TargetLang) {
    Write-Host "錯誤: 來源語言和目標語言不能相同" -ForegroundColor Red
    return
}

Write-Host "轉換配置:" -ForegroundColor Cyan
Write-Host "  • 來源語言: $SourceLang" -ForegroundColor White
Write-Host "  • 目標語言: $TargetLang" -ForegroundColor White
Write-Host ""

Write-Host "=== $SourceLang -> $TargetLang 轉換建議 ===" -ForegroundColor Yellow

$conversionKey = "$SourceLang-$TargetLang"

if ($conversionKey -eq "python-typescript") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • 使用 Pydantic -> TypeScript 介面轉換" -ForegroundColor White
    Write-Host "  • 注意 async/await 模式差異" -ForegroundColor Yellow
    Write-Host "  • 類型註解轉換為 TypeScript 型別" -ForegroundColor White
    Write-Host ""
    Write-Host "可用工具:" -ForegroundColor Cyan
    Write-Host "  python services/aiva_common/tools/schema_codegen_tool.py --language typescript" -ForegroundColor Gray
}
elseif ($conversionKey -eq "python-go") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • 錯誤處理從 try/except 改為 error 返回值" -ForegroundColor Yellow
    Write-Host "  • 類別轉換為結構體和方法" -ForegroundColor White
    Write-Host "  • 記憶體管理需要重新設計" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "可用工具:" -ForegroundColor Cyan
    Write-Host "  python services/aiva_common/tools/cross_language_interface.py --target go" -ForegroundColor Gray
}
elseif ($conversionKey -eq "python-rust") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • 學習 Rust 所有權系統" -ForegroundColor Yellow
    Write-Host "  • Option/Result 類型替代 None/Exception" -ForegroundColor White
    Write-Host "  • 建議完全重新設計而非直接轉換" -ForegroundColor Red
}
elseif ($conversionKey -eq "typescript-python") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • 介面轉換為 Pydantic 模型" -ForegroundColor White
    Write-Host "  • Promise 轉換為 async/await" -ForegroundColor White
    Write-Host "  • 類型檢查使用 mypy" -ForegroundColor White
}
elseif ($conversionKey -eq "go-python") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • 結構體轉換為 Python 類別" -ForegroundColor White
    Write-Host "  • Goroutines 轉換為 asyncio" -ForegroundColor Yellow
    Write-Host "  • 錯誤處理轉換為 try/except" -ForegroundColor White
}
elseif ($conversionKey -eq "rust-python") {
    Write-Host "轉換建議:" -ForegroundColor Green
    Write-Host "  • Option/Result 轉換為 Optional/Exception" -ForegroundColor White
    Write-Host "  • 所有權系統轉換為引用計數" -ForegroundColor Yellow
    Write-Host "  • 建議保留 Rust 作為 Python 擴展" -ForegroundColor Red
}
else {
    Write-Host "複雜轉換，建議參考完整指南" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "完整轉換指南:" -ForegroundColor Cyan
Write-Host "   guides/development/LANGUAGE_CONVERSION_GUIDE.md" -ForegroundColor Gray

Write-Host ""
Write-Host "相關工具和資源:" -ForegroundColor Cyan
Write-Host "  • Schema 代碼生成: services/aiva_common/tools/schema_codegen_tool.py" -ForegroundColor Gray
Write-Host "  • 跨語言接口: services/aiva_common/tools/cross_language_interface.py" -ForegroundColor Gray
Write-Host "  • 跨語言驗證: services/aiva_common/tools/cross_language_validator.py" -ForegroundColor Gray

Write-Host ""
Write-Host "轉換輔助完成！" -ForegroundColor Green