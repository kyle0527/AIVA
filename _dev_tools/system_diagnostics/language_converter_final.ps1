param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$SourceLang,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("python", "typescript", "go", "rust")]
    [string]$TargetLang
)

Write-Host "╔══════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   AIVA Language Converter    ║" -ForegroundColor Magenta  
Write-Host "╚══════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

if ($SourceLang -eq $TargetLang) {
    Write-Host "Error: Source and target languages cannot be the same" -ForegroundColor Red
    exit 1
}

Write-Host "Conversion Configuration:" -ForegroundColor Cyan
Write-Host "  • Source Language: $SourceLang" -ForegroundColor White
Write-Host "  • Target Language: $TargetLang" -ForegroundColor White
Write-Host ""

Write-Host "=== $SourceLang -> $TargetLang Conversion Guide ===" -ForegroundColor Yellow

$conversionKey = "$SourceLang-$TargetLang"

if ($conversionKey -eq "python-typescript") {
    Write-Host "Conversion Tips:" -ForegroundColor Green
    Write-Host "  • Use Pydantic -> TypeScript interface conversion" -ForegroundColor White
    Write-Host "  • Pay attention to async/await pattern differences" -ForegroundColor Yellow
    Write-Host "  • Convert type annotations to TypeScript types" -ForegroundColor White
    Write-Host ""
    Write-Host "Available Tools:" -ForegroundColor Cyan
    Write-Host "  python services/aiva_common/tools/schema_codegen_tool.py --language typescript" -ForegroundColor Gray
}
elseif ($conversionKey -eq "python-go") {
    Write-Host "Conversion Tips:" -ForegroundColor Green
    Write-Host "  • Change error handling from try/except to error return values" -ForegroundColor Yellow
    Write-Host "  • Convert classes to structs and methods" -ForegroundColor White
    Write-Host "  • Redesign memory management" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available Tools:" -ForegroundColor Cyan
    Write-Host "  python services/aiva_common/tools/cross_language_interface.py --target go" -ForegroundColor Gray
}
elseif ($conversionKey -eq "python-rust") {
    Write-Host "Conversion Tips:" -ForegroundColor Green
    Write-Host "  • Learn Rust ownership system" -ForegroundColor Yellow
    Write-Host "  • Replace None/Exception with Option/Result types" -ForegroundColor White
    Write-Host "  • Recommend complete redesign rather than direct conversion" -ForegroundColor Red
}
elseif ($conversionKey -eq "typescript-python") {
    Write-Host "Conversion Tips:" -ForegroundColor Green
    Write-Host "  • Convert interfaces to Pydantic models" -ForegroundColor White
    Write-Host "  • Convert Promise to async/await" -ForegroundColor White
    Write-Host "  • Use mypy for type checking" -ForegroundColor White
}
else {
    Write-Host "Complex conversion, please refer to the complete guide" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Complete Conversion Guide:" -ForegroundColor Cyan
Write-Host "   guides/development/LANGUAGE_CONVERSION_GUIDE.md" -ForegroundColor Gray

Write-Host ""
Write-Host "Related Tools and Resources:" -ForegroundColor Cyan
Write-Host "  • Schema Code Generation: services/aiva_common/tools/schema_codegen_tool.py" -ForegroundColor Gray
Write-Host "  • Cross-Language Interface: services/aiva_common/tools/cross_language_interface.py" -ForegroundColor Gray
Write-Host "  • Cross-Language Validator: services/aiva_common/tools/cross_language_validator.py" -ForegroundColor Gray

Write-Host ""
Write-Host "Conversion assistance completed!" -ForegroundColor Green