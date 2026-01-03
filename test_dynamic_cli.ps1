# 動態 CLI 測試腳本

$env:PYTHONPATH="C:\D\fold7\AIVA-git"
$CLI_CMD = "python -m services.core.aiva_core.core_capabilities.cli.aiva_cli"

Write-Host "=== AIVA 動態 Flow CLI 測試 ===" -ForegroundColor Cyan
Write-Host ""

# 測試 1: 列出前 10 個 flows
Write-Host "📋 測試 1: 列出前 10 個 flows" -ForegroundColor Yellow
& $CLI_CMD.Split() list-flows --limit 10
Write-Host ""

# 測試 2: 查看 flow4 幫助
Write-Host "📋 測試 2: 查看 flow4 幫助" -ForegroundColor Yellow
& $CLI_CMD.Split() flow4 --help
Write-Host ""

# 測試 3: Dry-run flow1（簡單的2步flow）
Write-Host "📋 測試 3: Dry-run flow1" -ForegroundColor Yellow
& $CLI_CMD.Split() flow1 --dry-run
Write-Host ""

# 測試 4: Dry-run flow4 帶參數
Write-Host "📋 測試 4: Dry-run flow4 帶多個參數" -ForegroundColor Yellow
& $CLI_CMD.Split() flow4 --data "/tmp/test.npz" --param model=v1 --param epochs=100 -i 0.8 --dry-run
Write-Host ""

# 測試 5: 查看 service_backbone 模組的 flows
Write-Host "📋 測試 5: 查看 service_backbone 模組" -ForegroundColor Yellow
& $CLI_CMD.Split() list-flows --module service_backbone --limit 5
Write-Host ""

Write-Host "✅ 所有測試完成！" -ForegroundColor Green
Write-Host ""
Write-Host "使用方式:" -ForegroundColor Cyan
Write-Host "  aiva flow<ID> --target <目標> -i <強度>" -ForegroundColor White
Write-Host "  aiva flow<ID> --data <路徑> --dry-run" -ForegroundColor White
Write-Host "  aiva list-flows --limit 20" -ForegroundColor White
Write-Host "  aiva flow<ID> --help" -ForegroundColor White
