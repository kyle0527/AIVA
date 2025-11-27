# 刪除 node_modules 中的 MD 文件

# 選項 1: 只刪除 MD 文件（執行 Python 腳本）
Write-Host "`n=== 選項 1: 刪除 439 個 MD 文件 ===" -ForegroundColor Yellow
Write-Host "執行: python _delete_node_modules_md.py" -ForegroundColor Cyan
Write-Host ""

# 選項 2: 刪除整個 node_modules（推薦）
Write-Host "=== 選項 2: 刪除整個 node_modules（推薦）===" -ForegroundColor Yellow
Write-Host "執行: Remove-Item -Recurse -Force services\scan\engines\typescript_engine\node_modules" -ForegroundColor Cyan
Write-Host ""

Write-Host "當前狀態:" -ForegroundColor Green
Write-Host "  ✅ 所有 439 個 MD 文件內容已整合到 DEPENDENCIES_GUIDE.md"
Write-Host "  ✅ 完整清單已儲存到 _node_modules_complete_inventory.json"
Write-Host "  ✅ Git 不會追蹤 node_modules（已在 .gitignore）"
Write-Host ""

Write-Host "建議:" -ForegroundColor Cyan
Write-Host "  - 如果不需要使用 TypeScript Engine，刪除整個 node_modules"
Write-Host "  - 如果還需要使用，保留 node_modules（MD 文件只佔 1-2 MB）"
Write-Host "  - 需要時執行 'npm install' 重建所有依賴"
Write-Host ""
