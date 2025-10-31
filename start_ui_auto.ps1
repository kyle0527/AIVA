#!/usr/bin/env pwsh
# AIVA UI 自動端口啟動腳本 (PowerShell)

Write-Host "🚀 啟動 AIVA UI 自動端口伺服器..." -ForegroundColor Green

try {
    # 檢查是否在專案根目錄
    if (-not (Test-Path "pyproject.toml")) {
        Write-Host "❌ 請在 AIVA 專案根目錄執行此腳本" -ForegroundColor Red
        exit 1
    }

    # 啟動 UI 伺服器
    python -m services.core.aiva_core.ui_panel.auto_server --mode hybrid --host 127.0.0.1 --ports 8080 8081 3000 5000 9000
}
catch {
    Write-Host "❌ 啟動失敗: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}