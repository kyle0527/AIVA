#!/usr/bin/env pwsh
# AIVA Core API 啟動腳本 (v2.0 AICommand 模式)
# 只需要 port 8000，不需要 RabbitMQ 或 gRPC

$env:PYTHONPATH = "C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services"

Write-Host "🚀 Starting AIVA Core API on port 8000..." -ForegroundColor Cyan
Write-Host "   - PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray

Set-Location "C:\D\fold7\AIVA-git"

python -m uvicorn aiva_core.service_backbone.api.app:app --host 127.0.0.1 --port 8000
