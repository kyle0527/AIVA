@echo off
chcp 65001 >nul 2>&1
REM ==========================================
REM AIVA 完整系統啟動腳本
REM 基於雙循環設計指南的完整流程
REM ==========================================

echo.
echo ========================================
echo   AIVA 智能滲透測試系統 v2.2.0
echo   啟動雙循環自我優化架構
echo ========================================
echo.

REM 設置環境變量
set PYTHONPATH=C:\D\fold7\AIVA-git;C:\D\fold7\AIVA-git\services
set PYTHONIOENCODING=utf-8
set AIVA_ENV=production

REM 檢查 RabbitMQ 是否運行
echo [1/5] 檢查 RabbitMQ 服務...
sc query RabbitMQ >nul 2>&1
if errorlevel 1 (
    echo ❌ RabbitMQ 未運行！請先啟動 RabbitMQ 服務
    echo.
    echo 啟動方式:
    echo   - Windows 服務: net start RabbitMQ
    echo   - Docker: docker-compose up -d rabbitmq
    echo.
    pause
    exit /b 1
)
echo ✅ RabbitMQ 服務正常

echo.
echo [2/5] 啟動 AIVA Core API (FastAPI)...
echo      功能: 系統唯一入口點
echo      端口: http://localhost:8000
echo      文檔: http://localhost:8000/docs
echo.

REM 在後台啟動 Core API
start "AIVA Core API" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git && python -m uvicorn services.core.aiva_core.service_backbone.api.app:app --host 0.0.0.0 --port 8000 --reload"

REM 等待 API 啟動
timeout /t 5 /nobreak >nul

echo.
echo [3/5] 啟動掃描引擎 Worker...

REM 啟動 Rust Engine Worker
start "Rust Engine Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git\services\scan\rust_engine && cargo run --release"

REM 啟動 Python Engine Worker  
start "Python Engine Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git && python -m services.scan.python_engine.worker"

REM 啟動 TypeScript Engine Worker
start "TypeScript Engine Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git\services\scan\typescript_engine && npm start"

echo ✅ 掃描引擎已啟動 (Rust, Python, TypeScript)

echo.
echo [4/5] 啟動功能 Worker...

REM 啟動 SSRF Worker
start "SSRF Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git && python -m services.features.function_ssrf.worker"

REM 啟動 BizLogic Worker
start "BizLogic Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git && python -m services.features.function_bizlogic.worker"

REM 啟動 XSS Worker
start "XSS Worker" /MIN cmd /c "cd /d C:\D\fold7\AIVA-git && python -m services.features.function_xss.worker"

echo ✅ 功能 Worker 已啟動

echo.
echo [5/5] 系統狀態檢查...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   ✅ AIVA 系統已完全啟動！
echo ========================================
echo.
echo 📡 API 服務: http://localhost:8000
echo 📚 API 文檔: http://localhost:8000/docs
echo 🔍 健康檢查: http://localhost:8000/health
echo.
echo ==========================================
echo   使用方式
echo ==========================================
echo.
echo 方式 1: 使用 API 發送掃描請求
echo   curl -X POST http://localhost:8000/api/v1/scan ^
echo        -H "Content-Type: application/json" ^
echo        -d "{\"target\": \"http://example.com\", \"scan_type\": \"full\"}"
echo.
echo 方式 2: 使用 Web 界面
echo   打開瀏覽器訪問: http://localhost:8000/docs
echo   在 Swagger UI 中測試 API
echo.
echo 方式 3: 使用 Python 客戶端
echo   python scripts\common\test_scan_api.py --target http://example.com
echo.
echo ==========================================
echo.
echo 按任意鍵查看實時日誌...
pause

REM 打開日誌查看
start "AIVA Logs" cmd /c "cd /d C:\D\fold7\AIVA-git\logs && powershell -Command Get-Content -Path aiva_core\app.log -Wait -Tail 50"

echo.
echo 日誌查看已啟動，按 Ctrl+C 可停止
echo.
pause
