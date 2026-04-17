@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
REM AIVA 能力選單快速啟動
REM 雙擊此檔案即可啟動互動式能力選單

REM 設置 Python 環境
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA 能力執行選單
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core"

python -m aiva_core.internal_exploration.aiva_internal_executor --menu

pause
