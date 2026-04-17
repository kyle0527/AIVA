@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
REM AIVA 統一執行器控制台 - 快速啟動
REM 雙擊此檔案啟動互動式選單

REM 設置 Python 環境
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA 統一執行器控制台
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core\aiva_core\internal_exploration"

python unified_executor_controller.py --menu

pause
