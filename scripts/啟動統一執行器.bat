@echo off
chcp 65001 >nul 2>&1
REM AIVA 統一執行器控制台 - 快速啟動
REM 雙擊此檔案啟動互動式選單

REM 設置 Python 環境
set PYTHONPATH=C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA 統一執行器控制台
echo ========================================
echo.

cd /d "C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration"

python unified_executor_controller.py --menu

pause
