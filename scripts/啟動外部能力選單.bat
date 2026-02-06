@echo off
chcp 65001 >nul 2>&1
REM AIVA 外部模組能力選單
REM 雙擊此檔案即可啟動互動式能力選單

REM 設置 Python 環境
set PYTHONPATH=C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA 外部模組能力執行選單
echo   (Python, Go, TypeScript)
echo ========================================
echo.

cd /d "C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration"

python aiva_external_executor.py --menu

pause
