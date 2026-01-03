@echo off
REM AIVA 能力選單快速啟動
REM 雙擊此檔案即可啟動互動式能力選單

echo ========================================
echo   AIVA 能力執行選單
echo ========================================
echo.

cd /d "C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools"

python aiva_cli_implementation.py --menu

pause
