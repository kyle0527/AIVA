@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
REM AIVA 快速執行能力
REM 用法: 快速執行能力.bat [capability] [target]
REM 範例: 快速執行能力.bat sqli http://test.com

REM 設置 Python 環境
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA 快速執行能力
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core\aiva_core\internal_exploration"

if "%1"=="" (
    echo 用法: 快速執行能力.bat [capability] [target]
    echo.
    echo 可用能力:
    python unified_executor_controller.py --list
) else if "%2"=="" (
    echo 錯誤: 需要提供目標 URL
    echo 用法: 快速執行能力.bat %1 [target_url]
) else (
    echo 執行能力: %1
    echo 目標: %2
    echo.
    python unified_executor_controller.py --capability %1 --target "%2" --dry-run
)

pause
