@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
REM AIVA Flow 預覽器（Dry Run）
REM 用法: 預覽Flow.bat [Flow ID]
REM 只顯示執行計畫，不實際運行

REM 設置 Python 環境
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA Flow 預覽器 (Dry Run)
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core"

if "%1"=="" (
    echo 用法: 預覽Flow.bat [Flow ID]
    echo 範例: 預覽Flow.bat 11
    echo.
    echo 此功能只顯示執行計畫，不實際運行代碼
    echo.
    echo 可用 Flow 列表:
    python -m aiva_core.internal_exploration.aiva_internal_executor --list
) else (
    echo 預覽 Flow %1 的執行計畫...
    python -m aiva_core.internal_exploration.aiva_internal_executor --flow %1 --dry-run
)

pause
