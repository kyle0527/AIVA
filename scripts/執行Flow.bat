@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
REM AIVA Flow 執行器
REM 用法: 執行Flow.bat [Flow ID]
REM 範例: 執行Flow.bat 11

REM 設置 Python 環境
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA Flow 執行器
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core"

if "%1"=="" (
    echo 用法: 執行Flow.bat [Flow ID]
    echo 範例: 執行Flow.bat 11
    echo.
    echo 可用 Flow 列表:
    python -m aiva_core.internal_exploration.aiva_internal_executor --list
) else (
    echo 執行 Flow %1 ...
    python -m aiva_core.internal_exploration.aiva_internal_executor --flow %1
)

pause
