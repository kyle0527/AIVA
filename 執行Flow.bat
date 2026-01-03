@echo off
REM AIVA Flow 執行器
REM 用法: 執行Flow.bat [Flow ID]
REM 範例: 執行Flow.bat 11

echo ========================================
echo   AIVA Flow 執行器
echo ========================================
echo.

cd /d "C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools"

if "%1"=="" (
    echo 用法: 執行Flow.bat [Flow ID]
    echo 範例: 執行Flow.bat 11
    echo.
    echo 可用 Flow 列表:
    python aiva_cli_implementation.py --list
) else (
    echo 執行 Flow %1 ...
    python aiva_cli_implementation.py --flow %1
)

pause
