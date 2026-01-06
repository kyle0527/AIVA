@echo off
REM AIVA CLI 工具統一啟動腳本 (Windows)
REM 自動設定 PYTHONPATH 並執行 CLI 工具

setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM 設定 Python 路徑
set "PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services;%PYTHONPATH%"

REM 切換到正確目錄
cd /d "%PROJECT_ROOT%\services\core"

REM 執行 CLI 工具
python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation %*

endlocal
