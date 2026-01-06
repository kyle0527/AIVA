@echo off
REM AIVA Capability CLI 工具啟動腳本 (Windows)
REM 用於查詢和執行能力

setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM 設定 Python 路徑
set "PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services;%PYTHONPATH%"

REM 切換到正確目錄
cd /d "%PROJECT_ROOT%\services\core"

REM 執行 Capability CLI
python -m aiva_core.internal_exploration.python_tools.aiva_capability_cli %*

endlocal
