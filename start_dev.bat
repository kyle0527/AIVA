@echo off
REM AIVA 開發環境啟動腳本
echo ========================================
echo AIVA 開發環境設定
echo ========================================

cd /d c:\D\E\AIVA\AIVA-main
set PYTHONPATH=c:\D\E\AIVA\AIVA-main

echo.
echo ✓ 工作目錄: %CD%
echo ✓ PYTHONPATH: %PYTHONPATH%
echo.
echo ========================================
echo 環境已就緒！
echo ========================================
echo.
echo 可用命令:
echo   python test_environment.py         - 測試環境
echo   python -m services.core.aiva_core.app      - 啟動 Core
echo   python -m services.scan.aiva_scan.worker   - 啟動 Scan
echo   code .                              - 開啟 VS Code
echo.

cmd /k
