@echo off
REM AIVA AI 快速啟動 - API 模式
REM 雙擊此文件即可啟動 AIVA AI 服務

echo ========================================
echo   AIVA AI Service - API Mode
echo ========================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo Starting AIVA AI Service...
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python start_ai_service.py --mode api --port 8000

pause
