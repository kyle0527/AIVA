@echo off
chcp 65001 >nul 2>&1

echo ========================================
echo   AIVA Module Executor (Direct Call)
echo ========================================
echo.

if "%1"=="" (
    echo Usage: script.bat [module] [url]
    echo.
    echo Modules:
    echo   xss      - XSS Scanner
    echo   sqli     - SQL Injection Scanner
    echo   ssrf     - SSRF Scanner
    echo   idor     - IDOR Scanner
    echo   bizlogic - Business Logic Scanner
    echo.
    echo Example:
    echo   script.bat xss http://localhost:3000
    pause
    exit /b 0
)

cd /d C:\D\fold7\AIVA-git
python scripts\run_module.py %1 %2

pause
