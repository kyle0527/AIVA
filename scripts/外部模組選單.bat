@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

:MENU
cls
echo ========================================
echo   AIVA External Module Menu
echo ========================================
echo.
echo 1. SQL Injection (function_sqli)
echo 2. XSS Detection (function_xss)
echo 3. SSRF Detection (function_ssrf)
echo 4. IDOR Detection (function_idor)
echo 5. Business Logic (function_bizlogic)
echo 6. List All Capabilities
echo 0. Exit
echo.
set /p choice=Select (0-6): 

if "%choice%"=="0" goto END
if "%choice%"=="6" goto LIST

cd /d "%PROJECT_ROOT%\services\core\aiva_core\internal_exploration\python_tools"

set /p target=Enter target URL: 

echo.
echo Running Python flow...
python aiva_external_module_cli.py --lang python --flow 1 --target %target%

pause
goto MENU

:LIST
cd /d "%PROJECT_ROOT%\services\core\aiva_core\internal_exploration\python_tools"
python aiva_external_module_cli.py --list --lang python
pause
goto MENU

:END
echo Goodbye!
