@echo off
set "PROJECT_ROOT=%~dp0.."
chcp 65001 >nul 2>&1
set PYTHONPATH=%PROJECT_ROOT%\services\core;%PROJECT_ROOT%\services
set PYTHONIOENCODING=utf-8

echo ========================================
echo   AIVA External Module Classifier
echo ========================================
echo.

cd /d "%PROJECT_ROOT%\services\core\aiva_core\internal_exploration\python_tools"

if "%1"=="" (
    echo Usage: script.bat [input_dir] [output_dir]
    echo Example: script.bat . ./reports
    pause
    exit /b 0
)

set INPUT_DIR=%1
set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=./reports

echo Input: %INPUT_DIR%
echo Output: %OUTPUT_DIR%
echo.
python aiva_external_module_batch_classifier.py -w %INPUT_DIR% -o %OUTPUT_DIR% -v

pause
