@echo off
setlocal

set "DEMO_FILE=%~dp0demo\index.html"

if not exist "%DEMO_FILE%" (
    echo [ERROR] Cannot find demo\index.html.
    echo Keep start_demo.bat, demo, and outputs in the same project directory.
    pause
    exit /b 1
)

start "" "%DEMO_FILE%"
exit /b 0
