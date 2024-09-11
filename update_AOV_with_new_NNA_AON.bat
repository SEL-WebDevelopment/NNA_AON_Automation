@echo off

REM Run service.py
echo Running service.py...
python service.py

REM Check if service.py was successful
IF %ERRORLEVEL% NEQ 0 (
    echo service.py failed. Exiting.
    exit /b 1
)

REM Run github_automation.py
echo Running github_automation.py...
python github_automation.py

REM Check if github_automation.py was successful
IF %ERRORLEVEL% NEQ 0 (
    echo github_automation.py failed.
    exit /b 1
)

echo Both scripts ran successfully.
