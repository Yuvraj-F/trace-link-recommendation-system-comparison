@echo off
cd /d "%~dp0"

set "ENV_DIR=.venv"

set "PYTHON=%ENV_DIR%\Scripts\python"

set "MAIN=package\main.py"


if not exist "%ENV_DIR%\Scripts\python.exe" (
    echo Python environment Not Found
    echo Creating new environment...
    python -m venv "%ENV_DIR%"
    "%ENV_DIR%\Scripts\python" -m pip install --upgrade pip
)

"%ENV_DIR%\Scripts\python" -m pip install -r requirements.txt

"%PYTHON%" "%MAIN%"