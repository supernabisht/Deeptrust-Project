@echo off
echo Setting up final Python environment for DeepTrust...

REM Remove existing virtual environment if it exists
if exist "venv_final" (
    echo Removing existing virtual environment...
    rmdir /s /q venv_final
)

REM Create new virtual environment
echo Creating new virtual environment...
python -m venv venv_final

REM Activate and install requirements
call venv_final\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r final_requirements_fixed.txt

echo.
echo Setup complete! Activate the environment with:
echo venv_final\Scripts\activate.bat
echo.
pause
