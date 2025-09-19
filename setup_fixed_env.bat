@echo off
echo Setting up fixed Python environment for DeepTrust...

REM Remove existing virtual environment if it exists
if exist "venv_fixed" (
    echo Removing existing virtual environment...
    rmdir /s /q venv_fixed
)

REM Create new virtual environment
echo Creating new virtual environment...
python -m venv venv_fixed

REM Activate and install requirements
call venv_fixed\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements_fixed.txt

echo.
echo Setup complete! Activate the environment with:
echo venv_fixed\Scripts\activate.bat
echo.
pause
