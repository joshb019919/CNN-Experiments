@echo off
cd /d "%~dp0"

echo Setting up Python virtual environment...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo Environment setup complete!
pause
