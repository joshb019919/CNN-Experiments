@echo off
echo Setting up Python virtual environment...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo Environment setup complete!
echo Run:   call venv\Scripts\activate
pause
