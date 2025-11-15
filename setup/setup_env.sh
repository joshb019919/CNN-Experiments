#!/bin/bash

echo "Setting up Python virtual environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete!"
echo "Activate with: source venv/bin/activate"
