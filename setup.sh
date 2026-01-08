#!/bin/bash

set -e

echo "Setting up AI Hardware Probe..."

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating..."
else
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the hardware probe:"
echo "  source venv/bin/activate"
echo "  python3 ai_hw_probe.py"
echo ""
echo "Or simply:"
echo "  ./ai_hw_probe.py"
echo ""
