#!/bin/bash
# Script to start the demo server

cd "$(dirname "$0")"

echo "================================"
echo " Starting Earthquake Demo "
echo "================================"
echo ""

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found!"
    echo "  Run: python3 -m venv venv"
    echo "  Run: source venv/bin/activate && pip install flask numpy torch pandas scikit-learn"
    exit 1
fi

# Check required packages
echo "Checking dependencies..."
python -c "import flask, numpy, torch, pandas, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✗ Missing dependencies!"
    echo "  Installing..."
    pip install flask numpy torch pandas scikit-learn
fi

echo "✓ Dependencies OK"
echo ""
echo "Starting Flask server..."
echo "Open browser: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

python demo.py
