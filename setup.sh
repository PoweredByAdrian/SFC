#!/bin/bash

# setup.sh - Automated environment setup (no sudo required)
# Usage: ./setup.sh

set -e  # Exit on any error

echo "================================================"
echo "   Project Environment Setup"
echo "================================================"

# ------------------------------------------------------
# STEP 1: Virtual Environment Setup
# ------------------------------------------------------
echo ""
echo "[Step 1/3] Creating Virtual Environment..."

if [ -d "venv" ]; then
    echo "✔ Existing virtual environment detected (./venv)."
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    echo "✔ Virtual environment created."
fi

VENV_PIP="./venv/bin/pip"
VENV_PYTHON="./venv/bin/python"

# ------------------------------------------------------
# STEP 2: Upgrade pip + setuptools + wheel
# ------------------------------------------------------
echo ""
echo "[Step 2/3] Upgrading pip & tools..."

$VENV_PIP install --upgrade pip setuptools wheel

echo "✔ pip, setuptools, wheel upgraded."

# ------------------------------------------------------
# STEP 3: Install Python Dependencies
# ------------------------------------------------------
echo ""
echo "[Step 3/3] Installing requirements..."

if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

$VENV_PIP install -r requirements.txt

echo ""
echo "================================================"
echo "   Setup Complete! Environment Ready."
echo "================================================"
echo ""
echo "HOW TO START:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo "  python watch.py"
echo ""
