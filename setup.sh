#!/bin/bash

# setup.sh - Automated setup script for CarRacing DQN Project
# Usage: ./setup.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "================================================"
echo "   CarRacing DQN Project - Setup Script"
echo "================================================"

# ------------------------------------------------------
# STEP 1: System Dependencies (Requires Sudo)
# ------------------------------------------------------
echo ""
echo "[Step 1/3] Checking System Dependencies..."

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is optimized for Linux. You may encounter issues on other OS."
fi

# We check if we have sudo access first
sudo -v

echo "Updating package lists..."
sudo apt-get update -qq

echo "Installing Physics Engine tools (SWIG & Box2D)..."
# Added python3-venv here - crucial for creating the environment
sudo apt-get install -y swig build-essential python3-dev python3-venv

echo "Installing Graphic drivers (SDL2)..."
sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

echo "✔ System dependencies ready."

# ------------------------------------------------------
# STEP 2: Virtual Environment Setup
# ------------------------------------------------------
echo ""
echo "[Step 2/3] Setting up Python Virtual Environment..."

# Logic: If 'venv' folder exists, we use it. If not, we create it.
if [ -d "venv" ]; then
    echo "✔ Found existing virtual environment ('venv/'). Using it."
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    echo "✔ Virtual environment created."
fi

# ------------------------------------------------------
# STEP 3: Python Dependencies Installation
# ------------------------------------------------------
echo ""
echo "[Step 3/3] Installing Python Libraries..."

# Check for requirements file
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: 'requirements.txt' not found in this folder."
    exit 1
fi

# CRITICAL: We use the pip INSIDE the venv directly.
# This ensures packages go into the venv without needing to 'source' actiavte in the script.
VENV_PIP="./venv/bin/pip"

echo "Upgrading internal pip..."
$VENV_PIP install --upgrade pip

echo "Installing project requirements..."
$VENV_PIP install -r requirements.txt

echo ""
echo "================================================"
echo "   Setup Complete! Ready to Race."
echo "================================================"
echo ""
echo "HOW TO START:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run your agent:"
echo "   python main.py"
echo ""