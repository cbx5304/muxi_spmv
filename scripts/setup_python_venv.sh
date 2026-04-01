#!/bin/bash
# Python virtual environment setup script
# Creates isolated Python environment with required dependencies

VENV_DIR="python_venv"

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating Python virtual environment..."

# Create virtual environment
python3 -m venv $VENV_DIR

# Activate and install packages
source $VENV_DIR/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install numpy scipy matplotlib pandas seaborn

echo ""
echo "=== Virtual Environment Setup Complete ==="
echo "Location: $VENV_DIR"
echo ""
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Installed packages:"
pip list