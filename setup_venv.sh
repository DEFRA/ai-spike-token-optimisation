#!/bin/bash
# Setup script for LLM Prompt Compression experiment environment

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when you're done, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your API keys:"
echo "   cp .env.example .env"
echo "   nano .env  # or use your preferred editor"
echo ""
echo "2. Run the example:"
echo "   python example_usage.py"
echo ""
echo "3. Run the full experiment:"
echo "   python experiment.py"
