#!/bin/bash
"""
NFL Draft Optimizer - Launch Script
Quick deployment for draft day
"""

echo "🏈 NFL Draft Optimizer - Launch Script"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install minimal required packages
echo "Installing dependencies..."
pip install pandas numpy scipy requests beautifulsoup4 sqlalchemy python-dotenv tqdm streamlit altair plotly --timeout=300

# Test the system
echo "Testing system..."
python simple_test.py

# Launch the app
echo "Launching NFL Draft Optimizer..."
echo "App will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop"

streamlit run complete_app.py --server.port 8501