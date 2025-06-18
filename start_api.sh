#!/bin/bash

# TTS IndicF5 API Startup Script

echo "🚀 Starting TTS IndicF5 API Server..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Virtual environment management
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check if required packages are installed
echo "📦 Checking dependencies..."
if ! python -c "import fastapi, uvicorn, transformers, soundfile" 2>/dev/null; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
fi

# Create required directories
mkdir -p data/out
mkdir -p data/prompts

# Check if prompts.json exists
if [ ! -f "data/prompts/prompts.json" ]; then
    echo "⚠️  data/prompts/prompts.json not found. The API will start but no prompts will be available."
    echo "💡 Make sure to place your prompts.json file and audio files in the data/prompts/ directory."
fi

echo "✅ Starting API server on http://localhost:8000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🌐 Web Interface: http://localhost:8000/web"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API server with activated virtual environment
python tts_api.py
