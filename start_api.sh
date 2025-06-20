#!/bin/bash

# TTS IndicF5 API Startup Script

echo "🚀 Starting TTS IndicF5 API Server..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the chatterbox directory
if [[ ! -f "tts_api.py" ]]; then
    echo -e "${RED}Error: Please run this script from the chatterbox directory${NC}"
    exit 1
fi

# Check if conda is available
if command -v conda >/dev/null 2>&1; then
    echo -e "${GREEN}Conda detected! Using conda environment...${NC}"
    
    # Check if indicf5 environment exists
    if ! conda env list | grep -q "^indicf5 "; then
        echo -e "${YELLOW}Creating conda environment 'indicf5'...${NC}"
        conda create -n indicf5 python=3.12 -y
    fi
    
    # Activate conda environment
    echo -e "${YELLOW}Activating conda environment 'indicf5'...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate indicf5
    
else
    echo -e "${YELLOW}Conda not found. Using virtual environment instead...${NC}"
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
fi

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
mkdir -p data/reference_voices

# Copy reference voices files if they exist and don't already exist in destination
if [ -d "prompts" ] && [ "$(ls -A prompts 2>/dev/null)" ]; then
    echo "📁 Copying reference voices files to data/reference_voices/..."
    for file in prompts/*.*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ ! -f "data/reference_voices/$filename" ]; then
                cp "$file" "data/reference_voices/" 2>/dev/null || true
            fi
        fi
    done
    if [ -f "data/reference_voices/prompts.json" ] && [ ! -f "data/reference_voices/reference_voices.json" ]; then
        mv data/reference_voices/prompts.json data/reference_voices/reference_voices.json 2>/dev/null || true
    fi
    echo "✅ Reference voices files copied (skipped existing files)"
fi

# Check if reference_voices.json exists
if [ ! -f "data/reference_voices/reference_voices.json" ]; then
    echo "⚠️  data/reference_voices/reference_voices.json not found. The API will start but no reference voices will be available."
    echo "💡 Make sure to place your reference_voices.json file and audio files in the data/reference_voices/ directory."
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
