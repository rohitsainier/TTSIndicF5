#!/bin/bash

# Docker entrypoint script for IndicF5 TTS API

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create data/reference_voices directory if it doesn't exist
mkdir -p data/reference_voices

# Copy reference voices files if they exist and don't already exist in destination
if [ -d "prompts" ] && [ "$(ls -A prompts 2>/dev/null)" ]; then
    echo "📁 Copying reference voices files to data/reference_voices/..."
    copied_files=0
    skipped_files=0
    
    for file in prompts/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ ! -f "data/reference_voices/$filename" ]; then
                if cp "$file" "data/reference_voices/" 2>/dev/null; then
                    echo "  ✓ Copied: $filename"
                    copied_files=$((copied_files + 1))
                else
                    echo "  ✗ Failed to copy: $filename"
                fi
            else
                echo "  - Skipped (already exists): $filename"
                skipped_files=$((skipped_files + 1))
            fi
        fi
    done
    
    # Handle prompts.json -> reference_voices.json conversion
    if [ -f "data/reference_voices/prompts.json" ] && [ ! -f "data/reference_voices/reference_voices.json" ]; then
        if mv "data/reference_voices/prompts.json" "data/reference_voices/reference_voices.json" 2>/dev/null; then
            echo "  ✓ Renamed prompts.json to reference_voices.json"
        else
            echo "  ✗ Failed to rename prompts.json to reference_voices.json"
        fi
    fi
    
    echo "✅ Reference voices files processing completed ($copied_files copied, $skipped_files skipped)"
else
    echo "📁 No prompts directory found or it's empty - skipping file copy"
fi

# Check if reference_voices.json exists
if [ ! -f "data/reference_voices/reference_voices.json" ]; then
    echo "⚠️  data/reference_voices/reference_voices.json not found. The API will start but no reference voices will be available."
    echo "💡 Make sure to place your reference_voices.json file and audio files in the data/reference_voices/ directory."
fi

echo -e "${GREEN}🎤 Starting IndicF5 TTS API${NC}"
echo "================================"

# Parse command line arguments
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Additional options
RELOAD=""
if [[ "${ENVIRONMENT:-production}" == "development" ]]; then
    RELOAD="--reload"
fi

echo -e "${YELLOW}Starting API server...${NC}"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo ""

# Start the server using uvicorn directly for better control
exec uvicorn tts_api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    $RELOAD
