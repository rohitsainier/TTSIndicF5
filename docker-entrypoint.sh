#!/bin/bash

# Docker entrypoint script for IndicF5 TTS API

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
