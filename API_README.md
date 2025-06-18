# TTS IndicF5 REST API

A comprehensive REST API for Text-to-Speech using the IndicF5 model, supporting multiple Indian languages with batch processing capabilities.

## Features

- **Single TTS Request**: Convert individual text to speech
- **Batch TTS Processing**: Process multiple texts in a single request
- **Multiple Output Formats**: Support for WAV and MP3 formats
- **Base64 Audio Response**: Get audio data directly in API response
- **Server-side File Saving**: Save generated audio files on the server
- **Prompt Management**: List and manage available voice prompts
- **Health Check**: Monitor API status and model loading
- **CORS Support**: Ready for web application integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Linux/macOS (recommended)
./start_api.sh

# Windows
start_api.bat

# Direct execution
python tts_api.py
```

The startup scripts will automatically:
- Create and activate a virtual environment
- Install required dependencies
- Start the server

### 3. View API Documentation

Open your browser and go to:
- API Interface: `http://localhost:8000/web`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check

```http
GET /api/health
```

Check if the API is running and model is loaded.

### Get Available Prompts

```http
GET /api/prompts
```

Returns all available voice prompts with their metadata.

### Web Interface

```http
GET /web
```

Serves the built-in web interface for testing the API.

### Single TTS Request

```http
POST /api/tts
```

Convert a single text to speech.

**Request Body:**
```json
{
  "text": "కాకులు ఒక పొలానికి వెళ్లి అక్కడ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి.",
  "prompt_key": "TEL_F_WIKI_00001",
  "output_format": "wav",
  "sample_rate": 24000,
  "normalize": true
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "UklGRiQEAABXQVZFZm10IBAAAAABAAEA...",
  "filename": "tts_TEL_F_WIKI_00001_20241217_143022.wav",
  "duration": 2.35,
  "sample_rate": 24000,
  "message": "TTS generation successful",
  "prompt_info": {
    "author": "AI Female Wiki Reader",
    "content": "ఒక ఊరి లో ప్రతి సంవత్సరం...",
    "file": "prompts/TEL_F_WIKI_00001.wav",
    "sample_rate": 16000
  }
}
```

### Batch TTS Request

```http
POST /api/tts/batch
```

Process multiple TTS requests in a single call.

**Request Body:**
```json
{
  "requests": [
    {
      "text": "మొదటి వాక్యం - ఇది తెలుగు భాషలో ఉంది.",
      "prompt_key": "TEL_F_WIKI_00001",
      "output_format": "wav",
      "sample_rate": 24000,
      "normalize": true
    },
    {
      "text": "दूसरा वाक्य - यह हिंदी में है।",
      "prompt_key": "HIN_F_HAPPY_00001",
      "output_format": "wav",
      "sample_rate": 24000,
      "normalize": true
    }
  ],
  "return_as_zip": false
}
```

### Save TTS to Server

```http
POST /api/tts/save
```

Generate TTS and save the audio file on the server.

## Available Prompt Keys

Based on the current prompts.json file:

- `TEL_F_WIKI_00001` - Telugu Female Wiki Reader
- `HIN_F_HAPPY_00001` - Hindi Female Happy Vibes
- `PAN_F_HAPPY_00001` - Punjabi Female Happy Vibes

## Usage Examples

### Python Client

```python
import requests
import base64

# Single TTS request
response = requests.post("http://localhost:8000/api/tts", json={
    "text": "హలో, ఇది తెలుగు వాక్యం.",
    "prompt_key": "TEL_F_WIKI_00001",
    "output_format": "wav",
    "sample_rate": 24000,
    "normalize": true
})

if response.status_code == 200:
    data = response.json()
    
    # Save audio from base64
    audio_bytes = base64.b64decode(data['audio_base64'])
    with open(data['filename'], 'wb') as f:
        f.write(audio_bytes)
    
    print(f"Audio saved: {data['filename']}")
```

### cURL

```bash
# Single TTS request
curl -X POST "http://localhost:8000/api/tts" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "హలో, ఇది తెలుగు వాక్యం.",
       "prompt_key": "TEL_F_WIKI_00001",
       "output_format": "wav",
       "sample_rate": 24000,
       "normalize": true
     }'

# Get available prompts
curl -X GET "http://localhost:8000/api/prompts"

# Health check
curl -X GET "http://localhost:8000/api/health"
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

async function generateTTS() {
  try {
    const response = await axios.post('http://localhost:8000/api/tts', {
      text: 'హలో, ఇది తెలుగు వాక్యం.',
      prompt_key: 'TEL_F_WIKI_00001',
      output_format: 'wav',
      sample_rate: 24000,
      normalize: true
    });

    if (response.data.success) {
      // Save audio from base64
      const audioBuffer = Buffer.from(response.data.audio_base64, 'base64');
      fs.writeFileSync(response.data.filename, audioBuffer);
      console.log(`Audio saved: ${response.data.filename}`);
    }
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

generateTTS();
```

## Testing

Run the example client to test all API endpoints:

```bash
python api_client_example.py
```

This will:
1. Check API health
2. List available prompts
3. Test single TTS generation
4. Test batch TTS processing
5. Test server-side file saving

## Configuration

### Prompts Configuration

Add new prompts by updating `data/prompts/prompts.json`:

```json
{
  "NEW_PROMPT_KEY": {
    "author": "Voice Actor Name",
    "content": "Reference text content",
    "file": "data/prompts/audio_file.wav",
    "sample_rate": 16000
  }
}
```

### Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `PROMPTS_FILE`: Path to prompts.json (default: `data/prompts/prompts.json`)
- `PROMPTS_DIR`: Directory for prompt files (default: `data/prompts`)
- `OUTPUT_DIR`: Directory for generated files (default: `data/out`)
- `MODEL_CACHE_DIR`: Directory for model cache

## Performance

- **Model Loading**: ~10-30 seconds on startup
- **Single TTS**: ~2-5 seconds per request
- **Batch Processing**: Parallel processing for multiple requests
- **Memory Usage**: ~2-4GB for model in memory

## Error Handling

The API provides detailed error responses:

```json
{
  "detail": "Prompt key 'INVALID_KEY' not found. Available keys: ['TEL_F_WIKI_00001', 'HIN_F_HAPPY_00001', 'PAN_F_HAPPY_00001']"
}
```

Common error codes:
- `400`: Bad request (invalid prompt key, malformed input)
- `500`: Internal server error (model failure, file I/O issues)

## Production Deployment

For production deployment:

1. Use a production ASGI server:
```bash
pip install gunicorn
gunicorn tts_api:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. Set up reverse proxy (nginx):
```nginx
location /tts/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

3. Configure CORS for your domain:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Limitations

- Model loads on startup (requires restart for model changes)
- Single worker recommended due to model memory usage
- Audio files limited by available memory
- No authentication/rate limiting (add middleware as needed)

## License

This API wrapper is provided under the same license as the IndicF5 model.
