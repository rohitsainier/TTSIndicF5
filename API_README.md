# TTS IndicF5 REST API

A comprehensive REST API for Text-to-Speech using the IndicF5 model, supporting multiple Indian languages with batch processing capabilities.

## Features

- **Single TTS Request**: Convert individual text to speech
- **Batch TTS Processing**: Process multiple texts in a single request
- **Multiple Output Formats**: Support for WAV and MP3 formats
- **Base64 Audio Response**: Get audio data directly in API response
- **Server-side File Saving**: Save generated audio files on the server
- **File Metadata Management**: Full CRUD operations for generated file metadata
- **Prompt Management**: List and manage available reference prompts
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

### Core TTS Endpoints

### `POST /api/tts`
Convert single text to speech. Audio files are automatically saved to the server's output directory.

**Request:**
```json
{
  "text": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
  "prompt_key": "hin_f_happy",
  "output_format": "wav",
  "sample_rate": 24000,
  "normalize": true
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "UklGRgABAABXQVZFZm10...",
  "filename": "tts_hin_f_happy_20231201_143052.wav",
  "duration": 2.45,
  "sample_rate": 24000,
  "message": "TTS generation successful",
  "prompt_info": {
    "author": "AI4Bharat",
    "content": "नमस्ते",
    "file": "HIN_F_HAPPY_00001.wav",
    "sample_rate": 24000
  }
}
```

### `POST /api/tts/batch`
Process multiple TTS requests in a single call. All generated files are saved to the server.

**Request:**
```json
{
  "requests": [
    {
      "text": "पहला वाक्य",
      "prompt_key": "hin_f_happy",
      "output_format": "wav",
      "sample_rate": 24000,
      "normalize": true
    },
    {
      "text": "दूसरा वाक्य", 
      "prompt_key": "hin_f_happy",
      "output_format": "wav",
      "sample_rate": 24000,
      "normalize": true
    }
  ],
  "return_as_zip": false
}
```

### File Management Endpoints

### `GET /api/files`
List all generated audio files from the server's output directory.

**Response:**
```json
{
  "files": [
    {
      "filename": "tts_hin_f_happy_20231201_143052.wav",
      "filepath": "/path/to/output/tts_hin_f_happy_20231201_143052.wav",
      "size": 98304,
      "created_time": "2023-12-01T14:30:52",
      "format": "wav"
    }
  ],
  "total_count": 15,
  "total_size": 2457600
}
```

### `GET /api/files/{filename}`
Download a specific generated audio file from the server.

**Response:** Binary audio file with appropriate Content-Type header.

### `DELETE /api/files/{filename}`
Delete a specific generated audio file from the server.

**Response:**
```json
{
  "success": true,
  "message": "File tts_hin_f_happy_20231201_143052.wav deleted successfully"
}
```

### `DELETE /api/files`
Delete all generated audio files from the server.

**Response:**
```json
{
  "success": true,
  "message": "Deleted 15 files successfully",
  "deleted_count": 15
}
```

### Prompt Management Endpoints

### `GET /api/prompts`
List all available reference prompts.

### `GET /api/prompts/{prompt_key}/audio`
Download the audio file for a specific prompt.

### `POST /api/prompts/upload`
Upload a new reference prompt to the server.

**Request:** Multipart form data with:
- `file`: Audio file (wav, mp3, flac)
- `name`: Display name for the prompt
- `author`: Author/source of the prompt
- `content`: Optional text content for the prompt

**Response:**
```json
{
  "success": true,
  "message": "Prompt 'my_voice' uploaded successfully",
  "prompt_key": "my_voice"
}
```

### `DELETE /api/prompts/{prompt_key}`
Delete a reference prompt from the server.

**Response:**
```json
{
  "success": true,
  "message": "Prompt 'my_voice' deleted successfully"  
}
```

### File Metadata Management Endpoints

### `GET /api/files/metadata`
Get metadata for all generated files with information about file name, prompt used, text input, creation date, size, and format.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "filename": "tts_hin_f_happy_20231201_143052.wav",
      "prompt_key": "hin_f_happy",
      "text_input": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
      "created_datetime": "2023-12-01T14:30:52.123456",
      "size_bytes": 98304,
      "format": "wav",
      "file_path": "/path/to/output/tts_hin_f_happy_20231201_143052.wav"
    }
  ],
  "total_count": 15,
  "message": "File metadata retrieved successfully"
}
```

### `GET /api/files/metadata/{filename}`
Get metadata for a specific generated file.

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "tts_hin_f_happy_20231201_143052.wav",
    "prompt_key": "hin_f_happy",
    "text_input": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
    "created_datetime": "2023-12-01T14:30:52.123456",
    "size_bytes": 98304,
    "format": "wav",
    "file_path": "/path/to/output/tts_hin_f_happy_20231201_143052.wav"
  },
  "message": "File metadata retrieved successfully"
}
```

### `POST /api/files/metadata`
Create metadata entry for an existing file (useful if file exists but metadata is missing).

**Request:**
```json
{
  "filename": "existing_file.wav",
  "prompt_key": "hin_f_happy",
  "text_input": "Original text used to generate this file",
  "format": "wav"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "existing_file.wav",
    "prompt_key": "hin_f_happy",
    "text_input": "Original text used to generate this file",
    "created_datetime": "2023-12-01T14:30:52.123456",
    "size_bytes": 98304,
    "format": "wav",
    "file_path": "/path/to/output/existing_file.wav"
  },
  "message": "File metadata created successfully"
}
```

### `PUT /api/files/metadata/{filename}`
Update metadata for an existing file.

**Request:**
```json
{
  "prompt_key": "new_prompt_key",
  "text_input": "Updated text description"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "tts_hin_f_happy_20231201_143052.wav",
    "prompt_key": "new_prompt_key",
    "text_input": "Updated text description",
    "created_datetime": "2023-12-01T14:30:52.123456",
    "size_bytes": 98304,
    "format": "wav",
    "file_path": "/path/to/output/tts_hin_f_happy_20231201_143052.wav"
  },
  "message": "File metadata updated successfully"
}
```

### `DELETE /api/files/metadata/{filename}`
Delete metadata for a specific file (file itself remains intact).

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "tts_hin_f_happy_20231201_143052.wav",
    "prompt_key": "hin_f_happy",
    "text_input": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
    "created_datetime": "2023-12-01T14:30:52.123456",
    "size_bytes": 98304,
    "format": "wav",
    "file_path": "/path/to/output/tts_hin_f_happy_20231201_143052.wav"
  },
  "message": "File metadata deleted successfully"
}
```

### `DELETE /api/files/metadata`
Delete all file metadata entries (files themselves remain intact).

**Response:**
```json
{
  "success": true,
  "data": [],
  "total_count": 15,
  "message": "All file metadata cleared successfully (15 entries removed)"
}
```

### Other Endpoints

### `GET /api/health`
Check API health and model status.

### `GET /api/`
Get API information and available endpoints.

### `GET /web`
Access the web interface for the TTS system.

## File Storage

- **Generated Audio Files**: Automatically saved to `PATHS["output_dir"]` (default: `./data/out/`)
- **Reference Prompts**: Stored in `PATHS["prompts_dir"]` (default: `./data/prompts/`)  
- **File Formats**: WAV, MP3, FLAC supported
- **Naming Convention**: `tts_{prompt_key}_{timestamp}.{format}`

## File Metadata Storage

Generated file metadata is automatically stored in `./data/out/files_metadata.json` with the following structure:

```json
{
  "tts_hin_f_happy_20231201_143052.wav": {
    "filename": "tts_hin_f_happy_20231201_143052.wav",
    "prompt_key": "hin_f_happy",
    "text_input": "नमस्ते, मैं एक टेक्स्ट-टू-स्पीच सिस्टम हूं।",
    "created_datetime": "2023-12-01T14:30:52.123456",
    "size_bytes": 98304,
    "format": "wav",
    "file_path": "/path/to/output/tts_hin_f_happy_20231201_143052.wav"
  }
}
```

The metadata includes:
- **filename**: Name of the generated audio file
- **prompt_key**: Which reference prompt was used for generation
- **text_input**: Original text that was converted to speech
- **created_datetime**: ISO format timestamp of when file was created
- **size_bytes**: File size in bytes
- **format**: Audio format (wav, mp3, flac)
- **file_path**: Full path to the audio file

## Web Interface CRUD Operations

The enhanced web interface (`/web`) now provides comprehensive file metadata management with full CRUD operations:

### 🎯 **Enhanced File Display**
- **Original Text Preview**: See the text that was used to generate each file
- **Reference Prompt Information**: View which prompt was used for generation
- **Metadata Status Indicators**: Visual indicators for files with/without metadata
- **Interactive File Cards**: Rich file information with embedded audio players

### 🔧 **Metadata Management**
- **View Full Details**: Click "📋 Details" to see complete metadata information
- **Edit Metadata**: Click "✏️" to modify prompt key or text input
- **Add Missing Metadata**: Click "➕" to add metadata to files that don't have it
- **Delete Confirmation**: Enhanced deletion with metadata cleanup

### 📊 **Statistics & Analytics**
- **Metadata Coverage**: Real-time statistics showing metadata completion percentage
- **File Counts**: Track total files vs files with metadata
- **Storage Information**: File size and storage usage tracking

### 🛠️ **Bulk Operations**
- **Metadata Management Section**: Collapsible tools panel with advanced features
- **Export Metadata**: Download all metadata as JSON file
- **Clear All Metadata**: Bulk deletion of metadata (files remain intact)
- **Show Missing**: Quick view of files that need metadata
- **Refresh All Data**: Update all statistics and file listings

### 💡 **User Experience Enhancements**
- **Modal Dialogs**: Professional forms for metadata editing
- **Real-time Updates**: Automatic refresh after operations
- **Error Handling**: Clear feedback for all operations
- **Keyboard Shortcuts**: ESC key closes modals
- **Responsive Design**: Works on desktop and mobile devices

All operations in the web interface interact directly with the server's file system and metadata JSON file, ensuring data consistency and persistence.

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

### File Metadata CRUD Testing

Test the file metadata management features:

```bash
python test_file_metadata.py
```

This will test:
1. Creating metadata entries for existing files
2. Reading all file metadata
3. Reading specific file metadata
4. Updating metadata entries
5. Deleting metadata entries

### Web Interface Integration Testing

Test the enhanced web interface with metadata features:

```bash
python test_web_integration.py
```

This will verify:
1. API endpoint connectivity
2. File and metadata endpoint integration
3. Web interface functionality
4. Metadata CRUD operations availability

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
