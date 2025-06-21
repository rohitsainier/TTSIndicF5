from fastapi import FastAPI, HTTPException, Response, APIRouter, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import numpy as np
import soundfile as sf
import io
import base64
import os
from datetime import datetime
import logging
from contextlib import asynccontextmanager
# System monitoring imports
import psutil
import GPUtil
import time
# Async support imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import get_config, validate_config, MODEL_CONFIG, PATHS, AUDIO_CONFIG, API_CONFIG, LOGGING_CONFIG
from tts_utils import TTSProcessor
from huggingface_hub import login

# Configure logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    filename=LOGGING_CONFIG["file"]
)
logger = logging.getLogger(__name__)

# Global TTS processor instance
tts_processor = None

# Thread pool executor for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)  # Limit to 2 workers to prevent overload

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_processor
    logger.info("Initializing TTS Processor...")

    if MODEL_CONFIG["hf_token"]:
        login(token=MODEL_CONFIG["hf_token"])

    # Initialize TTS processor
    tts_processor = TTSProcessor()
    
    # Load model and referenceVoices
    tts_processor.load_model()
    tts_processor.load_reference_voices()

    logger.info(f"TTS Processor initialized with {len(tts_processor.reference_voices)} referenceVoices")

    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        logger.warning(f"Configuration validation warnings: {config_errors}")
    
    # Ensure required directories exist
    os.makedirs(PATHS["output_dir"], exist_ok=True)
    os.makedirs(PATHS["reference_voices_dir"], exist_ok=True)

    yield
    # Shutdown
    logger.info("Shutting down...")
    executor.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan
)

# Add CORS middleware
if API_CONFIG["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Create API router for all API endpoints
api_router = APIRouter(prefix="/api", tags=["TTS API"])

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech. Supports prompt tags: <refvoice key='key'>text</refvoice>")
    reference_voice_key: str = Field(..., description="Key for the reference audio prompt (used as default for text outside prompt tags)")
    output_format: Optional[str] = Field("wav", description="Output audio format (wav, mp3)")
    sample_rate: Optional[int] = Field(24000, description="Audio sample rate")
    normalize: Optional[bool] = Field(True, description="Whether to normalize audio")
    seed: Optional[int] = Field(-1, description="Random seed for reproducible generation (-1 for random)")

class TTSBatchRequest(BaseModel):
    requests: List[TTSRequest] = Field(..., description="List of TTS requests to process")
    return_as_zip: Optional[bool] = Field(False, description="Return all audio files as a zip archive")

class TTSResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    filename: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: int
    message: Optional[str] = None
    reference_voice_info: Optional[Dict[str, Any]] = None
    used_seed: Optional[int] = None

class TTSBatchResponse(BaseModel):
    success: bool
    results: List[TTSResponse]
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: Optional[float] = None

class PromptInfo(BaseModel):
    author: str
    content: str
    file: str
    sample_rate: int

class ReferenceVoicesResponse(BaseModel):
    reference_voices: Dict[str, PromptInfo]
    total_count: int

class TTSPromptTaggedRequest(BaseModel):
    text: str = Field(..., description="Text with <refvoice key='key'>text</refvoice> tags")
    base_reference_voice_key: Optional[str] = Field(None, description="Default prompt key for text outside of prompt tags")
    output_format: Optional[str] = Field("wav", description="Output audio format (wav, mp3)")
    sample_rate: Optional[int] = Field(24000, description="Audio sample rate")
    normalize: Optional[bool] = Field(True, description="Whether to normalize audio")
    max_chunk_chars: Optional[int] = Field(300, description="Maximum characters per chunk for long texts")
    pause_duration: Optional[int] = Field(200, description="Duration of pause between segments in milliseconds")

class TTSPromptTaggedResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    filename: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: int
    message: Optional[str] = None
    segments_processed: Optional[int] = None
    segment_results: Optional[List[Dict[str, Any]]] = None

# Utility functions (keeping only API-specific ones)
def audio_to_base64(audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """Convert audio numpy array to base64 encoded string"""
    return tts_processor.audio_to_base64(audio_data, sample_rate, format)

async def generate_audio_async(text: str, reference_voice_key: str, seed: int = -1, sample_rate: int = 24000) -> tuple[np.ndarray, Dict[str, Any]]:
    """Async wrapper for TTS processor to run in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, tts_processor.generate_audio, text, reference_voice_key, seed, sample_rate)

# New helper functions
import re

def has_reference_voice_tags(text: str) -> bool:
    """Check if text contains prompt tags"""
    pattern = r'<refvoice\s+key\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</refvoice>'
    return bool(re.search(pattern, text, re.DOTALL))

async def process_text_with_intelligent_routing(text: str, reference_voice_key: str, output_path: str, 
                                               sample_rate: int = 24000, normalize: bool = True,
                                               output_format: str = "wav", seed: int = -1) -> Dict[str, Any]:
    """
    Intelligently route text processing based on whether it contains prompt tags
    """
    # Check if text contains prompt tags
    if has_reference_voice_tags(text):
        logger.info("Text contains prompt tags, using prompt-tagged processing")
        # Use prompt tagged processing
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            tts_processor.process_reference_voice_tagged_text,
            text,
            reference_voice_key,  # base_reference_voice_key for untagged text
            output_path,
            sample_rate,
            normalize,
            300,  # max_chunk_chars
            200,   # pause_duration
            seed,
        )
        return result
    else:
        logger.info("Text does not contain prompt tags, using normal processing")
        # Use normal processing logic
        if len(text) > 300:
            # Use chunked processing for long texts
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                tts_processor.process_single_text,
                text,
                reference_voice_key,
                output_path,
                sample_rate,
                normalize,
                300,  # max_chunk_chars
                seed
            )
            return result
        else:
            # Use direct generation for short texts
            audio, reference_voice_info = await generate_audio_async(text, reference_voice_key, seed)
            
            # Normalize audio if requested
            if normalize:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
            
            # Save audio file to disk
            tts_processor.save_audio_file(audio, output_path, sample_rate)
            
            return {
                "success": True,
                "audio_data": audio,
                "reference_voice_info": reference_voice_info,
                "message": "TTS generation successful"
            }

# API Routes
@app.get("/")
async def root():
    """Root endpoint - redirects to web interface"""
    return {
        "message": "TTS IndicF5 API Server",
        "version": "1.0.0",
        "model": "hareeshbabu82/TeluguIndicF5",
        "web_interface": "/api_demo.html",
        "api_docs": "/docs",
        "api_base": "/api",
        "health": "/health - Health check",
    }

@app.get("/api_demo.html", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface"""
    if not API_CONFIG["serve_web_interface"]:
        raise HTTPException(status_code=404, detail="Web interface is disabled")
    
    try:
        web_file_path = API_CONFIG["web_interface_path"]
        with open(web_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Web interface file not found: {API_CONFIG['web_interface_path']}")
    except Exception as e:
        logger.error(f"Failed to load web interface: {e}")
        raise HTTPException(status_code=500, detail="Failed to load web interface")

# API Router Endpoints
@api_router.get("/")
async def api_root():
    """API root endpoint with information"""
    return {
        "message": "TTS IndicF5 API",
        "version": "1.0.0",
        "model": "hareeshbabu82/TeluguIndicF5",
        "endpoints": {
            "single_tts": "/api/tts",
            "batch_tts": "/api/tts/batch",
            "reference_voices": "/api/referenceVoices",
            "health": "/api/health"
        },
        "features": {
            "reference_voice_tags": "The /api/tts endpoints automatically detect and process <refvoice key='key'>text</refvoice> tags"
        }
    }

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_processor is not None and tts_processor.model is not None,
        "reference_voices_loaded": tts_processor is not None and len(tts_processor.reference_voices) > 0,
        "available_reference_voices": len(tts_processor.reference_voices) if tts_processor else 0
    }

@api_router.get("/referenceVoices", response_model=ReferenceVoicesResponse)
async def get_reference_voices():
    """Get all available reference voices"""
    formatted_voices = {}
    for key, value in tts_processor.reference_voices.items():
        formatted_voices[key] = PromptInfo(**value)

    return ReferenceVoicesResponse(
        reference_voices=formatted_voices,
        total_count=len(tts_processor.reference_voices)
    )

@api_router.get("/referenceVoices/{voice_key}/audio")
async def get_reference_voice_audio(voice_key: str):
    """Get audio file for a specific reference voice"""
    if voice_key not in tts_processor.reference_voices:
        raise HTTPException(
            status_code=404, 
            detail=f"Voice key '{voice_key}' not found"
        )

    voice_info = tts_processor.reference_voices[voice_key]
    audio_file_path = f"{PATHS['reference_voices_dir']}/{voice_info['file']}"

    # Check if file exists
    if not os.path.exists(audio_file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Audio file not found: {audio_file_path}"
        )
    
    return FileResponse(
        path=audio_file_path,
        media_type="audio/wav",
        filename=os.path.basename(audio_file_path)
    )

@api_router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert single text to speech with intelligent prompt tag detection.
    
    Automatically detects and processes <refvoice key="key">text</refvoice> tags in the input text.
    If no prompt tags are found, uses the provided reference_voice_key for the entire text.
    """
    try:
        start_time = datetime.now()

        # Validate referenceVoices key
        if request.reference_voice_key not in tts_processor.reference_voices:
            raise HTTPException(
                status_code=400, 
                detail=f"Voice key '{request.reference_voice_key}' not found. Available keys: {list(tts_processor.reference_voices.keys())}"
            )
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Generate filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{request.reference_voice_key}_{timestamp}.{request.output_format}"
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Use intelligent routing based on prompt tags
        result = await process_text_with_intelligent_routing(
            text=request.text,
            reference_voice_key=request.reference_voice_key,
            output_path=file_path,
            sample_rate=request.sample_rate,
            normalize=request.normalize,
            output_format=request.output_format,
            seed=request.seed
        )
        
        if not result["success"]:
            raise Exception(result["message"])
        
        # Load the generated audio for base64 conversion
        audio_data = result["audio_data"]
        audio_base64 = audio_to_base64(audio_data, request.sample_rate or 24000, request.output_format or "wav")
        reference_voice_info = result["reference_voice_info"] if "reference_voice_info" in result else result.get("segment_results", [])[0] if "segment_results" in result else None
        used_seed = reference_voice_info.get("used_seed") if isinstance(reference_voice_info, dict) else None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return TTSResponse(
            success=True,
            audio_base64=audio_base64,
            filename=filename,
            duration=duration,
            sample_rate=request.sample_rate or 24000,
            reference_voice_info=reference_voice_info,
            used_seed=used_seed,
            message="TTS generation successful"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@api_router.post("/tts/batch", response_model=TTSBatchResponse)
async def batch_text_to_speech(batch_request: TTSBatchRequest):
    """
    Convert multiple texts to speech with intelligent prompt tag detection.
    
    Each request automatically detects and processes <refvoice key="key">text</refvoice> tags.
    If no prompt tags are found in a request, uses the provided reference_voice_key for the entire text.
    """
    """Convert multiple texts to speech"""
    try:
        start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Prepare data for batch processing
        texts = [req.text for req in batch_request.requests]
        reference_voice_keys = [req.reference_voice_key for req in batch_request.requests]
        
        # Use TTS processor for batch processing
        results = []
        successful_requests = 0
        failed_requests = 0
        
        for i, request in enumerate(batch_request.requests):
            try:
                # Validate prompt key
                if not tts_processor.validate_reference_voice_key(request.reference_voice_key):
                    raise ValueError(f"Prompt key '{request.reference_voice_key}' not found")
                
                # Generate filename and path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tts_batch_{i}_{request.reference_voice_key}_{timestamp}.{request.output_format}"
                file_path = os.path.join(PATHS["output_dir"], filename)
                
                # Use intelligent routing based on prompt tags
                result = await process_text_with_intelligent_routing(
                    text=request.text,
                    reference_voice_key=request.reference_voice_key,
                    output_path=file_path,
                    sample_rate=request.sample_rate,
                    normalize=request.normalize,
                    output_format=request.output_format
                )
                
                if result["success"]:
                    # Convert to base64 for response
                    audio_base64 = audio_to_base64(result["audio_data"], request.sample_rate or 24000, request.output_format or "wav")
                    
                    results.append(TTSResponse(
                        success=True,
                        audio_base64=audio_base64,
                        filename=filename,
                        sample_rate=request.sample_rate,
                        reference_voice_info=result["reference_voice_info"],
                        message="TTS generation successful"
                    ))
                    successful_requests += 1
                else:
                    results.append(TTSResponse(
                        success=False,
                        sample_rate=request.sample_rate,
                        message=result["message"]
                    ))
                    failed_requests += 1
                    
            except Exception as e:
                logger.error(f"Batch TTS generation failed for request {i}: {str(e)}")
                results.append(TTSResponse(
                    success=False,
                    sample_rate=request.sample_rate,
                    message=f"TTS generation failed: {str(e)}"
                ))
                failed_requests += 1
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        return TTSBatchResponse(
            success=failed_requests == 0,
            results=results,
            total_requests=len(batch_request.requests),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration
        )
        
    except Exception as e:
        logger.error(f"Batch TTS processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch TTS processing failed: {str(e)}")

@api_router.post("/tts/prompt-tagged", response_model=TTSPromptTaggedResponse)
async def reference_voice_tagged_text_to_speech(request: TTSPromptTaggedRequest):
    """Convert text with prompt tags to speech"""
    try:
        start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Generate filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_reference_voice_tagged_{timestamp}.{request.output_format}"
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Use TTS processor for prompt tagged processing
        result = tts_processor.process_reference_voice_tagged_text(
            text=request.text,
            base_reference_voice_key=request.base_reference_voice_key,
            output_path=file_path,
            sample_rate=request.sample_rate,
            normalize=request.normalize,
            max_chunk_chars=request.max_chunk_chars,
            pause_duration=request.pause_duration,
            seed=request.seed
        )
        
        if not result["success"]:
            raise Exception(result["message"])
        
        # Convert to base64 for response
        audio_base64 = audio_to_base64(result["audio_data"], request.sample_rate or 24000, request.output_format or "wav")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return TTSPromptTaggedResponse(
            success=True,
            audio_base64=audio_base64,
            filename=filename,
            duration=duration,
            sample_rate=request.sample_rate or 24000,
            segments_processed=result.get("segments_processed", 0),
            segment_results=result.get("segment_results", []),
            message="Prompt tagged TTS generation successful"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prompt tagged TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prompt tagged TTS generation failed: {str(e)}")

# New file management endpoints
class FileInfo(BaseModel):
    filename: str
    filepath: str
    size: int
    created_time: str
    format: str

class FilesListResponse(BaseModel):
    files: List[FileInfo]
    total_count: int
    total_size: int

@api_router.get("/files", response_model=FilesListResponse)
async def list_generated_files():
    """List all generated audio files"""
    try:
        output_dir = PATHS["output_dir"]
        if not os.path.exists(output_dir):
            return FilesListResponse(files=[], total_count=0, total_size=0)
        
        files = []
        total_size = 0
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.wav', '.mp3', '.flac')):
                file_stat = os.stat(file_path)
                file_size = file_stat.st_size
                total_size += file_size
                
                # Get file format from extension
                file_format = filename.split('.')[-1].lower()
                
                files.append(FileInfo(
                    filename=filename,
                    filepath=file_path,
                    size=file_size,
                    created_time=datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    format=file_format
                ))
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x.created_time, reverse=True)
        
        return FilesListResponse(
            files=files,
            total_count=len(files),
            total_size=total_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@api_router.get("/files/{filename}")
async def download_file(filename: str):
    """Download a generated audio file"""
    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)
        
        # Check if file has a valid audio extension
        valid_extensions = {'.wav', '.mp3', '.flac'}
        file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
        if file_ext not in valid_extensions:
            logger.warning(f"Invalid file extension requested: {filename}")
            raise HTTPException(status_code=403, detail="Only audio files are allowed")
        
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Convert to absolute paths for comparison
        abs_file_path = os.path.abspath(file_path)
        abs_output_dir = os.path.abspath(PATHS["output_dir"])
        
        # Security check - ensure the file is in the output directory
        if not abs_file_path.startswith(abs_output_dir):
            logger.warning(f"Security check failed for file: {filename}")
            raise HTTPException(status_code=403, detail="Access denied - file outside allowed directory")
        
        if not os.path.exists(abs_file_path):
            logger.warning(f"File not found: {abs_file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.isfile(abs_file_path):
            logger.warning(f"Path is not a file: {abs_file_path}")
            raise HTTPException(status_code=404, detail="Path is not a file")
        
        # Determine media type based on file extension
        media_type_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac'
        }
        media_type = media_type_map.get(file_ext, 'application/octet-stream')
        
        logger.info(f"Serving file: {abs_file_path}")
        return FileResponse(
            path=abs_file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@api_router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a generated audio file"""
    try:
        # Security check - prevent directory traversal attacks
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=403, detail="Invalid filename")
        
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Additional security check - ensure the resolved path is still within output directory
        output_dir_real = os.path.realpath(PATHS["output_dir"])
        file_path_real = os.path.realpath(file_path)
        
        if not file_path_real.startswith(output_dir_real):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        
        return {
            "success": True,
            "message": f"File {filename} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@api_router.delete("/files")
async def clear_all_files():
    """Delete all generated audio files"""
    try:
        output_dir = PATHS["output_dir"]
        if not os.path.exists(output_dir):
            return {
                "success": True,
                "message": "No files to delete",
                "deleted_count": 0
            }
        
        deleted_count = 0
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.wav', '.mp3', '.flac')):
                os.remove(file_path)
                deleted_count += 1
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} files successfully",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")

# Add upload and delete endpoints for reference voices
@api_router.post("/referenceVoices/upload")
async def upload_reference_voice(file: UploadFile = File(...), name: str = Form(...), author: str = Form(...), content: str = Form("")):
    """Upload a new reference voice audio file"""
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Ensure reference voices directory exists
        os.makedirs(PATHS["reference_voices_dir"], exist_ok=True)

        # Generate filename
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'wav'
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.{file_extension}"
        file_path = os.path.join(PATHS["reference_voices_dir"], filename)

        # Save file
        with open(file_path, "wb") as buffer:
            content_bytes = await file.read()
            buffer.write(content_bytes)

        # Update reference_voices.json
        voice_key = safe_name.lower().replace(' ', '_')
        tts_processor.reference_voices[voice_key] = {
            "author": author,
            "content": content or f"Sample content for {name}",
            "file": filename,
            "sample_rate": AUDIO_CONFIG["default_sample_rate"]
        }

        # Save updated reference_voices.json
        with open(PATHS["reference_voices_file"], "w", encoding="utf-8") as f:
            json.dump(tts_processor.reference_voices, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"Voice '{name}' uploaded successfully",
            "voice_key": voice_key
        }
        
    except Exception as e:
        logger.error(f"Failed to upload voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

@api_router.delete("/referenceVoices/{voice_key}")
async def delete_reference_voice(voice_key: str):
    """Delete a reference voice"""
    try:
        if voice_key not in tts_processor.reference_voices:
            raise HTTPException(status_code=404, detail=f"Voice key '{voice_key}' not found")

        # Remove audio file
        voice_info = tts_processor.reference_voices[voice_key]
        audio_file_path = os.path.join(PATHS["reference_voices_dir"], voice_info["file"])
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        # Remove from referenceVoices dict
        del tts_processor.reference_voices[voice_key]

        # Save updated reference_voices.json
        with open(PATHS["reference_voices_file"], "w", encoding="utf-8") as f:
            json.dump(tts_processor.reference_voices, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"Voice '{voice_key}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice {voice_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

# System monitoring endpoint
@api_router.get("/system/monitor")
async def system_monitor():
    """Get system CPU, memory, and GPU usage"""
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_total = memory_info.total
        memory_available = memory_info.available
        
        # GPU usage (if available)
        gpu_usage = 0
        gpu_memory_total = 0
        gpu_memory_free = 0
        gpu_memory_used = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100
            gpu_memory_total = gpus[0].memoryTotal
            gpu_memory_free = gpus[0].memoryFree
            gpu_memory_used = gpus[0].memoryUsed
        
        return {
            "success": True,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "gpu_usage": gpu_usage,
            "gpu_memory_total": gpu_memory_total,
            "gpu_memory_free": gpu_memory_free,
            "gpu_memory_used": gpu_memory_used
        }
    except Exception as e:
        logger.error(f"Failed to get system monitor data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system monitor data: {str(e)}")

# Include API router first (before static mounts) to ensure API routes take precedence
app.include_router(api_router)

# Mount static files for web interface (after API routes to avoid conflicts)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Note: Removed /data mount to avoid conflicts with /api/files endpoints
# Files should be accessed through /api/files/{filename} instead

# Add test endpoint for chunking demo
@api_router.post("/tts/chunk-demo")
async def chunk_demo(text: str):
    """Demo endpoint to show how text would be chunked"""
    try:
        chunks = tts_processor.split_text_into_chunks(text, max_chars=300)
        
        return {
            "success": True,
            "original_text": text,
            "original_length": len(text),
            "will_be_chunked": len(text) > 300,
            "num_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_number": i + 1,
                    "text": chunk,
                    "length": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ],
            "message": f"Text {'will be split into' if len(chunks) > 1 else 'will be processed as'} {len(chunks)} chunk{'s' if len(chunks) > 1 else ''}"
        }
    except Exception as e:
        logger.error(f"Chunk demo failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunk demo failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    from config import SERVER_CONFIG
    
    logger.info("🚀 ========================================")
    logger.info("🚀 IndicF5 TTS API Server Starting...")
    logger.info("🚀 ========================================")
    
    parser = argparse.ArgumentParser(description="IndicF5 TTS REST API Server")
    parser.add_argument("--host", default=SERVER_CONFIG["host"], help="Host to bind to")
    parser.add_argument("--port", type=int, default=SERVER_CONFIG["port"], help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=SERVER_CONFIG["reload"], help="Enable auto-reload for development")

    args = parser.parse_args()
    
    logger.info(f"🌐 Starting server on {args.host}:{args.port}")
    logger.info(f"🔄 Auto-reload: {args.reload}")
    try:
        import uvicorn
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except ImportError:
        logger.error("uvicorn not installed. Please install with: pip install uvicorn[standard]")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")