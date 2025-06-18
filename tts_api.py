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
    
    # Initialize TTS processor
    tts_processor = TTSProcessor()
    
    # Load model and prompts
    tts_processor.load_model()
    tts_processor.load_prompts()
    
    logger.info(f"TTS Processor initialized with {len(tts_processor.prompts)} prompts")
    
    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        logger.warning(f"Configuration validation warnings: {config_errors}")
    
    # Ensure required directories exist
    os.makedirs(PATHS["output_dir"], exist_ok=True)
    os.makedirs(PATHS["prompts_dir"], exist_ok=True)
    
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
    text: str = Field(..., description="Text to convert to speech. Supports prompt tags: <prompt key='key'>text</prompt>")
    prompt_key: str = Field(..., description="Key for the reference audio prompt (used as default for text outside prompt tags)")
    output_format: Optional[str] = Field("wav", description="Output audio format (wav, mp3)")
    sample_rate: Optional[int] = Field(24000, description="Audio sample rate")
    normalize: Optional[bool] = Field(True, description="Whether to normalize audio")

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
    prompt_info: Optional[Dict[str, Any]] = None

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

class PromptsResponse(BaseModel):
    prompts: Dict[str, PromptInfo]
    total_count: int

# File metadata models for CRUD operations
class FileMetadata(BaseModel):
    filename: str
    prompt_key: str
    text_input: str
    created_datetime: str
    size_bytes: int
    format: str
    file_path: str

class CreateFileMetadataRequest(BaseModel):
    filename: str
    prompt_key: str
    text_input: str
    format: str = "wav"

class UpdateFileMetadataRequest(BaseModel):
    prompt_key: Optional[str] = None
    text_input: Optional[str] = None

class FileMetadataResponse(BaseModel):
    success: bool
    data: Optional[FileMetadata] = None
    message: str

class FileMetadataListResponse(BaseModel):
    success: bool
    data: List[FileMetadata]
    total_count: int
    message: str

class TTSPromptTaggedRequest(BaseModel):
    text: str = Field(..., description="Text with <prompt key='key'>text</prompt> tags")
    base_prompt_key: Optional[str] = Field(None, description="Default prompt key for text outside of prompt tags")
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

# File metadata management functions
def get_metadata_file_path() -> str:
    """Get the path to the metadata JSON file"""
    return os.path.join(PATHS["output_dir"], "files_metadata.json")

def load_file_metadata() -> Dict[str, FileMetadata]:
    """Load file metadata from JSON file"""
    metadata_file = get_metadata_file_path()
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {filename: FileMetadata(**metadata) for filename, metadata in data.items()}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load metadata file: {e}")
            return {}
    return {}

def save_file_metadata(metadata_dict: Dict[str, FileMetadata]) -> bool:
    """Save file metadata to JSON file"""
    try:
        metadata_file = get_metadata_file_path()
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        
        # Convert FileMetadata objects to dict for JSON serialization
        data_to_save = {filename: metadata.dict() for filename, metadata in metadata_dict.items()}
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata file: {e}")
        return False

def create_file_metadata(filename: str, prompt_key: str, text_input: str, format: str = "wav") -> FileMetadata:
    """Create metadata for a generated file"""
    file_path = os.path.join(PATHS["output_dir"], filename)
    size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    
    return FileMetadata(
        filename=filename,
        prompt_key=prompt_key,
        text_input=text_input,
        created_datetime=datetime.now().isoformat(),
        size_bytes=size_bytes,
        format=format,
        file_path=file_path
    )

def add_file_metadata(filename: str, prompt_key: str, text_input: str, format: str = "wav") -> bool:
    """Add metadata for a new file"""
    try:
        metadata_dict = load_file_metadata()
        metadata = create_file_metadata(filename, prompt_key, text_input, format)
        metadata_dict[filename] = metadata
        return save_file_metadata(metadata_dict)
    except Exception as e:
        logger.error(f"Failed to add file metadata for {filename}: {e}")
        return False

def update_file_metadata(filename: str, updates: Dict[str, Any]) -> bool:
    """Update metadata for an existing file"""
    try:
        metadata_dict = load_file_metadata()
        if filename not in metadata_dict:
            return False
        
        # Update the metadata
        current_metadata = metadata_dict[filename]
        for key, value in updates.items():
            if hasattr(current_metadata, key) and value is not None:
                setattr(current_metadata, key, value)
        
        return save_file_metadata(metadata_dict)
    except Exception as e:
        logger.error(f"Failed to update file metadata for {filename}: {e}")
        return False

def delete_file_metadata(filename: str) -> bool:
    """Delete metadata for a file"""
    try:
        metadata_dict = load_file_metadata()
        if filename in metadata_dict:
            del metadata_dict[filename]
            return save_file_metadata(metadata_dict)
        return True  # File wasn't in metadata, so deletion is successful
    except Exception as e:
        logger.error(f"Failed to delete file metadata for {filename}: {e}")
        return False

async def generate_audio_async(text: str, prompt_key: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """Async wrapper for TTS processor to run in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, tts_processor.generate_audio, text, prompt_key)

# New helper functions
import re

def has_prompt_tags(text: str) -> bool:
    """Check if text contains prompt tags"""
    pattern = r'<prompt\s+key\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</prompt>'
    return bool(re.search(pattern, text, re.DOTALL))

async def process_text_with_intelligent_routing(text: str, prompt_key: str, output_path: str, 
                                               sample_rate: int = 24000, normalize: bool = True,
                                               output_format: str = "wav") -> Dict[str, Any]:
    """
    Intelligently route text processing based on whether it contains prompt tags
    """
    # Check if text contains prompt tags
    if has_prompt_tags(text):
        logger.info("Text contains prompt tags, using prompt-tagged processing")
        # Use prompt tagged processing
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            tts_processor.process_prompt_tagged_text,
            text,
            prompt_key,  # base_prompt_key for untagged text
            output_path,
            sample_rate,
            normalize,
            300,  # max_chunk_chars
            200   # pause_duration
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
                prompt_key,
                output_path,
                sample_rate,
                normalize,
                300  # max_chunk_chars
            )
            return result
        else:
            # Use direct generation for short texts
            audio, prompt_info = await generate_audio_async(text, prompt_key)
            
            # Normalize audio if requested
            if normalize:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
            
            # Save audio file to disk
            tts_processor.save_audio_file(audio, output_path, sample_rate)
            
            return {
                "success": True,
                "audio_data": audio,
                "prompt_info": prompt_info,
                "message": "TTS generation successful"
            }

# API Routes
@app.get("/")
async def root():
    """Root endpoint - redirects to web interface"""
    return {
        "message": "TTS IndicF5 API Server",
        "version": "1.0.0",
        "model": "ai4bharat/IndicF5",
        "web_interface": "/web",
        "api_docs": "/docs",
        "api_base": "/api"
    }

@app.get("/web", response_class=HTMLResponse)
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
        "model": "ai4bharat/IndicF5",
        "endpoints": {
            "single_tts": "/api/tts",
            "batch_tts": "/api/tts/batch",
            "prompts": "/api/prompts",
            "health": "/api/health"
        },
        "features": {
            "prompt_tags": "The /api/tts endpoints automatically detect and process <prompt key='key'>text</prompt> tags"
        }
    }

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_processor is not None and tts_processor.model is not None,
        "prompts_loaded": tts_processor is not None and len(tts_processor.prompts) > 0,
        "available_prompts": len(tts_processor.prompts) if tts_processor else 0
    }

@api_router.get("/prompts", response_model=PromptsResponse)
async def get_prompts():
    """Get all available prompts"""
    formatted_prompts = {}
    for key, value in tts_processor.prompts.items():
        formatted_prompts[key] = PromptInfo(**value)
    
    return PromptsResponse(
        prompts=formatted_prompts,
        total_count=len(tts_processor.prompts)
    )

@api_router.get("/prompts/{prompt_key}/audio")
async def get_prompt_audio(prompt_key: str):
    """Get audio file for a specific prompt"""
    if prompt_key not in tts_processor.prompts:
        raise HTTPException(
            status_code=404, 
            detail=f"Prompt key '{prompt_key}' not found"
        )
    
    prompt_info = tts_processor.prompts[prompt_key]
    audio_file_path = f"{PATHS['prompts_dir']}/{prompt_info['file']}"

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
    
    Automatically detects and processes <prompt key="key">text</prompt> tags in the input text.
    If no prompt tags are found, uses the provided prompt_key for the entire text.
    """
    try:
        start_time = datetime.now()
        
        # Validate prompt key
        if request.prompt_key not in tts_processor.prompts:
            raise HTTPException(
                status_code=400, 
                detail=f"Prompt key '{request.prompt_key}' not found. Available keys: {list(tts_processor.prompts.keys())}"
            )
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Generate filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{request.prompt_key}_{timestamp}.{request.output_format}"
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Use intelligent routing based on prompt tags
        result = await process_text_with_intelligent_routing(
            text=request.text,
            prompt_key=request.prompt_key,
            output_path=file_path,
            sample_rate=request.sample_rate,
            normalize=request.normalize,
            output_format=request.output_format
        )
        
        if not result["success"]:
            raise Exception(result["message"])
        
        # Load the generated audio for base64 conversion
        audio_data = result["audio_data"]
        audio_base64 = audio_to_base64(audio_data, request.sample_rate, request.output_format)
        prompt_info = result["prompt_info"]
        
        # Add file metadata
        add_file_metadata(filename, request.prompt_key, request.text, request.output_format)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return TTSResponse(
            success=True,
            audio_base64=audio_base64,
            filename=filename,
            duration=duration,
            sample_rate=request.sample_rate,
            prompt_info=prompt_info,
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
    
    Each request automatically detects and processes <prompt key="key">text</prompt> tags.
    If no prompt tags are found in a request, uses the provided prompt_key for the entire text.
    """
    """Convert multiple texts to speech"""
    try:
        start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Prepare data for batch processing
        texts = [req.text for req in batch_request.requests]
        prompt_keys = [req.prompt_key for req in batch_request.requests]
        
        # Use TTS processor for batch processing
        results = []
        successful_requests = 0
        failed_requests = 0
        
        for i, request in enumerate(batch_request.requests):
            try:
                # Validate prompt key
                if not tts_processor.validate_prompt_key(request.prompt_key):
                    raise ValueError(f"Prompt key '{request.prompt_key}' not found")
                
                # Generate filename and path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tts_batch_{i}_{request.prompt_key}_{timestamp}.{request.output_format}"
                file_path = os.path.join(PATHS["output_dir"], filename)
                
                # Use intelligent routing based on prompt tags
                result = await process_text_with_intelligent_routing(
                    text=request.text,
                    prompt_key=request.prompt_key,
                    output_path=file_path,
                    sample_rate=request.sample_rate,
                    normalize=request.normalize,
                    output_format=request.output_format
                )
                
                if result["success"]:
                    # Convert to base64 for response
                    audio_base64 = audio_to_base64(result["audio_data"], request.sample_rate, request.output_format)
                    
                    # Add file metadata
                    add_file_metadata(filename, request.prompt_key, request.text, request.output_format)
                    
                    results.append(TTSResponse(
                        success=True,
                        audio_base64=audio_base64,
                        filename=filename,
                        sample_rate=request.sample_rate,
                        prompt_info=result["prompt_info"],
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
async def prompt_tagged_text_to_speech(request: TTSPromptTaggedRequest):
    """Convert text with prompt tags to speech"""
    try:
        start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        
        # Generate filename and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_prompt_tagged_{timestamp}.{request.output_format}"
        file_path = os.path.join(PATHS["output_dir"], filename)
        
        # Use TTS processor for prompt tagged processing
        result = tts_processor.process_prompt_tagged_text(
            text=request.text,
            base_prompt_key=request.base_prompt_key,
            output_path=file_path,
            sample_rate=request.sample_rate,
            normalize=request.normalize,
            max_chunk_chars=request.max_chunk_chars,
            pause_duration=request.pause_duration
        )
        
        if not result["success"]:
            raise Exception(result["message"])
        
        # Convert to base64 for response
        audio_base64 = audio_to_base64(result["audio_data"], request.sample_rate, request.output_format)
        
        # Add file metadata (use the first prompt key or base prompt key for metadata)
        prompt_key_for_metadata = request.base_prompt_key or "multi_prompt"
        add_file_metadata(filename, prompt_key_for_metadata, request.text, request.output_format)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return TTSPromptTaggedResponse(
            success=True,
            audio_base64=audio_base64,
            filename=filename,
            duration=duration,
            sample_rate=request.sample_rate,
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
        
        # Delete metadata if exists
        delete_file_metadata(filename)
        
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
                delete_file_metadata(filename)  # Remove metadata
                deleted_count += 1
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} files successfully",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")

# Add upload and delete endpoints for prompts
@api_router.post("/prompts/upload")
async def upload_prompt(file: UploadFile = File(...), name: str = Form(...), author: str = Form(...), content: str = Form("")):
    """Upload a new prompt audio file"""
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Ensure prompts directory exists
        os.makedirs(PATHS["prompts_dir"], exist_ok=True)
        
        # Generate filename
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'wav'
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.{file_extension}"
        file_path = os.path.join(PATHS["prompts_dir"], filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content_bytes = await file.read()
            buffer.write(content_bytes)
        
        # Update prompts.json
        prompt_key = safe_name.lower().replace(' ', '_')
        tts_processor.prompts[prompt_key] = {
            "author": author,
            "content": content or f"Sample content for {name}",
            "file": filename,
            "sample_rate": AUDIO_CONFIG["default_sample_rate"]
        }
        
        # Save updated prompts.json
        with open(PATHS["prompts_file"], "w", encoding="utf-8") as f:
            json.dump(tts_processor.prompts, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": f"Prompt '{name}' uploaded successfully",
            "prompt_key": prompt_key
        }
        
    except Exception as e:
        logger.error(f"Failed to upload prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload prompt: {str(e)}")

@api_router.delete("/prompts/{prompt_key}")
async def delete_prompt(prompt_key: str):
    """Delete a prompt"""
    try:
        if prompt_key not in tts_processor.prompts:
            raise HTTPException(status_code=404, detail=f"Prompt key '{prompt_key}' not found")
        
        # Remove audio file
        prompt_info = tts_processor.prompts[prompt_key]
        audio_file_path = os.path.join(PATHS["prompts_dir"], prompt_info["file"])
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        # Remove from prompts dict
        del tts_processor.prompts[prompt_key]
        
        # Save updated prompts.json
        with open(PATHS["prompts_file"], "w", encoding="utf-8") as f:
            json.dump(tts_processor.prompts, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "message": f"Prompt '{prompt_key}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prompt {prompt_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")

# File Metadata CRUD Endpoints
@api_router.get("/files/metadata", response_model=FileMetadataListResponse)
async def get_all_file_metadata():
    """Get metadata for all generated files"""
    try:
        metadata_dict = load_file_metadata()
        metadata_list = list(metadata_dict.values())
        
        return FileMetadataListResponse(
            success=True,
            data=metadata_list,
            total_count=len(metadata_list),
            message="File metadata retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to retrieve file metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file metadata: {str(e)}")

@api_router.get("/files/metadata/{filename}", response_model=FileMetadataResponse)
async def get_file_metadata(filename: str):
    """Get metadata for a specific file"""
    try:
        # Sanitize filename
        filename = os.path.basename(filename)
        
        metadata_dict = load_file_metadata()
        if filename not in metadata_dict:
            raise HTTPException(status_code=404, detail=f"Metadata for file '{filename}' not found")
        
        return FileMetadataResponse(
            success=True,
            data=metadata_dict[filename],
            message="File metadata retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve metadata for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file metadata: {str(e)}")

@api_router.post("/files/metadata", response_model=FileMetadataResponse)
async def create_file_metadata_entry(request: CreateFileMetadataRequest):
    """Create metadata entry for a file (if file exists but metadata doesn't)"""
    try:
        # Sanitize filename
        filename = os.path.basename(request.filename)
        
        # Check if file exists
        file_path = os.path.join(PATHS["output_dir"], filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        # Check if metadata already exists
        metadata_dict = load_file_metadata()
        if filename in metadata_dict:
            raise HTTPException(status_code=409, detail=f"Metadata for file '{filename}' already exists")
        
        # Validate prompt key
        if request.prompt_key not in tts_processor.prompts:
            raise HTTPException(status_code=400, detail=f"Prompt key '{request.prompt_key}' not found")
        
        # Create and save metadata
        success = add_file_metadata(filename, request.prompt_key, request.text_input, request.format)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create file metadata")
        
        # Return the created metadata
        metadata_dict = load_file_metadata()
        return FileMetadataResponse(
            success=True,
            data=metadata_dict[filename],
            message="File metadata created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create metadata for {request.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create file metadata: {str(e)}")

@api_router.put("/files/metadata/{filename}", response_model=FileMetadataResponse)
async def update_file_metadata_entry(filename: str, request: UpdateFileMetadataRequest):
    """Update metadata for a specific file"""
    try:
        # Sanitize filename
        filename = os.path.basename(filename)
        
        # Check if metadata exists
        metadata_dict = load_file_metadata()
        if filename not in metadata_dict:
            raise HTTPException(status_code=404, detail=f"Metadata for file '{filename}' not found")
        
        # Prepare updates
        updates = {}
        if request.prompt_key is not None:
            # Validate prompt key
            if request.prompt_key not in tts_processor.prompts:
                raise HTTPException(status_code=400, detail=f"Prompt key '{request.prompt_key}' not found")
            updates["prompt_key"] = request.prompt_key
        
        if request.text_input is not None:
            updates["text_input"] = request.text_input
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        # Update metadata
        success = update_file_metadata(filename, updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update file metadata")
        
        # Return updated metadata
        metadata_dict = load_file_metadata()
        return FileMetadataResponse(
            success=True,
            data=metadata_dict[filename],
            message="File metadata updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update metadata for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update file metadata: {str(e)}")

@api_router.delete("/files/metadata/{filename}", response_model=FileMetadataResponse)
async def delete_file_metadata_entry(filename: str):
    """Delete metadata for a specific file"""
    try:
        # Sanitize filename
        filename = os.path.basename(filename)
        
        # Check if metadata exists
        metadata_dict = load_file_metadata()
        if filename not in metadata_dict:
            raise HTTPException(status_code=404, detail=f"Metadata for file '{filename}' not found")
        
        # Store the metadata before deletion for response
        deleted_metadata = metadata_dict[filename]
        
        # Delete metadata
        success = delete_file_metadata(filename)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete file metadata")
        
        return FileMetadataResponse(
            success=True,
            data=deleted_metadata,
            message="File metadata deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete metadata for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file metadata: {str(e)}")

@api_router.delete("/files/metadata", response_model=FileMetadataListResponse)
async def clear_all_file_metadata():
    """Delete all file metadata"""
    try:
        metadata_dict = load_file_metadata()
        deleted_metadata = list(metadata_dict.values())
        deleted_count = len(deleted_metadata)
        
        # Clear all metadata
        success = save_file_metadata({})
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear file metadata")
        
        return FileMetadataListResponse(
            success=True,
            data=deleted_metadata,
            total_count=deleted_count,
            message=f"All file metadata cleared successfully ({deleted_count} entries removed)"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear all file metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear file metadata: {str(e)}")

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

if __name__ == "__main__":
    try:
        import uvicorn
        from config import SERVER_CONFIG
        uvicorn.run(
            app, 
            host=SERVER_CONFIG["host"], 
            port=SERVER_CONFIG["port"],
            reload=SERVER_CONFIG["reload"]
        )
    except ImportError:
        logger.error("uvicorn not installed. Please install with: pip install uvicorn[standard]")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

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
