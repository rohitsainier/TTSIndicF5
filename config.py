"""
Configuration settings for TTS IndicF5 API
"""

import os
from typing import Dict, Any

# Server Configuration
SERVER_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", 8000)),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "reload": os.getenv("RELOAD", "false").lower() == "true",
}

# Model Configuration
MODEL_CONFIG = {
    "repo_id": os.getenv("MODEL_REPO_ID", "ai4bharat/IndicF5"),
    "trust_remote_code": True,
    "cache_dir": os.getenv("MODEL_CACHE_DIR", None),
}

# Audio Configuration
AUDIO_CONFIG = {
    "default_sample_rate": int(os.getenv("DEFAULT_SAMPLE_RATE", 24000)),
    "default_format": os.getenv("DEFAULT_FORMAT", "wav"),
    "normalize_by_default": os.getenv("NORMALIZE_DEFAULT", "true").lower() == "true",
    "max_audio_length": int(os.getenv("MAX_AUDIO_LENGTH", 300)),  # seconds
}

# File Paths
PATHS = {
    "prompts_file": os.getenv("PROMPTS_FILE", "data/prompts/prompts.json"),
    "prompts_dir": os.getenv("PROMPTS_DIR", "data/prompts"),
    "output_dir": os.getenv("OUTPUT_DIR", "data/out"),
    "temp_dir": os.getenv("TEMP_DIR", "/tmp/tts_api"),
}

# API Configuration
API_CONFIG = {
    "title": "TTS IndicF5 API",
    "description": "REST API for Text-to-Speech using IndicF5 model with support for Indian languages",
    "version": "1.0.0",
    "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", 10)),
    "enable_cors": os.getenv("ENABLE_CORS", "true").lower() == "true",
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "serve_web_interface": os.getenv("SERVE_WEB_INTERFACE", "true").lower() == "true",
    "web_interface_path": os.getenv("WEB_INTERFACE_PATH", "web.html"),
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.getenv("LOG_FILE", None),  # None means log to stdout
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "model_loading_timeout": int(os.getenv("MODEL_LOADING_TIMEOUT", 300)),  # seconds
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT", 60)),  # seconds
    "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", 5)),
}

# Security Configuration (for production)
SECURITY_CONFIG = {
    "enable_rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true",
    "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
    "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", 3600)),  # seconds
    "api_key_required": os.getenv("API_KEY_REQUIRED", "false").lower() == "true",
    "api_key": os.getenv("API_KEY", None),
}

def get_config() -> Dict[str, Any]:
    """Get all configuration settings"""
    return {
        "server": SERVER_CONFIG,
        "model": MODEL_CONFIG,
        "audio": AUDIO_CONFIG,
        "paths": PATHS,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "security": SECURITY_CONFIG,
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if prompts file exists
    if not os.path.exists(PATHS["prompts_file"]):
        errors.append(f"Prompts file not found: {PATHS['prompts_file']}")
    
    # Check if prompts directory exists
    if not os.path.exists(PATHS["prompts_dir"]):
        errors.append(f"Prompts directory not found: {PATHS['prompts_dir']}")
    
    # Check if output directory is writable
    try:
        os.makedirs(PATHS["output_dir"], exist_ok=True)
        test_file = os.path.join(PATHS["output_dir"], ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        errors.append(f"Output directory not writable: {PATHS['output_dir']} - {e}")
    
    # Validate numeric settings
    if AUDIO_CONFIG["default_sample_rate"] <= 0:
        errors.append("Default sample rate must be positive")
    
    if API_CONFIG["max_batch_size"] <= 0:
        errors.append("Max batch size must be positive")
    
    return errors
