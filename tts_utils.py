"""
TTS Utility Module - Core TTS logic for batch processing and programmatic use
Contains all the core TTS functionality separated from API endpoints.
"""

import json
import numpy as np
import soundfile as sf
import io
import base64
import os
from datetime import datetime
from transformers import AutoModel
import logging
from typing import List, Optional, Dict, Any, Tuple
import re
from pydub import AudioSegment
import tempfile
from config import get_config, MODEL_CONFIG, PATHS, AUDIO_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class TTSProcessor:
    """Main TTS processor class for handling text-to-speech operations"""
    
    def __init__(self, model_repo_id: str = None, cache_dir: str = None, prompts_file: str = None):
        """
        Initialize TTS processor
        
        Args:
            model_repo_id: Hugging Face model repository ID
            cache_dir: Directory to cache the model
            prompts_file: Path to prompts.json file
        """
        self.model = None
        self.prompts = {}
        
        # Use config values if not provided
        self.model_repo_id = model_repo_id or MODEL_CONFIG["repo_id"]
        self.cache_dir = cache_dir or MODEL_CONFIG["cache_dir"]
        self.prompts_file = prompts_file or PATHS["prompts_file"]
        self.prompts_dir = PATHS["prompts_dir"]
        
        logger.info(f"Initializing TTS Processor with model: {self.model_repo_id}")
    
    def load_model(self):
        """Load the TTS model from Hugging Face"""
        if self.model is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading IndicF5 model from {self.model_repo_id}...")
        self.model = AutoModel.from_pretrained(
            self.model_repo_id,
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
            cache_dir=self.cache_dir
        )
        logger.info(f"Model {self.model_repo_id} loaded successfully")
    
    def load_prompts(self):
        """Load prompts from prompts.json file"""
        try:
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
        except FileNotFoundError:
            logger.warning(f"Prompts file {self.prompts_file} not found, prompts will be empty")
            self.prompts = {}
    
    def get_available_prompts(self) -> Dict[str, Any]:
        """Get all available prompts"""
        return self.prompts.copy()
    
    def validate_prompt_key(self, prompt_key: str) -> bool:
        """Validate if a prompt key exists"""
        return prompt_key in self.prompts
    
    def get_prompt_info(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific prompt"""
        return self.prompts.get(prompt_key)
    
    def generate_audio(self, text: str, prompt_key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate audio using the model
        
        Args:
            text: Text to convert to speech
            prompt_key: Key for the reference audio prompt
            
        Returns:
            Tuple of (audio_array, prompt_info)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if prompt_key not in self.prompts:
            raise ValueError(f"Prompt key '{prompt_key}' not found")
        
        prompt_info = self.prompts[prompt_key]
        
        # Construct full path to reference audio file
        ref_audio_path = os.path.join(self.prompts_dir, prompt_info["file"])
        
        # Check if the reference audio file exists
        if not os.path.exists(ref_audio_path):
            raise ValueError(f"Reference audio file {prompt_info['file']} not found at {ref_audio_path}")
        
        # Generate speech
        audio = self.model(
            text,
            ref_audio_path=ref_audio_path,
            ref_text=prompt_info["content"],
        )
        
        return audio, prompt_info
    
    def split_text_into_chunks(self, text: str, max_chars: int = 300) -> List[str]:
        """
        Split text into chunks of maximum character length, preserving sentence structure
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first (using period, exclamation, question mark)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # If a single sentence is longer than max_chars, split it further
            if len(sentence) > max_chars:
                # Split by commas or other punctuation
                sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                for part in sub_parts:
                    if len(part) > max_chars:
                        # If still too long, split by words
                        words = part.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk + " " + word) <= max_chars:
                                temp_chunk = temp_chunk + " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        if temp_chunk:
                            if len(current_chunk + " " + temp_chunk) <= max_chars:
                                current_chunk = current_chunk + " " + temp_chunk if current_chunk else temp_chunk
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = temp_chunk
                    else:
                        if len(current_chunk + " " + part) <= max_chars:
                            current_chunk = current_chunk + " " + part if current_chunk else part
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
            else:
                # Check if adding this sentence would exceed max_chars
                if len(current_chunk + " " + sentence) <= max_chars:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        # Add any remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def combine_audio_files(self, audio_files: List[str], output_path: str, sample_rate: int = 24000) -> bool:
        """
        Combine multiple audio files into a single file
        
        Args:
            audio_files: List of paths to audio files to combine
            output_path: Path where combined audio will be saved
            sample_rate: Sample rate for the output
            
        Returns:
            True if successful, False otherwise
        """
        try:
            combined_audio = AudioSegment.empty()
            
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    # Load audio file
                    audio_segment = AudioSegment.from_wav(audio_file)
                    
                    # Set frame rate to ensure consistency
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
                    
                    # Add to combined audio
                    combined_audio += audio_segment
                    
                    # Add a small pause between chunks (100ms)
                    silence = AudioSegment.silent(duration=100)
                    combined_audio += silence
            
            # Remove the last silence
            if len(combined_audio) > 100:
                combined_audio = combined_audio[:-100]
            
            # Export combined audio
            combined_audio.export(output_path, format="wav")
            
            return True
        except Exception as e:
            logger.error(f"Error combining audio files: {str(e)}")
            return False
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary audio files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
    
    def audio_to_base64(self, audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
        """
        Convert audio numpy array to base64 encoded string
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            format: Output format (wav, mp3, etc.)
            
        Returns:
            Base64 encoded audio string
        """
        buffer = io.BytesIO()
        
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Write audio to buffer
        sf.write(buffer, audio_data, sample_rate, format=format.upper())
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    
    def save_audio_file(self, audio_data: np.ndarray, file_path: str, sample_rate: int):
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data as numpy array
            file_path: Path where to save the file
            sample_rate: Sample rate of the audio
        """
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        sf.write(file_path, np.array(audio_data, dtype=np.float32), samplerate=sample_rate)
    
    def process_single_text(self, text: str, prompt_key: str, output_path: str = None, 
                           sample_rate: int = 24000, normalize: bool = True, 
                           max_chunk_chars: int = 300) -> Dict[str, Any]:
        """
        Process a single text and generate audio
        
        Args:
            text: Text to convert to speech
            prompt_key: Key for the reference audio prompt
            output_path: Path to save the audio file (optional)
            sample_rate: Sample rate for the output
            normalize: Whether to normalize the audio
            max_chunk_chars: Maximum characters per chunk for long texts
            
        Returns:
            Dictionary with processing results
        """
        try:
            start_time = datetime.now()
            
            # Validate prompt key
            if not self.validate_prompt_key(prompt_key):
                raise ValueError(f"Prompt key '{prompt_key}' not found. Available keys: {list(self.prompts.keys())}")
            
            # Check if text needs to be chunked
            if len(text) > max_chunk_chars:
                logger.info(f"Text length {len(text)} > {max_chunk_chars} chars, splitting into chunks")
                
                # Split text into chunks
                text_chunks = self.split_text_into_chunks(text, max_chars=max_chunk_chars)
                logger.info(f"Split text into {len(text_chunks)} chunks")
                
                # Generate audio for each chunk
                temp_audio_files = []
                try:
                    for i, chunk in enumerate(text_chunks):
                        logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                        
                        # Generate audio for this chunk
                        chunk_audio, prompt_info = self.generate_audio(chunk, prompt_key)
                        
                        # Normalize audio if requested
                        if normalize:
                            if chunk_audio.dtype == np.int16:
                                chunk_audio = chunk_audio.astype(np.float32) / 32768.0
                        
                        # Save chunk to temporary file
                        temp_filename = f"temp_chunk_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_file_path = os.path.join(temp_dir, temp_filename)
                            self.save_audio_file(chunk_audio, temp_file_path, sample_rate)
                            
                            # Copy to a persistent location for combining
                            persistent_temp_path = temp_file_path.replace(temp_dir, tempfile.gettempdir())
                            os.makedirs(os.path.dirname(persistent_temp_path), exist_ok=True)
                            sf.write(persistent_temp_path, np.array(chunk_audio, dtype=np.float32), samplerate=sample_rate)
                            temp_audio_files.append(persistent_temp_path)
                    
                    # Combine all audio chunks
                    if output_path:
                        success = self.combine_audio_files(temp_audio_files, output_path, sample_rate)
                        if not success:
                            raise Exception("Failed to combine audio chunks")
                        
                        # Load the combined audio
                        final_audio, sr = sf.read(output_path)
                    else:
                        # Create a temporary combined file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_combined:
                            success = self.combine_audio_files(temp_audio_files, temp_combined.name, sample_rate)
                            if not success:
                                raise Exception("Failed to combine audio chunks")
                            
                            # Load the combined audio
                            final_audio, sr = sf.read(temp_combined.name)
                            
                            # Clean up the temporary combined file
                            os.unlink(temp_combined.name)
                    
                    # Clean up temporary files
                    self.cleanup_temp_files(temp_audio_files)
                    
                except Exception as e:
                    # Clean up temporary files in case of error
                    self.cleanup_temp_files(temp_audio_files)
                    raise e
            
            else:
                # Process single text (original logic)
                logger.info(f"Text length {len(text)} <= {max_chunk_chars} chars, processing as single chunk")
                
                # Generate audio
                final_audio, prompt_info = self.generate_audio(text, prompt_key)
                
                # Normalize audio if requested
                if normalize:
                    if final_audio.dtype == np.int16:
                        final_audio = final_audio.astype(np.float32) / 32768.0
                
                # Save audio file if output path provided
                if output_path:
                    self.save_audio_file(final_audio, output_path, sample_rate)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "audio_data": final_audio,
                "sample_rate": sample_rate,
                "duration": duration,
                "prompt_info": prompt_info,
                "output_path": output_path,
                "message": "TTS generation successful"
            }
            
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"TTS generation failed: {str(e)}"
            }
    
    def process_batch_texts(self, texts: List[str], prompt_keys: List[str], 
                           output_dir: str = None, sample_rate: int = 24000, 
                           normalize: bool = True, max_chunk_chars: int = 300,
                           filename_prefix: str = "tts_batch") -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of texts to convert to speech
            prompt_keys: List of prompt keys (should match length of texts)
            output_dir: Directory to save audio files (optional)
            sample_rate: Sample rate for the output
            normalize: Whether to normalize the audio
            max_chunk_chars: Maximum characters per chunk for long texts
            filename_prefix: Prefix for generated filenames
            
        Returns:
            List of processing results for each text
        """
        if len(texts) != len(prompt_keys):
            raise ValueError("Number of texts must match number of prompt keys")
        
        results = []
        start_time = datetime.now()
        
        # Ensure output directory exists if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (text, prompt_key) in enumerate(zip(texts, prompt_keys)):
            try:
                logger.info(f"Processing batch item {i+1}/{len(texts)}")
                
                # Generate output path if output_dir provided
                output_path = None
                if output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{filename_prefix}_{i}_{prompt_key}_{timestamp}.wav"
                    output_path = os.path.join(output_dir, filename)
                
                # Process the text
                result = self.process_single_text(
                    text=text,
                    prompt_key=prompt_key,
                    output_path=output_path,
                    sample_rate=sample_rate,
                    normalize=normalize,
                    max_chunk_chars=max_chunk_chars
                )
                
                result["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch TTS generation failed for item {i}: {str(e)}")
                results.append({
                    "success": False,
                    "batch_index": i,
                    "error": str(e),
                    "message": f"TTS generation failed: {str(e)}"
                })
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Add summary statistics
        successful_count = sum(1 for r in results if r.get("success", False))
        failed_count = len(results) - successful_count
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed, total time: {total_duration:.2f}s")
        
        return results


# Convenience functions for quick usage
def create_tts_processor(model_repo_id: str = None, cache_dir: str = None, 
                        prompts_file: str = None, auto_load: bool = True) -> TTSProcessor:
    """
    Create and optionally auto-load a TTS processor
    
    Args:
        model_repo_id: Hugging Face model repository ID
        cache_dir: Directory to cache the model
        prompts_file: Path to prompts.json file
        auto_load: Whether to automatically load model and prompts
        
    Returns:
        TTSProcessor instance
    """
    processor = TTSProcessor(model_repo_id, cache_dir, prompts_file)
    
    if auto_load:
        processor.load_model()
        processor.load_prompts()
    
    return processor


def generate_speech(text: str, prompt_key: str, output_path: str = None,
                   model_repo_id: str = None, sample_rate: int = 24000) -> Dict[str, Any]:
    """
    Simple function to generate speech from text (creates processor automatically)
    
    Args:
        text: Text to convert to speech
        prompt_key: Key for the reference audio prompt
        output_path: Path to save the audio file (optional)
        model_repo_id: Hugging Face model repository ID
        sample_rate: Sample rate for the output
        
    Returns:
        Dictionary with processing results
    """
    processor = create_tts_processor(model_repo_id=model_repo_id, auto_load=True)
    return processor.process_single_text(text, prompt_key, output_path, sample_rate)


def generate_speech_batch(texts: List[str], prompt_keys: List[str], 
                         output_dir: str = None, model_repo_id: str = None,
                         sample_rate: int = 24000) -> List[Dict[str, Any]]:
    """
    Simple function to generate speech from multiple texts (creates processor automatically)
    
    Args:
        texts: List of texts to convert to speech
        prompt_keys: List of prompt keys (should match length of texts)
        output_dir: Directory to save audio files (optional)
        model_repo_id: Hugging Face model repository ID
        sample_rate: Sample rate for the output
        
    Returns:
        List of processing results
    """
    processor = create_tts_processor(model_repo_id=model_repo_id, auto_load=True)
    return processor.process_batch_texts(texts, prompt_keys, output_dir, sample_rate)
