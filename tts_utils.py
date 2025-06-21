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
from pydub import AudioSegment, silence
import tempfile
import torch
from config import get_config, MODEL_CONFIG, PATHS, AUDIO_CONFIG
from f5_tts.api import F5TTS
from f5_tts.model.utils import seed_everything

# Configure logging
logger = logging.getLogger(__name__)

class TTSProcessor:
    """Main TTS processor class for handling text-to-speech operations"""
    
    def __init__(self, model_repo_id: str = None, cache_dir: str = None, reference_voices_file: str = None):
        """
        Initialize TTS processor
        
        Args:
            model_repo_id: Hugging Face model repository ID
            cache_dir: Directory to cache the model
            reference_voices_file: Path to reference_voices.json file
        """
        self.model = None
        self.modelF5TTS = None
        self.reference_voices = {}
        
        # Use config values if not provided
        self.model_repo_id = model_repo_id or MODEL_CONFIG["repo_id"]
        self.cache_dir = cache_dir or MODEL_CONFIG["cache_dir"]
        self.reference_voices_file = reference_voices_file or PATHS["reference_voices_file"]
        self.reference_voices_dir = PATHS["reference_voices_dir"]
        
        logger.info(f"Initializing TTS Processor with model: {self.model_repo_id}")
    
    def load_model(self):
        """Load the TTS model from Hugging Face"""
        if self.model is not None and self.modelF5TTS is not None:
            logger.info("Models already loaded")
            return

        if not self.model:            
            logger.info(f"Loading IndicF5 model from {self.model_repo_id}...")
            self.model = AutoModel.from_pretrained(
                self.model_repo_id,
                trust_remote_code=MODEL_CONFIG["trust_remote_code"],
                cache_dir=self.cache_dir
            )
            logger.info(f"Model {self.model_repo_id} loaded successfully")

        if not self.modelF5TTS:
            logger.info(f"Loading F5TTS model using F5TTS_Base...")
            self.modelF5TTS = F5TTS(model="F5TTS_Base", hf_cache_dir=self.cache_dir)
            logger.info(f"F5TTS model F5TTS_Base loaded successfully")

    def load_reference_voices(self):
        """Load referenceVoices from reference_voices.json file"""
        try:
            with open(self.reference_voices_file, "r", encoding="utf-8") as f:
                self.reference_voices = json.load(f)
            logger.info(f"Loaded {len(self.reference_voices)} referenceVoices from {self.reference_voices_file}")
        except FileNotFoundError:
            logger.warning(f"ReferenceVoices file {self.reference_voices_file} not found, referenceVoices will be empty")
            self.reference_voices = {}

    def get_available_reference_voices(self) -> Dict[str, Any]:
        """Get all available referenceVoices"""
        return self.reference_voices.copy()

    def validate_reference_voice_key(self, reference_voice_key: str) -> bool:
        """Validate if a referenceVoices key exists"""
        return reference_voice_key in self.reference_voices

    def get_reference_voices_info(self, reference_voice_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific referenceVoices"""
        return self.reference_voices.get(reference_voice_key)

    def generate_audio(self, text: str, reference_voice_key: str, seed: int = -1, sample_rate: int = 24000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate audio using the model
        
        Args:
            text: Text to convert to speech
            reference_voice_key: Key for the reference audio prompt
            seed: Random seed for reproducible generation (-1 for random)
            sample_rate: Sample rate for the output audio

        Returns:
            Tuple of (audio_array, reference_voice_info)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.modelF5TTS is None:
            raise RuntimeError("F5TTS model not loaded. Call load_model() first.")

        if reference_voice_key not in self.reference_voices:
            raise ValueError(f"ReferenceVoices key '{reference_voice_key}' not found")

        # Set inference seed
        if seed < 0 or seed > 2**31 - 1:
            print("Warning: Seed must be in range 0 ~ 2147483647. Using random seed instead.")
            seed = np.random.randint(0, 2**31 - 1)
        seed_everything(seed)
        used_seed = seed

        reference_voice_info = self.reference_voices[reference_voice_key]
        
        # Construct full path to reference audio file
        ref_audio_path = os.path.join(self.reference_voices_dir, reference_voice_info["file"])

        # Check if the reference audio file exists
        if not os.path.exists(ref_audio_path):
            raise ValueError(f"Reference audio file {reference_voice_info['file']} not found at {ref_audio_path}")
        
        # Check what is the model of Reference voice
        if "model" not in reference_voice_info or reference_voice_info["model"] in ["IndicF5"]:            
            logger.info("using IndicF5 model...")
            # Generate speech using IndicF5 model
            audio = self.model(
                text,
                ref_audio_path=ref_audio_path,
                ref_text=reference_voice_info["content"],
            ) 
        elif reference_voice_info["model"] in ["F5TTS"]:
            logger.info("using F5TTS model...")
            # Generate speech using F5TTS model
            wav = self.modelF5TTS.infer(
                    ref_file=ref_audio_path,
                    ref_text=reference_voice_info["content"],
                    gen_text=text,
                    target_rms=0.1,
                    seed=used_seed,
                    # file_wave=str(files("data").joinpath("out/api_out.wav")),
                    # file_spec=str(files("data").joinpath("out/api_out.png")),
            )
            if wav is None:
                raise ValueError(f"F5TTS model failed to generate audio")
            audio = convert_wav_and_remove_silence(audio=wav,  # type: ignore
                        sample_rate=sample_rate,
                        )
            # audio = wav  # Assuming wav is already in the correct format
        else:
            raise ValueError(f"Invalid model type in reference voice info: {reference_voice_info}")
        
        # Add used seed to reference voice info
        reference_voice_info_with_seed = reference_voice_info.copy()
        reference_voice_info_with_seed["used_seed"] = used_seed

        return audio, reference_voice_info_with_seed # type: ignore

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
    
    def process_single_text(self, text: str, reference_voice_key: str, output_path: str = None, 
                           sample_rate: int = 24000, normalize: bool = True, 
                           max_chunk_chars: int = 300, seed: int = -1) -> Dict[str, Any]:
        """
        Process a single text and generate audio
        
        Args:
            text: Text to convert to speech
            reference_voice_key: Key for the reference audio prompt
            output_path: Path to save the audio file (optional)
            sample_rate: Sample rate for the output
            normalize: Whether to normalize the audio
            max_chunk_chars: Maximum characters per chunk for long texts
            seed: Random seed for reproducible generation (-1 for random)
            
        Returns:
            Dictionary with processing results
        """
        try:
            start_time = datetime.now()
            
            # Validate referenceVoices key
            if not self.validate_reference_voice_key(reference_voice_key):
                raise ValueError(f"ReferenceVoices key '{reference_voice_key}' not found. Available keys: {list(self.reference_voices.keys())}")

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
                        chunk_audio, reference_voice_info = self.generate_audio(chunk, reference_voice_key, seed, sample_rate)
                        
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
                final_audio, reference_voice_info = self.generate_audio(text, reference_voice_key, seed, sample_rate)

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
                "reference_voice_info": reference_voice_info,
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
    
    def process_batch_texts(self, texts: List[str], reference_voice_keys: List[str], 
                           output_dir: str = None, sample_rate: int = 24000, 
                           normalize: bool = True, max_chunk_chars: int = 300,
                           filename_prefix: str = "tts_batch") -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of texts to convert to speech
            reference_voice_keys: List of prompt keys (should match length of texts)
            output_dir: Directory to save audio files (optional)
            sample_rate: Sample rate for the output
            normalize: Whether to normalize the audio
            max_chunk_chars: Maximum characters per chunk for long texts
            filename_prefix: Prefix for generated filenames
            
        Returns:
            List of processing results for each text
        """
        if len(texts) != len(reference_voice_keys):
            raise ValueError("Number of texts must match number of prompt keys")
        
        results = []
        start_time = datetime.now()
        used_seed = -1
        
        # Ensure output directory exists if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (text, reference_voice_key) in enumerate(zip(texts, reference_voice_keys)):
            try:
                logger.info(f"Processing batch item {i+1}/{len(texts)}")
                
                # Generate output path if output_dir provided
                output_path = None
                if output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{filename_prefix}_{i}_{reference_voice_key}_{timestamp}.wav"
                    output_path = os.path.join(output_dir, filename)
                
                # Process the text
                result = self.process_single_text(
                    text=text,
                    reference_voice_key=reference_voice_key,
                    output_path=output_path,
                    sample_rate=sample_rate,
                    normalize=normalize,
                    max_chunk_chars=max_chunk_chars,
                    seed=used_seed
                )
                
                result["batch_index"] = i
                used_seed = result.get("reference_voice_info", {}).get("used_seed", -1)
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
    
    def parse_reference_voice_tags(self, text: str, base_reference_voice_key: str = None) -> List[Dict[str, str]]:
        """
        Parse text containing <refvoice key="reference_voice_key">text to speak</refvoice> tags
        
        Args:
            text: Input text containing prompt tags
            base_reference_voice_key: Default prompt key for text outside of prompt tags
            
        Returns:
            List of dictionaries with 'reference_voice_key' and 'text' keys
        """
        # If no base_reference_voice_key is provided, only extract tagged content
        if base_reference_voice_key is None:
            # Regular expression to match <refvoice key="reference_voice_key">text</refvoice> tags
            pattern = r'<refvoice\s+key\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</refvoice>'
            matches = re.findall(pattern, text, re.DOTALL)
            
            segments = []
            for reference_voice_key, segment_text in matches:
                segments.append({
                    'reference_voice_key': reference_voice_key.strip(),
                    'text': segment_text.strip()
                })
            
            return segments
        
        # With base_reference_voice_key, we need to handle text outside tags too
        segments = []
        pattern = r'<refvoice\s+key\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</refvoice>'
        
        # Find all matches with their positions
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        last_end = 0
        for match in matches:
            # Add any text before this tag using base_reference_voice_key
            before_text = text[last_end:match.start()].strip()
            if before_text:
                segments.append({
                    'reference_voice_key': base_reference_voice_key,
                    'text': before_text
                })
            
            # Add the tagged content
            reference_voice_key = match.group(1).strip()
            segment_text = match.group(2).strip()
            if segment_text:
                segments.append({
                    'reference_voice_key': reference_voice_key,
                    'text': segment_text
                })
            
            last_end = match.end()
        
        # Add any remaining text after the last tag
        after_text = text[last_end:].strip()
        if after_text:
            segments.append({
                'reference_voice_key': base_reference_voice_key,
                'text': after_text
            })
        
        return segments
    
    def process_reference_voice_tagged_text(self, text: str, base_reference_voice_key: str = None, output_path: str = None, 
                                 sample_rate: int = 24000, normalize: bool = True,
                                 max_chunk_chars: int = 300, pause_duration: int = 200, seed: int = -1) -> Dict[str, Any]:
        """
        Process text containing multiple <refvoice key="reference_voice_key">text</refvoice> tags
        and generate combined audio
        
        Args:
            text: Input text with prompt tags
            base_reference_voice_key: Default prompt key for text outside of prompt tags
            output_path: Path to save the combined audio file (optional)
            sample_rate: Sample rate for the output
            normalize: Whether to normalize the audio
            max_chunk_chars: Maximum characters per chunk for long texts
            pause_duration: Duration of pause between segments in milliseconds
            
        Returns:
            Dictionary with processing results including combined audio
        """
        try:
            start_time = datetime.now()
            
            # Parse the prompt tags
            segments = self.parse_reference_voice_tags(text, base_reference_voice_key)
            
            if not segments:
                if base_reference_voice_key is None:
                    raise ValueError("No valid <refvoice key='key'>text</refvoice> tags found in input text")
                else:
                    raise ValueError("No text content found to process")
            
            logger.info(f"Found {len(segments)} prompt segments to process")
            
            # Validate all prompt keys exist
            for segment in segments:
                if not self.validate_reference_voice_key(segment['reference_voice_key']):
                    raise ValueError(f"ReferenceVoices key '{segment['reference_voice_key']}' not found. Available keys: {list(self.reference_voices.keys())}")

            # Generate audio for each segment
            temp_audio_files = []
            segment_results = []
            
            try:
                for i, segment in enumerate(segments):
                    logger.info(f"Processing segment {i+1}/{len(segments)} with referenceVoices '{segment['reference_voice_key']}'...")

                    # Process this segment
                    result = self.process_single_text(
                        text=segment['text'],
                        reference_voice_key=segment['reference_voice_key'],
                        output_path=None,  # Don't save individual segments
                        sample_rate=sample_rate,
                        normalize=normalize,
                        max_chunk_chars=max_chunk_chars,
                        seed=seed
                    )
                    used_seed = result.get("reference_voice_info", {}).get("used_seed", -1)
                    
                    if not result['success']:
                        raise Exception(f"Failed to process segment {i+1}: {result.get('error', 'Unknown error')}")
                    
                    # Save segment to temporary file
                    temp_filename = f"temp_segment_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                    temp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
                    
                    self.save_audio_file(result['audio_data'], temp_file_path, sample_rate)
                    temp_audio_files.append(temp_file_path)
                    
                    # Store segment result
                    segment_results.append({
                        'segment_index': i,
                        'reference_voice_key': segment['reference_voice_key'],
                        'text': segment['text'],
                        'success': True,
                        'audio_duration': len(result['audio_data']) / sample_rate,
                        'used_seed': used_seed,
                    })
                
                # Combine all audio segments with pauses
                if output_path:
                    success = self.combine_audio_files_with_pauses(temp_audio_files, output_path, sample_rate, pause_duration)
                    if not success:
                        raise Exception("Failed to combine audio segments")
                    
                    # Load the combined audio
                    final_audio, sr = sf.read(output_path)
                else:
                    # Create a temporary combined file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_combined:
                        success = self.combine_audio_files_with_pauses(temp_audio_files, temp_combined.name, sample_rate, pause_duration)
                        if not success:
                            raise Exception("Failed to combine audio segments")
                        
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
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": True,
                "audio_data": final_audio,
                "sample_rate": sample_rate,
                "duration": duration,
                "segments_processed": len(segments),
                "segment_results": segment_results,
                "output_path": output_path,
                "message": f"Successfully processed {len(segments)} segments and combined audio"
            }
            
        except Exception as e:
            logger.error(f"Prompt-tagged TTS generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Prompt-tagged TTS generation failed: {str(e)}"
            }
    
    def combine_audio_files_with_pauses(self, audio_files: List[str], output_path: str, 
                                      sample_rate: int = 24000, pause_duration: int = 200) -> bool:
        """
        Combine multiple audio files into a single file with custom pause duration
        
        Args:
            audio_files: List of paths to audio files to combine
            output_path: Path where combined audio will be saved
            sample_rate: Sample rate for the output
            pause_duration: Duration of pause between segments in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            combined_audio = AudioSegment.empty()
            
            for i, audio_file in enumerate(audio_files):
                if os.path.exists(audio_file):
                    # Load audio file
                    audio_segment = AudioSegment.from_wav(audio_file)
                    
                    # Set frame rate to ensure consistency
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
                    
                    # Add to combined audio
                    combined_audio += audio_segment
                    
                    # Add pause between segments (except after the last one)
                    if i < len(audio_files) - 1 and pause_duration > 0:
                        silence = AudioSegment.silent(duration=pause_duration)
                        combined_audio += silence
            
            # Export combined audio
            combined_audio.export(output_path, format="wav")
            
            return True
        except Exception as e:
            logger.error(f"Error combining audio files with pauses: {str(e)}")
            return False

    # ...existing code...


# Convenience functions for quick usage
def create_tts_processor(model_repo_id: str = None, cache_dir: str = None, 
                        reference_voices_file: str = None, auto_load: bool = True) -> TTSProcessor:
    """
    Create and optionally auto-load a TTS processor
    
    Args:
        model_repo_id: Hugging Face model repository ID
        cache_dir: Directory to cache the model
        reference_voices_file: Path to reference_voices.json file
        auto_load: Whether to automatically load model and referenceVoices

    Returns:
        TTSProcessor instance
    """
    processor = TTSProcessor(model_repo_id, cache_dir, reference_voices_file)

    if auto_load:
        processor.load_model()
        processor.load_reference_voices()

    return processor


def generate_speech(text: str, reference_voice_key: str, output_path: str = None,
                   model_repo_id: str = None, sample_rate: int = 24000) -> Dict[str, Any]:
    """
    Simple function to generate speech from text (creates processor automatically)
    
    Args:
        text: Text to convert to speech
        reference_voice_key: Key for the reference audio prompt
        output_path: Path to save the audio file (optional)
        model_repo_id: Hugging Face model repository ID
        sample_rate: Sample rate for the output
        
    Returns:
        Dictionary with processing results
    """
    processor = create_tts_processor(model_repo_id=model_repo_id, auto_load=True)
    return processor.process_single_text(text, reference_voice_key, output_path, sample_rate)


def generate_speech_batch(texts: List[str], reference_voices_keys: List[str], 
                         output_dir: str = None, model_repo_id: str = None,
                         sample_rate: int = 24000) -> List[Dict[str, Any]]:
    """
    Simple function to generate speech from multiple texts (creates processor automatically)
    
    Args:
        texts: List of texts to convert to speech
        reference_voices_keys: List of referenceVoices keys (should match length of texts)
        output_dir: Directory to save audio files (optional)
        model_repo_id: Hugging Face model repository ID
        sample_rate: Sample rate for the output
        
    Returns:
        List of processing results
    """
    processor = create_tts_processor(model_repo_id=model_repo_id, auto_load=True)
    return processor.process_batch_texts(texts, reference_voices_keys, output_dir, sample_rate)


def generate_speech_from_reference_voice_tags(text: str, base_reference_voice_key: str = None, output_path: str = None,
                                   model_repo_id: str = None, sample_rate: int = 24000,
                                   pause_duration: int = 200) -> Dict[str, Any]:
    """
    Simple function to generate speech from text containing prompt tags
    
    Args:
        text: Input text with <refvoice key="key">text</refvoice> tags
        base_reference_voice_key: Default referenceVoices key for text outside of prompt tags
        output_path: Path to save the combined audio file (optional)
        model_repo_id: Hugging Face model repository ID
        sample_rate: Sample rate for the output
        pause_duration: Duration of pause between segments in milliseconds
        
    Returns:
        Dictionary with processing results
    """
    processor = create_tts_processor(model_repo_id=model_repo_id, auto_load=True)
    return processor.process_reference_voice_tagged_text(
        text=text, 
        base_reference_voice_key=base_reference_voice_key,
        output_path=output_path, 
        sample_rate=sample_rate,
        pause_duration=pause_duration
    )

def is_english_or_latin(text: str) -> bool:
    # Check for English/Latin characters (basic ASCII + common Latin extensions)
    latin_pattern = re.compile(r'^[a-zA-Z0-9äöüÄÖÜßèéêëàáâãäåæçéêëìíîïñòóôõöøùúûüýÿ\s\.,!?\'`";\-:()&\^%\$#@~<>/]+$')
    return bool(latin_pattern.match(text.strip()))

def convert_wav_and_remove_silence(audio: np.ndarray, sample_rate: int = 24000, 
                                   remove_silence: bool = True) -> np.ndarray:
    # Convert to pydub format and remove silence if needed
    # Handle different input types - audio might be a tuple from F5TTS
    if isinstance(audio, tuple):
        # If it's a tuple, take the first element (usually the audio data)
        audio = audio[0]
    
    # Convert to numpy array if it's not already
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
    buffer.seek(0)
    audio_segment = AudioSegment.from_file(buffer, format="wav")

    if remove_silence:
        non_silent_segs = silence.split_on_silence(
            audio_segment,
            min_silence_len=1000,
            silence_thresh=-50,
            keep_silence=500,
            seek_step=10,
        )
        non_silent_wave = sum(non_silent_segs, AudioSegment.silent(duration=0))
        audio_segment = non_silent_wave

    # Normalize loudness
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    audio_segment = audio_segment.apply_gain(change_in_dBFS)
    print(f"Audio loudness normalized to {target_dBFS} dBFS")

    return np.array(audio_segment.get_array_of_samples())        