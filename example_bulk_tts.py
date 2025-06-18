#!/usr/bin/env python3
"""
Example script demonstrating how to use tts_utils.py for bulk TTS operations
This script shows various ways to use the TTS utilities programmatically
"""

import os
import sys
from pathlib import Path

# Add the project directory to the path so we can import our modules
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from tts_utils import TTSProcessor, create_tts_processor, generate_speech, generate_speech_batch


def example_1_simple_generation():
    """Example 1: Simple single text generation"""
    print("=== Example 1: Simple Single Text Generation ===")
    
    # Simple one-line generation (automatically loads model and prompts)
    result = generate_speech(
        text="Hello, this is a test of the TTS system.",
        prompt_key="hin_f_happy_00001",  # Make sure this prompt exists in your prompts.json
        output_path="output_simple.wav"
    )
    
    if result["success"]:
        print(f"✅ Success! Audio saved to: {result['output_path']}")
        print(f"   Duration: {result['duration']:.2f} seconds")
    else:
        print(f"❌ Failed: {result['message']}")


def example_2_processor_instance():
    """Example 2: Using TTSProcessor instance for multiple operations"""
    print("\n=== Example 2: Using TTSProcessor Instance ===")
    
    # Create and configure processor
    processor = create_tts_processor(auto_load=True)
    
    # Check available prompts
    print(f"Available prompts: {list(processor.prompts.keys())}")
    
    # Generate multiple audios with the same processor
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "And this is the third sentence with some additional content to make it longer."
    ]
    
    for i, text in enumerate(texts):
        result = processor.process_single_text(
            text=text,
            prompt_key="hin_f_happy_00001",  # Change this to match your available prompts
            output_path=f"output_example2_{i+1}.wav",
            sample_rate=24000,
            normalize=True
        )
        
        if result["success"]:
            print(f"✅ Generated audio {i+1}: output_example2_{i+1}.wav")
        else:
            print(f"❌ Failed audio {i+1}: {result['message']}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple texts"""
    print("\n=== Example 3: Batch Processing ===")
    
    # Texts to process
    texts = [
        "Welcome to the TTS system demonstration.",
        "This is a batch processing example with multiple texts being converted to speech simultaneously.",
        "The system can handle various lengths of text and will automatically chunk longer texts for better quality.",
        "Each text gets its own audio file with proper naming conventions."
    ]
    
    # Corresponding prompt keys (you can use different prompts for each text)
    prompt_keys = [
        "hin_f_happy_00001",  # Change these to match your available prompts
        "hin_f_happy_00001",
        "hin_f_happy_00001", 
        "hin_f_happy_00001"
    ]
    
    # Process batch
    results = generate_speech_batch(
        texts=texts,
        prompt_keys=prompt_keys,
        output_dir="batch_output",
        sample_rate=24000
    )
    
    # Show results
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Batch processing completed: {successful}/{total} successful")
    
    for i, result in enumerate(results):
        if result["success"]:
            print(f"✅ Text {i+1}: {result['output_path']}")
        else:
            print(f"❌ Text {i+1}: {result['message']}")


def example_4_advanced_processor_usage():
    """Example 4: Advanced processor usage with custom settings"""
    print("\n=== Example 4: Advanced Processor Usage ===")
    
    # Create processor with custom settings
    processor = TTSProcessor(
        # model_repo_id="custom/model",  # Use custom model if needed
        # cache_dir="./custom_cache",    # Use custom cache directory
        # prompts_file="./custom_prompts.json"  # Use custom prompts file
    )
    
    # Load model and prompts
    processor.load_model()
    processor.load_prompts()
    
    # Test text chunking
    long_text = """
    This is a very long text that will be automatically chunked by the TTS system.
    The system intelligently splits text at sentence boundaries, preserving natural speech flow.
    When sentences are too long, it further splits at punctuation marks like commas and semicolons.
    Finally, if needed, it splits at word boundaries to ensure each chunk is within the optimal length.
    This ensures high-quality speech synthesis even for very long documents or articles.
    """
    
    # Show how text would be chunked
    chunks = processor.split_text_into_chunks(long_text.strip(), max_chars=300)
    print(f"Long text would be split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")
    
    # Process the long text
    result = processor.process_single_text(
        text=long_text.strip(),
        prompt_key="hin_f_happy_00001",  # Change this to match your available prompts
        output_path="output_long_text.wav",
        max_chunk_chars=300
    )
    
    if result["success"]:
        print(f"✅ Long text processed successfully: {result['output_path']}")
        print(f"   Processing time: {result['duration']:.2f} seconds")
    else:
        print(f"❌ Long text processing failed: {result['message']}")


def example_5_base64_generation():
    """Example 5: Generate audio as base64 (useful for APIs)"""
    print("\n=== Example 5: Base64 Audio Generation ===")
    
    processor = create_tts_processor(auto_load=True)
    
    # Generate audio without saving to file
    result = processor.process_single_text(
        text="This audio will be converted to base64 format.",
        prompt_key="hin_f_happy_00001",  # Change this to match your available prompts
        output_path=None,  # No file output
        sample_rate=24000
    )
    
    if result["success"]:
        # Convert to base64
        audio_base64 = processor.audio_to_base64(
            result["audio_data"], 
            result["sample_rate"]
        )
        print(f"✅ Audio generated as base64 (length: {len(audio_base64)} characters)")
        print(f"   First 100 characters: {audio_base64[:100]}...")
        
        # You can now use this base64 string in APIs, databases, etc.
    else:
        print(f"❌ Base64 generation failed: {result['message']}")


def main():
    """Run all examples"""
    print("TTS Utils Examples - Bulk TTS Operations")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("tts_examples_output", exist_ok=True)
    os.chdir("tts_examples_output")
    
    try:
        # Run examples
        example_1_simple_generation()
        example_2_processor_instance()
        example_3_batch_processing()
        example_4_advanced_processor_usage()
        example_5_base64_generation()
        
        print("\n" + "=" * 50)
        print("🎉 All examples completed!")
        print("Check the 'tts_examples_output' directory for generated audio files.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("1. Proper prompts.json file with valid prompt keys")
        print("2. Corresponding audio files in the prompts directory")
        print("3. Required Python packages installed")


if __name__ == "__main__":
    main()
