#!/usr/bin/env python3
"""
Example client script for TTS IndicF5 API
Demonstrates how to use the REST API for single and batch TTS requests

Note: This script expects the API server to be running with reference voices configured in data/reference_voices/
"""

import requests
import json
import base64
import os
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

def save_audio_from_base64(audio_base64: str, filename: str):
    """Save base64 encoded audio to file"""
    audio_bytes = base64.b64decode(audio_base64)
    with open(filename, 'wb') as f:
        f.write(audio_bytes)
    print(f"Audio saved to: {filename}")

def test_health_check():
    """Test API health check"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        print("✓ API is healthy")
        print(json.dumps(response.json(), indent=2))
    else:
        print("✗ API health check failed")
        print(response.text)
    print()

def test_get_reference_voices():
    """Test getting available reference voices"""
    print("=== Getting Available Reference Voices ===")
    response = requests.get(f"{API_BASE_URL}/referenceVoices")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total_count']} reference voices:")
        for key, voice in data['reference_voices'].items():
            print(f"  - {key}: {voice['author']}")
        return list(data['reference_voices'].keys())
    else:
        print("✗ Failed to get reference voices")
        print(response.text)
        return []

def test_single_tts(voice_keys):
    """Test single TTS request with chunking support"""
    print("=== Testing Single TTS Request (with Chunking) ===")

    if not voice_keys:
        print("No voices available for testing")
        return
    
    # Test with a long text that will trigger chunking
    long_text = ("కాకులు ఒక పొలానికి వెళ్లి అక్కడ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి. " +
                "రైతు వాటిని భయపెట్టడానికి వివిధ మార్గాలను అనుసరించాడు కానీ కాకులు తిరిగి వచ్చాయి. " +
                "చివరికి రైతు ఒక తెలివైన పథకం రూపొందించాడు మరియు అతను కాకులను పరిష్కరించగలిగాడు. " +
                "ఈ కథ మనకు సహనం మరియు తెలివితేటల గురించి బోధిస్తుంది. " +
                "ఇది 300 అక్షరాలకు మించిన పాఠ్యం కాబట్టి ఇది స్వయంచాలకంగా అనేక భాగాలుగా విభజించబడుతుంది.")
    
    print(f"Text length: {len(long_text)} characters")
    print(f"Text preview: {long_text[:100]}...")
    print(f"Will be automatically chunked: {'Yes' if len(long_text) > 300 else 'No'}")

    # Use the first available voice
    voice_key = voice_keys[0]

    request_data = {
        "text": long_text,
        "reference_voice_key": voice_key,
        "output_format": "wav",
        "sample_rate": 24000,
        "normalize": True
    }

    print(f"Sending request with voice: {voice_key}")
    response = requests.post(f"{API_BASE_URL}/tts", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ TTS generation successful")
        print(f"  Duration: {data['duration']:.2f} seconds")
        print(f"  Filename: {data['filename']}")
        
        # Check if filename indicates it was combined (chunked)
        if "combined" in data['filename']:
            print("  ✓ Text was automatically chunked and audio combined")
        
        # Save audio file
        os.makedirs("client_output", exist_ok=True)
        output_path = f"client_output/{data['filename']}"
        save_audio_from_base64(data['audio_base64'], output_path)
        
    else:
        print("✗ TTS generation failed")
        print(response.text)
    print()

def test_batch_tts(reference_voices_keys):
    """Test batch TTS request"""
    print("=== Testing Batch TTS Request ===")

    if not reference_voices_keys:
        print("No referenceVoices available for testing")
        return
    
    # Create multiple requests with different texts and referenceVoices
    texts = [
        "మొదటి వాక్యం - ఇది తెలుగు భాషలో ఉంది.",
        "दूसरा वाक्य - यह हिंदी में है।",
        "ਤੀਜਾ ਵਾਕ - ਇਹ ਪੰਜਾਬੀ ਵਿੱਚ ਹੈ।"
    ]
    
    requests_list = []
    for i, text in enumerate(texts):
        voice_key = reference_voices_keys[i % len(reference_voices_keys)]  # Cycle through available referenceVoices
        requests_list.append({
            "text": text,
            "reference_voice_key": voice_key,
            "output_format": "wav",
            "sample_rate": 24000,
            "normalize": True
        })
    
    batch_request = {
        "requests": requests_list,
        "return_as_zip": False
    }
    
    print(f"Sending batch request with {len(requests_list)} items")
    response = requests.post(f"{API_BASE_URL}/tts/batch", json=batch_request)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Batch TTS generation completed")
        print(f"  Total requests: {data['total_requests']}")
        print(f"  Successful: {data['successful_requests']}")
        print(f"  Failed: {data['failed_requests']}")
        print(f"  Total duration: {data['total_duration']:.2f} seconds")
        
        # Save all audio files
        os.makedirs("client_output/batch", exist_ok=True)
        for i, result in enumerate(data['results']):
            if result['success']:
                output_path = f"client_output/batch/{result['filename']}"
                save_audio_from_base64(result['audio_base64'], output_path)
            else:
                print(f"  ✗ Request {i} failed: {result['message']}")
                
    else:
        print("✗ Batch TTS generation failed")
        print(response.text)
    print()

def test_save_tts(reference_voices_keys):
    """Test TTS with file saving on server"""
    print("=== Testing TTS with Server-side Save ===")

    if not reference_voices_keys:
        print("No referenceVoices available for testing")
        return

    voice_key = reference_voices_keys[0]

    request_data = {
        "text": "సర్వర్‌లో ఫైల్ సేవ్ చేయడం టెస్ట్ చేస్తున్నాము.",
        "voice_key": voice_key,
        "output_format": "wav",
        "sample_rate": 24000,
        "normalize": True
    }

    print(f"Sending save request with voice: {voice_key}")
    response = requests.post(f"{API_BASE_URL}/tts/save", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ TTS generation and save successful")
        print(f"  Server filename: {data['filename']}")
        
    else:
        print("✗ TTS save failed")
        print(response.text)
    print()

def test_chunk_demo():
    """Test the text chunking demo endpoint"""
    print("=== Testing Text Chunking Demo ===")
    
    # Test with different text lengths
    test_texts = [
        "Short text that won't be chunked.",
        "This is a medium-length text that approaches but doesn't exceed the 300-character limit for automatic chunking, so it should be processed as a single unit without any splitting or combining of audio files.",
        "This is a very long text that definitely exceeds the 300-character limit and will be automatically split into multiple chunks by the TTS API. The intelligent chunking algorithm will split this text at sentence boundaries first, then at punctuation marks like commas and semicolons, and finally by words if absolutely necessary. This ensures that the resulting audio maintains completely natural speech patterns without breaking words or sentences in awkward places during the text-to-speech conversion process.",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test Text {i+1} ({len(text)} chars) ---")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        response = requests.post(f"{API_BASE_URL}/tts/chunk-demo", params={"text": text})
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Will be chunked: {result['will_be_chunked']}")
            print(f"✓ Number of chunks: {result['num_chunks']}")
            
            for chunk in result['chunks']:
                chunk_preview = chunk['text'][:60] + ('...' if len(chunk['text']) > 60 else '')
                print(f"  Chunk {chunk['chunk_number']} ({chunk['length']} chars): {chunk_preview}")
        else:
            print(f"✗ Error: {response.text}")
    print()

def main():
    """Main function to run all tests"""
    print("TTS IndicF5 API Client Test")
    print("=" * 40)
    
    # Test health check
    test_health_check()
    
    # Test text chunking demo
    test_chunk_demo()

    # Get available referenceVoices
    reference_voices_keys = test_get_reference_voices()

    if not reference_voices_keys:
        print("No referenceVoices available. Cannot proceed with TTS tests.")
        return
    
    # Test single TTS (with chunking support)
    test_single_tts(reference_voices_keys)

    # Test batch TTS
    test_batch_tts(reference_voices_keys)

    # Test server-side save
    test_save_tts(reference_voices_keys)

    # Test chunk demo
    test_chunk_demo()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
