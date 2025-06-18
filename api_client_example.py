#!/usr/bin/env python3
"""
Example client script for TTS IndicF5 API
Demonstrates how to use the REST API for single and batch TTS requests

Note: This script expects the API server to be running with prompts configured in data/prompts/
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

def test_get_prompts():
    """Test getting available prompts"""
    print("=== Getting Available Prompts ===")
    response = requests.get(f"{API_BASE_URL}/prompts")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total_count']} prompts:")
        for key, prompt in data['prompts'].items():
            print(f"  - {key}: {prompt['author']}")
        return list(data['prompts'].keys())
    else:
        print("✗ Failed to get prompts")
        print(response.text)
        return []

def test_single_tts(prompt_keys):
    """Test single TTS request"""
    print("=== Testing Single TTS Request ===")
    
    if not prompt_keys:
        print("No prompts available for testing")
        return
    
    # Use the first available prompt
    prompt_key = prompt_keys[0]
    
    request_data = {
        "text": "కాకులు ఒక పొలానికి వెళ్లి అక్కడ మొక్కలన్నిటిని ధ్వంసం చేయ సాగాయి.",
        "prompt_key": prompt_key,
        "output_format": "wav",
        "sample_rate": 24000,
        "normalize": True
    }
    
    print(f"Sending request with prompt: {prompt_key}")
    response = requests.post(f"{API_BASE_URL}/tts", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ TTS generation successful")
        print(f"  Duration: {data['duration']:.2f} seconds")
        print(f"  Filename: {data['filename']}")
        
        # Save audio file
        os.makedirs("client_output", exist_ok=True)
        output_path = f"client_output/{data['filename']}"
        save_audio_from_base64(data['audio_base64'], output_path)
        
    else:
        print("✗ TTS generation failed")
        print(response.text)
    print()

def test_batch_tts(prompt_keys):
    """Test batch TTS request"""
    print("=== Testing Batch TTS Request ===")
    
    if not prompt_keys:
        print("No prompts available for testing")
        return
    
    # Create multiple requests with different texts and prompts
    texts = [
        "మొదటి వాక్యం - ఇది తెలుగు భాషలో ఉంది.",
        "दूसरा वाक्य - यह हिंदी में है।",
        "ਤੀਜਾ ਵਾਕ - ਇਹ ਪੰਜਾਬੀ ਵਿੱਚ ਹੈ।"
    ]
    
    requests_list = []
    for i, text in enumerate(texts):
        prompt_key = prompt_keys[i % len(prompt_keys)]  # Cycle through available prompts
        requests_list.append({
            "text": text,
            "prompt_key": prompt_key,
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

def test_save_tts(prompt_keys):
    """Test TTS with file saving on server"""
    print("=== Testing TTS with Server-side Save ===")
    
    if not prompt_keys:
        print("No prompts available for testing")
        return
    
    prompt_key = prompt_keys[0]
    
    request_data = {
        "text": "సర్వర్‌లో ఫైల్ సేవ్ చేయడం టెస్ట్ చేస్తున్నాము.",
        "prompt_key": prompt_key,
        "output_format": "wav",
        "sample_rate": 24000,
        "normalize": True
    }
    
    print(f"Sending save request with prompt: {prompt_key}")
    response = requests.post(f"{API_BASE_URL}/tts/save", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✓ TTS generation and save successful")
        print(f"  Server filename: {data['filename']}")
        
    else:
        print("✗ TTS save failed")
        print(response.text)
    print()

def main():
    """Main function to run all tests"""
    print("TTS IndicF5 API Client Test")
    print("=" * 40)
    
    # Test health check
    test_health_check()
    
    # Get available prompts
    prompt_keys = test_get_prompts()
    
    if not prompt_keys:
        print("No prompts available. Cannot proceed with TTS tests.")
        return
    
    # Test single TTS
    test_single_tts(prompt_keys)
    
    # Test batch TTS
    test_batch_tts(prompt_keys)
    
    # Test server-side save
    test_save_tts(prompt_keys)
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
