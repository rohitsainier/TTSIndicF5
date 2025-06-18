#!/usr/bin/env python3
"""
Quick test script to verify API endpoints are working
"""

import requests
import json

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API endpoints...")
    print("=" * 40)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint: OK")
            data = response.json()
            print(f"   API base: {data.get('api_base', 'Not found')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test API root
    try:
        response = requests.get(f"{base_url}/api/")
        if response.status_code == 200:
            print("✅ API root: OK")
            data = response.json()
            print(f"   Endpoints: {data.get('endpoints', 'Not found')}")
        else:
            print(f"❌ API root failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API root error: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health endpoint: OK")
            data = response.json()
            print(f"   Model loaded: {data.get('model_loaded', 'Unknown')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test prompts endpoint
    try:
        response = requests.get(f"{base_url}/api/prompts")
        if response.status_code == 200:
            print("✅ Prompts endpoint: OK")
            data = response.json()
            print(f"   Available prompts: {data.get('total_count', 0)}")
        else:
            print(f"❌ Prompts endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prompts endpoint error: {e}")
    
    # Test web interface
    try:
        response = requests.get(f"{base_url}/web")
        if response.status_code == 200:
            print("✅ Web interface: OK")
        else:
            print(f"❌ Web interface failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Web interface error: {e}")
    
    print("\n🎯 Test complete! Start the server with ./start_api.sh and run this script to verify.")

if __name__ == "__main__":
    test_endpoints()
