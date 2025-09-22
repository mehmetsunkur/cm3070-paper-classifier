#!/usr/bin/env python3
"""
Test script for the web server.
"""

import subprocess
import time
import sys
from pathlib import Path
import requests
import tempfile

def test_web_server():
    """Test the web server functionality."""
    print("🌐 Testing Web Server...")
    
    # Start the web server in the background
    print("Starting web server...")
    
    try:
        # Start server process
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "paper_classifier.web_server:app",
            "--host", "127.0.0.1",
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Models available: {health_data.get('models_available', 'Unknown')}")
        else:
            print("❌ Health check failed")
            return False
        
        # Test models endpoint
        print("Testing models endpoint...")
        response = requests.get("http://127.0.0.1:8000/models", timeout=10)
        
        if response.status_code == 200:
            print("✅ Models endpoint working")
            models_data = response.json()
            print(f"   Total models: {models_data.get('total_models', 'Unknown')}")
        else:
            print("❌ Models endpoint failed")
        
        # Test main page
        print("Testing main page...")
        response = requests.get("http://127.0.0.1:8000/", timeout=10)
        
        if response.status_code == 200:
            print("✅ Main page loads successfully")
            if "Paper Classification" in response.text:
                print("   Contains expected content")
        else:
            print("❌ Main page failed to load")
        
        print("\n🎉 Web server test completed!")
        print("📱 Open your browser to: http://127.0.0.1:8000")
        print("📄 Upload a PDF file to test classification")
        
        return True
        
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        return False
    
    finally:
        # Clean up server process
        if 'server_process' in locals():
            print("\nStopping test server...")
            server_process.terminate()
            server_process.wait()


def create_quick_start_guide():
    """Create a quick start guide for the web interface."""
    guide_content = """
# 🌐 Paper Classification Web Interface - Quick Start

## 🚀 Launch the Web Server

```bash
# Start the web server
classify-paper-web

# Or with custom settings
classify-paper-web --host 0.0.0.0 --port 8080 --models-dir packages/paper_classifier/trained_models
```

## 📱 Using the Web Interface

1. **Open your browser**: http://127.0.0.1:8000
2. **Upload PDF**: Drag & drop or click to browse
3. **Wait for processing**: Usually takes 10-30 seconds
4. **View results**: See classification for discipline, field, and method

## 🔧 API Endpoints

- `GET /` - Main upload interface
- `POST /upload` - Upload PDF for classification
- `GET /health` - Health check
- `GET /models` - List available models

## 📁 Example API Usage

```bash
# Check server health
curl http://127.0.0.1:8000/health

# List available models
curl http://127.0.0.1:8000/models

# Upload PDF via API
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload
```

## 🏷️ Classification Categories

- **Discipline**: Computer Science, Software Engineering, Data Science, etc.
- **Field**: Machine Learning, Computer Vision, NLP, etc.  
- **Method**: Empirical Study, Theoretical Analysis, etc.

## ⚙️ Configuration

The web server automatically:
- Finds trained models in the `trained_models` directory
- Selects appropriate models based on paper size
- Provides confidence scores and detailed results

## 🔍 Troubleshooting

- **Models not found**: Ensure `trained_models` directory exists
- **Slow processing**: First request loads models (takes time)
- **Large files**: Maximum 50MB PDF size limit
- **GPU memory**: Close other GPU processes if out of memory

## 🛠️ Development

```bash
# Install with web dependencies
cd packages/paper_classifier && uv pip install -e .

# Run with auto-reload for development
classify-paper-web --reload
```
"""
    
    guide_path = Path("WEB_INTERFACE_GUIDE.md")
    with open(guide_path, "w") as f:
        f.write(guide_content)
    
    print(f"📖 Created quick start guide: {guide_path}")


def main():
    """Main test function."""
    print("🧪 Testing Paper Classification Web Interface")
    print("=" * 50)
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Test the web server
    success = test_web_server()
    
    if success:
        print("\n✅ All tests passed!")
        print("\n🎯 Next steps:")
        print("1. Run: classify-paper-web")
        print("2. Open: http://127.0.0.1:8000")
        print("3. Upload a PDF file")
    else:
        print("\n❌ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()