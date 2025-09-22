#!/usr/bin/env python3
"""
Simple demo launcher for the web interface.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    """Launch the web interface demo."""
    print("🚀 Starting Paper Classification Web Demo")
    print("=" * 50)
    
    # Check if models directory exists
    models_dir = Path("packages/paper_classifier/trained_models")
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("Please ensure trained models are available before starting the web interface.")
        return 1
    
    print(f"✅ Found models directory: {models_dir}")
    
    # Start the web server
    print("\n🌐 Starting web server...")
    print("📱 The interface will open in your browser shortly...")
    print("🔄 Processing may take 10-30 seconds for the first request (model loading)")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start server
        cmd = [
            "classify-paper-web",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--models-dir", str(models_dir)
        ]
        
        # Add auto-reload for development
        if "--dev" in sys.argv:
            cmd.append("--reload")
            print("🔧 Development mode: auto-reload enabled")
        
        # Start the server process
        server_process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Open browser
        url = "http://127.0.0.1:8000"
        print(f"🌐 Opening browser: {url}")
        webbrowser.open(url)
        
        # Wait for the server process
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping web server...")
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()
        print("✅ Server stopped successfully")
        return 0
    
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return 1


if __name__ == "__main__":
    exit(main())