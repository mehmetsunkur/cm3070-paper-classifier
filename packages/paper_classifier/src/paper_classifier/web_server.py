#!/usr/bin/env python3
"""
FastAPI web server for paper classification.
Simple single-user development server using existing inference code.
"""

import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import existing inference code
from .inference_utils import InferenceEngine, format_result_human_readable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine (singleton for single-user server)
inference_engine: Optional[InferenceEngine] = None

# Create FastAPI app
app = FastAPI(
    title="Paper Classification Web Interface",
    description="Upload PDF papers for automatic classification by discipline, field, and method",
    version="1.0.0"
)

# Create templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Create static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def get_inference_engine():
    """Get or create the global inference engine."""
    global inference_engine
    if inference_engine is None:
        logger.info("Initializing inference engine...")
        models_dir = "trained_models"
        
        # Try different model directory paths
        possible_paths = [
            "trained_models",
            "packages/paper_classifier/trained_models",
            "../trained_models",
            "./packages/paper_classifier/trained_models"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                models_dir = path
                break
        
        logger.info(f"Using models directory: {models_dir}")
        inference_engine = InferenceEngine(models_dir=models_dir)
        logger.info("Inference engine initialized successfully")
    
    return inference_engine


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload and classification."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size (limit to 50MB for development)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "paper_classifier_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"
    
    try:
        # Write file
        with open(temp_file, 'wb') as f:
            f.write(content)
        
        logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")
        
        # Get inference engine
        engine = get_inference_engine()
        
        # Run classification
        result = engine.classify_pdf(temp_file)
        
        # Format results
        response_data = {
            "success": True,
            "filename": file.filename,
            "file_size": len(content),
            "results": result.to_dict(),
            "human_readable": format_result_human_readable(result, include_confidence=True)
        }
        
        logger.info(f"Classification successful for {file.filename}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Classification failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_inference_engine()
        models = engine.loader.model_manager.discover_models()
        return {
            "status": "healthy",
            "models_available": len(models),
            "inference_engine": "ready"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/models")
async def list_models():
    """List available models."""
    try:
        engine = get_inference_engine()
        models = engine.loader.model_manager.discover_models()
        
        model_info = []
        for model in models:
            info = {
                "name": model.name,
                "path": str(model.final_model_path),
                "has_mappings": model.has_label_mappings,
            }
            if model.metrics:
                info["eval_loss"] = model.metrics.get("eval_loss")
                info["eval_accuracy"] = model.metrics.get("eval_accuracy")
            
            model_info.append(info)
        
        return {
            "total_models": len(models),
            "models": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


def create_html_template():
    """Create the HTML template file."""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Classification</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #ecf0f1;
            transition: background-color 0.3s;
        }
        .upload-area:hover {
            background-color: #d5dbdb;
        }
        .upload-area.dragover {
            background-color: #3498db;
            color: white;
        }
        #file-input {
            display: none;
        }
        .upload-btn {
            background-color: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        .upload-btn:hover {
            background-color: #2980b9;
        }
        .upload-btn:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loading .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background-color: #e8f5e8;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
            display: none;
        }
        .error {
            margin-top: 20px;
            padding: 20px;
            background-color: #fdf2f2;
            border-radius: 5px;
            border-left: 4px solid #e74c3c;
            color: #c0392b;
            display: none;
        }
        .result-section {
            margin: 15px 0;
        }
        .result-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .confidence {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .file-info {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #6c757d;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.4;
        }
        .example-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }
        .example-section h3 {
            color: #34495e;
            margin-bottom: 10px;
        }
        .example-section ul {
            color: #7f8c8d;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 Paper Classification System</h1>
        <p style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
            Upload a PDF paper to automatically classify it by academic discipline, research field, and methodology
        </p>
        
        <div class="upload-area" id="upload-area">
            <h3>📎 Drop PDF file here or click to browse</h3>
            <p>Maximum file size: 50MB</p>
            <input type="file" id="file-input" accept=".pdf">
            <button class="upload-btn" onclick="document.getElementById('file-input').click()">
                Choose PDF File
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing your paper... This may take 10-30 seconds.</p>
        </div>
        
        <div class="error" id="error"></div>
        <div class="results" id="results"></div>
        
        <div class="example-section">
            <h3>🏷️ Classification Categories</h3>
            <ul>
                <li><strong>Discipline:</strong> Computer Science, Software Engineering, Data Science, etc.</li>
                <li><strong>Field:</strong> Machine Learning, Computer Vision, Natural Language Processing, etc.</li>
                <li><strong>Method:</strong> Empirical Study, Theoretical Analysis, Algorithm Development, etc.</li>
            </ul>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const error = document.getElementById('error');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        async function handleFile(file) {
            // Validate file
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showError('Please select a PDF file.');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                showError('File size must be less than 50MB.');
                return;
            }

            // Show loading
            hideAll();
            loading.style.display = 'block';

            // Upload and process
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    showResults(data);
                } else {
                    showError(data.detail || 'An error occurred during processing.');
                }
            } catch (err) {
                showError('Network error. Please try again.');
                console.error('Upload error:', err);
            } finally {
                loading.style.display = 'none';
            }
        }

        function showResults(data) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="file-info">
                    <strong>📁 File:</strong> ${data.filename} (${formatFileSize(data.file_size)})
                </div>
            `;

            const res = data.results;
            
            if (res.discipline) {
                html += `
                    <div class="result-section">
                        <div class="result-label">🏛️ Academic Discipline</div>
                        <div>${res.discipline.label}</div>
                        <div class="confidence">Confidence: ${(res.discipline.confidence * 100).toFixed(1)}%</div>
                    </div>
                `;
            }

            if (res.field) {
                html += `
                    <div class="result-section">
                        <div class="result-label">🔬 Research Field</div>
                        <div>${res.field.label}</div>
                        <div class="confidence">Confidence: ${(res.field.confidence * 100).toFixed(1)}%</div>
                    </div>
                `;
            }

            if (res.method) {
                html += `
                    <div class="result-section">
                        <div class="result-label">🔍 Research Method</div>
                        <div>${res.method.label}</div>
                        <div class="confidence">Confidence: ${(res.method.confidence * 100).toFixed(1)}%</div>
                    </div>
                `;
            }

            if (res.processing_time_seconds) {
                html += `
                    <div class="file-info">
                        ⏱️ Processing time: ${res.processing_time_seconds.toFixed(2)} seconds
                    </div>
                `;
            }

            // Add formatted output
            html += `
                <details style="margin-top: 20px;">
                    <summary style="cursor: pointer; font-weight: bold;">📋 Detailed Results</summary>
                    <pre>${data.human_readable}</pre>
                </details>
            `;

            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideAll() {
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
    </script>
</body>
</html>
    '''
    
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Created HTML template at {templates_dir}/index.html")


def main():
    """Main entry point for the web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Classification Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", default="trained_models", help="Models directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create template if it doesn't exist
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        create_html_template()
    
    logger.info(f"Starting Paper Classification Web Server...")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Server will be available at: http://{args.host}:{args.port}")
    
    # Set models directory in environment for the inference engine
    os.environ["PAPER_CLASSIFIER_MODELS_DIR"] = args.models_dir
    
    uvicorn.run(
        "paper_classifier.web_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()