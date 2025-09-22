
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
