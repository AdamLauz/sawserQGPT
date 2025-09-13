# SawserQ GPT 2.0 - Modern RAG Application

A modern, production-ready RAG (Retrieval-Augmented Generation) application built with FastAPI, lightweight models, and async patterns.

## 🚀 Features

- **Lightweight Models**: Uses efficient, smaller models for faster inference
- **Modern Architecture**: Built with FastAPI, async/await, and dependency injection
- **Production Ready**: Proper error handling, logging, health checks, and monitoring
- **Streaming Support**: Real-time response streaming for better UX
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Interactive UI**: Modern Streamlit client with real-time chat

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     FastAPI     │    │   Vector DB     │
│     Client      │◄──►│     Server      │◄──►│   (LlamaIndex)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Lightweight   │
                       │   LLM Models    │
                       └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI with async/await
- **LLM**: Microsoft DialoGPT-medium (lightweight alternative)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: LlamaIndex with HuggingFace embeddings
- **Frontend**: Streamlit with async HTTP client
- **Deployment**: Docker + Docker Compose
- **Monitoring**: Structured logging + health checks

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker (optional)
- 4GB+ RAM recommended
- **GPU (optional)**: NVIDIA GPU with CUDA support for faster inference

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd sawserQGPT
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install GPU support (optional)**:
   ```bash
   # For GPU support (NVIDIA CUDA)
   python install_gpu.py
   
   # Or manually install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

5. **Test GPU setup (optional)**:
   ```bash
   python test_gpu.py
   ```

6. **Add your documents**:
   ```bash
   mkdir -p resources
   # Add your PDF/text files to resources/
   ```

7. **Start the server**:
   ```bash
   python start_server.py
   ```

8. **Start the client** (in another terminal):
   ```bash
   streamlit run client_streamlit.py
   ```

### Docker Deployment

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Client: Run `streamlit run client_streamlit.py`

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `microsoft/DialoGPT-medium` | LLM model to use |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `MAX_TOKENS` | `512` | Maximum tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `USE_GPU` | `true` (auto-detect) | Enable GPU acceleration |
| `DEVICE` | `cuda` (auto-detect) | Device to use (cuda/cpu) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `PERSIST_DIR` | `./storage` | Vector DB storage directory |
| `RESOURCES_DIR` | `./resources` | Documents directory |

### Model Options

**Lightweight LLM Models**:
- `microsoft/DialoGPT-medium` (345M parameters) - Default
- `microsoft/DialoGPT-small` (117M parameters) - Faster
- `distilgpt2` (82M parameters) - Smallest

**Embedding Models**:
- `sentence-transformers/all-MiniLM-L6-v2` (22M parameters) - Default
- `sentence-transformers/all-MiniLM-L12-v2` (33M parameters) - Better quality
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (22M parameters) - Alternative

## 🚀 GPU Support

### Automatic GPU Detection
The system automatically detects and uses GPU when available:
- **Auto-detection**: Uses GPU if CUDA is available
- **Environment override**: Set `USE_GPU=false` to force CPU
- **Device selection**: Automatically chooses optimal device

### GPU Configuration
```bash
# Enable GPU (default)
export USE_GPU=true
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

# Force CPU
export USE_GPU=false
export DEVICE=cpu
```

### Performance Benefits
| Configuration | Memory | Speed | Quality |
|---------------|--------|-------|---------|
| **CPU Only** | ~2GB RAM | 1x | Good |
| **GPU (345M)** | ~1GB VRAM + 1GB RAM | 3-5x faster | Good |
| **GPU (774M)** | ~2GB VRAM + 1GB RAM | 2-3x faster | Better |

### Testing GPU Setup
```bash
# Test GPU availability
python test_gpu.py

# Install GPU support
python install_gpu.py
```

## 📚 API Documentation

### Endpoints

- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `POST /api/v1/query` - Query with RAG
- `POST /api/v1/query/stream` - Streaming query

### Example Usage

```python
import httpx

# Query the API
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/query",
        json={"query": "Tell me about Circassian history"}
    )
    result = response.json()
    print(result["response"])
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Lint code
flake8 app/
black app/
isort app/
```

## 🚀 Production Deployment

### Docker Compose (Recommended)

```bash
# Production deployment with nginx
docker-compose --profile production up -d
```

### Manual Deployment

1. **Configure production settings**:
   ```bash
   export DEBUG=false
   export WORKERS=4
   export USE_GPU=true  # If available
   ```

2. **Start with gunicorn**:
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Use nginx as reverse proxy** (see `nginx/nginx.conf`)

## 📊 Monitoring

- **Health Checks**: `/api/v1/health`
- **Metrics**: Prometheus-compatible endpoints
- **Logging**: Structured JSON logging
- **Tracing**: Request/response tracing

## 🔍 Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Check available memory (4GB+ recommended)
   - Verify model names in configuration
   - Check internet connection for model downloads

2. **Vector DB not ready**:
   - Ensure documents are in `resources/` directory
   - Check `storage/` directory permissions
   - Verify embedding model compatibility

3. **API connection errors**:
   - Check server is running on correct port
   - Verify firewall settings
   - Check Docker port mappings

### Debug Mode

```bash
export DEBUG=true
python start_server.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- FastAPI team for the excellent framework
- Hugging Face for model hosting
- LlamaIndex for RAG capabilities
- Streamlit for the UI framework