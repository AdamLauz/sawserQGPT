# SawserQ GPT 2.0 - MCP-Based RAG Application

A modern, production-ready RAG (Retrieval-Augmented Generation) application built with MCP (Model Context Protocol), lightweight models, and async patterns.

## 🚀 Features

- **Knowledge Graph**: Extracts and reasons over facts from documents
- **Multi-hop Reasoning**: Connects related facts through graph traversal
- **Enhanced LLM**: DialoGPT-large for better reasoning capabilities
- **Fact Extraction**: Automatically extracts entities and relations from text
- **MCP Architecture**: Built with FastMCP for modular, tool-based interactions
- **Production Ready**: Proper error handling, logging, and health checks
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Interactive UI**: Modern Streamlit client with real-time chat

## ️ Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐    Direct    ┌─────────────────┐
│   Streamlit     │◄──────────►│   FastAPI       │◄───────────►│   RAG Agent     │
│     Client      │   Requests │   Server        │   Calls     │   (FastMCP)     │
└─────────────────┘            └─────────────────┘             └─────────────────┘
                                                                    │
                                                                    │ uses
                                                                    ▼
                                                           ┌─────────────────┐
                                                           │ Knowledge Graph │
                                                           │   MCP Tool      │
                                                           │   (NetworkX)    │
                                                           └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI server that calls RAG Agent (FastMCP)
- **RAG Agent**: FastMCP-based agent that uses Knowledge Graph tool
- **Knowledge Graph**: NetworkX-based MCP tool for fact extraction and storage
- **LLM**: Enhanced DialoGPT-large for better reasoning
- **Frontend**: Streamlit with HTTP client
- **Deployment**: Docker + Docker Compose
- **Monitoring**: Structured logging + health checks

## 📁 Project Structure

```
sawserQGPT/
├── server/              # FastAPI backend + MCP agent
│   ├── agent/          # MCP tools and RAG agent
│   │   ├── resources/  # Document storage
│   │   ├── storage/    # Vector database storage
│   │   └── *.py        # Agent files
│   ├── api/            # FastAPI routes
│   ├── services/       # Core services
│   ├── core/           # Core components
│   ├── scripts/        # Utility scripts
│   └── start_server.py # Server startup
├── client/             # Streamlit frontend
├── docker/             # Docker configuration
├── test/               # Testing and setup
└── requirements.txt    # Main project dependencies
```

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

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Quick setup with MCP approach**:
   ```bash
   python test/setup_mcp_approach.py
   ```

4. **Install GPU support (optional)**:
   ```bash
   # For GPU support (NVIDIA CUDA)
   python install_gpu.py
   
   # Or manually install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

6. **Test the system**:
   ```bash
   python test/test_mcp_system.py
   ```

7. **Add your documents**:
   ```bash
   # Add your PDF/text files to resources/
   ```

8. **Run the system**:
   ```bash
   # Option 1: Start FastAPI server (recommended)
   python server/start_server.py
   
   # Option 2: Use Streamlit client (in another terminal)
   streamlit run client/client_streamlit.py
   
   # Option 3: Run MCP system directly (alternative)
   python test/run_mcp_system.py
   ```

### Docker Deployment

1. **Build and run**:
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the application**:
   - FastAPI Server: http://localhost:8000
   - Client: Run `streamlit run client/client_streamlit.py`

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `microsoft/DialoGPT-large` | Open source LLM model to use |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Open source embedding model |
| `MAX_TOKENS` | `512` | Maximum tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `USE_GPU` | `true` (auto-detect) | Enable GPU acceleration |
| `DEVICE` | `cuda` (auto-detect) | Device to use (cuda/cpu) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `PERSIST_DIR` | `./storage` | Vector DB storage directory |
| `RESOURCES_DIR` | `./resources` | Documents directory |

### Open Source Model Options

**Conversational LLM Models**:
- `microsoft/DialoGPT-large` (774M parameters) - **Default, best quality**
- `microsoft/DialoGPT-medium` (345M parameters) - Balanced performance
- `facebook/blenderbot-400M-distill` (400M parameters) - Good alternative
- `EleutherAI/gpt-neo-125M` (125M parameters) - Very fast, low memory
- `EleutherAI/gpt-neo-1.3B` (1.3B parameters) - High quality, more memory

**Embedding Models**:
- `sentence-transformers/all-MiniLM-L6-v2` (22M parameters) - **Default, fast and efficient**
- `sentence-transformers/all-MiniLM-L12-v2` (33M parameters) - Better than L6
- `sentence-transformers/all-mpnet-base-v2` (420M parameters) - Best quality, more memory

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

## 🔄 Data Flow

The system follows this data flow:

1. **Streamlit Client** → HTTP request → **FastAPI Server**
2. **FastAPI Server** → Direct call → **RAG Agent** (FastMCP)
3. **RAG Agent** → MCP tool call → **Knowledge Graph Tool**
4. **Knowledge Graph Tool** → Query NetworkX graph → **Response**
5. **RAG Agent** → Generate response with LLM → **FastAPI Server**
6. **FastAPI Server** → HTTP response → **Streamlit Client**

## 🔧 MCP Tools

The system uses MCP (Model Context Protocol) tools for modular functionality:

### Available Tools

1. **RAG Agent** (`app/agent/rag_agent.py`)
   - FastMCP-based agent that orchestrates the entire RAG process
   - Uses Knowledge Graph tool internally
   - Manages LLM interactions and response generation

2. **Knowledge Graph Tool** (`app/tools/knowledge_graph_tool.py`)
   - NetworkX-based MCP tool for knowledge graph operations
   - Entity extraction from text
   - Knowledge graph queries and reasoning

3. **Document Processor Tool** (`app/tools/document_processor_tool.py`)
   - Processes documents from resources directory
   - Extracts entities and relationships
   - Populates knowledge graph

### Using the System

```python
# Example: Query through FastAPI
import httpx

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
# Test the complete MCP system
python test_mcp_system.py

# Test GPU setup
python test_gpu.py

# Run individual tests
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
# Production deployment
docker-compose --profile production up -d
```

### Manual Deployment

1. **Configure production settings**:
   ```bash
   export DEBUG=false
   export USE_GPU=true  # If available
   ```

2. **Run the MCP system**:
   ```bash
   python run_mcp_system.py
   ```

3. **Use nginx as reverse proxy** (see `nginx/nginx.conf`)

## 📊 Monitoring

- **Health Checks**: Built into MCP orchestrator
- **Logging**: Structured JSON logging
- **Status**: System status through orchestrator

## 🔍 Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Check available memory (4GB+ recommended)
   - Verify model names in configuration
   - Check internet connection for model downloads

2. **Knowledge graph not ready**:
   - Ensure documents are in `resources/` directory
   - Check `knowledge_graph/` directory permissions
   - Verify spaCy model installation

3. **MCP system connection errors**:
   - Check orchestrator initialization
   - Verify all tools are loaded
   - Check Docker port mappings

### Debug Mode

```bash
export DEBUG=true
python run_mcp_system.py
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

- FastMCP team for the excellent MCP framework
- Hugging Face for model hosting
- NetworkX for knowledge graph capabilities
- Streamlit for the UI framework