# Server - FastAPI Backend

This directory contains the FastAPI server that acts as a bridge between HTTP clients and MCP tools.

## Structure

```
server/
├── agent/                # MCP tools and RAG agent
│   ├── resources/        # Document storage
│   ├── storage/         # Vector database storage
│   ├── rag_agent.py     # Main RAG agent (with integrated LLM)
│   ├── knowledge_graph_tool.py
│   └── document_processor_tool.py
├── api/                  # FastAPI application
│   ├── main.py          # FastAPI application entry point
│   ├── dependencies.py  # FastAPI dependency injection
│   ├── exceptions.py    # Custom exceptions
│   ├── models.py        # Pydantic models
│   ├── routers/         # API route modules
│   │   ├── health.py    # Health endpoints
│   │   └── query.py     # Query endpoints
│   ├── middleware/      # Custom middleware
│   │   └── cors.py      # CORS configuration
│   └── dependencies/     # API-specific dependencies
│       └── rate_limit.py # Rate limiting
├── reverse_proxy/        # Nginx configuration
│   ├── nginx.conf       # Nginx config
│   └── Dockerfile       # Nginx containerization
# No global config - each component has its own
├── start_server.py      # Server startup script
├── Dockerfile           # Server containerization
└── requirements.txt     # Server dependencies
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python start_server.py

# Or with uvicorn directly
uvicorn server.api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `POST /api/v1/query` - Query with RAG
- `POST /api/v1/query/stream` - Streaming query
