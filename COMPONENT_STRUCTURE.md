# Component-Based Architecture

The project has been reorganized into component-specific configurations and requirements, making it more modular and maintainable.

## New Structure

```
server/
├── agent/                    # RAG Agent Component
│   ├── config.py            # Agent-specific configuration
│   ├── requirements.txt     # Agent dependencies
│   ├── rag_agent.py         # Main RAG agent
│   ├── knowledge_graph_tool.py
│   ├── document_processor_tool.py
│   ├── resources/           # Document storage
│   └── storage/             # Vector database storage
├── api/                      # FastAPI Component
│   ├── config.py            # API-specific configuration
│   ├── requirements.txt     # API dependencies
│   ├── main.py              # FastAPI application
│   ├── dependencies.py      # FastAPI dependency injection
│   ├── exceptions.py        # Custom exceptions
│   ├── models.py            # Pydantic models
│   ├── routers/             # API route modules
│   │   ├── health.py        # Health endpoints
│   │   └── query.py         # Query endpoints
│   ├── middleware/          # Custom middleware
│   │   └── cors.py          # CORS configuration
│   └── dependencies/        # API-specific dependencies
│       └── rate_limit.py    # Rate limiting
├── reverse_proxy/            # Nginx Component
│   ├── config.py            # Reverse proxy configuration
│   ├── requirements.txt     # Reverse proxy dependencies
│   ├── nginx.conf           # Nginx configuration
│   └── Dockerfile           # Nginx containerization
├── start_server.py          # Server startup script
├── Dockerfile               # Server containerization
└── requirements.txt         # Combined dependencies
```

## Component Configurations

### 1. Agent Configuration (`server/agent/config.py`)
- **LLM Model Settings**: Model names, tokens, temperature
- **Knowledge Graph Settings**: Entity extraction, relations
- **Vector Database Settings**: Chunk size, similarity thresholds
- **GPU Settings**: Device detection, memory management
- **Resource Directories**: Storage paths

### 2. API Configuration (`server/api/config.py`)
- **Application Settings**: Name, version, debug mode
- **Server Settings**: Host, port, workers
- **CORS Settings**: Origins, credentials, methods, headers
- **Rate Limiting**: Query limits, health check limits

### 3. Reverse Proxy Configuration (`server/reverse_proxy/config.py`)
- **Nginx Settings**: Worker connections, body size limits
- **Proxy Settings**: Timeouts, upstream configuration
- **Server Settings**: Port, server name

## Component Requirements

### 1. Agent Requirements (`server/agent/requirements.txt`)
- PyTorch and transformers
- Knowledge graph tools (NetworkX, spaCy)
- MCP tools (FastMCP)
- ML utilities (NumPy, SciPy)

### 2. API Requirements (`server/api/requirements.txt`)
- FastAPI and uvicorn
- HTTP clients (httpx)
- Development tools (pytest, black, flake8)

### 3. Reverse Proxy Requirements (`server/reverse_proxy/requirements.txt`)
- Configuration management
- Nginx (via system/Docker)

### 4. Main Server Requirements (`server/requirements.txt`)
- Combined dependencies for full server deployment

## Benefits of Component-Based Structure

### 1. **Modularity**
- Each component has its own configuration
- Dependencies are clearly separated
- Easy to understand what each component needs

### 2. **Maintainability**
- Changes to one component don't affect others
- Clear separation of concerns
- Easier to debug and troubleshoot

### 3. **Scalability**
- Components can be deployed independently
- Easy to add new components
- Clear interfaces between components

### 4. **Development**
- Developers can work on specific components
- Clear dependency management
- Easier testing and development

## Usage Examples

### Install Component Dependencies
```bash
# Install agent dependencies
pip install -r server/agent/requirements.txt

# Install API dependencies
pip install -r server/api/requirements.txt

# Install all dependencies
pip install -r server/requirements.txt
```

### Component-Specific Configuration
```python
# Use agent configuration
from server.agent.config import agent_config
print(agent_config.llm_model_name)

# Use API configuration
from server.api.config import api_config
print(api_config.host)

# Use reverse proxy configuration
from server.reverse_proxy.config import reverse_proxy_config
print(reverse_proxy_config.nginx_worker_connections)
```

## Migration Notes

- **Old**: Single `server/config.py` with all settings
- **New**: Component-specific config files
- **Old**: Single `server/requirements.txt`
- **New**: Component-specific requirements files
- **Benefits**: Better organization, easier maintenance, clearer dependencies

This component-based structure makes the project much more maintainable and follows modern software architecture best practices!
