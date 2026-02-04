# FastAPI Application Structure

This directory contains the FastAPI application organized according to best practices.

## Structure

```
server/api/
├── main.py                 # FastAPI application entry point
├── __init__.py            # Package initialization
├── routers/               # API route modules
│   ├── __init__.py
│   ├── health.py         # Health check endpoints
│   └── query.py          # Query endpoints
├── middleware/            # Custom middleware
│   ├── __init__.py
│   └── cors.py           # CORS configuration
├── dependencies/          # API-specific dependencies
│   ├── __init__.py
│   ├── auth.py           # Authentication dependencies
│   └── rate_limit.py     # Rate limiting dependencies
└── README.md             # This file
```

## Best Practices Implemented

### 1. **Separation of Concerns**
- **Routers**: Handle HTTP requests and responses
- **Middleware**: Handle cross-cutting concerns (CORS, logging, etc.)
- **Dependencies**: Handle authentication, rate limiting, etc.
- **Main**: Application configuration and setup

### 2. **Router Organization**
- Each router handles a specific domain (health, query)
- Routers are included in main.py
- Each router has its own prefix and tags

### 3. **Middleware Structure**
- CORS middleware is properly configured
- Easy to add new middleware (logging, authentication, etc.)
- Middleware is applied at the application level

### 4. **Dependency Injection**
- API-specific dependencies are separated from core dependencies
- Rate limiting is implemented as a dependency
- Authentication is prepared for future implementation

### 5. **Error Handling**
- Custom exception handlers in main.py
- Proper HTTP status codes
- Consistent error response format

## API Endpoints

### Health Endpoints
- `GET /api/v1/health` - Main health check
- `GET /api/v1/health/llm` - LLM service health
- `GET /api/v1/health/vector` - Vector database health

### Query Endpoints
- `POST /api/v1/query` - Process RAG query
- `POST /api/v1/query/stream` - Streaming RAG query
- `GET /api/v1/query/health` - Query service health

## Features

### Rate Limiting
- Query endpoints: 50 requests per minute
- Health endpoints: 100 requests per minute
- Configurable per endpoint

### CORS Support
- Configurable origins
- Supports credentials
- Production-ready configuration

### Authentication (Prepared)
- JWT token support
- Optional authentication
- Easy to extend for protected endpoints

## Usage

### Development
```bash
# Run the server
python server/api/main.py

# Or with uvicorn
uvicorn server.api.main:app --reload
```

### Production
```bash
# Use the main server startup
python server/start_server.py
```

## Adding New Features

### New Router
1. Create new file in `routers/`
2. Define APIRouter with appropriate prefix
3. Add to main.py: `app.include_router(new_router)`

### New Middleware
1. Create new file in `middleware/`
2. Define setup function
3. Call in main.py

### New Dependencies
1. Create new file in `dependencies/`
2. Define dependency functions
3. Use in router endpoints

## Configuration

All configuration is handled through `server/config.py` and environment variables.
