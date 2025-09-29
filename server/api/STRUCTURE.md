# FastAPI Structure Overview

## New API Organization

The server API has been reorganized following FastAPI best practices:

```
server/api/
├── main.py                    # FastAPI app entry point (moved to root)
├── __init__.py               # Package initialization
├── routers/                  # Route modules
│   ├── __init__.py
│   ├── health.py            # Health check endpoints
│   └── query.py             # Query endpoints
├── middleware/               # Custom middleware
│   ├── __init__.py
│   └── cors.py              # CORS configuration
├── dependencies/             # API-specific dependencies
│   ├── __init__.py
│   ├── auth.py              # Authentication (prepared)
│   └── rate_limit.py        # Rate limiting
├── README.md                 # API documentation
└── STRUCTURE.md             # This file
```

## Key Improvements

### 1. **Main.py at Root**
- `main.py` is now at the root of the `api/` folder
- Contains FastAPI app configuration
- Handles lifespan events
- Sets up middleware and routers

### 2. **Router Organization**
- **Before**: `health.py`, `query.py` directly in `api/`
- **After**: Organized in `routers/` subfolder
- Each router handles specific domain logic
- Clean separation of concerns

### 3. **Middleware Structure**
- **Before**: CORS middleware inline in main.py
- **After**: Dedicated `middleware/` folder
- CORS configuration is modular
- Easy to add new middleware

### 4. **Dependencies**
- **Before**: All dependencies in server root
- **After**: API-specific dependencies in `dependencies/`
- Rate limiting implemented
- Authentication prepared for future use

### 5. **Best Practices Implemented**
- ✅ Separation of concerns
- ✅ Modular structure
- ✅ Easy to extend
- ✅ Production-ready
- ✅ Rate limiting
- ✅ CORS configuration
- ✅ Error handling
- ✅ Documentation

## Benefits

1. **Maintainability**: Clear structure makes it easy to find and modify code
2. **Scalability**: Easy to add new routers, middleware, and dependencies
3. **Testing**: Each component can be tested independently
4. **Documentation**: Clear organization makes the code self-documenting
5. **Production Ready**: Includes rate limiting, CORS, and error handling

## Usage

### Development
```bash
# Run directly
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
1. Create file in `routers/`
2. Define APIRouter
3. Add to main.py: `app.include_router(new_router)`

### New Middleware
1. Create file in `middleware/`
2. Define setup function
3. Call in main.py

### New Dependencies
1. Create file in `dependencies/`
2. Define dependency functions
3. Use in router endpoints

This structure follows FastAPI best practices and makes the codebase more maintainable and scalable.
