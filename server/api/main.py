"""FastAPI application with modern RAG architecture."""

import logging
import asyncio
import os
from contextlib import asynccontextmanager

# Disable OpenAI by setting environment variable
os.environ["OPENAI_API_KEY"] = ""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from server.api.config import api_config
from server.api.dependencies import get_rag_orchestrator_client
from server.api.exceptions import SawserQGPTError
from server.api.routers import health, query
from server.api.middleware.cors import setup_cors_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    A FastAPI-specific pattern for managing application lifecycle.
    @asynccontextmanager is It's a Python decorator that turns a function into an async context manager. Think of it like a with statement but for async code.
    The "yield" statement is KEY here, splitting the function into two parts:
    - The part before the yield is executed when the context manager is entered (e.g., when the application starts).
    - The part after the yield is executed when the context manager is exited (e.g., when the application shuts down).
    This pattern is used to manage the lifecycle of the application, ensuring that the services are properly initialized and shut down.
    FastAPI automatically calls this function, note that the function is sent as an argument to the FastAPI application.
    """
    # Startup
    logger.info("Starting SawserQ GPT application")
    
    try:
        # Initialize RAG orchestrator client
        rag_client = get_rag_orchestrator_client()
        
        # Check if services are healthy
        logger.info("Checking service health...")
        health_status = await rag_client.get_health_status()
        if not health_status.get("overall_healthy", False):
            logger.warning("Some services are not healthy, but continuing...")
        logger.info("Services checked successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield # This is a yield statement that allows us to pause the execution of the function and return a value. In this case we don't need to return a value, we just want to pause the execution of the function.
    
    # Shutdown - when the application is shutting down, we need to unload the models to free the memory.
    logger.info("Shutting down SawserQ GPT application")
    try:
        # RAG agent cleanup is handled automatically when the process exits
        logger.info("RAG agent cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=api_config.app_name,
    version=api_config.app_version,
    description="Modern RAG application with lightweight models",
    lifespan=lifespan # lifespan is a context manager that allows us to start and stop the application.
)

# Setup middleware
setup_cors_middleware(app)

# Include routers
app.include_router(health.router)
app.include_router(query.router)


@app.exception_handler(SawserQGPTError)
async def sawserq_gpt_exception_handler(request, exc: SawserQGPTError):
    """Handle custom exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "error_type": type(exc).__name__}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "error_type": "InternalError"}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {api_config.app_name}",
        "version": api_config.app_version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server.api.main:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,
        workers=1 if api_config.debug else api_config.workers
    )