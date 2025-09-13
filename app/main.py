"""FastAPI application with modern RAG architecture."""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import get_llm_service, get_vector_service
from app.exceptions import SawserQGPTError
from app.routers import health, query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    This is a context manager that allows us to start and stop the application.
    """
    # Startup
    logger.info("Starting SawserQ GPT application")
    
    try:
        # Initialize services
        llm_service = get_llm_service()
        vector_service = get_vector_service()
        
        # Load models asynchronously, wait for both of them to complete before moving on (await)
        logger.info("Loading models...")
        await asyncio.gather( # gather alone runs multiple tasks in parallel does not wait for them to complete before moving on. (that is why we use await)
            llm_service.load_model(),
            vector_service.initialize()
        )
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield # This is a yield statement that allows us to pause the execution of the function and return a value. In this case we don't need to return a value, we just want to pause the execution of the function.
    
    # Shutdown - when the application is shutting down, we need to unload the models to free the memory.
    logger.info("Shutting down SawserQ GPT application")
    try:
        llm_service = get_llm_service()
        await llm_service.unload_model()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Modern RAG application with lightweight models",
    lifespan=lifespan # lifespan is a context manager that allows us to start and stop the application.
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )
