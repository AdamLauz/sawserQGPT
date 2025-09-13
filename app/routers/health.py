"""Health check router."""

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.dependencies import get_llm_service, get_vector_service
from app.models import HealthResponse
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    llm_service: LLMService = Depends(get_llm_service),
    vector_service: VectorService = Depends(get_vector_service)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check LLM service
        llm_loaded = llm_service.is_loaded
        
        # Check vector service
        vector_ready = vector_service.is_ready
        
        # Overall health
        overall_healthy = llm_loaded and vector_ready
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            version=settings.app_version,
            model_loaded=llm_loaded,
            vector_db_ready=vector_ready
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/llm")
async def llm_health(
    llm_service: LLMService = Depends(get_llm_service)
) -> dict:
    """LLM service health check."""
    try:
        return {
            "loaded": llm_service.is_loaded,
            "model_name": settings.llm_model_name,
            "device": settings.device
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {"loaded": False, "error": str(e)}


@router.get("/health/vector")
async def vector_health(
    vector_service: VectorService = Depends(get_vector_service)
) -> dict:
    """Vector database health check."""
    try:
        stats = await vector_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        return {"status": "error", "error": str(e)}
