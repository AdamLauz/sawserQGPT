"""Health check router."""

import logging
from fastapi import APIRouter, Depends, HTTPException

from server.api.config import api_config
from server.api.dependencies import get_rag_orchestrator_client
from server.api.models import HealthResponse
from server.api.services.rag_orchestrator_client import RAGOrchestratorClient
from server.api.dependencies.rate_limit import health_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client),
    _: None = Depends(health_rate_limiter)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Get health status from RAG orchestrator
        health_status = await rag_client.get_health_status()
        
        return HealthResponse(
            status="healthy" if health_status.get("overall_healthy", False) else "unhealthy",
            version=api_config.app_version,
            llm_loaded=health_status.get("llm_loaded", False),
            vector_db_ready=health_status.get("knowledge_graph_ready", False)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/llm")
async def llm_health(
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client)
) -> dict:
    """LLM service health check."""
    try:
        health_status = await rag_client.get_health_status()
        return {
            "loaded": health_status.get("llm_loaded", False),
            "model_name": "microsoft/DialoGPT-large",  # Default model name
            "device": "cuda" if health_status.get("llm_loaded", False) else "cpu"
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {"loaded": False, "error": str(e)}


@router.get("/health/vector")
async def vector_health(
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client)
) -> dict:
    """Vector database health check."""
    try:
        health_status = await rag_client.get_health_status()
        return {
            "status": "ready" if health_status.get("knowledge_graph_ready", False) else "not_ready",
            "knowledge_graph_stats": {
                "nodes": 0,  # Will be populated by actual service
                "edges": 0,
                "density": 0.0
            }
        }
    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        return {"status": "error", "error": str(e)}