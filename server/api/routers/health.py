"""Health check router."""

import logging
from fastapi import APIRouter, Depends, HTTPException

from server.api.config import api_config
from server.api.dependencies import get_rag_agent
from server.api.models import HealthResponse
from server.agent.rag_agent import RAGAgent
from server.api.dependencies.rate_limit import health_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag_agent: RAGAgent = Depends(get_rag_agent),
    _: None = Depends(health_rate_limiter)
) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        # Get health status from RAG agent
        health_status = await rag_agent.get_health_status()
        
        return HealthResponse(
            status="healthy" if health_status.overall_healthy else "unhealthy",
            version=api_config.app_version,
            llm_loaded=health_status.llm_loaded,
            vector_db_ready=health_status.knowledge_graph_ready
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/llm")
async def llm_health(
    rag_agent: RAGAgent = Depends(get_rag_agent)
) -> dict:
    """LLM service health check."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        health_status = await rag_agent.get_health_status()
        from server.agent.config import agent_config
        return {
            "loaded": health_status.llm_loaded,
            "model_name": agent_config.llm_model_name,
            "device": agent_config.device
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {"loaded": False, "error": str(e)}


@router.get("/health/vector")
async def vector_health(
    rag_agent: RAGAgent = Depends(get_rag_agent)
) -> dict:
    """Vector database health check."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        health_status = await rag_agent.get_health_status()
        return {
            "status": "ready" if health_status.knowledge_graph_ready else "not_ready",
            "knowledge_graph_stats": health_status.knowledge_graph_stats
        }
    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        return {"status": "error", "error": str(e)}