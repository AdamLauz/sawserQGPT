"""Query router for RAG endpoints."""

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from server.api.dependencies import get_rag_orchestrator_client
from server.api.exceptions import QueryError
from server.api.models import QueryRequest, QueryResponse, ErrorResponse
from server.api.services.rag_orchestrator_client import RAGOrchestratorClient
from server.api.dependencies.rate_limit import query_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, 
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client),
    _: None = Depends(query_rate_limiter)
) -> QueryResponse:
    """Process a query using RAG."""
    try:
        # Use the RAG orchestrator client
        result = await rag_client.query(question=request.query)
        
        return QueryResponse(
            response=result["answer"],
            context_used=result["context_used"],
            tokens_generated=len(result["answer"].split())  # Rough token count
        )
        
    except QueryError as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest, 
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client),
    _: None = Depends(query_rate_limiter)
) -> StreamingResponse:
    """Process a query using RAG with streaming response."""
    try:
        async def generate_stream() -> AsyncGenerator[str, None]:
            async for chunk in rag_client.query_stream(question=request.query):
                yield chunk
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except QueryError as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/query/health")
async def query_health(
    rag_client: RAGOrchestratorClient = Depends(get_rag_orchestrator_client)
) -> dict:
    """Get health status of the query service."""
    try:
        health_status = await rag_client.get_health_status()
        return {
            "healthy": health_status.get("overall_healthy", False),
            "llm_loaded": health_status.get("llm_loaded", False),
            "knowledge_graph_ready": health_status.get("knowledge_graph_ready", False),
            "document_processor_ready": health_status.get("document_processor_ready", False)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"error": str(e), "healthy": False}