"""Query router for RAG endpoints."""

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from server.api.dependencies import get_rag_agent
from server.api.exceptions import QueryError
from server.api.models import QueryRequest, QueryResponse, ErrorResponse
from server.agent.rag_agent import RAGAgent
from server.api.dependencies.rate_limit import query_rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, 
    rag_agent: RAGAgent = Depends(get_rag_agent),
    _: None = Depends(query_rate_limiter)
) -> QueryResponse:
    """Process a query using RAG."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        # Use the RAG agent's query method
        result = await rag_agent.query(question=request.query)
        
        return QueryResponse(
            response=result.answer,
            context_used=result.context_used,
            tokens_generated=len(result.answer.split())  # Rough token count
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
    rag_agent: RAGAgent = Depends(get_rag_agent),
    _: None = Depends(query_rate_limiter)
) -> StreamingResponse:
    """Process a query using RAG with streaming response."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        async def generate_stream() -> AsyncGenerator[str, None]:
            # For now, we'll simulate streaming by yielding the full response
            # The RAG agent doesn't have streaming yet, so we'll return the full response
            result = await rag_agent.query(question=request.query)
            # Split response into words and yield them with delay
            words = result.answer.split()
            for word in words:
                yield word + " "
        
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
    rag_agent: RAGAgent = Depends(get_rag_agent)
) -> dict:
    """Get health status of the query service."""
    try:
        # Initialize RAG agent if not already done
        if not rag_agent._initialized:
            await rag_agent.initialize()
        
        health_status = await rag_agent.get_health_status()
        return {
            "healthy": health_status.overall_healthy,
            "llm_loaded": health_status.llm_loaded,
            "knowledge_graph_ready": health_status.knowledge_graph_ready,
            "knowledge_graph_stats": health_status.knowledge_graph_stats
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"error": str(e), "healthy": False}