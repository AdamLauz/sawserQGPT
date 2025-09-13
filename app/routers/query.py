"""Query router for RAG endpoints."""

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.dependencies import get_rag_service
from app.exceptions import QueryError
from app.models import QueryRequest, QueryResponse, ErrorResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)) -> QueryResponse:
    """Process a query using RAG."""
    try:
        response, context_used, context_info = await rag_service.query(
            user_query=request.query,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_context=True
        )
        
        return QueryResponse(
            response=response,
            context_used=context_used,
            tokens_generated=len(response.split())  # Rough token count
        )
        
    except QueryError as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query/stream")
async def query_stream(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)) -> StreamingResponse:
    """Process a query using RAG with streaming response."""
    try:
        async def generate_stream() -> AsyncGenerator[str, None]:
            async for token in rag_service.query_stream(
                user_query=request.query,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                use_context=True
            ):
                yield token
        
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
    rag_service: RAGService = Depends(get_rag_service)
) -> dict:
    """Get health status of the query service."""
    try:
        health_status = await rag_service.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"error": str(e), "healthy": False}
