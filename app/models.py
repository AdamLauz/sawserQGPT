"""Pydantic models for request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    max_tokens: Optional[int] = Field(None, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate and clean query input."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    response: str = Field(..., description="Generated response")
    context_used: bool = Field(..., description="Whether context was used in generation")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    llm_loaded: bool = Field(..., description="Whether the LLM model is loaded")
    vector_db_ready: bool = Field(..., description="Whether vector database is ready")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: Optional[str] = Field(None, description="Additional error details")


class ContextInfo(BaseModel):
    """Context information model."""
    
    source_nodes: List[str] = Field(..., description="Source node texts")
    similarity_scores: List[float] = Field(..., description="Similarity scores")
    total_sources: int = Field(..., description="Total number of source nodes")
