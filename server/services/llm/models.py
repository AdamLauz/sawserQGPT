"""LLM Service models."""

from pydantic import BaseModel, Field
from typing import List, Optional


class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")


class GenerateResponse(BaseModel):
    """Response for text generation."""
    text: str = Field(..., description="Generated text")


class EmbedRequest(BaseModel):
    """Request for text embedding."""
    text: str = Field(..., description="Text to embed")


class EmbedResponse(BaseModel):
    """Response for text embedding."""
    embeddings: List[float] = Field(..., description="Text embeddings")


class ModelInfo(BaseModel):
    """Information about loaded models."""
    llm_model: str = Field(..., description="LLM model name")
    embedding_model: str = Field(..., description="Embedding model name")
    device: str = Field(..., description="Device being used")
    llm_loaded: bool = Field(..., description="Whether LLM is loaded")
    embedding_loaded: bool = Field(..., description="Whether embedding model is loaded")
