"""Knowledge Graph Service models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction."""
    text: str = Field(..., description="Text to extract entities from")
    source: str = Field(default="unknown", description="Source of the text")


class EntityExtractionResponse(BaseModel):
    """Response for entity extraction."""
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    relations: List[Dict[str, Any]] = Field(..., description="Extracted relations")


class AddEntitiesRequest(BaseModel):
    """Request to add entities to graph."""
    entities: List[Dict[str, Any]] = Field(..., description="Entities to add")


class AddEntitiesResponse(BaseModel):
    """Response for adding entities."""
    status: str = Field(..., description="Status of the operation")
    added_entities: int = Field(default=0, description="Number of entities added")
    error: Optional[str] = Field(default=None, description="Error message if any")


class QueryRequest(BaseModel):
    """Request to query knowledge graph."""
    query: str = Field(..., description="Query string")
    max_results: int = Field(default=10, description="Maximum number of results")


class QueryResponse(BaseModel):
    """Response for knowledge graph query."""
    answer: str = Field(..., description="Query answer")
    relevant_entities: List[Dict[str, Any]] = Field(..., description="Relevant entities")
    reasoning_path: List[Dict[str, Any]] = Field(..., description="Reasoning path")


class GraphStatsResponse(BaseModel):
    """Response for graph statistics."""
    status: str = Field(..., description="Graph status")
    nodes: int = Field(..., description="Number of nodes")
    edges: int = Field(..., description="Number of edges")
    density: float = Field(..., description="Graph density")
    is_connected: bool = Field(..., description="Whether graph is connected")
    error: Optional[str] = Field(default=None, description="Error message if any")
