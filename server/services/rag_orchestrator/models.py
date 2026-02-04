"""RAG Orchestrator Service models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    question: str = Field(..., description="Question to ask")
    max_context_entities: int = Field(default=10, description="Maximum number of context entities")


class RAGQueryResponse(BaseModel):
    """Response for RAG query."""
    answer: str = Field(..., description="Generated answer")
    context_used: bool = Field(..., description="Whether knowledge graph context was used")
    relevant_entities: List[Dict[str, Any]] = Field(default=[], description="Relevant entities from knowledge graph")
    reasoning_path: List[Dict[str, Any]] = Field(default=[], description="Reasoning path through knowledge graph")


class ProcessDocumentsRequest(BaseModel):
    """Request for processing documents."""
    resources_dir: str = Field(..., description="Path to resources directory")


class ProcessDocumentsResponse(BaseModel):
    """Response for processing documents."""
    success: bool = Field(..., description="Whether processing was successful")
    processed_files: int = Field(default=0, description="Number of files processed")
    total_entities: int = Field(default=0, description="Total entities extracted")
    knowledge_graph_stats: Dict[str, Any] = Field(default={}, description="Knowledge graph statistics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthStatusResponse(BaseModel):
    """Response for health status."""
    llm_loaded: bool = Field(..., description="Whether LLM is loaded")
    knowledge_graph_ready: bool = Field(..., description="Whether knowledge graph is ready")
    document_processor_ready: bool = Field(..., description="Whether document processor is ready")
    overall_healthy: bool = Field(..., description="Overall system health")
    error: Optional[str] = Field(default=None, description="Error message if any")
