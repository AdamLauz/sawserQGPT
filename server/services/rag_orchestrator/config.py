"""RAG Orchestrator Service configuration."""

from pydantic import BaseSettings


class RAGOrchestratorConfig(BaseSettings):
    """RAG orchestrator service configuration."""
    
    # Service URLs
    knowledge_graph_host: str = "localhost"
    knowledge_graph_port: int = 8001
    document_processor_host: str = "localhost"
    document_processor_port: int = 8002
    llm_host: str = "localhost"
    llm_port: int = 8004
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8003
    
    # Timeout settings
    service_timeout: int = 30  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
