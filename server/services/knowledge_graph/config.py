"""Knowledge Graph Service configuration."""

from pydantic import BaseSettings


class KnowledgeGraphConfig(BaseSettings):
    """Knowledge graph service configuration."""
    
    # Knowledge graph settings
    knowledge_graph_dir: str = "./knowledge_graph"
    max_entities_per_document: int = 50
    max_relations_per_document: int = 100
    entity_similarity_threshold: float = 0.8
    relation_confidence_threshold: float = 0.7
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
