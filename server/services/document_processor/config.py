"""Document Processor Service configuration."""

from pydantic import BaseSettings


class DocumentProcessorConfig(BaseSettings):
    """Document processor service configuration."""
    
    # Document processing settings
    resources_dir: str = "./agent/resources"
    chunk_size: int = 256
    chunk_overlap: int = 25
    similarity_cutoff: float = 0.5
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8002
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
