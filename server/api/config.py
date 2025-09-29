"""API-specific configuration."""

from pydantic import BaseSettings


class APIConfig(BaseSettings):
    """API-specific configuration."""
    
    # Application settings
    app_name: str = "SawserQ GPT"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # CORS settings
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # Rate limiting settings
    query_rate_limit: int = 50  # requests per minute
    health_rate_limit: int = 100  # requests per minute
    rate_limit_window: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global API config instance
api_config = APIConfig()
