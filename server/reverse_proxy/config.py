"""Reverse proxy configuration."""

from pydantic import BaseSettings


class ReverseProxyConfig(BaseSettings):
    """Reverse proxy configuration."""
    
    # Nginx settings
    nginx_worker_connections: int = 1024
    nginx_client_max_body_size: str = "10M"
    nginx_client_body_buffer_size: str = "10M"
    
    # Proxy settings
    proxy_connect_timeout: str = "60s"
    proxy_send_timeout: str = "60s"
    proxy_read_timeout: str = "60s"
    
    # Upstream settings
    upstream_server: str = "server:8000"
    upstream_health_endpoint: str = "/api/v1/health"
    
    # Server settings
    server_port: int = 80
    server_name: str = "localhost"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global reverse proxy config instance
reverse_proxy_config = ReverseProxyConfig()
