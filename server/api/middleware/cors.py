"""CORS middleware configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors_middleware(app: FastAPI) -> None:
    """Setup CORS middleware for the FastAPI application."""
    from server.api.config import api_config
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors_origins,
        allow_credentials=api_config.cors_credentials,
        allow_methods=api_config.cors_methods,
        allow_headers=api_config.cors_headers,
    )
