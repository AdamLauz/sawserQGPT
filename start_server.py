#!/usr/bin/env python3
"""Startup script for SawserQ GPT server."""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main startup function."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"GPU Enabled: {settings.use_gpu}")
    logger.info(f"LLM Model: {settings.llm_model_name}")
    logger.info(f"Embedding Model: {settings.embedding_model_name}")
    
    # Log GPU information
    settings.log_gpu_status()
    
    # Ensure directories exist
    Path(settings.persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.resources_dir).mkdir(parents=True, exist_ok=True)
    
    # Start server
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            workers=1 if settings.debug else settings.workers,
            log_level="info" if not settings.debug else "debug"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
