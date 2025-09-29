"""Dependency injection for FastAPI."""

import logging
from functools import lru_cache

from fastapi import Depends
from server.api.services.rag_orchestrator_client import RAGOrchestratorClient

logger = logging.getLogger(__name__)


@lru_cache()
def get_rag_orchestrator_client() -> RAGOrchestratorClient:
    """Get RAG orchestrator client instance (singleton)."""
    return RAGOrchestratorClient()