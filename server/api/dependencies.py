"""Dependency injection for FastAPI."""

import logging
from functools import lru_cache # lru_cache is a decorator that allows us to cache the result of a function.

from fastapi import Depends

from server.config import settings
from server.agent.rag_agent import RAGAgent

logger = logging.getLogger(__name__)


@lru_cache()
def get_rag_agent() -> RAGAgent:
    """Get RAG agent instance (singleton)."""
    return RAGAgent()