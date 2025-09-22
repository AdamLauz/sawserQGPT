"""Dependency injection for FastAPI."""

import logging
from functools import lru_cache # lru_cache is a decorator that allows us to cache the result of a function.

from fastapi import Depends

from app.config import settings
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


@lru_cache()
def get_llm_service() -> LLMService:
    """Get LLM service instance (singleton).
    @lru_cache will return the same instance for the same input, in this case there is no input so it will return the same instance for every call."""
    return LLMService()


@lru_cache()
def get_vector_service() -> VectorService:
    """Get vector service instance (singleton)."""
    return VectorService()


def get_rag_service(
    llm_service: LLMService = Depends(get_llm_service), # Depends is a function that allows us to get the dependencies of a function.
    vector_service: VectorService = Depends(get_vector_service) # Depends is a function that allows us to get the dependencies of a function.
) -> RAGService:
    """Get RAG service instance with dependencies.
    Note that calling this function will use the same llm_service and vector_service which were initialized only once in the lifespan of the application."""
    return RAGService(llm_service, vector_service) # RAGService is a class that allows us to use the RAG service.
