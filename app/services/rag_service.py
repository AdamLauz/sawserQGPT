"""RAG service that combines LLM and vector database."""

import logging
from typing import AsyncGenerator, Optional, Tuple

from app.config import settings
from app.exceptions import QueryError
from app.models import ContextInfo
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service that combines LLM and vector database."""
    
    def __init__(self, llm_service: LLMService, vector_service: VectorService):
        self.llm_service = llm_service
        self.vector_service = vector_service
    
    async def query(self, user_query: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, use_context: bool = True) -> Tuple[str, bool, ContextInfo]:
        """Process a query using RAG."""
        try:
            context = ""
            context_used = False
            context_info = ContextInfo(
                source_nodes=[],
                similarity_scores=[],
                total_sources=0
            )
            
            # Get context from vector database if requested
            if use_context:
                try:
                    context, source_texts, similarity_scores = await self.vector_service.get_context(user_query)
                    if context and source_texts:
                        context_used = True
                        context_info = ContextInfo(
                            source_nodes=source_texts,
                            similarity_scores=similarity_scores,
                            total_sources=len(source_texts)
                        )
                except Exception as e:
                    logger.warning(f"Failed to get context: {e}")
                    context = ""
            
            # Create prompt with context
            prompt = self._create_prompt(user_query, context)
            
            # Generate response
            response = await self.llm_service.generate_response(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response, context_used, context_info
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise QueryError(f"Failed to process query: {e}")
    
    async def query_stream(self, user_query: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, use_context: bool = True) -> AsyncGenerator[str, None]:
        """Process a query using RAG with streaming response."""
        try:
            context = ""
            context_used = False
            
            # Get context from vector database if requested
            if use_context:
                try:
                    context, _, _ = await self.vector_service.get_context(user_query)
                    if context:
                        context_used = True
                except Exception as e:
                    logger.warning(f"Failed to get context: {e}")
                    context = ""
            
            # Create prompt with context
            prompt = self._create_prompt(user_query, context)
            
            # Generate streaming response
            async for token in self.llm_service.generate_stream(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature
            ):
                yield token
                
        except Exception as e:
            logger.error(f"Error processing streaming query: {e}")
            yield f"Error: {str(e)}"
    
    def _create_prompt(self, user_query: str, context: str = "") -> str:
        """Create a prompt with optional context."""
        if context:
            return f"""You are SawserQGPT, a virtual Circassian history and culture expert. 
You communicate in clear, accessible language and use facts and reliable numbers when available.
You tailor your responses to match the user's input, providing helpful and accurate information.

{context}

Please respond to the following user's input. Use the context above if it is helpful.

User: {user_query}

Assistant:"""
        else:
            return f"""You are SawserQGPT, a virtual Circassian history and culture expert. 
You communicate in clear, accessible language and use facts and reliable numbers when available.
You tailor your responses to match the user's input, providing helpful and accurate information.

User: {user_query}

Assistant:"""
    
    async def get_health_status(self) -> dict:
        """Get health status of the RAG service."""
        try:
            llm_loaded = self.llm_service.is_loaded
            vector_ready = self.vector_service.is_ready
            
            return {
                "llm_loaded": llm_loaded,
                "vector_ready": vector_ready,
                "overall_healthy": llm_loaded and vector_ready
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "llm_loaded": False,
                "vector_ready": False,
                "overall_healthy": False,
                "error": str(e)
            }
