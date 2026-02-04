"""RAG Orchestrator Service client using FastMCP."""

import httpx
from typing import Dict, Any, AsyncGenerator


class RAGOrchestratorClient:
    """Client for RAG Orchestrator FastMCP service."""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
    
    async def process_documents(self, resources_dir: str) -> Dict[str, Any]:
        """Process documents and build knowledge graph."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/process_documents",
                    json={"resources_dir": resources_dir}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def query(self, question: str, max_context_entities: int = 10) -> Dict[str, Any]:
        """Process a RAG query."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/query_rag",
                    json={"question": question, "max_context_entities": max_context_entities}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "context_used": False, "relevant_entities": [], "reasoning_path": []}
    
    async def query_stream(self, question: str, max_context_entities: int = 10) -> AsyncGenerator[str, None]:
        """Process a streaming RAG query."""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/query_rag_stream",
                    json={"question": question, "max_context_entities": max_context_entities}
                ) as response:
                    async for chunk in response.aiter_text():
                        yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/get_health_status")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"overall_healthy": False, "error": str(e)}