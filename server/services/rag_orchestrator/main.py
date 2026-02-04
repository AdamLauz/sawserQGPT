"""RAG Orchestrator Service - FastMCP-based orchestrator for RAG operations."""

import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import httpx

from .config import RAGOrchestratorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config
config = RAGOrchestratorConfig()


class RAGOrchestratorService:
    """RAG orchestrator service that coordinates between other FastMCP services."""
    
    def __init__(self):
        self.knowledge_graph_url = f"http://{config.knowledge_graph_host}:{config.knowledge_graph_port}"
        self.document_processor_url = f"http://{config.document_processor_host}:{config.document_processor_port}"
        self.llm_url = f"http://{config.llm_host}:{config.llm_port}"
    
    async def process_documents(self, resources_dir: str) -> Dict[str, Any]:
        """Process documents and build knowledge graph."""
        try:
            async with httpx.AsyncClient() as client:
                # Process documents
                doc_response = await client.post(
                    f"{self.document_processor_url}/process_directory",
                    json={"directory_path": resources_dir, "extract_entities": True}
                )
                doc_result = doc_response.json()
                
                if not doc_result["success"]:
                    return {"success": False, "error": doc_result.get("error", "Document processing failed")}
                
                # Add entities to knowledge graph
                total_entities = 0
                for result in doc_result["results"]:
                    if result["success"] and result["entities_count"] > 0:
                        # Here we would extract entities and add them to knowledge graph
                        # For now, we'll just count them
                        total_entities += result["entities_count"]
                
                # Get knowledge graph stats
                kg_response = await client.get(f"{self.knowledge_graph_url}/get_graph_stats")
                kg_stats = kg_response.json()
                
                return {
                    "success": True,
                    "processed_files": doc_result["processed_files"],
                    "total_entities": total_entities,
                    "knowledge_graph_stats": kg_stats
                }
                
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_rag(self, question: str, max_context_entities: int = 10) -> Dict[str, Any]:
        """Process a RAG query."""
        try:
            async with httpx.AsyncClient() as client:
                # Query knowledge graph
                kg_response = await client.post(
                    f"{self.knowledge_graph_url}/query_graph",
                    json={"query": question, "max_results": max_context_entities}
                )
                kg_result = kg_response.json()
                
                # Create context from knowledge graph results
                context = f"Knowledge Graph Context:\n{kg_result['answer']}"
                if kg_result["relevant_entities"]:
                    context += f"\n\nRelevant Entities:\n"
                    for entity in kg_result["relevant_entities"]:
                        context += f"- {entity['text']} ({entity['label']}): {entity.get('description', '')}\n"
                
                # Generate response using LLM
                prompt = f"""You are SawserQGPT, a virtual Circassian history and culture expert.

{context}

Please answer the following question based on the knowledge graph context above. If the context doesn't contain enough information, say so.

Question: {question}

Answer:"""
                
                llm_response = await client.post(
                    f"{self.llm_url}/generate_text",
                    json={"prompt": prompt}
                )
                llm_result = llm_response.json()
                
                return {
                    "answer": llm_result,
                    "context_used": bool(kg_result["relevant_entities"]),
                    "relevant_entities": kg_result["relevant_entities"],
                    "reasoning_path": kg_result["reasoning_path"]
                }
                
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "context_used": False,
                "relevant_entities": [],
                "reasoning_path": []
            }
    
    async def query_rag_stream(self, question: str, max_context_entities: int = 10) -> AsyncGenerator[str, None]:
        """Process a streaming RAG query."""
        try:
            async with httpx.AsyncClient() as client:
                # Query knowledge graph
                kg_response = await client.post(
                    f"{self.knowledge_graph_url}/query_graph",
                    json={"query": question, "max_results": max_context_entities}
                )
                kg_result = kg_response.json()
                
                # Create context from knowledge graph results
                context = f"Knowledge Graph Context:\n{kg_result['answer']}"
                if kg_result["relevant_entities"]:
                    context += f"\n\nRelevant Entities:\n"
                    for entity in kg_result["relevant_entities"]:
                        context += f"- {entity['text']} ({entity['label']}): {entity.get('description', '')}\n"
                
                # Generate streaming response using LLM
                prompt = f"""You are SawserQGPT, a virtual Circassian history and culture expert.

{context}

Please answer the following question based on the knowledge graph context above. If the context doesn't contain enough information, say so.

Question: {question}

Answer:"""
                
                async with client.stream(
                    "POST",
                    f"{self.llm_url}/generate_text_stream",
                    json={"prompt": prompt}
                ) as response:
                    async for chunk in response.aiter_text():
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Error processing streaming RAG query: {e}")
            yield f"Error: {str(e)}"
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        try:
            health_status = {
                "llm_loaded": False,
                "knowledge_graph_ready": False,
                "document_processor_ready": False,
                "overall_healthy": False
            }
            
            async with httpx.AsyncClient() as client:
                # Check LLM service
                try:
                    llm_response = await client.get(f"{self.llm_url}/health_check")
                    llm_health = llm_response.json()
                    health_status["llm_loaded"] = llm_health.get("llm_loaded", False)
                except:
                    health_status["llm_loaded"] = False
                
                # Check Knowledge Graph service
                try:
                    kg_response = await client.get(f"{self.knowledge_graph_url}/health_check")
                    kg_health = kg_response.json()
                    health_status["knowledge_graph_ready"] = kg_health.get("status") == "healthy"
                except:
                    health_status["knowledge_graph_ready"] = False
                
                # Check Document Processor service
                try:
                    doc_response = await client.get(f"{self.document_processor_url}/health_check")
                    doc_health = doc_response.json()
                    health_status["document_processor_ready"] = doc_health.get("status") == "healthy"
                except:
                    health_status["document_processor_ready"] = False
            
            health_status["overall_healthy"] = (
                health_status["llm_loaded"] and 
                health_status["knowledge_graph_ready"] and 
                health_status["document_processor_ready"]
            )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "llm_loaded": False,
                "knowledge_graph_ready": False,
                "document_processor_ready": False,
                "overall_healthy": False,
                "error": str(e)
            }


# Global RAG orchestrator service instance
rag_service = RAGOrchestratorService()

# Create FastMCP application
mcp = FastMCP("RAG Orchestrator Service")


@mcp.tool()
async def process_documents(
    resources_dir: str = Field(..., description="Path to resources directory")
) -> Dict[str, Any]:
    """Process documents and build knowledge graph."""
    try:
        result = await rag_service.process_documents(resources_dir)
        return result
    except Exception as e:
        logger.error(f"Error in process_documents: {e}")
        raise


@mcp.tool()
async def query_rag(
    question: str = Field(..., description="Question to ask"),
    max_context_entities: int = Field(default=10, description="Maximum number of context entities")
) -> Dict[str, Any]:
    """Process a RAG query."""
    try:
        result = await rag_service.query_rag(question, max_context_entities)
        return result
    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        raise


@mcp.tool()
async def query_rag_stream(
    question: str = Field(..., description="Question to ask"),
    max_context_entities: int = Field(default=10, description="Maximum number of context entities")
) -> List[str]:
    """Process a streaming RAG query."""
    try:
        chunks = []
        async for chunk in rag_service.query_rag_stream(question, max_context_entities):
            chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error in query_rag_stream: {e}")
        raise


@mcp.tool()
async def get_health_status() -> Dict[str, Any]:
    """Get health status of all services."""
    try:
        status = await rag_service.get_health_status()
        return status
    except Exception as e:
        logger.error(f"Error in get_health_status: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    
    # Run as FastAPI app
    app = mcp.create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=True
    )