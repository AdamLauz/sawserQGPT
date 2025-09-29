"""RAG Agent wrapped with FastMCP for consistent tool architecture."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from server.agent.config import agent_config
from server.api.exceptions import ModelLoadError
from server.agent.knowledge_graph_tool import knowledge_graph_tool
from server.agent.document_processor_tool import document_processor_tool

logger = logging.getLogger(__name__)


class ProcessDocumentsRequest(BaseModel):
    """Request for processing documents."""
    resources_dir: str = Field(..., description="Path to resources directory")


class ProcessDocumentsResponse(BaseModel):
    """Response for processing documents."""
    success: bool = Field(..., description="Whether processing was successful")
    processed_files: int = Field(default=0, description="Number of files processed")
    total_entities: int = Field(default=0, description="Total entities extracted")
    knowledge_graph_stats: Dict[str, Any] = Field(default={}, description="Knowledge graph statistics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    question: str = Field(..., description="Question to ask")
    max_context_entities: int = Field(default=10, description="Maximum number of context entities")


class QueryResponse(BaseModel):
    """Response for querying the RAG system."""
    answer: str = Field(..., description="Generated answer")
    context_used: bool = Field(..., description="Whether knowledge graph context was used")
    relevant_entities: List[Dict[str, Any]] = Field(default=[], description="Relevant entities from knowledge graph")
    reasoning_path: List[Dict[str, Any]] = Field(default=[], description="Reasoning path through knowledge graph")


class HealthStatusResponse(BaseModel):
    """Response for health status."""
    llm_loaded: bool = Field(..., description="Whether LLM is loaded")
    knowledge_graph_ready: bool = Field(..., description="Whether knowledge graph is ready")
    knowledge_graph_stats: Dict[str, Any] = Field(default={}, description="Knowledge graph statistics")
    overall_healthy: bool = Field(..., description="Overall system health")
    error: Optional[str] = Field(default=None, description="Error message if any")


class RAGAgent:
    """RAG Agent with integrated LLM functionality and FastMCP for consistent tool architecture."""
    
    def __init__(self):
        # LLM components
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # MCP tools
        self.knowledge_graph = knowledge_graph_tool
        self.document_processor = document_processor_tool
        self.mcp = FastMCP("RAGAgent")
        self._setup_tools()
    
    async def initialize(self):
        """Initialize the RAG agent."""
        try:
            logger.info("Initializing RAG agent...")
            await self.load_model()
            logger.info("RAG agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG agent: {e}")
            raise
    
    async def load_model(self) -> None:
        """Load the lightweight LLM model asynchronously."""
        try:
            logger.info(f"Loading LLM model: {agent_config.llm_model_name}")
            logger.info(f"Device: {agent_config.device} (GPU: {agent_config.use_gpu})")
            
            # Log GPU status
            agent_config.log_gpu_status()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                agent_config.llm_model_name,
                use_fast=True,
                trust_remote_code=False
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine optimal torch dtype and device map
            if agent_config.use_gpu and torch.cuda.is_available():
                torch_dtype = torch.float16
                device_map = "auto"
                logger.info("Using GPU with half precision (float16)")
            else:
                torch_dtype = torch.float32
                device_map = None
                logger.info("Using CPU with full precision (float32)")
            
            # Load model with optimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                agent_config.llm_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Ensure model is on the correct device
            if not agent_config.use_gpu or device_map is None:
                self.model = self.model.to(agent_config.device)
            
            # Log memory usage
            if agent_config.use_gpu and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            self._is_loaded = True
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate a response for the given prompt."""
        if not self.is_loaded:
            await self.load_model()
        
        max_tokens = max_tokens or agent_config.max_tokens
        temperature = temperature or agent_config.temperature
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_response_stream(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response for the given prompt."""
        if not self.is_loaded:
            await self.load_model()
        
        max_tokens = max_tokens or agent_config.max_tokens
        temperature = temperature or agent_config.temperature
        
        try:
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=30.0, 
                skip_special_tokens=True, 
                skip_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate in a separate thread
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the response
            for new_text in streamer:
                yield new_text
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}"
    
    def _setup_tools(self):
        """Setup MCP tools for the RAG agent."""
        
        @self.mcp.tool()
        async def process_documents(request: ProcessDocumentsRequest) -> ProcessDocumentsResponse:
            """Process all documents in a directory and build knowledge graph."""
            try:
                logger.info(f"Processing documents in {request.resources_dir}")
                
                # Process all documents in directory
                process_result = await self.document_processor.process_directory(
                    directory_path=request.resources_dir,
                    extract_entities=True
                )
                
                if not process_result["success"]:
                    return ProcessDocumentsResponse(
                        success=False,
                        error=process_result["error"]
                    )
                
                # Add entities to knowledge graph
                total_entities = 0
                for result in process_result["results"]:
                    if result["success"] and result["entities_count"] > 0:
                        total_entities += result["entities_count"]
                
                # Get knowledge graph stats
                kg_stats = await self.knowledge_graph.get_graph_stats()
                
                return ProcessDocumentsResponse(
                    success=True,
                    processed_files=process_result["processed_files"],
                    total_entities=total_entities,
                    knowledge_graph_stats=kg_stats
                )
                
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                return ProcessDocumentsResponse(
                    success=False,
                    error=str(e)
                )
        
        @self.mcp.tool()
        async def query(request: QueryRequest) -> QueryResponse:
            """Query the knowledge graph and generate a response."""
            try:
                logger.info(f"Processing query: {request.question}")
                
                # Query knowledge graph
                kg_response = await self.knowledge_graph.query_knowledge_graph(
                    query=request.question,
                    max_results=request.max_context_entities
                )
                
                # Create context from knowledge graph results
                context = f"Knowledge Graph Context:\n{kg_response.answer}"
                if kg_response.relevant_entities:
                    context += f"\n\nRelevant Entities:\n"
                    for entity in kg_response.relevant_entities:
                        context += f"- {entity['text']} ({entity['label']}): {entity.get('description', '')}\n"
                
                # Generate response using integrated LLM
                prompt = f"""You are SawserQGPT, a virtual Circassian history and culture expert.

{context}

Please answer the following question based on the knowledge graph context above. If the context doesn't contain enough information, say so.

Question: {request.question}

Answer:"""
                
                response = await self.generate_response(prompt)
                
                return QueryResponse(
                    answer=response,
                    context_used=bool(kg_response.relevant_entities),
                    relevant_entities=kg_response.relevant_entities,
                    reasoning_path=kg_response.reasoning_path
                )
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return QueryResponse(
                    answer=f"I encountered an error processing your question: {str(e)}",
                    context_used=False,
                    relevant_entities=[],
                    reasoning_path=[]
                )
        
        @self.mcp.tool()
        async def get_health_status() -> HealthStatusResponse:
            """Get health status of the agent and its tools."""
            try:
                llm_loaded = self.is_loaded
                kg_stats = await self.knowledge_graph.get_graph_stats()
                
                return HealthStatusResponse(
                    llm_loaded=llm_loaded,
                    knowledge_graph_ready=kg_stats.get("status") == "ready",
                    knowledge_graph_stats=kg_stats,
                    overall_healthy=llm_loaded and kg_stats.get("status") == "ready"
                )
                
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                return HealthStatusResponse(
                    llm_loaded=False,
                    knowledge_graph_ready=False,
                    knowledge_graph_stats={"status": "error"},
                    overall_healthy=False,
                    error=str(e)
                )
    
    async def start(self):
        """Start the MCP agent."""
        await self.mcp.run()
    
