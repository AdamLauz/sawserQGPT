"""LLM Service - FastMCP-based service for language model operations."""

import logging
import asyncio
import os
from contextlib import asynccontextmanager

# Disable OpenAI by setting environment variable
os.environ["OPENAI_API_KEY"] = ""

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from .config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config
config = LLMConfig()


class LLMService:
    """LLM service for text generation and embeddings."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.embedding_tokenizer = None
        self._is_loaded = False
        self._embedding_loaded = False
    
    async def load_model(self) -> None:
        """Load the LLM model."""
        try:
            logger.info(f"Loading LLM model: {config.llm_model_name}")
            logger.info(f"Device: {config.device} (GPU: {config.use_gpu})")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm_model_name,
                use_fast=True,
                trust_remote_code=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine optimal settings
            if config.use_gpu and torch.cuda.is_available():
                torch_dtype = torch.float16
                device_map = "auto"
                logger.info("Using GPU with half precision (float16)")
            else:
                torch_dtype = torch.float32
                device_map = None
                logger.info("Using CPU with full precision (float32)")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            if not config.use_gpu or device_map is None:
                self.model = self.model.to(config.device)
            
            self._is_loaded = True
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    async def load_embedding_model(self) -> None:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {config.embedding_model_name}")
            
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(config.embedding_model_name)
            self._embedding_loaded = True
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate text response."""
        if not self._is_loaded:
            await self.load_model()
        
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
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
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_stream(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
        """Generate streaming text response."""
        if not self._is_loaded:
            await self.load_model()
        
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature
        
        try:
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=30.0, 
                skip_special_tokens=True, 
                skip_prompt=True
            )
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
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
            
            for new_text in streamer:
                yield new_text
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self._embedding_loaded:
            await self.load_embedding_model()
        
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


# Global LLM service instance
llm_service = LLMService()

# Create FastMCP application
mcp = FastMCP("LLM Service")


@mcp.tool()
async def generate_text(
    prompt: str = Field(..., description="Input prompt for text generation"),
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate"),
    temperature: Optional[float] = Field(None, description="Sampling temperature")
) -> str:
    """Generate text using the language model."""
    try:
        response = await llm_service.generate(prompt, max_tokens, temperature)
        return response
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        raise


@mcp.tool()
async def generate_text_stream(
    prompt: str = Field(..., description="Input prompt for text generation"),
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate"),
    temperature: Optional[float] = Field(None, description="Sampling temperature")
) -> List[str]:
    """Generate streaming text using the language model."""
    try:
        chunks = []
        async for chunk in llm_service.generate_stream(prompt, max_tokens, temperature):
            chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error in generate_text_stream: {e}")
        raise


@mcp.tool()
async def generate_embeddings(
    text: str = Field(..., description="Text to generate embeddings for")
) -> List[float]:
    """Generate embeddings for text."""
    try:
        embeddings = await llm_service.embed(text)
        return embeddings
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {e}")
        raise


@mcp.tool()
async def get_model_info() -> dict:
    """Get information about loaded models."""
    return {
        "llm_model": config.llm_model_name,
        "embedding_model": config.embedding_model_name,
        "device": config.device,
        "llm_loaded": llm_service._is_loaded,
        "embedding_loaded": llm_service._embedding_loaded
    }


@mcp.tool()
async def health_check() -> dict:
    """Health check for the LLM service."""
    return {
        "status": "healthy",
        "llm_loaded": llm_service._is_loaded,
        "embedding_loaded": llm_service._embedding_loaded,
        "model": config.llm_model_name,
        "device": config.device
    }


@asynccontextmanager
async def lifespan():
    """Application lifespan manager."""
    logger.info("Starting LLM Service")
    
    try:
        # Initialize LLM service
        await llm_service.load_model()
        await llm_service.load_embedding_model()
        logger.info("LLM Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM Service: {e}")
        raise
    
    yield
    
    logger.info("Shutting down LLM Service")


# Set up lifespan
mcp.add_lifespan(lifespan)

if __name__ == "__main__":
    import uvicorn
    
    # Run as FastAPI app
    app = mcp.create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        reload=True
    )