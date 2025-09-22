"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field, validator # Field is a class that allows us to create a field for the settings object. validator is a function that allows us to validate the settings object.
from pydantic_settings import BaseSettings # BaseSettings is a class that allows us to create a settings object.


def _detect_gpu_availability() -> bool:
    """Detect if GPU is available and should be used."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return False
    
    # Check if explicitly disabled via environment
    if os.getenv("USE_GPU", "").lower() in ("false", "0", "no"):
        return False
    
    # Check if explicitly enabled via environment
    if os.getenv("USE_GPU", "").lower() in ("true", "1", "yes"):
        return True
    
    # Auto-detect: use GPU if available and not explicitly disabled
    return True


def _get_optimal_device() -> str:
    """Get the optimal device for computation."""
    if _detect_gpu_availability():
        return "cuda"
    return "cpu"


def _get_gpu_info() -> dict:
    """Get GPU information for logging."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "count": torch.cuda.device_count(), # count of GPUs
        "current_device": torch.cuda.current_device(), # current GPU.
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3, # total memory of the GPU. Units are in GiB.
            "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3, # memory allocated to the GPU. Units are GiB.
            "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3, # memory reserved for the GPU. Units are GiB.
        }
        info["devices"].append(device_info)
    
    return info


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "SawserQ GPT"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Model settings - Simple open source models
    llm_model_name: str = "microsoft/DialoGPT-medium"  # Simple, reliable model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2" # Simple embeddings model
    max_tokens: int = 512 # Maximum tokens to generate
    temperature: float = 0.7 # Sampling temperature
    top_k: int = 3 # Top-k sampling, the number of tokens to sample from the model.
    
    # Vector database settings
    persist_dir: str = "./storage" # Directory to store the vector database.
    chunk_size: int = 256 # Chunk size for the vector database. This is the number of tokens in each chunk.
    chunk_overlap: int = 25 # Chunk overlap for the vector database. This is the number of tokens to overlap between chunks.
    similarity_cutoff: float = 0.5 # Similarity cutoff for the vector database. This is the similarity score below which chunks are considered similar.
    
    # Resources directory
    resources_dir: str = "./resources" # Directory to store the resources. e.g. PDF files.
    
    # Server settings
    host: str = "0.0.0.0" # Host to bind to.    
    port: int = 8000 # Port to bind to.
    workers: int = 1 # Number of workers to use.
    
    # GPU settings
    use_gpu: bool = Field(default_factory=lambda: _detect_gpu_availability())
    device: str = Field(default_factory=lambda: _get_optimal_device())
    cuda_visible_devices: Optional[str] = Field(default=None, description="CUDA_VISIBLE_DEVICES setting")
    
    @validator("persist_dir", "resources_dir")
    def validate_directories(cls, v):
        """Ensure directories exist or create them."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("device")
    def validate_device(cls, v, values):
        """Validate device setting based on GPU availability."""
        if v == "cuda" and not values.get("use_gpu", False):
            return "cpu"
        return v
    
    @validator("cuda_visible_devices")
    def validate_cuda_devices(cls, v):
        """Validate CUDA_VISIBLE_DEVICES setting."""
        if v is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = v
        return v
    
    def get_gpu_info(self) -> dict:
        """Get GPU information for logging and monitoring."""
        return _get_gpu_info()
    
    def log_gpu_status(self) -> None:
        """Log GPU status for debugging."""
        gpu_info = self.get_gpu_info()
        if gpu_info["available"]:
            print(f"🚀 GPU Available: {gpu_info['count']} device(s)")
            for device in gpu_info["devices"]:
                print(f"   - GPU {device['id']}: {device['name']} ({device['memory_total']:.1f} GB)")
        else:
            print("💻 Using CPU (GPU not available)")
    
    class Config: # What is this? It is a class that allows us to configure the settings object.
        env_file = ".env" # The environment file to use.
        env_file_encoding = "utf-8" # The encoding of the environment file.
        case_sensitive = False # Whether to case sensitive the environment variables.


# Global settings instance
settings = Settings()
