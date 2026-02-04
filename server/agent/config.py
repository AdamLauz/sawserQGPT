"""Agent-specific configuration."""

import os
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field, validator
from pydantic_settings import BaseSettings


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
        "count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,
            "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,
        }
        info["devices"].append(device_info)
    
    return info


class AgentConfig(BaseSettings):
    """Agent-specific configuration."""
    
    # Model settings
    llm_model_name: str = "microsoft/DialoGPT-large"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Knowledge Graph settings
    knowledge_graph_dir: str = "./knowledge_graph"
    max_entities_per_document: int = 50
    max_relations_per_document: int = 100
    entity_similarity_threshold: float = 0.8
    relation_confidence_threshold: float = 0.7
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 3
    
    # Vector database settings
    persist_dir: str = "./agent/storage"
    chunk_size: int = 256
    chunk_overlap: int = 25
    similarity_cutoff: float = 0.5
    
    # Resources directory
    resources_dir: str = "./agent/resources"
    
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global agent config instance
agent_config = AgentConfig()
