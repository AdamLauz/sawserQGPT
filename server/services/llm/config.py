"""LLM Service configuration."""

import os
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field, validator
from pydantic_settings import BaseSettings


def _detect_gpu_availability() -> bool:
    """Detect if GPU is available and should be used."""
    if not torch.cuda.is_available():
        return False
    
    if os.getenv("USE_GPU", "").lower() in ("false", "0", "no"):
        return False
    
    if os.getenv("USE_GPU", "").lower() in ("true", "1", "yes"):
        return True
    
    return True


def _get_optimal_device() -> str:
    """Get the optimal device for computation."""
    if _detect_gpu_availability():
        return "cuda"
    return "cpu"


class LLMConfig(BaseSettings):
    """LLM service configuration."""
    
    # Model settings
    llm_model_name: str = "microsoft/DialoGPT-large"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 3
    
    # GPU settings
    use_gpu: bool = Field(default_factory=lambda: _detect_gpu_availability())
    device: str = Field(default_factory=lambda: _get_optimal_device())
    cuda_visible_devices: Optional[str] = Field(default=None, description="CUDA_VISIBLE_DEVICES setting")
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8004
    
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
