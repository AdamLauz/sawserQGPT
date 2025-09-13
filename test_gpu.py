#!/usr/bin/env python3
"""Test GPU availability and configuration."""

import torch
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.config import settings

def test_gpu():
    """Test GPU availability and configuration."""
    print("🔍 GPU Configuration Test")
    print("=" * 50)
    
    # Check PyTorch CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ No CUDA GPUs available")
        print("   - Make sure you have CUDA installed")
        print("   - Install PyTorch with CUDA support:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Check environment variables
    print(f"\nEnvironment Variables:")
    print(f"USE_GPU: {os.getenv('USE_GPU', 'Not set')}")
    print(f"DEVICE: {os.getenv('DEVICE', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Check app configuration
    print(f"\nApp Configuration:")
    print(f"use_gpu: {settings.use_gpu}")
    print(f"device: {settings.device}")
    print(f"llm_model: {settings.llm_model_name}")
    print(f"embedding_model: {settings.embedding_model_name}")
    
    # Test tensor operations
    print(f"\nTesting GPU Operations:")
    try:
        if cuda_available and settings.use_gpu:
            # Test tensor creation on GPU
            print("Creating test tensors on GPU...")
            x = torch.randn(1000, 1000, device=settings.device)
            y = torch.randn(1000, 1000, device=settings.device)
            z = torch.mm(x, y)
            print(f"✅ GPU tensor operations working")
            print(f"   - Created tensors on {settings.device}")
            print(f"   - Matrix multiplication successful")
            print(f"   - Result shape: {z.shape}")
            
            # Test memory usage
            if settings.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"   - Memory allocated: {memory_allocated:.1f} MB")
                print(f"   - Memory reserved: {memory_reserved:.1f} MB")
            
        else:
            print("💻 Testing CPU operations...")
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = torch.mm(x, y)
            print(f"✅ CPU tensor operations working")
            print(f"   - Result shape: {z.shape}")
            
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    
    # Test model loading capability
    print(f"\nTesting Model Loading Capability:")
    try:
        from transformers import AutoTokenizer
        print(f"Testing tokenizer loading for {settings.llm_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
        print(f"✅ Tokenizer loaded successfully")
        
        # Test a simple tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        if settings.use_gpu and cuda_available:
            tokens = {k: v.to(settings.device) for k, v in tokens.items()}
        print(f"✅ Tokenization test successful")
        print(f"   - Input: '{test_text}'")
        print(f"   - Tokens: {tokens['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False
    
    print(f"\n🎉 GPU setup test completed successfully!")
    print(f"   - Device: {settings.device}")
    print(f"   - GPU Enabled: {settings.use_gpu}")
    print(f"   - Ready to run: python start_server.py")
    
    return True

if __name__ == "__main__":
    try:
        success = test_gpu()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
