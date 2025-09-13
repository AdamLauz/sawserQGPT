#!/usr/bin/env python3
"""Install PyTorch with GPU support."""

import subprocess
import sys
import platform

def install_pytorch_gpu():
    """Install PyTorch with GPU support."""
    print("🚀 Installing PyTorch with GPU support...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is already available")
            print(f"   - PyTorch version: {torch.__version__}")
            print(f"   - CUDA version: {torch.version.cuda}")
            return True
    except ImportError:
        print("PyTorch not installed, proceeding with installation...")
    
    # Determine the right PyTorch installation command
    system = platform.system().lower()
    
    if system == "windows":
        # Windows with CUDA 12.1
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif system == "linux":
        # Linux with CUDA 12.1
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    else:
        # macOS or other systems
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
        print("⚠️  macOS detected - installing CPU-only version")
        print("   For GPU support on macOS, you'll need to use MPS (Metal Performance Shaders)")
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch installed successfully")
        
        # Test installation
        print("\n🧪 Testing installation...")
        import torch
        print(f"   - PyTorch version: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   - CUDA version: {torch.version.cuda}")
            print(f"   - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   - No CUDA GPUs detected")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def install_cpu_only():
    """Install PyTorch CPU-only version."""
    print("💻 Installing PyTorch CPU-only version...")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch CPU-only installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def main():
    """Main installation function."""
    print("🔧 PyTorch Installation Script")
    print("=" * 40)
    
    # Check if user wants GPU or CPU
    choice = input("Do you want to install GPU support? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        success = install_pytorch_gpu()
    else:
        success = install_cpu_only()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Test GPU setup: python test_gpu.py")
        print("2. Start the server: python start_server.py")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
