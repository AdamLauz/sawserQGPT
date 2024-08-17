# Pull image with CUDA support
FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Transfer content from the current directory to /app in the container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install essential tools and add Python PPA
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    git \
    build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Check if Python is installed and print the version
RUN python3.10 --version

# Upgrade pip for Python 3.10
RUN python3.10 -m ensurepip --upgrade && python3.10 -m pip install --upgrade pip

# Install Python packages from requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA
RUN python3.10 -m pip install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Add the nvidia-smi command to check GPU status
CMD nvidia-smi && uwsgi app.ini
