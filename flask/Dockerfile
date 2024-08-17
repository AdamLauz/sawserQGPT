# Pull image with CUDA support
FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Transfer content from the current directory to /app in the container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install audio libraries and essential build tools
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean

# Install python3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev

# Check if Python is installed and print the version
RUN python3 --version

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA
RUN pip install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Add the nvidia-smi command to check GPU status
CMD nvidia-smi && uwsgi app.ini
