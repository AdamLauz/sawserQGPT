# Pull image with CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

# Transfer content from the current directory to /app in the container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install essential build tools and Python 3.10
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y git build-essential python3.10 python3.10-venv python3.10-dev && \
    apt-get clean

# Create and activate a Python virtual environment
RUN python3.10 -m venv /venv

# Ensure the virtual environment is used for running commands
ENV PATH="/venv/bin:$PATH"

# Check if Python is installed and print the version
RUN python --version

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA
RUN pip install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# install wheel and ninja
RUN pip install wheel ninja

# Add the nvidia-smi command to check GPU status
CMD nvidia-smi && uwsgi app.ini
