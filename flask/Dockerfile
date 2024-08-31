# Pull image with CUDA support
ARG CUDA_VERSION="12.1.1"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION

# Base NVidia CUDA Ubuntu image
FROM $DOCKER_FROM AS base

# Set the working directory to /app
WORKDIR /app

# Transfer content from the current directory to /app in the container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install Python plus openssh, which is our minimum set of required packages.
RUN apt-get update -y && \
    apt-get install -y python3.10 python3.10-pip python3.10-venv python3.10-dev && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git git-lfs && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
ARG PYTORCH="2.1.1"
ARG CUDA="121"
RUN pip3 install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu$CUDA

# install wheel and ninja
RUN pip install wheel ninja

RUN python3 -m pip cache purge

# Add the nvidia-smi command to check GPU status
CMD nvidia-smi && uwsgi app.ini
