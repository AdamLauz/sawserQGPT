# Pull Python3.10
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install audio libraries and essential build tools
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Install python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install wheel and ninja
RUN pip install wheel ninja


# Install torch with CUDA
RUN pip install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]
