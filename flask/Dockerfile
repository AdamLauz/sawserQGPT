# Pull Python3.10
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# List contents of /app to verify directories
RUN ls -l /app

# Install audio libraries
RUN apt-get clean
RUN apt-get update && apt-get install

# Upgrade pip
RUN pip install --upgrade pip

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# autoGPT build from source
RUN pip install wheel ninja
RUN git clone https://github.com/PanQiWei/AutoGPTQ
RUN cd AutoGPTQ
RUN pip install .
RUN cd ..


# Install torch with cuda
RUN pip install torch torchvision torchaudio --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]