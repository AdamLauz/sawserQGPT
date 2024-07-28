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

# Install python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for GPU usage
ENV USE_GPU=false

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]