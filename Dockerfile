FROM python:3.8-slim-buster

# metainformation
LABEL org.opencontainers.image.version = "0.0.0"
LABEL org.opencontainers.image.authors = "Vinícius Mello"
LABEL org.opencontainers.image.source = "https://github.com/viniciusdsmello/weightless-neural-network"
LABEL org.opencontainers.image.licenses = "MIT"

RUN apt-get update && apt-get upgrade -y && apt-get clean

# Essential Installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \ 
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    python3-opencv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade -r requirements.txt
