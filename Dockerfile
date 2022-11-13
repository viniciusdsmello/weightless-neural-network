FROM python:3.9-slim

# metainformation
LABEL org.opencontainers.image.version = "0.0.0"
LABEL org.opencontainers.image.authors = "Vinícius Mello"
LABEL org.opencontainers.image.source = "https://github.com/viniciusdsmello/weightless-neural-network"
LABEL org.opencontainers.image.licenses = "MIT"

RUN apt-get update && apt-get upgrade -y && apt-get clean

# Essential Installs
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app
ENV PYTHONPATH /app

COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade -r requirements.txt

COPY . /app/