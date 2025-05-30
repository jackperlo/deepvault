# Base Python image
FROM python:3.10-slim

# Metadata
LABEL maintainer="perlogiacomo@gmail.com"
LABEL description="Docker image for DeepVault - auto-built from GitHub"

# Set working directory
WORKDIR /deepvault

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone latest DeepVault repo
RUN git clone https://github.com/jackperlo/deepvault.git .

# Copy your requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default execution (change if different main script)
CMD ["/bin/bash"]
