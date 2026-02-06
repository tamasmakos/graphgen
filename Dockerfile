FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Core system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Python 3.11 and minimal build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    wget \
    git \
    procps \
    socat \
    netcat-openbsd \
    openssh-client \
    net-tools \
    iproute2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch (Separated for caching)
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install Heavy Libraries (Separated for caching)
RUN python -m pip install --no-cache-dir gliner[gpu] sentence-transformers nltk spacy

# Copy only requirements first to leverage cache
COPY requirements.txt .

# Install remaining dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -m spacy download en_core_web_lg
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4');"

# Copy source code
COPY graphgen/ ./graphgen
