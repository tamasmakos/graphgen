FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11 and system build deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    git \
    wget \
    procps \
    socat \
    netcat-openbsd \
    iproute2 \
    net-tools \
    ca-certificates \
    lsb-release \
    gnupg \
    openssh-client \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install modern Node.js (v20 LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python and python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install pip for Python 3.11 explicitly if needed, but using python -m pip is safer
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# LAYER 1: Heavy Frameworks (Cached)
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# LAYER 2: Heavy Libraries & NLTK (Cached)
# Installing nltk here ensures it's available for the download step
RUN python -m pip install --no-cache-dir gliner[gpu] sentence-transformers nltk

COPY requirements.txt .

# LAYER 3: Remaining Dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Download Spacy model (Re-enabled now that pip is fixed)
RUN python -m spacy download en_core_web_lg

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4');"

COPY graphgen/ ./graphgen
