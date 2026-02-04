FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11 and system build deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy model
RUN python -m spacy download en_core_web_lg

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4');"

COPY graphgen/ ./graphgen
