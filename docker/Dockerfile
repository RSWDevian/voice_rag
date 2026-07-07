# docker/Dockerfile (Updated - Remove COPY data/)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt

# Create directories inside container
RUN mkdir -p /app/models /app/logs /app/data /app/data/embeddings

# Create empty sample documents file
RUN echo '[]' > /app/data/sample_documents.json

# Pre-download models (skip if fails)
RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)" || echo "VAD model download skipped"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || echo "Sentence transformer download skipped"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')" || echo "DistilBERT download skipped"

# Copy application (NO data/ here - it will be mounted as volume)
COPY src/ ./src/
COPY static/ ./static/
# REMOVED: COPY data/ ./data/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]