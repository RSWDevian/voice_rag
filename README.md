# Streaming RAG Voice Assistant

A real-time voice-based Retrieval-Augmented Generation (RAG) system with **< 800ms end-to-end latency**. This project implements a production-ready voice assistant that processes speech input, retrieves relevant context from a knowledge base, generates responses using an LLM, and speaks back to the user.

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)

## Features

- **Voice Input**: Real-time speech-to-text using OpenRouter STT (supports Whisper, Groq, etc.)
- **Intelligent RAG**: Vector search with FAISS + Sentence Transformers for document retrieval
- **Streaming LLM**: Low-latency streaming responses via OpenRouter (GPT-4, Claude, Gemini, etc.)
- **Voice Output**: High-quality text-to-speech using ElevenLabs
- **Document Ingestion**: Upload JSON documents to build your knowledge base
- **Ultra-Low Latency**: Optimized pipeline with < 800ms end-to-end response time
- **WebSocket Streaming**: Real-time bidirectional audio streaming
- **Performance Metrics**: Built-in latency tracking and health monitoring
- **Docker Support**: Easy deployment with Docker Compose


## Prerequisites

- Python 3.10+
- OpenRouter API Key (for LLM, STT)
- ElevenLabs API Key (for TTS)
- (Optional) GCP Service Account for Google TTS

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/streaming-rag-voice.git
cd streaming-rag-voice
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Setup the environment variables in `.env` or `.env.local` etc.

### 5. Run the server
```bash
python -m src.main
```

## Docker Deployment
### Using Docker Compose
```bash
# Build and start all services
docker-compose up --build
```
### Manusl Docker build
```bash
docker build -f docker/Dockerfile -t streaming-rag .
docker run -p 8000:8000 --env-file .env streaming-rag
```

## API Endpoints

| Category | Method | Endpoint | Description | Request Body | Response |
|----------|--------|----------|-------------|--------------|----------|
| **WebSocket** | WS | `/stream` | Real-time audio streaming. Send PCM audio chunks, receive PCM audio response. | Binary audio (PCM 16kHz, int16) | Binary audio (PCM) |
| **Voice & Query** | POST | `/query` | Process a text query through the RAG pipeline | `{"text": "Your question here", "context": []}` | `{"success": true, "response": "...", "intent": "...", "confidence": 0.9, "latency_ms": 123}` |
| | GET | `/` | Demo UI interface | - | HTML page |
| **RAG - Ingest** | POST | `/rag/ingest` | Ingest documents via JSON payload | `{"documents": [{"text": "...", "metadata": {...}}], "collection": "default"}` | `{"success": true, "count": 2, "message": "...", "timestamp": 123}` |
| | POST | `/rag/ingest/file` | Ingest documents from uploaded JSON file | Multipart form with `file` | `{"success": true, "count": 2, "filename": "...", "message": "...", "timestamp": 123}` |
| **RAG - Retrieve** | POST | `/rag/search` | Search for relevant documents | `{"query": "search text", "top_k": 3, "threshold": 0.0}` | `{"success": true, "query": "...", "results": [...], "count": 3, "timestamp": 123}` |
| **RAG - List** | GET | `/rag/documents` | List all ingested documents (paginated) | Query params: `limit=50`, `offset=0` | `{"success": true, "count": 50, "total": 150, "documents": [...], "timestamp": 123}` |
| **RAG - Delete** | DELETE | `/rag/documents` | Clear all documents | - | `{"success": true, "count": 150, "message": "...", "timestamp": 123}` |
| | DELETE | `/rag/documents/{id}` | Delete a specific document by ID | - | `{"success": true, "message": "Document 0 deleted", "timestamp": 123}` |
| **RAG - Reindex** | POST | `/rag/reindex` | Rebuild index from existing documents | - | `{"success": true, "count": 150, "message": "...", "timestamp": 123}` |
| **Health** | GET | `/health` | Health check with component status | - | `{"status": "healthy", "timestamp": 123, "config": {...}, "health": {...}}` |
| | GET | `/health/ready` | Readiness probe for container orchestration | - | `{"status": "ready", "timestamp": 123}` |
| | GET | `/health/live` | Liveness probe for container orchestration | - | `{"status": "alive", "timestamp": 123}` |
| **Monitoring** | GET | `/metrics` | All performance metrics | - | `{"timestamp": 123, "components": {...}, "error_count": 0, "total_metrics": 100}` |
| | GET | `/metrics/latency` | Latency statistics for all components | - | `{"vad": {"avg": 18.5, "p95": 28.0}, "llm": {...}}` |
| | GET | `/metrics/component/{component}` | Component-specific metrics | - | Component statistics |
| | POST | `/metrics/reset` | Reset all metrics | - | `{"status": "reset", "timestamp": 123}` |
| **Info** | GET | `/info` | Application information | - | `{"name": "...", "version": "...", "models": {...}, "endpoints": {...}}` |
| **Status** | GET | `/pipeline/status` | Current pipeline status | - | `{"is_processing": false, "session_id": "...", "transcripts": {...}, "timestamp": 123}` |
| | GET | `/pipeline/stats` | Detailed pipeline statistics | - | `{"is_processing": false, "metrics": {...}, "component_stats": {...}}` |
| | POST | `/pipeline/reset` | Reset pipeline state | - | `{"status": "reset", "timestamp": 123}` |

