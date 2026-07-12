"""
Main FastAPI Application with WebSocket Support
Entry point for the Streaming RAG Voice Assistant
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.config import config
from src.pipeline import get_pipeline, get_simple_pipeline
from src.metrics import get_metrics_collector
from src.vector_search import get_vector_search
from src.utils.logger import get_logger
if config.TTS_ENGINE == "v2":
    from src.tts_v2 import get_tts
else:
    from src.tts import get_tts

logger = get_logger(__name__)


def _active_component_models() -> Dict[str, str]:
    """
    Model actually serving requests right now, accounting for engine
    selection - config.STT_MODEL/LLM_MODEL/TTS_MODEL are the v1 (API-backed)
    model names and stay unchanged regardless of which engine is selected, so
    reporting them unconditionally in /health, /info, and the startup log
    would misreport reality whenever STT_ENGINE/LLM_ENGINE/TTS_ENGINE is "v2".
    """
    return {
        "stt": "moonshine (local, v2)" if config.STT_ENGINE == "v2" else config.STT_MODEL,
        "llm": f"{config.OLLAMA_MODEL} (local, v2)" if config.LLM_ENGINE == "v2" else config.LLM_MODEL,
        "tts": f"piper/{config.PIPER_VOICE} (local, v2)" if config.TTS_ENGINE == "v2" else config.TTS_MODEL,
    }


# ============================================
# Pydantic Models for RAG Endpoints
# ============================================

class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion"""
    documents: List[Dict[str, Any]]
    collection: Optional[str] = "default"

class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion"""
    success: bool
    count: int
    message: str
    timestamp: float

class SearchRequest(BaseModel):
    """Request model for document search"""
    query: str
    top_k: Optional[int] = None
    threshold: Optional[float] = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Streaming RAG Voice Assistant Starting")
    active_models = _active_component_models()
    logger.info(f"Active models: STT={active_models['stt']}, LLM={active_models['llm']}, TTS={active_models['tts']}")
    logger.info(f"Latency Target: < 800ms")
    logger.info("=" * 60)
    
    # Initialize pipeline components
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    # Initialize vector search
    try:
        vs = get_vector_search()
        logger.info(f"Vector search initialized: {vs.get_document_count()} documents")
    except Exception as e:
        logger.warning(f"Vector search initialization warning: {e}")
    
    # Start metrics session
    metrics = get_metrics_collector()
    metrics.start_session("app_startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Streaming RAG Voice Assistant...")
    
    try:
        # Close pipeline resources
        await pipeline.close()
        logger.info("Pipeline closed successfully")
    except Exception as e:
        logger.error(f"Error closing pipeline: {e}")
    
    # End metrics session
    metrics.end_session()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Streaming RAG Voice Assistant",
    description="Real-time voice interaction with < 800ms latency",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Get component instances
pipeline = get_pipeline()
simple_pipeline = get_simple_pipeline()
metrics = get_metrics_collector()
vector_search = get_vector_search()

# Tracks the asyncio Task currently driving the shared `pipeline` singleton
# over /stream, so a new connection can preempt a stale one (see
# websocket_endpoint) instead of either corrupting shared state or rejecting
# the new, legitimate connection.
_active_stream_task: Optional[asyncio.Task] = None


# ============================================
# Web UI Endpoints
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the demo interface"""
    try:
        with open(f"{config.STATIC_DIR}/demo.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("Demo HTML file not found")
        return HTMLResponse(content="""
            <html>
                <body>
                    <h1>Demo Interface Not Found</h1>
                    <p>Please ensure static/demo.html exists</p>
                </body>
            </html>
        """, status_code=404)


# ============================================
# Health Check Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = metrics.get_health_status()
    
    active_models = _active_component_models()
    return {
        "status": health_status["status"],
        "timestamp": time.time(),
        "version": "1.0.0",
        "config": {
            "stt_model": active_models["stt"],
            "llm_model": active_models["llm"],
            "tts_model": active_models["tts"],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE
        },
        "vector_search": {
            "ready": vector_search.is_ready(),
            "document_count": vector_search.get_document_count()
        },
        "health": health_status["checks"]
    }


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe for container orchestration"""
    try:
        # Check if pipeline is ready
        if pipeline.vad and pipeline.stt and pipeline.llm and pipeline.tts:
            return {
                "status": "ready",
                "vector_search_ready": vector_search.is_ready(),
                "document_count": vector_search.get_document_count(),
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "timestamp": time.time()}
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e), "timestamp": time.time()}
        )


@app.get("/health/live")
async def liveness_check():
    """Liveness probe for container orchestration"""
    return {
        "status": "alive",
        "timestamp": time.time()
    }


# ============================================
# Metrics Endpoints
# ============================================

@app.get("/metrics")
async def get_metrics():
    """Get all performance metrics"""
    return metrics.get_all_stats()


@app.get("/metrics/component/{component}")
async def get_component_metrics(component: str):
    """Get metrics for specific component"""
    if component not in metrics.metrics:
        raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
    
    return metrics.get_component_stats(component)


@app.get("/metrics/export")
async def export_metrics():
    """Export metrics as JSON"""
    return metrics.export_metrics("dict")


@app.get("/metrics/latency")
async def get_latency_stats():
    """Get latency statistics for all components"""
    return metrics.get_latency_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics"""
    metrics.reset_all()
    return {"status": "reset", "timestamp": time.time()}


# ============================================
# Pipeline Status Endpoints
# ============================================

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status"""
    return {
        "is_processing": pipeline.is_processing,
        "session_id": pipeline.current_session_id,
        "transcripts": {
            "count": len(pipeline.partial_transcripts),
            "final": pipeline.final_transcript if pipeline.final_transcript else None
        },
        "buffer": {
            "size": len(pipeline.audio_buffer),
            "duration": pipeline.buffer_duration
        },
        "is_responding": pipeline.is_responding,
        "timestamp": time.time()
    }


@app.get("/pipeline/stats")
async def get_pipeline_stats():
    """Get detailed pipeline statistics"""
    return pipeline.get_pipeline_stats()


@app.post("/pipeline/reset")
async def reset_pipeline():
    """Reset pipeline state"""
    pipeline._reset_session_state()
    return {"status": "reset", "timestamp": time.time()}


# ============================================
# Simple Query Endpoint (Text-based)
# ============================================

@app.post("/query")
async def process_query(request: Request):
    """
    Process a text query through the pipeline
    
    Request body:
    {
        "text": "What is the weather today?",
        "context": []  # Optional
    }
    """
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text query is required")
        
        logger.info(f"Processing text query: '{text[:50]}...'")
        
        start_time = time.time()
        result = await simple_pipeline.process_query(text)
        latency = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics.record_latency("query_api", latency)
        metrics.record_success("query_api")
        
        return {
            "success": True,
            "response": result["response"],
            "context": result["context"],
            "latency_ms": result["latency_ms"],
            "timestamp": result["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        metrics.record_error("query_api", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# WebSocket Endpoint (Audio Streaming)
# ============================================

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio

    Client sends audio chunks (PCM 16kHz, int16)
    Server streams back audio chunks (PCM)
    """
    global _active_stream_task

    await websocket.accept()
    logger.info("WebSocket connection established")

    # Track connection
    connection_id = f"ws_{int(time.time())}"
    metrics.record_count("websocket_connections", 1)
    metrics.record_success("websocket_connect")

    # `pipeline` is a shared singleton whose audio_buffer/VAD state is NOT
    # connection-scoped, so two simultaneous streams would interleave audio
    # into the same buffer and corrupt each other. A new connection almost
    # always means the previous one is stale (hard reload, closed tab) rather
    # than a genuine second user - rejecting the new one would make the user
    # wait for the stale session to notice its socket is dead on its own
    # (which can take seconds, since it's usually blocked on an LLM/TTS call,
    # not on reading the socket). Instead, cancel the stale session's task so
    # the new connection can proceed immediately.
    if _active_stream_task is not None and not _active_stream_task.done():
        stale_session = pipeline.current_session_id
        logger.warning(f"Cancelling stale session {stale_session} to make way for {connection_id}")
        _active_stream_task.cancel()
        try:
            await _active_stream_task
        except (asyncio.CancelledError, Exception):
            pass

    _active_stream_task = asyncio.current_task()

    # Set pipeline session
    pipeline.current_session_id = connection_id
    pipeline._reset_session_state()

    # Reuse the shared component instances - no re-instantiation.
    vad = pipeline.vad
    stt = pipeline.stt
    llm = pipeline.llm
    tts = pipeline.tts
    query_builder = pipeline.query_builder

    audio_buffer = bytearray()
    buffer_duration = 0.0
    speech_detected = False
    silence_counter = 0
    max_silence = 5  # ~5 * 256ms chunks (~1.3s) of trailing silence = end of utterance, same as /stt_stream

    try:
        pipeline.is_processing = True

        # One flat loop: VAD -> buffer -> STT -> send transcript -> vector
        # search -> LLM -> TTS -> send audio. Structured like /stt_stream
        # (proven reliable in testing) instead of the old nested
        # process_audio_stream/_process_transcript/_stream_tts generators.
        while True:
            try:
                audio_chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # 1. VAD
            vad_start = time.time()
            is_speech = vad.process_chunk(audio_chunk)
            metrics.record_latency("vad", (time.time() - vad_start) * 1000)

            if is_speech is None:
                continue

            ready_to_process = False
            if is_speech:
                speech_detected = True
                silence_counter = 0
                audio_buffer.extend(audio_chunk)
                buffer_duration += len(audio_chunk) / 2 / config.VAD_SAMPLE_RATE
                if buffer_duration >= config.ASR_CHUNK_DURATION:
                    ready_to_process = True
            elif speech_detected:
                silence_counter += 1
                if silence_counter >= max_silence:
                    ready_to_process = True

            if not ready_to_process:
                continue

            utterance_start = time.time()

            # 2. STT
            stt_start = time.time()
            transcript = await stt.process_audio([bytes(audio_buffer)])
            metrics.record_latency("stt", (time.time() - stt_start) * 1000)

            audio_buffer = bytearray()
            buffer_duration = 0.0
            speech_detected = False
            silence_counter = 0

            if not transcript or not transcript.strip():
                continue

            transcript = transcript.strip()
            logger.info(f"STT result: '{transcript}'")
            pipeline.partial_transcripts.append(transcript)
            pipeline.final_transcript = transcript

            # Surface the transcript immediately, before LLM/TTS run, so it's
            # visible in the UI as a direct correctness check on VAD+STT.
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "transcript": transcript,
                "timestamp": time.time()
            }))

            # 3. Vector search + query build
            query_start = time.time()
            query = query_builder.build_query(transcript, intent="general")
            metrics.record_latency("query_builder", (time.time() - query_start) * 1000)

            vector_start = time.time()
            search_results = vector_search.search(transcript, top_k=config.TOP_K_RESULTS)
            metrics.record_latency("vector_search", (time.time() - vector_start) * 1000)
            context = [r["text"] for r in search_results]

            # 4. LLM (streamed) + 5. TTS, pipelined through StreamingTTS's own
            # buffering instead of one synthesize_text() call per tiny LLM
            # chunk. StreamingTTS.stream_text() (src/tts_v2.py) already
            # buffers/splits text at sentence boundaries across an entire
            # input stream - but calling tts.synthesize_text(llm_chunk) fresh
            # for every ~3-token LLM chunk resets that buffer to empty on
            # every single call, so one response fragmented into dozens of
            # tiny TTS calls, each paying its own ~100ms overhead serially,
            # all counted against the "llm" timer below (which wrapped the
            # whole loop, TTS included). Feeding the LLM's own generator
            # straight into stream_text() once lets it batch properly.
            response_text_parts = []
            first_token_latency_ms = None
            llm_tts_start = time.time()

            async def _llm_text_stream():
                nonlocal first_token_latency_ms
                async for llm_chunk in llm.stream_response(query, context):
                    if first_token_latency_ms is None:
                        first_token_latency_ms = (time.time() - llm_tts_start) * 1000
                        logger.info(f"LLM first token: {first_token_latency_ms:.0f}ms")
                    response_text_parts.append(llm_chunk)
                    yield llm_chunk

            async for audio_out in tts.stream_text(_llm_text_stream()):
                try:
                    await websocket.send_bytes(audio_out)
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected during send")
                    raise

            # "llm" is time-to-first-token (matches the CLAUDE.md target);
            # the combined figure - LLM generation with TTS synthesis and
            # websocket sends interleaved throughout - is inherently
            # inseparable now that they're pipelined together, so it's
            # recorded under its own name rather than mislabeled as either.
            # Per-chunk TTS cost is still visible via tts.get_performance_stats()
            # (PiperTTS tracks its own avg_latency_ms internally).
            llm_tts_latency_ms = (time.time() - llm_tts_start) * 1000
            metrics.record_latency("llm", first_token_latency_ms if first_token_latency_ms is not None else llm_tts_latency_ms)
            metrics.record_latency("llm_tts_stream", llm_tts_latency_ms)

            response_text = "".join(response_text_parts)
            logger.info(f"Response: '{response_text}'")

            if response_text:
                # The frontend's "Response" display box (see demo.html
                # DOM.responseDisplay) is only ever populated by the /query
                # HTTP endpoint otherwise - it stays stuck on its placeholder
                # text through a live /stream voice interaction without this.
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "response": response_text,
                    "timestamp": time.time()
                }))

                query_builder.update_context(query, response_text)
                metrics.record_latency("total_pipeline", (time.time() - utterance_start) * 1000)

    except asyncio.CancelledError:
        # Preempted by a newer connection (see the check at the top of this
        # function) - already logged there and fully cleaned up below, so
        # swallow it here rather than letting Starlette log a scary-looking
        # (but expected/harmless) traceback for a normal cancellation.
        logger.info(f"Session {connection_id} cancelled (superseded by a newer connection)")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        metrics.record_count("websocket_disconnections", 1)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        metrics.record_error("websocket", e)
        
        try:
            # Try to send error message
            error_msg = json.dumps({"error": str(e)})
            await websocket.send_text(error_msg)
        except:
            pass
    
    finally:
        # Clean up
        pipeline.is_processing = False
        metrics.record_count("websocket_closed", 1)
        logger.info(f"WebSocket connection closed: {connection_id}")


# ============================================
# RAG Data Upload & Search Endpoints
# ============================================

@app.post("/rag/ingest", response_model=DocumentIngestResponse)
async def ingest_documents(request: DocumentIngestRequest):
    """
    Ingest documents into the RAG system
    
    Request body:
    {
        "documents": [
            {
                "text": "Document content here...",
                "metadata": {
                    "category": "weather",
                    "source": "api",
                    "id": "doc_001"
                }
            }
        ],
        "collection": "default"  # Optional
    }
    """
    try:
        docs = request.documents
        collection = request.collection
        
        if not docs:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Format documents for ingestion
        formatted_docs = []
        for doc in docs:
            text = doc.get("text", "")
            if not text or not text.strip():
                continue
            formatted_docs.append({
                "text": text,
                "metadata": doc.get("metadata", {})
            })
        
        if not formatted_docs:
            raise HTTPException(status_code=400, detail="No valid documents with text content")
        
        # Add to index
        vector_search.add_documents(formatted_docs)
        
        logger.info(f"Ingested {len(formatted_docs)} documents into collection: {collection}")
        metrics.record_count("rag_ingest", len(formatted_docs))
        metrics.record_success("rag_ingest")
        
        return DocumentIngestResponse(
            success=True,
            count=len(formatted_docs),
            message=f"Successfully ingested {len(formatted_docs)} documents",
            timestamp=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        metrics.record_error("rag_ingest", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ingest/file")
async def ingest_from_file(file: UploadFile = File(...)):
    """
    Ingest documents from a JSON file
    
    Expected JSON format:
    [
        {
            "text": "Document content...",
            "metadata": {"category": "general"}
        }
    ]
    OR
    {
        "documents": [
            {
                "text": "Document content...",
                "metadata": {"category": "general"}
            }
        ]
    }
    """
    try:
        # Read file content
        content = await file.read()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Handle both array and object with 'documents' field
        if isinstance(data, dict) and "documents" in data:
            docs = data["documents"]
        elif isinstance(data, list):
            docs = data
        else:
            raise HTTPException(status_code=400, detail="Invalid document format. Expected array or {'documents': [...]}")
        
        if not docs:
            raise HTTPException(status_code=400, detail="No documents found in file")
        
        # Process documents
        formatted_docs = []
        for doc in docs:
            text = doc.get("text", "")
            if not text or not text.strip():
                continue
            formatted_docs.append({
                "text": text,
                "metadata": doc.get("metadata", {})
            })
        
        if not formatted_docs:
            raise HTTPException(status_code=400, detail="No valid documents with text content")
        
        vector_search.add_documents(formatted_docs)
        
        logger.info(f"Ingested {len(formatted_docs)} documents from file: {file.filename}")
        metrics.record_count("rag_ingest_file", len(formatted_docs))
        metrics.record_success("rag_ingest_file")
        
        return {
            "success": True,
            "count": len(formatted_docs),
            "filename": file.filename,
            "message": f"Successfully ingested {len(formatted_docs)} documents from {file.filename}",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        metrics.record_error("rag_ingest_file", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/documents")
async def get_documents(limit: int = 50, offset: int = 0):
    """
    Get list of ingested documents
    
    Args:
        limit: Maximum number of documents to return (default: 50)
        offset: Number of documents to skip (default: 0)
    """
    try:
        documents = vector_search.documents
        metadata = vector_search.metadata
        
        total = len(documents)
        
        # Apply pagination
        start = offset
        end = min(offset + limit, total)
        
        results = []
        for i in range(start, end):
            results.append({
                "id": i,
                "text": documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i],
                "full_text": documents[i],
                "metadata": metadata[i] if i < len(metadata) else {},
                "length": len(documents[i])
            })
        
        return {
            "success": True,
            "count": len(results),
            "total": total,
            "offset": offset,
            "limit": limit,
            "documents": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/documents")
async def clear_all_documents():
    """
    Clear all documents from the index
    """
    try:
        count = vector_search.get_document_count()
        
        # Reset the index
        vector_search.documents = []
        vector_search.metadata = []
        vector_search._load_empty_index()
        vector_search._is_ready = False
        
        logger.info(f"Cleared all {count} documents")
        metrics.record_count("rag_clear", count)
        metrics.record_success("rag_clear")
        
        return {
            "success": True,
            "count": count,
            "message": f"Cleared {count} documents",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        metrics.record_error("rag_clear", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete a specific document by ID
    """
    try:
        if doc_id < 0 or doc_id >= len(vector_search.documents):
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Get text for logging
        doc_text = vector_search.documents[doc_id][:100] if doc_id < len(vector_search.documents) else ""
        
        # Remove document and metadata
        vector_search.documents.pop(doc_id)
        if doc_id < len(vector_search.metadata):
            vector_search.metadata.pop(doc_id)
        
        # Rebuild index if documents remain
        if vector_search.documents and vector_search.encoder:
            # Rebuild from remaining documents
            embeddings = vector_search.encoder.encode(
                vector_search.documents,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            dimension = embeddings.shape[1]
            import faiss
            vector_search.index = faiss.IndexFlatIP(dimension)
            vector_search.index.add(embeddings.astype(np.float32))
            vector_search._is_ready = True
        else:
            vector_search._load_empty_index()
        
        logger.info(f"Deleted document {doc_id}: '{doc_text[:50]}...'")
        metrics.record_count("rag_delete", 1)
        metrics.record_success("rag_delete")
        
        return {
            "success": True,
            "message": f"Document {doc_id} deleted",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        metrics.record_error("rag_delete", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/search")
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents
    
    Request body:
    {
        "query": "What is the weather like?",
        "top_k": 3,
        "threshold": 0.0  # Optional similarity threshold
    }
    """
    try:
        query = request.query.strip()
        top_k = request.top_k or config.TOP_K_RESULTS
        threshold = request.threshold or 0.0
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        if threshold > 0:
            results = vector_search.search_with_threshold(query, threshold)
        else:
            results = vector_search.search(query, top_k)
        
        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "results": results,
            "count": len(results),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/reindex")
async def reindex_documents():
    """
    Rebuild the index from existing documents
    Useful if the index gets corrupted or for testing
    """
    try:
        count = len(vector_search.documents)
        
        if count == 0:
            return {
                "success": False,
                "message": "No documents to reindex",
                "timestamp": time.time()
            }
        
        # Rebuild index
        embeddings = vector_search.encoder.encode(
            vector_search.documents,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        dimension = embeddings.shape[1]
        import faiss
        vector_search.index = faiss.IndexFlatIP(dimension)
        vector_search.index.add(embeddings.astype(np.float32))
        vector_search._is_ready = True
        
        logger.info(f"Reindexed {count} documents")
        metrics.record_count("rag_reindex", count)
        metrics.record_success("rag_reindex")
        
        return {
            "success": True,
            "count": count,
            "message": f"Reindexed {count} documents",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error reindexing: {e}")
        metrics.record_error("rag_reindex", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    metrics.record_error("app_error", exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": time.time()
        }
    )


# ============================================
# Application Info Endpoint
# ============================================

@app.get("/info")
async def get_info():
    """Get application information"""
    active_models = _active_component_models()
    return {
        "name": "Streaming RAG Voice Assistant",
        "version": "1.0.0",
        "description": "Real-time voice interaction with < 800ms latency",
        "features": {
            "voice_activity_detection": True,
            "streaming_asr": True,
            "intent_classification": True,
            "vector_search": True,
            "streaming_llm": True,
            "streaming_tts": True,
            "context_management": True,
            "performance_metrics": True,
            "document_ingestion": True
        },
        "engines": {
            "stt": config.STT_ENGINE,
            "llm": config.LLM_ENGINE,
            "tts": config.TTS_ENGINE
        },
        "models": {
            "stt": active_models["stt"],
            "llm": active_models["llm"],
            "tts": active_models["tts"],
            "embedding": config.EMBEDDING_MODEL
        },
        "vector_search": {
            "ready": vector_search.is_ready(),
            "document_count": vector_search.get_document_count()
        },
        "performance": {
            "target_latency_ms": 800,
            "current_latency_ms": metrics.get_average_latency("total_pipeline") if metrics.get_average_latency("total_pipeline") > 0 else "measuring..."
        },
        "endpoints": {
            "websocket": "/stream",
            "query": "/query",
            "rag_ingest": "/rag/ingest",
            "rag_search": "/rag/search",
            "health": "/health",
            "metrics": "/metrics",
            "demo": "/"
        },
        "timestamp": time.time()
    }

# ============================================
# TTS Test
# ============================================
@app.websocket("/tts_stream")
async def tts_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for TTS streaming test
    Sends raw PCM audio data (16kHz, mono, 16-bit)
    """
    await websocket.accept()
    logger.info("TTS WebSocket connection established")
    
    try:
        # Receive request
        data = await websocket.receive_text()
        request = json.loads(data)
        text = request.get("text", "")
        voice = request.get("voice", "21m00Tcm4TlvDq8ikWAM")
        
        if not text:
            await websocket.send_text(json.dumps({"error": "No text provided"}))
            await websocket.close()
            return
        
        logger.info(f"TTS Request: text='{text[:50]}...', voice={voice}")

        # The per-request voice override only makes sense for the ElevenLabs
        # engine (v1) - Piper (v2) uses a single voice fixed at model-load
        # time (PIPER_VOICE), so mutating ELEVENLABS_VOICE_ID here would be a
        # silent no-op against tts_test.html's voice buttons (all ElevenLabs
        # voice IDs) whenever TTS_ENGINE=v2.
        using_elevenlabs = config.TTS_ENGINE != "v2"
        original_voice = config.ELEVENLABS_VOICE_ID if using_elevenlabs else None
        if using_elevenlabs:
            config.ELEVENLABS_VOICE_ID = voice
        else:
            logger.debug(
                f"Ignoring requested voice='{voice}' - active TTS engine is Piper (v2), "
                f"which always uses PIPER_VOICE={config.PIPER_VOICE}"
            )

        try:
            tts = get_tts()
            chunk_count = 0

            # Stream audio chunks as raw PCM
            async for audio_chunk in tts.synthesize_text(text):
                # Raw PCM bytes, 16kHz mono int16 - from ElevenLabs (v1) or
                # Piper (v2) depending on TTS_ENGINE.
                chunk_count += 1
                logger.debug(f"Sending chunk {chunk_count}: {len(audio_chunk)} bytes")

                # Send as binary (raw PCM)
                await websocket.send_bytes(audio_chunk)

                # Small delay for streaming feel
                await asyncio.sleep(0.01)

            logger.info(f"TTS complete: {chunk_count} chunks sent")

        finally:
            # Restore original voice
            if using_elevenlabs:
                config.ELEVENLABS_VOICE_ID = original_voice
        
        await websocket.close()
        
    except Exception as e:
        logger.error(f"TTS WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
        await websocket.close()

# ============================================
# STT Test
# ============================================
# src/main.py - Complete fixed /stt_stream endpoint

@app.websocket("/stt_stream")
async def stt_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for STT testing only - returns JSON with transcripts
    """
    await websocket.accept()
    logger.info("STT WebSocket connection established")
    
    # Get pipeline components
    pipeline = get_pipeline()
    vad = pipeline.vad
    stt = pipeline.stt
    
    # Audio buffer
    audio_buffer = bytearray()
    buffer_duration = 0.0
    speech_detected = False
    silence_counter = 0
    max_silence = 5  # 150ms silence = end of speech

    # stt_test.html reads data.vad_latency/stt_latency/total_latency off every
    # transcript message - last_vad_latency_ms tracks the most recent VAD call
    # (VAD runs once per incoming chunk, not once per utterance) and
    # utterance_start_time marks when the current utterance's buffering began,
    # so total_latency reflects speech-start to transcript-ready.
    last_vad_latency_ms = 0.0
    utterance_start_time = None

    async def _finish_utterance(reason: str):
        nonlocal audio_buffer, buffer_duration, speech_detected, silence_counter, utterance_start_time
        logger.info(f"Processing buffer ({reason}): {len(audio_buffer)} bytes")
        try:
            await websocket.send_text(json.dumps({"type": "processing"}))
        except Exception:
            pass
        transcript, stt_latency_ms = await process_stt_buffer(audio_buffer, stt)
        if transcript:
            total_latency_ms = (time.time() - utterance_start_time) * 1000 if utterance_start_time else stt_latency_ms
            await send_transcript(websocket, transcript, last_vad_latency_ms, stt_latency_ms, total_latency_ms)
        audio_buffer = bytearray()
        buffer_duration = 0.0
        speech_detected = False
        silence_counter = 0
        utterance_start_time = None

    try:
        while True:
            try:
                audio_chunk = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=0.5
                )
                logger.debug(f"Received audio chunk: {len(audio_chunk)} bytes")
            except asyncio.TimeoutError:
                # If we have buffered audio, process it after timeout
                if len(audio_buffer) > 0:
                    await _finish_utterance("timeout")
                continue
            except WebSocketDisconnect:
                logger.info("STT WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"STT WebSocket receive error: {e}")
                break

            # Check if audio chunk is valid
            if not audio_chunk or len(audio_chunk) < 100:
                continue

            # Run VAD
            try:
                vad_start = time.time()
                is_speech = vad.process_chunk(audio_chunk)
                last_vad_latency_ms = (time.time() - vad_start) * 1000
            except Exception as e:
                logger.error(f"VAD error: {e}")
                continue

            if is_speech is None:
                continue

            if is_speech:
                # Speech detected
                if not speech_detected:
                    utterance_start_time = time.time()
                speech_detected = True
                silence_counter = 0
                audio_buffer.extend(audio_chunk)
                buffer_duration += len(audio_chunk) / 2 / config.VAD_SAMPLE_RATE

                # Process when buffer is full (2 seconds)
                if buffer_duration >= config.ASR_CHUNK_DURATION:
                    await _finish_utterance("speech buffer full")
            else:
                # Silence
                if speech_detected:
                    silence_counter += 1
                    # End of speech after 150ms silence
                    if silence_counter >= max_silence:
                        await _finish_utterance("after silence")
                else:
                    # No speech yet, ignore silence
                    pass

        # Process remaining buffer on close
        if len(audio_buffer) > 0:
            await _finish_utterance("final flush")

    except WebSocketDisconnect:
        logger.info("STT WebSocket disconnected")
    except Exception as e:
        logger.error(f"STT WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
    finally:
        logger.info("STT WebSocket connection closed")


async def process_stt_buffer(audio_buffer: bytearray, stt) -> tuple[Optional[str], float]:
    """Process the audio buffer through STT, returning (transcript, stt_latency_ms)"""
    if len(audio_buffer) < 512:  # Minimum 512 samples
        logger.debug(f"Buffer too small: {len(audio_buffer)} bytes")
        return None, 0.0

    try:
        stt_start = time.time()

        # Convert buffer to list for STT
        audio_chunks = [bytes(audio_buffer)]
        transcript = await stt.process_audio(audio_chunks)

        stt_latency = (time.time() - stt_start) * 1000
        if transcript:
            logger.info(f"STT result: '{transcript}' ({stt_latency:.0f}ms)")
        else:
            logger.debug(f"STT returned None after {stt_latency:.0f}ms")
        return transcript, stt_latency

    except Exception as e:
        logger.error(f"STT processing error: {e}")
        return None, 0.0


async def send_transcript(
    websocket,
    transcript: str,
    vad_latency: Optional[float] = None,
    stt_latency: Optional[float] = None,
    total_latency: Optional[float] = None
):
    """Send transcript (with per-stage latencies, for stt_test.html's readouts) to the client"""
    if not transcript or transcript.strip() == '':
        return

    try:
        response = {
            "type": "transcript",
            "transcript": transcript.strip(),
            "vad_latency": vad_latency,
            "stt_latency": stt_latency,
            "total_latency": total_latency,
            "timestamp": time.time()
        }
        await websocket.send_text(json.dumps(response))
        logger.info(f"Sent transcript: {transcript}")
    except Exception as e:
        logger.error(f"Error sending transcript: {e}")

# src/main.py - Add these routes after the static mount

# ============================================
# Test Routes
# ============================================

@app.get("/test/stt")
async def get_stt_test():
    """STT Test Page"""
    try:
        with open(f"{config.STATIC_DIR}/stt_test.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error(f"STT test page not found at {config.STATIC_DIR}/stt_test.html")
        return HTMLResponse(content="""
            <html>
                <body>
                    <h1>STT Test Page Not Found</h1>
                    <p>Please create static/stt_test.html</p>
                </body>
            </html>
        """, status_code=404)


@app.get("/test/tts")
async def get_tts_test():
    """TTS Test Page"""
    try:
        with open(f"{config.STATIC_DIR}/tts_test.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error(f"TTS test page not found at {config.STATIC_DIR}/tts_test.html")
        return HTMLResponse(content="""
            <html>
                <body>
                    <h1>TTS Test Page Not Found</h1>
                    <p>Please create static/tts_test.html</p>
                </body>
            </html>
        """, status_code=404)

# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "src.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,  # Single worker for streaming
        log_level=config.LOG_LEVEL.lower(),
        reload=config.DEBUG,
        ws_ping_interval=5.0,
        ws_ping_timeout=10.0
    )