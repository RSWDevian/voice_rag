"""
Main FastAPI Application with WebSocket Support
Entry point for the Streaming RAG Voice Assistant
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import config
from src.pipeline import get_pipeline, get_simple_pipeline
from src.metrics import get_metrics_collector
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Streaming RAG Voice Assistant Starting")
    logger.info(f"OpenRouter Models: STT={config.STT_MODEL}, LLM={config.LLM_MODEL}, TTS={config.TTS_MODEL}")
    logger.info(f"Latency Target: < 800ms")
    logger.info("=" * 60)
    
    # Initialize pipeline components
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
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
    
    return {
        "status": health_status["status"],
        "timestamp": time.time(),
        "version": "1.0.0",
        "config": {
            "stt_model": config.STT_MODEL,
            "llm_model": config.LLM_MODEL,
            "tts_model": config.TTS_MODEL,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE
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
            "intent": result["intent"],
            "confidence": result["confidence"],
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
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Track connection
    connection_id = f"ws_{int(time.time())}"
    metrics.record_count("websocket_connections", 1)
    metrics.record_success("websocket_connect")
    
    # Set pipeline session
    pipeline.current_session_id = connection_id
    pipeline._reset_session_state()
    
    try:
        # Process audio stream
        async for audio_chunk in pipeline.process_audio_stream(websocket):
            try:
                await websocket.send_bytes(audio_chunk)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected during send")
                break
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                break
        
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
            "performance_metrics": True
        },
        "models": {
            "stt": config.STT_MODEL,
            "llm": config.LLM_MODEL,
            "tts": config.TTS_MODEL,
            "embedding": config.EMBEDDING_MODEL
        },
        "performance": {
            "target_latency_ms": 800,
            "current_latency_ms": metrics.get_average_latency("total_pipeline") if metrics.get_average_latency("total_pipeline") > 0 else "measuring..."
        },
        "endpoints": {
            "websocket": "/stream",
            "query": "/query",
            "health": "/health",
            "metrics": "/metrics",
            "demo": "/"
        },
        "timestamp": time.time()
    }


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