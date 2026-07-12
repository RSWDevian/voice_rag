# src/pipeline.py
"""
Main Pipeline Orchestrator
Ties together all components for end-to-end streaming RAG
"""

import asyncio
import time
from typing import Dict, Any
from collections import deque

from src.config import config
from src.vad import get_vad
if config.STT_ENGINE == "v2":
    from src.stt_v2 import get_stt
else:
    from src.stt import get_stt
from src.vector_search import get_vector_search
from src.query_builder import get_query_builder
if config.LLM_ENGINE == "v2":
    from src.llm_v2 import get_llm
else:
    from src.llm import get_llm
if config.TTS_ENGINE == "v2":
    from src.tts_v2 import get_tts
else:
    from src.tts import get_tts
from src.metrics import get_metrics_collector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingRAGPipeline:
    """
    Main pipeline orchestrator for streaming RAG
    Manages the complete flow from audio input to speech output
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        # Initialize components
        self.vad = get_vad()
        self.stt = get_stt()
        self.vector_search = get_vector_search()
        self.query_builder = get_query_builder()
        self.llm = get_llm()
        self.tts = get_tts()
        self.metrics = get_metrics_collector()
        
        # Pipeline state
        self.is_processing = False
        self.current_session_id = None
        
        # Audio buffering
        self.audio_buffer = deque(maxlen=100)  # up to 100 incoming chunks
        self.buffer_duration = 0.0
        
        # Transcript tracking
        self.partial_transcripts = []
        self.final_transcript = ""
        self.transcript_confidence = 0.0
        
        # Response tracking
        self.response_buffer = []
        self.is_responding = False
        
        # Performance tracking
        self.pipeline_start_time = 0.0
        self.utterance_start_time = 0.0
        self.component_timings = {}
        
        logger.info("Streaming RAG Pipeline initialized")

    def _reset_session_state(self):
        """Reset pipeline state for new session"""
        self.audio_buffer.clear()
        self.buffer_duration = 0.0
        self.partial_transcripts = []
        self.final_transcript = ""
        self.response_buffer = []
        self.is_responding = False
        
        # Reset STT
        self.stt.reset()

        # Reset VAD's internal partial-frame buffer (shared singleton across sessions)
        self.vad.reset_buffer()
        
        # Reset metrics for this session
        self.metrics.reset_session()
        
        logger.debug("Pipeline state reset")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Dict: Pipeline statistics
        """
        return {
            "is_processing": self.is_processing,
            "transcripts": {
                "count": len(self.partial_transcripts),
                "final": self.final_transcript
            },
            "audio_buffer": {
                "size": len(self.audio_buffer),
                "duration": self.buffer_duration
            },
            "metrics": self.metrics.get_all_stats(),
            "component_stats": {
                "vad": self.vad.get_performance_stats(),
                "stt": self.stt.get_performance_stats(),
                "vector_search": self.vector_search.get_performance_stats(),
                "query_builder": self.query_builder.get_performance_stats(),
                "llm": self.llm.get_performance_stats(),
                "tts": self.tts.get_performance_stats()
            }
        }
    
    async def close(self):
        """Close all resources"""
        await self.stt.close()
        await self.llm.close()
        await self.tts.close()
        logger.info("Pipeline resources closed")


# Simple pipeline for single queries (non-streaming)
class SimplePipeline:
    """
    Simple pipeline for single query processing (non-streaming)
    Useful for testing and batch processing
    """
    
    def __init__(self):
        """Initialize simple pipeline"""
        self.vector_search = get_vector_search()
        self.query_builder = get_query_builder()
        self.llm = get_llm()
        
        logger.info("Simple pipeline initialized")
    
    async def process_query(self, text: str) -> Dict[str, Any]:
        """
        Process a single query
        
        Args:
            text: Query text
            
        Returns:
            Dict: Response with metadata
        """
        start_time = time.time()
        
        # 1. Build Query
        query = self.query_builder.build_query(text, intent="general")
        
        # 2. Vector Search
        search_results = self.vector_search.search(text, top_k=config.TOP_K_RESULTS)
        context = [r["text"] for r in search_results]
        
        # 3. Get Response
        response_text = await self.llm.complete(query, context)
        
        # 4. Update Context
        self.query_builder.update_context(query, response_text)
        
        # 5. Prepare response
        total_time = (time.time() - start_time) * 1000
        
        return {
            "query": text,
            "context": context,
            "response": response_text,
            "latency_ms": total_time,
            "timestamp": time.time()
        }


# Singleton instances
_pipeline_instance = None
_simple_pipeline_instance = None


def get_pipeline() -> StreamingRAGPipeline:
    """Get or create global pipeline instance"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = StreamingRAGPipeline()
    
    return _pipeline_instance


def get_simple_pipeline() -> SimplePipeline:
    """Get or create global simple pipeline instance"""
    global _simple_pipeline_instance
    
    if _simple_pipeline_instance is None:
        _simple_pipeline_instance = SimplePipeline()
    
    return _simple_pipeline_instance


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_simple_pipeline():
        """Test simple pipeline"""
        pipeline = get_simple_pipeline()
        
        test_queries = [
            "What is the weather like?",
            "Tell me about machine learning",
            "Who wrote 1984?",
            "Hello, how are you?"
        ]
        
        print("Testing Simple Pipeline:")
        print("=" * 60)
        
        for query in test_queries:
            result = await pipeline.process_query(query)
            print(f"\nQuery: {query}")
            print(f"Response: {result['response']}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
            if result['context']:
                print(f"Context: {result['context'][0][:50]}...")
    
    # Run tests
    asyncio.run(test_simple_pipeline())