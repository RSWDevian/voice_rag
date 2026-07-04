# src/pipeline.py
"""
Main Pipeline Orchestrator
Ties together all components for end-to-end streaming RAG
"""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
from collections import deque

from src.config import config
from src.vad import get_vad
from src.stt import get_stt
from src.intent import get_intent_classifier
from src.vector_search import get_vector_search
from src.query_builder import get_query_builder
from src.llm import get_llm
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
        self.intent_classifier = get_intent_classifier()
        self.vector_search = get_vector_search()
        self.query_builder = get_query_builder()
        self.llm = get_llm()
        self.tts = get_tts()
        self.metrics = get_metrics_collector()
        
        # Pipeline state
        self.is_processing = False
        self.current_session_id = None
        
        # Audio buffering
        self.audio_buffer = deque(maxlen=100)  # 3 seconds of audio
        self.buffer_duration = 0.0
        self.chunk_duration = 0.03  # 30ms per chunk
        
        # Transcript tracking
        self.partial_transcripts = []
        self.final_transcript = ""
        self.transcript_confidence = 0.0
        
        # Response tracking
        self.response_buffer = []
        self.is_responding = False
        
        # Performance tracking
        self.pipeline_start_time = 0.0
        self.component_timings = {}
        
        logger.info("Streaming RAG Pipeline initialized")
    
    async def process_audio_stream(self, websocket) -> AsyncGenerator[bytes, None]:
        """
        Process streaming audio from websocket
        
        Args:
            websocket: WebSocket connection for receiving audio
            
        Yields:
            bytes: Audio chunks for response
        """
        self.is_processing = True
        self.pipeline_start_time = time.time()
        
        try:
            # Reset state for new session
            self._reset_session_state()
            
            # Main processing loop
            while self.is_processing:
                # Receive audio chunk
                try:
                    audio_chunk = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # Check if we should timeout
                    if time.time() - self.pipeline_start_time > 10.0:
                        logger.warning("Pipeline timeout - no audio received")
                        break
                    continue
                except Exception as e:
                    logger.error(f"WebSocket receive error: {e}")
                    break
                
                # 1. Voice Activity Detection
                vad_start = time.time()
                is_speech = self.vad.is_speech(audio_chunk)
                vad_latency = (time.time() - vad_start) * 1000
                self.metrics.record_latency("vad", vad_latency)
                
                if not is_speech:
                    # If we have buffered audio, process it
                    if self.audio_buffer and self.buffer_duration >= config.ASR_CHUNK_DURATION:
                        transcript = await self._process_audio_buffer()
                        if transcript:
                            async for response_audio in self._process_transcript(transcript):
                                yield response_audio
                    continue
                
                # Add to buffer
                self.audio_buffer.append(audio_chunk)
                self.buffer_duration += self.chunk_duration
                
                # 2. Process audio buffer when full
                if self.buffer_duration >= config.ASR_CHUNK_DURATION:
                    transcript = await self._process_audio_buffer()
                    if transcript and transcript != self.final_transcript:
                        # 3. Process transcript through pipeline
                        async for response_audio in self._process_transcript(transcript):
                            yield response_audio
                        self.final_transcript = transcript
            
            # Process any remaining audio
            if self.audio_buffer:
                transcript = await self._process_audio_buffer()
                if transcript:
                    async for response_audio in self._process_transcript(transcript):
                        yield response_audio
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Send error response
            async for audio in self._send_error_response(str(e)):
                yield audio
        finally:
            self.is_processing = False
            logger.info("Pipeline processing complete")
    
    async def _process_audio_buffer(self) -> Optional[str]:
        """
        Process buffered audio through STT
        
        Returns:
            str: Transcript or None
        """
        if not self.audio_buffer:
            return None
        
        stt_start = time.time()
        
        # Convert buffer to list
        audio_chunks = list(self.audio_buffer)
        
        # Process through STT
        transcript = await self.stt.process_audio(audio_chunks)
        
        stt_latency = (time.time() - stt_start) * 1000
        self.metrics.record_latency("stt", stt_latency)
        
        # Clear buffer
        self.audio_buffer.clear()
        self.buffer_duration = 0.0
        
        if transcript:
            self.partial_transcripts.append(transcript)
            logger.debug(f"STT: '{transcript[:50]}...' ({stt_latency:.0f}ms)")
            return transcript
        
        return None
    
    async def _process_transcript(self, transcript: str) -> AsyncGenerator[bytes, None]:
        """
        Process transcript through the RAG pipeline
        
        Args:
            transcript: Transcribed text
            
        Yields:
            bytes: Audio response chunks
        """
        if not transcript or not transcript.strip():
            return
        
        try:
            # 1. Intent Classification
            intent_start = time.time()
            intent, service_route, confidence = self.intent_classifier.classify(transcript)
            intent_latency = (time.time() - intent_start) * 1000
            self.metrics.record_latency("intent", intent_latency)
            
            logger.debug(f"Intent: {intent} (confidence: {confidence:.2f})")
            
            # 2. Build Query
            query_start = time.time()
            query = self.query_builder.build_query(transcript, intent)
            query_latency = (time.time() - query_start) * 1000
            self.metrics.record_latency("query_builder", query_latency)
            
            # 3. Vector Search (if applicable)
            context = []
            if intent in ["knowledge_query", "question", "general"]:
                vector_start = time.time()
                search_results = self.vector_search.search(transcript, top_k=config.TOP_K_RESULTS)
                vector_latency = (time.time() - vector_start) * 1000
                self.metrics.record_latency("vector_search", vector_latency)
                
                context = [r["text"] for r in search_results]
                logger.debug(f"Vector search: {len(context)} results ({vector_latency:.0f}ms)")
            
            # 4. Get routing chain
            routing_chain = self.intent_classifier.get_routing_chain(intent)
            logger.debug(f"Routing chain: {[s.service_name for s in routing_chain]}")
            
            # 5. LLM Streaming
            response_text = ""
            first_token = True
            
            async for llm_chunk in self.llm.stream_response(query, context):
                if first_token:
                    llm_latency = self.metrics.get_latest_latency("llm")
                    logger.debug(f"LLM first token: {llm_latency:.0f}ms")
                    first_token = False
                
                response_text += llm_chunk
                
                # 6. TTS Streaming
                async for audio_chunk in self._stream_tts(llm_chunk):
                    yield audio_chunk
            
            # 7. Update context
            if response_text:
                self.query_builder.update_context(query, response_text)
                
                # 8. Track complete pipeline latency
                total_latency = (time.time() - self.pipeline_start_time) * 1000
                self.metrics.record_latency("total_pipeline", total_latency)
                logger.info(f"Pipeline complete: {total_latency:.0f}ms")
            
        except Exception as e:
            logger.error(f"Transcript processing error: {e}")
            async for audio in self._send_error_response(str(e)):
                yield audio
    
    async def _stream_tts(self, text_chunk: str) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS for a text chunk
        
        Args:
            text_chunk: Text chunk to synthesize
            
        Yields:
            bytes: Audio chunks
        """
        if not text_chunk or not text_chunk.strip():
            return
        
        tts_start = time.time()
        
        # Synthesize text chunk
        async for audio in self.tts.synthesize_text(text_chunk):
            tts_latency = (time.time() - tts_start) * 1000
            self.metrics.record_latency("tts", tts_latency)
            yield audio
    
    async def _send_error_response(self, error_message: str) -> AsyncGenerator[bytes, None]:
        """
        Send error response as TTS
        
        Args:
            error_message: Error message to speak
        """
        error_text = f"I encountered an error: {error_message}"
        logger.error(f"Sending error response: {error_text}")
        
        async for audio in self.tts.synthesize_text(error_text):
            yield audio
    
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
                "intent": self.intent_classifier.get_performance_stats(),
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
        self.intent_classifier = get_intent_classifier()
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
        
        # 1. Intent Classification
        intent, service_route, confidence = self.intent_classifier.classify(text)
        
        # 2. Build Query
        query = self.query_builder.build_query(text, intent)
        
        # 3. Vector Search
        context = []
        if intent in ["knowledge_query", "question", "general"]:
            search_results = self.vector_search.search(text, top_k=config.TOP_K_RESULTS)
            context = [r["text"] for r in search_results]
        
        # 4. Get Response
        response_text = await self.llm.complete(query, context)
        
        # 5. Update Context
        self.query_builder.update_context(query, response_text)
        
        # 6. Prepare response
        total_time = (time.time() - start_time) * 1000
        
        return {
            "query": text,
            "intent": intent,
            "confidence": confidence,
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
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            print(f"Response: {result['response']}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
            if result['context']:
                print(f"Context: {result['context'][0][:50]}...")
    
    async def test_streaming_pipeline():
        """Test streaming pipeline (simulated)"""
        pipeline = get_pipeline()
        
        # Simulate audio stream
        async def simulate_audio():
            # This would be real audio in production
            import numpy as np
            # Generate silence
            silence = np.zeros(480, dtype=np.int16).tobytes()
            # Generate some speech (simulated)
            speech = np.random.randn(480).astype(np.int16).tobytes()
            
            yield silence * 10  # 300ms silence
            yield speech * 20   # 600ms speech
            yield silence * 5   # 150ms silence
        
        print("\nTesting Streaming Pipeline:")
        print("=" * 60)
        
        # Process audio stream
        audio_count = 0
        async for audio in pipeline.process_audio_stream(simulate_audio()):
            audio_count += 1
            print(f"Received audio chunk: {len(audio)} bytes")
        
        print(f"Total audio chunks: {audio_count}")
        print(f"Pipeline stats: {pipeline.get_pipeline_stats()}")
        
        await pipeline.close()
    
    # Run tests
    asyncio.run(test_simple_pipeline())
    asyncio.run(test_streaming_pipeline())