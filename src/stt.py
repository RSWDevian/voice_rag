"""
Speech-to-text module using Openrouter Unified API
Supports multiple models with latency optimization
"""

import base64
import asyncio
import time
from typing import List, Optional, AsyncGenerator
from io import BytesIO
import httpx
import numpy as np

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenRouterSTT:

    def __init__(self):
        """Initialize STT client using OpenRouter API configuration"""
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        self.model = config.STT_MODEL
        self.timeout = config.OPENROUTER_TIMEOUT
        self.connections = config.MAX_CONCURRENT_USERS

        # Performance tracking
        self.total_transcription = 0.0
        self.total_latency_ms = 0.0

        if not self.api_key or not self.api_url:
            logger.error("OpenRouter API key or URL is not configured")
            raise ValueError("OpenRouter API key or URL is missing")

        logger.info(f"STT initialized: model={self.model}, sample_rate={self.sample_rate}, timeout={self.timeout}s")

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

        logger.info(f"STT initialized: model={self.model}")
    
    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> Optional[str]:
        """
        Transcribe audio bytes to text using OpenRouter STT
        
        Args:
            audio_bytes: Raw audio data (int16 PCM at 16kHz)
            language: Language code (default: 'en')
            
        Returns:
            str: Transcribed text or None on error
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes received")
            return None
        
        try:
            start_time = time.time()
            
            # Encode audio as base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Prepare request
            payload = {
                "model": self.model,
                "audio": audio_base64,
                "response_format": "text",
                "language": language,
            }
            
            # Add model-specific optimizations
            if "whisper" in self.model:
                payload["temperature"] = 0.0  # Deterministic for speed
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            
            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.total_transcriptions += 1
            self.total_latency_ms += latency_ms
            
            # Log performance occasionally
            if self.total_transcriptions % 10 == 0:
                avg_latency = self.total_latency_ms / self.total_transcriptions
                logger.debug(f"STT stats: avg_latency={avg_latency:.2f}ms, "
                           f"transcriptions={self.total_transcriptions}")
            
            # Parse response
            if response.status_code == 200:
                transcript = response.text.strip()
                logger.debug(f"STT: '{transcript[:50]}...' ({latency_ms:.0f}ms)")
                return transcript
            else:
                logger.error(f"STT API error: {response.status_code} - {response.text}")
                return None
                
        except httpx.TimeoutException:
            logger.error("STT API timeout")
            return None
        except Exception as e:
            logger.error(f"STT exception: {e}")
            return None
    
    async def transcribe_chunked(self, audio_chunks: List[bytes]) -> Optional[str]:
        """
        Transcribe multiple audio chunks combined
        
        Args:
            audio_chunks: List of audio byte chunks
            
        Returns:
            str: Combined transcription or None on error
        """
        if not audio_chunks:
            return None
        
        # Combine chunks
        combined = b''.join(audio_chunks)
        return await self.transcribe(combined)
    
    async def stream_transcribe(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Stream audio and transcribe in chunks
        
        Args:
            audio_stream: Async generator of audio bytes
            
        Yields:
            str: Partial transcriptions as they become available
        """
        buffer = []
        buffer_duration = 0.0
        chunk_duration = 0.03  # 30ms per chunk
        target_duration = config.ASR_CHUNK_DURATION
        
        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            buffer_duration += chunk_duration
            
            if buffer_duration >= target_duration:
                # Process buffer
                combined = b''.join(buffer)
                transcript = await self.transcribe(combined)
                
                # Clear buffer
                buffer = []
                buffer_duration = 0.0
                
                if transcript:
                    yield transcript
        
        # Process remaining buffer
        if buffer:
            combined = b''.join(buffer)
            transcript = await self.transcribe(combined)
            if transcript:
                yield transcript
    
    def get_performance_stats(self) -> dict:
        """Get STT performance statistics"""
        if self.total_transcriptions == 0:
            return {
                "total_transcriptions": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0
            }
        
        return {
            "total_transcriptions": self.total_transcriptions,
            "avg_latency_ms": self.total_latency_ms / self.total_transcriptions,
            "total_latency_ms": self.total_latency_ms
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'client'):
            try:
                asyncio.create_task(self.client.aclose())
            except:
                pass


class StreamingSTT:
    """
    Streaming STT with buffering and partial transcript support
    Wraps OpenRouterSTT with streaming logic
    """
    
    def __init__(self):
        self.stt = OpenRouterSTT()
        self.buffer = []
        self.buffer_duration = 0.0
        self.chunk_duration = 0.03
        self.target_duration = config.ASR_CHUNK_DURATION
        self.last_transcript = ""
        self.transcript_history = []
        
        logger.info(f"StreamingSTT initialized: chunk_duration={self.target_duration}s")
    
    async def process_audio(self, audio_chunks: List[bytes]) -> Optional[str]:
        """
        Process audio chunks and return transcript
        
        Args:
            audio_chunks: List of audio byte chunks
            
        Returns:
            str: Transcript or None
        """
        if not audio_chunks:
            return None
        
        return await self.stt.transcribe_chunked(audio_chunks)
    
    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Process streaming audio and yield partial transcripts
        
        Args:
            audio_stream: Async generator of audio bytes
            
        Yields:
            str: Partial transcripts
        """
        async for transcript in self.stt.stream_transcribe(audio_stream):
            if transcript and transcript != self.last_transcript:
                self.last_transcript = transcript
                self.transcript_history.append(transcript)
                yield transcript
    
    def get_last_transcript(self) -> str:
        """Get the most recent transcript"""
        return self.last_transcript
    
    def get_transcript_history(self) -> List[str]:
        """Get all transcripts from current session"""
        return self.transcript_history
    
    def reset(self):
        """Reset streaming state"""
        self.buffer = []
        self.buffer_duration = 0.0
        self.last_transcript = ""
        self.transcript_history = []
    
    async def close(self):
        """Close STT client"""
        await self.stt.close()


# Singleton instance
_stt_instance = None


def get_stt() -> StreamingSTT:
    """
    Get or create global STT instance
    
    Returns:
        StreamingSTT: Global STT instance
    """
    global _stt_instance
    
    if _stt_instance is None:
        _stt_instance = StreamingSTT()
    
    return _stt_instance


# Test usage
if __name__ == "__main__":
    import asyncio
    
    async def test_stt():
        """Test STT with sample audio"""
        stt = StreamingSTT()
        
        # Test with generated audio (simulated)
        # In practice, you'd use real audio data
        sample_audio = np.zeros(16000 * 2, dtype=np.int16).tobytes()
        
        # Test transcribe
        transcript = await stt.process_audio([sample_audio])
        print(f"Transcript: {transcript}")
        
        # Test performance
        stats = stt.stt.get_performance_stats()
        print(f"Performance: {stats}")
        
        await stt.close()
    
    asyncio.run(test_stt())