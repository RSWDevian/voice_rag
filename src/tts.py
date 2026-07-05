"""
Text-to-Speech using ElevenLabs API
Direct integration with streaming support
Supports English voices
"""

import time
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any
from pathlib import Path

import httpx
import numpy as np

from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ElevenLabsTTS:
    """
    ElevenLabs TTS client with streaming support
    Optimized for low latency with PCM format
    Uses English voices
    """
    
    def __init__(self):
        """Initialize ElevenLabs TTS client"""
        self.api_key = config.ELEVENLABS_API_KEY
        self.voice_id = config.ELEVENLABS_VOICE_ID
        self.model_id = getattr(config, 'ELEVENLABS_MODEL_ID', 'eleven_turbo_v2')
        self.response_format = getattr(config, 'TTS_RESPONSE_FORMAT', 'pcm')
        self.speed = getattr(config, 'TTS_SPEED', 1.0)
        self.timeout = getattr(config, 'API_TIMEOUT', 10.0)
        
        # Performance tracking
        self.total_synthesizes = 0
        self.total_latency_ms = 0.0
        self.total_audio_bytes = 0
        
        # Create HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            )
        )
        
        logger.info(f"ElevenLabs TTS initialized: voice_id={self.voice_id}, "
                   f"model={self.model_id}, format={self.response_format}")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio (non-streaming)
        
        Args:
            text: Text to synthesize
            
        Returns:
            bytes: Audio data
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return b''
        
        try:
            start_time = time.time()
            
            # Prepare request with updated model
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Make API request
            response = await self.client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/pcm" if self.response_format == "pcm" else "audio/mpeg"
                },
                json=payload
            )
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.total_synthesizes += 1
            self.total_latency_ms += latency_ms
            
            # Parse response
            if response.status_code == 200:
                audio_data = response.content
                self.total_audio_bytes += len(audio_data)
                
                logger.debug(f"TTS synthesis: '{text[:30]}...' ({len(audio_data)} bytes, "
                           f"{latency_ms:.0f}ms)")
                return audio_data
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return b''
                
        except httpx.TimeoutException:
            logger.error("ElevenLabs API timeout")
            return b''
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            return b''
    
    async def synthesize_stream(self, text: str, chunk_size: int = 50) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS by splitting text into chunks
        
        Args:
            text: Text to synthesize
            chunk_size: Number of characters per chunk
            
        Yields:
            bytes: Audio chunks
        """
        if not text or not text.strip():
            return
        
        # Split text into sentences or chunks
        chunks = self._split_text(text, chunk_size)
        
        for chunk in chunks:
            if chunk.strip():
                audio = await self.synthesize(chunk)
                if audio:
                    yield audio
                    
                # Small delay to ensure streaming order
                await asyncio.sleep(0.01)
    
    async def synthesize_streaming(self, text_chunks: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS from async text chunks
        
        Args:
            text_chunks: Async generator of text chunks
            
        Yields:
            bytes: Audio chunks
        """
        buffer = []
        buffer_chars = 0
        min_chunk_size = getattr(config, 'TTS_MIN_CHUNK_SIZE', 10)
        max_chunk_size = getattr(config, 'TTS_MAX_CHUNK_SIZE', 50)
        
        async for text_chunk in text_chunks:
            buffer.append(text_chunk)
            buffer_chars += len(text_chunk)
            
            if buffer_chars >= min_chunk_size:
                text = ''.join(buffer)
                
                if len(text) > max_chunk_size:
                    split_point = self._find_sentence_boundary(text, max_chunk_size)
                    
                    if split_point > 0:
                        first_part = text[:split_point]
                        audio = await self.synthesize(first_part)
                        if audio:
                            yield audio
                        
                        buffer = [text[split_point:]]
                        buffer_chars = len(buffer[0])
                    else:
                        audio = await self.synthesize(text)
                        if audio:
                            yield audio
                        buffer = []
                        buffer_chars = 0
                else:
                    continue
        
        if buffer:
            text = ''.join(buffer)
            audio = await self.synthesize(text)
            if audio:
                yield audio
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks at sentence boundaries
        
        Args:
            text: Text to split
            chunk_size: Target chunk size
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        import re
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, max_len: int) -> int:
        """
        Find a sentence boundary within max_len characters
        
        Args:
            text: Text to search
            max_len: Maximum length to search
            
        Returns:
            int: Position of sentence boundary (0 if none found)
        """
        # Look for sentence-ending punctuation
        for i in range(min(max_len, len(text) - 1), 0, -1):
            if text[i] in '.!?':
                return i + 1
        
        # Look for comma or semicolon as fallback
        for i in range(min(max_len, len(text) - 1), 0, -1):
            if text[i] in ',;':
                return i + 1
        
        # Look for space as fallback
        for i in range(min(max_len, len(text) - 1), 0, -1):
            if text[i] == ' ':
                return i + 1
        
        # No good boundary found
        return 0
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if self.total_synthesizes == 0:
            return {
                "total_synthesizes": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "total_audio_bytes": 0
            }
        
        return {
            "total_synthesizes": self.total_synthesizes,
            "avg_latency_ms": self.total_latency_ms / self.total_synthesizes,
            "total_latency_ms": self.total_latency_ms,
            "total_audio_bytes": self.total_audio_bytes,
            "avg_bytes_per_synthesis": self.total_audio_bytes / self.total_synthesizes
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_synthesizes = 0
        self.total_latency_ms = 0.0
        self.total_audio_bytes = 0
    
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


class StreamingTTS:
    """
    Streaming TTS wrapper with buffering and callbacks
    Optimized for low-latency speech synthesis
    """
    
    def __init__(self):
        """Initialize streaming TTS wrapper"""
        self.tts = ElevenLabsTTS()
        
        # Buffering
        self.buffer = []
        self.buffer_chars = 0
        self.min_buffer_size = getattr(config, 'TTS_MIN_CHUNK_SIZE', 10)
        self.max_buffer_size = getattr(config, 'TTS_MAX_CHUNK_SIZE', 50)
        
        # Streaming state
        self.is_streaming = False
        self.is_processing = False
        
        # Callbacks
        self.on_audio_callbacks = []
        self.on_chunk_callbacks = []
        self.on_complete_callbacks = []
        
        # Performance tracking
        self.total_chunks = 0
        self.total_audio_bytes = 0
        
        logger.info(f"Streaming TTS initialized: min_buffer={self.min_buffer_size}, "
                   f"max_buffer={self.max_buffer_size}")
    
    async def stream_text(self, text_chunks: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Stream text to audio with intelligent buffering
        
        Args:
            text_chunks: Async generator of text chunks
            
        Yields:
            bytes: Audio chunks
        """
        self.is_streaming = True
        self.buffer = []
        self.buffer_chars = 0
        
        start_time = time.time()
        
        try:
            async for text_chunk in text_chunks:
                if not text_chunk:
                    continue
                
                # Add to buffer
                self.buffer.append(text_chunk)
                self.buffer_chars += len(text_chunk)
                
                # Check if buffer is ready for synthesis
                if self.buffer_chars >= self.min_buffer_size:
                    # Process buffer
                    text = ''.join(self.buffer)
                    
                    # Check if we need to split
                    if len(text) > self.max_buffer_size:
                        # Split at a good boundary
                        split_point = self._find_split_point(text)
                        
                        if split_point > 0:
                            # Synthesize first part
                            first_part = text[:split_point]
                            audio = await self.tts.synthesize(first_part)
                            
                            if audio:
                                self.total_chunks += 1
                                self.total_audio_bytes += len(audio)
                                
                                # Trigger callbacks
                                for callback in self.on_audio_callbacks:
                                    try:
                                        await callback(audio)
                                    except Exception as e:
                                        logger.debug(f"Audio callback error: {e}")
                                
                                yield audio
                            
                            # Keep remaining text in buffer
                            self.buffer = [text[split_point:]]
                            self.buffer_chars = len(self.buffer[0])
                        else:
                            # Synthesize entire buffer
                            audio = await self.tts.synthesize(text)
                            
                            if audio:
                                self.total_chunks += 1
                                self.total_audio_bytes += len(audio)
                                
                                # Trigger callbacks
                                for callback in self.on_audio_callbacks:
                                    try:
                                        await callback(audio)
                                    except Exception as e:
                                        logger.debug(f"Audio callback error: {e}")
                                
                                yield audio
                            
                            self.buffer = []
                            self.buffer_chars = 0
                    else:
                        # Buffer not ready, continue accumulating
                        continue
                    
                    # Trigger chunk callbacks
                    for callback in self.on_chunk_callbacks:
                        try:
                            await callback(text)
                        except Exception as e:
                            logger.debug(f"Chunk callback error: {e}")
            
            # Process remaining buffer
            if self.buffer:
                text = ''.join(self.buffer)
                audio = await self.tts.synthesize(text)
                
                if audio:
                    self.total_chunks += 1
                    self.total_audio_bytes += len(audio)
                    
                    # Trigger callbacks
                    for callback in self.on_audio_callbacks:
                        try:
                            await callback(audio)
                        except Exception as e:
                            logger.debug(f"Audio callback error: {e}")
                    
                    yield audio
        
        finally:
            self.is_streaming = False
            
            # Trigger complete callbacks
            total_time = (time.time() - start_time) * 1000
            for callback in self.on_complete_callbacks:
                try:
                    await callback({
                        "total_chunks": self.total_chunks,
                        "total_audio_bytes": self.total_audio_bytes,
                        "total_time_ms": total_time
                    })
                except Exception as e:
                    logger.debug(f"Complete callback error: {e}")
            
            logger.debug(f"TTS stream complete: {self.total_chunks} chunks, "
                       f"{self.total_audio_bytes} bytes, {total_time:.0f}ms")
    
    async def synthesize_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize complete text to audio stream
        
        Args:
            text: Text to synthesize
            
        Yields:
            bytes: Audio chunks
        """
        async def text_generator():
            yield text
        
        async for audio in self.stream_text(text_generator()):
            yield audio
    
    def _find_split_point(self, text: str) -> int:
        """
        Find a good split point in text
        
        Args:
            text: Text to split
            
        Returns:
            int: Split position
        """
        # Try to split at sentence boundary
        for i in range(min(self.max_buffer_size, len(text) - 1), 0, -1):
            if text[i] in '.!?':
                return i + 1
        
        # Try to split at comma or semicolon
        for i in range(min(self.max_buffer_size, len(text) - 1), 0, -1):
            if text[i] in ',;':
                return i + 1
        
        # Try to split at space
        for i in range(min(self.max_buffer_size, len(text) - 1), 0, -1):
            if text[i] == ' ':
                return i + 1
        
        # No good split point
        return 0
    
    def add_audio_callback(self, callback):
        """Add callback for each audio chunk"""
        self.on_audio_callbacks.append(callback)
    
    def add_chunk_callback(self, callback):
        """Add callback for each text chunk"""
        self.on_chunk_callbacks.append(callback)
    
    def add_complete_callback(self, callback):
        """Add callback for completion"""
        self.on_complete_callbacks.append(callback)
    
    def clear_callbacks(self):
        """Clear all callbacks"""
        self.on_audio_callbacks.clear()
        self.on_chunk_callbacks.clear()
        self.on_complete_callbacks.clear()
    
    def reset_buffer(self):
        """Reset the text buffer"""
        self.buffer = []
        self.buffer_chars = 0
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        stats = self.tts.get_performance_stats()
        stats.update({
            "total_chunks": self.total_chunks,
            "is_streaming": self.is_streaming
        })
        return stats
    
    async def close(self):
        """Close underlying TTS client"""
        await self.tts.close()


# Singleton instance
_tts_instance = None


def get_tts() -> StreamingTTS:
    """Get or create global TTS instance"""
    global _tts_instance
    
    if _tts_instance is None:
        _tts_instance = StreamingTTS()
    
    return _tts_instance


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_tts():
        """Test TTS streaming"""
        tts = StreamingTTS()
        
        test_text = "Hello! This is a test of the text to speech system. " \
                   "It should stream audio with low latency."
        
        print("Testing ElevenLabs TTS Streaming:")
        print("=" * 60)
        
        # Test complete synthesis
        print("Synthesizing complete text...")
        audio_chunks = []
        async for audio in tts.synthesize_text(test_text):
            audio_chunks.append(audio)
            print(f"Received audio chunk: {len(audio)} bytes")
        
        print(f"\nTotal audio chunks: {len(audio_chunks)}")
        print(f"Total audio bytes: {sum(len(c) for c in audio_chunks)}")
        print(f"\nPerformance: {tts.get_performance_stats()}")
        
        await tts.close()
    
    asyncio.run(test_tts())