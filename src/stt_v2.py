"""
Speech-to-text module using Moonshine (moonshine-ai/moonshine) - v2
Runs the Moonshine Medium Streaming model fully on-device (no network round trip),
~269ms latency on Linux x86. Same public interface as src/stt.py (get_stt() ->
StreamingSTT with process_audio/process_stream/reset/close), so callers only need
to change their import from `src.stt` to `src.stt_v2`.
"""

import asyncio
import time
from typing import List, Optional, AsyncGenerator

import numpy as np
from moonshine_voice import Transcriber, ModelArch, get_model_for_language

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MoonshineSTT:

    def __init__(self):
        """Initialize STT using a local Moonshine Medium Streaming model"""
        self.language = getattr(config, 'STT_LANGUAGE', 'en')
        # self.model_arch = ModelArch.MEDIUM_STREAMING
        self.model_arch = ModelArch.SMALL_STREAMING
        self.sample_rate = getattr(config, 'VAD_SAMPLE_RATE', 16000)
        self.connections = config.MAX_CONCURRENT_USERS
        self._closed = False

        # Performance tracking
        self.total_transcriptions = 0
        self.total_latency_ms = 0.0

        # Downloads (and caches under models/) the quantized weights on first run,
        # then loads them into memory for the lifetime of this instance.
        model_path, model_arch = get_model_for_language(
            self.language,
            self.model_arch,
            cache_root=config.MODELS_DIR,
        )
        self.transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

        # The underlying C transcriber handle isn't safe for concurrent calls,
        # so every transcribe() serializes through this lock.
        self._lock = asyncio.Lock()

        logger.info(f"STT initialized: model=moonshine-{model_arch.name.lower()}, sample_rate={self.sample_rate}")

    @staticmethod
    def _pcm16_to_float(pcm_bytes: bytes) -> List[float]:
        """Convert raw 16-bit mono PCM to normalized float32 samples (-1.0 to 1.0)."""
        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (int16.astype(np.float32) / 32768.0).tolist()

    def _transcribe_sync(self, audio_data: List[float]) -> str:
        transcript = self.transcriber.transcribe_without_streaming(
            audio_data, sample_rate=self.sample_rate
        )
        return " ".join(line.text for line in transcript.lines).strip()

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> Optional[str]:
        """
        Transcribe audio bytes to text using the local Moonshine model

        Args:
            audio_bytes: Raw audio data (int16 PCM at 16kHz)
            language: Language code (default: 'en') - kept for API parity with
                the OpenRouter-backed STT; the loaded model's language is fixed
                at construction time via STT_LANGUAGE

        Returns:
            str: Transcribed text or None on error
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes received")
            return None

        try:
            start_time = time.time()

            audio_data = self._pcm16_to_float(audio_bytes)

            async with self._lock:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, self._transcribe_sync, audio_data)

            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.total_transcriptions += 1
            self.total_latency_ms += latency_ms

            # Log performance occasionally
            if self.total_transcriptions % 10 == 0:
                avg_latency = self.total_latency_ms / self.total_transcriptions
                logger.debug(f"STT stats: avg_latency={avg_latency:.2f}ms, "
                           f"transcriptions={self.total_transcriptions}")

            logger.debug(f"STT: '{text[:50]}...' ({latency_ms:.0f}ms)")
            return text or None

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
                combined = b''.join(buffer)
                transcript = await self.transcribe(combined)

                buffer = []
                buffer_duration = 0.0

                if transcript:
                    yield transcript

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
        """Free the underlying transcriber"""
        if not self._closed:
            self.transcriber.close()
            self._closed = True

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'transcriber') and not self._closed:
            try:
                self.transcriber.close()
            except Exception:
                pass


class StreamingSTT:
    """
    Streaming STT with buffering and partial transcript support
    Wraps MoonshineSTT with streaming logic
    """

    def __init__(self):
        self.stt = MoonshineSTT()
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

    def get_performance_stats(self) -> dict:
        """Get STT performance statistics"""
        return self.stt.get_performance_stats()

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

        # Generate test audio (1 second of silence)
        sample_audio = np.zeros(16000, dtype=np.int16).tobytes()

        # Test transcribe
        transcript = await stt.process_audio([sample_audio])
        print(f"Transcript: {transcript}")

        # Test performance
        stats = stt.stt.get_performance_stats()
        print(f"Performance: {stats}")

        await stt.close()

    asyncio.run(test_stt())
