# src/vad.py
"""
Voice Activity Detection (VAD) using Silero VAD.
Optimized for CPU inference with minimal latency. (<30ms per chunk)
"""
import os
import torch
import numpy as np
from typing import Union, Optional, List
import time
from collections import deque

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SileroVAD:
    """Silero VAD with proper audio chunk buffering."""

    def __init__(self):
        """Initialize the Silero VAD model."""
        self.sample_rate = config.VAD_SAMPLE_RATE
        self.threshold = config.VAD_THRESHOLD
        self._model = None
        
        # VAD expects exactly 512 samples at 16kHz (32ms)
        self.chunk_size = 512  # samples
        self.chunk_bytes = self.chunk_size * 2  # 1024 bytes (int16)
        
        # Buffer for accumulating audio
        self.buffer = bytearray()
        self.total_inferences = 0
        self.total_latency_ms = 0.0

        # Diagnostics: track the probability/amplitude distribution seen since
        # the last log line, to tell "mic capturing silence" apart from
        # "capturing real speech but below threshold" without guessing.
        self._diag_prob_max = 0.0
        self._diag_prob_sum = 0.0
        self._diag_amp_max = 0
        
        # ========== FIX: Set TORCH_HUB_DIR before loading ==========
        hub_dir = str(os.path.join(config.MODELS_DIR, "silero_vad"))
        os.makedirs(hub_dir, exist_ok=True)
        os.environ['TORCH_HUB_DIR'] = hub_dir
        logger.info(f"TORCH_HUB_DIR set to: {hub_dir}")
        
        self._load_model()
        logger.info(f"VAD initialized: sample_rate={self.sample_rate}, threshold={self.threshold}")

    def _load_model(self):
        """Load the Silero VAD model from torchhub."""
        try:
            start_time = time.time()
            
            # ========== FIX: Remove download_root parameter ==========
            # Silero VAD returns a tuple (model, utils)
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
                source='github'
                # download_root REMOVED - use TORCH_HUB_DIR instead
            )
            
            self._model.eval()
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Silero VAD model loaded in {load_time:.2f} ms")
            
            # Log where the model was cached
            hub_dir = os.environ.get('TORCH_HUB_DIR', 'default cache')
            logger.info(f"Model cached at: {hub_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise RuntimeError("Could not load Silero VAD model") from e
    
    def _prepare_chunk(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Prepare audio data for VAD inference.
        Ensures proper size and format.
        """
        if not audio_data:
            return None
        
        # Ensure data length is even (multiple of 2 bytes for int16)
        if len(audio_data) % 2 != 0:
            logger.warning(f"Odd byte length {len(audio_data)}, truncating")
            audio_data = audio_data[:len(audio_data) - len(audio_data) % 2]
        
        if len(audio_data) == 0:
            return None
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Failed to convert audio data: {e}")
            return None
        
        # Ensure we have the right number of samples
        if len(audio_np) != self.chunk_size:
            # If too long, take first chunk_size samples
            if len(audio_np) > self.chunk_size:
                audio_np = audio_np[:self.chunk_size]
            # If too short, pad with zeros
            else:
                padded = np.zeros(self.chunk_size, dtype=np.float32)
                padded[:len(audio_np)] = audio_np
                audio_np = padded
        
        return audio_np
    
    def process_chunk(self, audio_chunk: bytes) -> Optional[bool]:
        """
        Process a chunk of audio data through VAD.
        Handles buffering and chunking automatically.
        
        Args:
            audio_chunk: Raw audio bytes (int16 PCM)
            
        Returns:
            bool: True if speech detected, None if not enough data
        """
        if not audio_chunk:
            return None
        
        # Add to buffer
        self.buffer.extend(audio_chunk)
        
        # Process while we have enough data
        results = []
        while len(self.buffer) >= self.chunk_bytes:
            # Extract one VAD chunk
            chunk_data = bytes(self.buffer[:self.chunk_bytes])
            self.buffer = self.buffer[self.chunk_bytes:]
            
            # Run VAD on this chunk
            is_speech = self.is_speech(chunk_data)
            results.append(is_speech)
        
        # Return True if ANY sub-chunk was speech, not just the last one.
        # Each incoming chunk (4096 samples/256ms) is split into 8 Silero
        # sub-chunks (512 samples/32ms each); returning only results[-1]
        # meant a natural word boundary or brief pause landing in the final
        # 32ms silently discarded the other 7 sub-chunks' speech - causing
        # real, clearly-detected speech (see the per-sub-chunk diagnostics
        # logged in is_speech()) to never accumulate the pipeline's 2s
        # buffer threshold, so STT/LLM/TTS would never trigger.
        return any(results) if results else None
    
    def is_speech(self, audio_chunk: Union[np.ndarray, bytes]) -> bool:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as bytes (int16) or numpy array (float32)
                        Must be exactly 512 samples (32ms at 16kHz)
            
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if self._model is None:
            raise RuntimeError("VAD model is not loaded")
        
        try:
            start_time = time.time()
            
            # Convert and prepare chunk
            if isinstance(audio_chunk, bytes):
                audio_np = self._prepare_chunk(audio_chunk)
                if audio_np is None:
                    return False
            else:
                audio_np = audio_chunk.astype(np.float32)
                if len(audio_np) != self.chunk_size:
                    if len(audio_np) > self.chunk_size:
                        audio_np = audio_np[:self.chunk_size]
                    else:
                        padded = np.zeros(self.chunk_size, dtype=np.float32)
                        padded[:len(audio_np)] = audio_np
                        audio_np = padded
            
            # Convert to torch tensor with batch dimension
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

            # Run inference with gradient tracking disabled
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate)

            prob_value = float(speech_prob)

            # Check if speech probability exceeds threshold
            is_speech = speech_prob > self.threshold

            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.total_inferences += 1
            self.total_latency_ms += latency_ms

            # Diagnostics
            self._diag_prob_max = max(self._diag_prob_max, prob_value)
            self._diag_prob_sum += prob_value
            self._diag_amp_max = max(self._diag_amp_max, float(np.abs(audio_np).max()))

            # Occasional logging
            if self.total_inferences % 100 == 0:
                avg_latency = self.total_latency_ms / self.total_inferences
                avg_prob = self._diag_prob_sum / 100
                logger.info(
                    f"VAD average latency over {self.total_inferences} inferences: {avg_latency:.2f} ms "
                    f"| speech_prob max={self._diag_prob_max:.3f} avg={avg_prob:.3f} (threshold={self.threshold}) "
                    f"| input amplitude max={self._diag_amp_max:.4f}"
                )
                self._diag_prob_max = 0.0
                self._diag_prob_sum = 0.0
                self._diag_amp_max = 0.0

            return bool(is_speech)
        
        except Exception as e:
            logger.error(f"Error during VAD inference: {e}")
            return False
    
    def get_speech_probability(self, audio_chunk: Union[np.ndarray, bytes]) -> float:
        """Get raw speech probability without threshold."""
        if self._model is None:
            return 0.0
        
        try:
            audio_np = self._prepare_chunk(audio_chunk) if isinstance(audio_chunk, bytes) else audio_chunk
            if audio_np is None:
                return 0.0
            
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate)
            
            return float(speech_prob)
        except Exception as e:
            logger.debug(f"Speech probability error: {e}")
            return 0.0
    
    def flush_buffer(self) -> List[bool]:
        """Process any remaining audio in the buffer."""
        results = []
        while len(self.buffer) >= self.chunk_bytes:
            chunk_data = bytes(self.buffer[:self.chunk_bytes])
            self.buffer = self.buffer[self.chunk_bytes:]
            results.append(self.is_speech(chunk_data))
        return results
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if self.total_inferences == 0:
            return {
                "total_inferences": 0,
                "avg_latency_ms": 0.0,
                "total_latency_ms": 0.0
            }
        
        return {
            "total_inferences": self.total_inferences,
            "avg_latency_ms": self.total_latency_ms / self.total_inferences,
            "total_latency_ms": self.total_latency_ms
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_inferences = 0
        self.total_latency_ms = 0.0

    def reset_buffer(self):
        """Clear buffered partial-frame audio (call between sessions - the VAD
        instance is a shared singleton, and a leftover partial frame from a
        prior session would misalign all subsequent 512-sample chunking)."""
        self.buffer = bytearray()

    def update_threshold(self, new_threshold: float):
        """Update VAD threshold dynamically."""
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            logger.info(f"VAD threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid threshold value: {new_threshold}. Must be between 0.0 and 1.0")


# Singleton instance for global use
_vad_instance = None

def get_vad() -> SileroVAD:
    """Get the singleton instance of SileroVAD."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = SileroVAD()
    return _vad_instance


# Testing usage
if __name__ == "__main__":
    import time
    
    vad = SileroVAD()
    
    # Create a test audio chunk (512 samples of silence)
    silence = np.zeros(512, dtype=np.float32)
    
    # Create a test audio chunk (512 samples of noise)
    noise = np.random.randn(512).astype(np.float32) * 0.1
    
    print("Testing VAD...")
    print(f"Silence detection: {vad.is_speech(silence)}")
    print(f"Noise detection: {vad.is_speech(noise)}")
    print(f"Performance stats: {vad.get_performance_stats()}")