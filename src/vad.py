"""
Voice Activity Detection (VAD) using Silero VAD.
Optimized for CPU interference with minimal latency. (<30ms per chunk)
"""
import torch
import numpy as np
from typing import Union, Optional
import time

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SileroVAD:

    def __init__(self):
        """
        Initialize the Silero VAD model.
        Loads the pre-trained model from the torchhub.
        """
        self.sample_rate = config.SAMPLE_RATE
        self.threshold = config.VAD_THRESHOLD
        self._model = None
        self._load_model()

        logger.info(f"VAD initialized: sample_rate={self.sample_rate}, threshold={self.threshold}")

    def _load_model(self):
        """Load the Silero VAD model from torchhub."""
        try:
            start_time = time.time()
            self._model = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            # Set the eval mode
            self._model.eval()
            loaf_time = (time.time() - start_time)*1000
            logger.info(f"Silero VAD model loaded in {loaf_time:.2f} ms")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise RuntimeError("Could not load Silero VAD model") from e
        
    def is_speech(self, audio_chunk: Union[np.ndarray, bytes]) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio data as numpy array (float32) or bytes (int16).

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if self._model is None:
            raise RuntimeError("VAD model is not loaded")
            return False
        
        try:
            start_time = time.time()

            # Convert bytes to numpy array if necessary
            if isinstance(audio_chunk, bytes):
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_chunk, np.ndarray):
                audio_np = audio_chunk.astype(np.float32)
            else:
                logger.warning(f"Unsupported audio chunk type: {type(audio_chunk)}")
                return False
            
            # Validate audio link
            if len(audio_np) == 0:
                logger.warning("Received empty audio chunk")
                return False
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np)

            # Run interference with gradient tracking
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate)

            # Check if speech probability exceeds threshhold
            is_speech = speech_prob > self.threshold

            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.total_interferences += 1
            self.total_latency_ms += latency_ms

            # Occational logging info
            if self.total_interferences % 100 == 0:
                avg_latency = self.total_latency_ms / self.total_interferences
                logger.info(f"VAD average latency over {self.total_interferences} interferences: {avg_latency:.2f} ms")

            return bool(is_speech)
        
        except Exception as e:
            logger.error(f"Error during VAD inference: {e}")
            return False
        
    def get_speech_probability(self, audio_chunk: Union[np.ndarray, bytes]) -> float:
        """
        Get raw speech probability without threshold
        
        Args:
            audio_chunk: Audio data as numpy array (float32) or bytes (int16)
            
        Returns:
            float: Speech probability between 0.0 and 1.0
        """
        if self._model is None:
            return 0.0
        
        try:
            # Convert bytes to numpy if needed
            if isinstance(audio_chunk, bytes):
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_chunk, np.ndarray):
                audio_np = audio_chunk.astype(np.float32)
            else:
                return 0.0
            
            if len(audio_np) == 0:
                return 0.0
            
            audio_tensor = torch.from_numpy(audio_np)
            
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.sample_rate)
            
            return float(speech_prob)
            
        except Exception as e:
            logger.debug(f"Speech probability error: {e}")
            return 0.0
        
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the VAD
        
        Returns:
            dict: Performance metrics
        """
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
        """Reset performance statistics"""
        self.total_inferences = 0
        self.total_latency_ms = 0.0

    def update_threshold(self, new_threshold: float):
        """
        Update VAD threshold dynamically
        
        Args:
            new_threshold: New threshold value (0.0 - 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            logger.info(f"VAD threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid threshold value: {new_threshold}. Must be between 0.0 and 1.0")

# Singletone instance for global use
_vad_instance = None

def get_vad() -> SileroVAD:
    """
    Get the singleton instance of SileroVAD.
    
    Returns:
        SileroVAD: The VAD instance
    """
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = SileroVAD()
    return _vad_instance


# Testing usage
if __name__ == "__main__":
    # Test VAD with sample audio
    import time
    
    vad = SileroVAD()
    
    # Create a test audio chunk (1 second of silence)
    silence = np.zeros(16000, dtype=np.float32)
    
    # Create a test audio chunk (1 second of noise)
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    
    print("Testing VAD...")
    print(f"Silence detection: {vad.is_speech(silence)}")
    print(f"Noise detection: {vad.is_speech(noise)}")
    print(f"Performance stats: {vad.get_performance_stats()}")