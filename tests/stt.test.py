import asyncio
import numpy as np
from src.stt import StreamingSTT

async def test_stt_basic():
    """Test basic STT functionality"""
    stt = StreamingSTT()
    
    # Create test audio (1 second of silence + beep)
    audio = np.zeros(16000, dtype=np.int16)
    audio[8000:9000] = 1000  # Add beep
    
    audio_bytes = audio.tobytes()
    transcript = await stt.process_audio([audio_bytes])
    
    print(f"Transcript: {transcript}")
    # Note: Actual transcript will depend on model
    
    await stt.close()

async def test_stt_latency():
    """Test STT latency meets target (< 300ms)"""
    import time
    
    stt = StreamingSTT()
    
    # Create test audio
    audio = np.zeros(16000 * 2, dtype=np.int16)
    audio_bytes = audio.tobytes()
    
    # Measure latency
    start = time.time()
    await stt.process_audio([audio_bytes])
    latency_ms = (time.time() - start) * 1000
    
    print(f"STT Latency: {latency_ms:.2f}ms")
    assert latency_ms < 300, f"STT too slow: {latency_ms:.2f}ms"
    
    await stt.close()

if __name__ == "__main__":
    asyncio.run(test_stt_basic())
    asyncio.run(test_stt_latency())