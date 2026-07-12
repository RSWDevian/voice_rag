"""
Ad hoc timing test for local Piper TTS (src/tts_v2.py) - runs fully on-device.
Compares against the < 150ms TTS first-audio target from CLAUDE.md.

Note: PiperTTS.synthesize() returns the full clip in one call (no incremental
audio callback), so this measures time-to-complete-audio, not time-to-first-
audio-byte - same caveat as tests/groq_platground_test/groq_tts.test.py.

Usage: venv/bin/python tests/local_tests/piper_tts_test.py
"""

import asyncio
import time
import wave
from pathlib import Path

from src.tts_v2 import PiperTTS, TARGET_SAMPLE_RATE

TEXT = "Hello! This is a test of the text to speech system."
OUTPUT_PATH = Path(__file__).parent / "output" / "piper_tts_output.wav"


async def test_tts_synthesis_latency():
    tts = PiperTTS()

    # Warm-up call so model load / first-inference overhead isn't counted
    # against the measured latency.
    await tts.synthesize("Warm up.")

    start = time.time()
    audio = await tts.synthesize(TEXT)
    latency_ms = (time.time() - start) * 1000

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(audio)

    print(f"Model: piper ({tts.voice_name})")
    print(f"Saved audio: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print(f"Latency (full clip, post warm-up): {latency_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 150ms first audio (not directly comparable - see note above)")

    await tts.close()


if __name__ == "__main__":
    asyncio.run(test_tts_synthesis_latency())
