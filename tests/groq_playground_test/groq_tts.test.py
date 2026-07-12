"""
Ad hoc timing test for Groq's TTS (Orpheus) API (not part of src/ yet).
Compares against the < 150ms TTS first-audio target from CLAUDE.md.

Note: unlike src/tts_v2.py's streaming synthesis, the Groq SDK's
audio.speech.create() returns the full clip in one response - this measures
time-to-complete-audio, not time-to-first-audio-byte.

Usage: venv/bin/python tests/groq_tts.test.py
Requires: pip install groq, GROQ_API_KEY set in .env or the shell.
"""

import os
import time
from pathlib import Path

from src.config import config  # noqa: F401  (side effect: load_dotenv())
from groq import Groq

MODEL = "canopylabs/orpheus-v1-english"
VOICE = "troy"
TEXT = "Hello! This is a test of the text to speech system."
OUTPUT_PATH = Path(__file__).parent / "output" / "groq_tts_output.wav"


def test_tts_synthesis_latency():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    start = time.time()
    response = client.audio.speech.create(
        model=MODEL,
        voice=VOICE,
        input=TEXT,
        response_format="wav",
    )
    latency_ms = (time.time() - start) * 1000

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    response.write_to_file(OUTPUT_PATH)

    print(f"Model: {MODEL}, voice: {VOICE}")
    print(f"Saved audio: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print(f"Latency (full clip): {latency_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 150ms first audio (not directly comparable - see note above)")


if __name__ == "__main__":
    test_tts_synthesis_latency()
