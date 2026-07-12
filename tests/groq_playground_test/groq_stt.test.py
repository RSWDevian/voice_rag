"""
Ad hoc timing test for Groq's Whisper transcription API (not part of src/ yet).
Compares against the < 300ms ASR first-token target from CLAUDE.md.

Uses the sample speech clip bundled with moonshine-voice (already a project
dependency for stt_v2.py) so no extra audio asset needs to be committed.

Usage: venv/bin/python tests/groq_stt.test.py
Requires: pip install groq, GROQ_API_KEY set in .env or the shell.
"""

import os
import time
from pathlib import Path

import moonshine_voice
from src.config import config  # noqa: F401  (side effect: load_dotenv())
from groq import Groq

MODEL = "whisper-large-v3-turbo"
SAMPLE_AUDIO = Path(moonshine_voice.__file__).parent / "assets" / "clone-test.wav"


def test_stt_transcription_latency():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    start = time.time()
    with open(SAMPLE_AUDIO, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=MODEL,
            response_format="json",
            language="en",
            temperature=0.0,
        )
    latency_ms = (time.time() - start) * 1000

    print(f"Model: {MODEL}")
    print(f"Audio: {SAMPLE_AUDIO.name}")
    print(f"Transcript: {transcription.text}")
    print(f"Latency: {latency_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 300ms ASR first token")


if __name__ == "__main__":
    test_stt_transcription_latency()
