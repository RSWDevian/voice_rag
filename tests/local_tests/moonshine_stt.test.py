"""
Ad hoc timing test for local Moonshine STT (src/stt_v2.py) - runs fully on-device.
Compares against the < 300ms ASR first-token target from CLAUDE.md.

Note: named moonshine_stt_test.py (not *_tts_*) - Moonshine is an ASR/STT model,
it has no TTS counterpart; see src/stt_v2.py.

Uses the same sample speech clip as tests/groq_platground_test/groq_stt.test.py
(bundled with moonshine-voice, already a project dependency) so results are
directly comparable to the Groq Whisper timing.

Two benchmarks:
  1. One-shot transcribe() - the whole clip handed over in a single call.
  2. True streaming stream_transcribe() - the clip is fed in as ~30ms PCM
     frames (simulating a live mic/VAD pipeline), same as process_stream()
     would see in production. Reports time-to-first-partial-transcript
     (comparable to the < 300ms ASR first-token target) as well as total
     time to the final transcript.

Usage: PYTHONPATH=. venv/bin/python tests/local_tests/moonshine_stt.test.py
"""

import asyncio
import time
from pathlib import Path

import moonshine_voice

from src.stt_v2 import MoonshineSTT

SAMPLE_AUDIO = Path(moonshine_voice.__file__).parent / "assets" / "clone-test.wav"
FRAME_MS = 30  # matches the 30ms chunk_duration assumed by stream_transcribe()
SAMPLE_RATE = 16000
BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 mono


def _load_pcm() -> bytes:
    with open(SAMPLE_AUDIO, "rb") as f:
        f.seek(44)  # skip the 44-byte WAV header - raw PCM16 from here on
        return f.read()


async def _frame_generator(pcm_bytes: bytes):
    for i in range(0, len(pcm_bytes), BYTES_PER_FRAME):
        yield pcm_bytes[i : i + BYTES_PER_FRAME]


async def test_one_shot_latency(stt: MoonshineSTT, pcm_bytes: bytes):
    await stt.transcribe(pcm_bytes)  # warm-up, excluded from the measured latency

    start = time.time()
    text = await stt.transcribe(pcm_bytes)
    latency_ms = (time.time() - start) * 1000

    print("--- One-shot transcribe() ---")
    print(f"Transcript: {text}")
    print(f"Latency (post warm-up): {latency_ms:.0f}ms")


async def test_streaming_latency(stt: MoonshineSTT, pcm_bytes: bytes):
    # Warm-up pass through the streaming path too, so CUDA/session-init-style
    # one-time costs on the first Stream aren't counted below.
    async for _ in stt.stream_transcribe(_frame_generator(pcm_bytes)):
        pass

    start = time.time()
    first_partial_ms = None
    final_text = ""

    async for text in stt.stream_transcribe(_frame_generator(pcm_bytes)):
        if first_partial_ms is None:
            first_partial_ms = (time.time() - start) * 1000
        final_text = text

    total_ms = (time.time() - start) * 1000

    print("\n--- True streaming stream_transcribe() ---")
    print(f"Frame size: {FRAME_MS}ms, fed as {len(pcm_bytes) // BYTES_PER_FRAME} frames")
    print(f"Final transcript: {final_text}")
    print(f"Time to first partial: {first_partial_ms:.0f}ms" if first_partial_ms else "No partials yielded")
    print(f"Total time to final transcript: {total_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 300ms ASR first token")


async def main():
    stt = MoonshineSTT()
    pcm_bytes = _load_pcm()

    print(f"Model: moonshine ({stt.model_arch.name.lower()})")
    print(f"Audio: {SAMPLE_AUDIO.name}\n")

    await test_one_shot_latency(stt, pcm_bytes)
    await test_streaming_latency(stt, pcm_bytes)

    await stt.close()


if __name__ == "__main__":
    asyncio.run(main())
