"""
End-to-end latency test for the /stream websocket (VAD -> STT -> vector
search -> LLM -> TTS), run entirely locally with no mic/browser needed.

Closes the loop on-device: synthesizes a spoken question with the same
local Piper TTS used for output (src/tts_v2.py, 16kHz mono PCM16 - exactly
what VAD/STT expect), streams it to /stream in 4096-sample chunks (same
chunk size static/demo.html's ScriptProcessorNode uses), then a tail of
silence chunks to trigger end-of-utterance, and times the response.

Prereqs: the FastAPI server must already be running (see command printed
at the bottom of this file, or just run `venv/bin/python -m src.main`).

Usage: venv/bin/python tests/local_tests/full_pipeline_stream.test.py ["question text"]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tts_v2 import PiperTTS, TARGET_SAMPLE_RATE  # noqa: E402

WS_URL = "ws://localhost:8000/stream"
CHUNK_SAMPLES = 4096  # matches static/demo.html's createScriptProcessor(4096, 1, 1)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16
TRAILING_SILENCE_CHUNKS = 8  # > max_silence(5) chunks of ~256ms each, to trip end-of-utterance
SEND_INTERVAL_S = CHUNK_SAMPLES / TARGET_SAMPLE_RATE  # simulate real-time mic streaming


async def synthesize_question(text: str) -> bytes:
    tts = PiperTTS()
    await tts.synthesize("Warm up.")  # exclude model warm-up from the test itself
    audio = await tts.synthesize(text)
    await tts.close()
    return audio


async def run(question: str):
    print(f"Synthesizing input speech locally: '{question}'")
    speech_pcm = await synthesize_question(question)
    speech_duration_s = len(speech_pcm) / 2 / TARGET_SAMPLE_RATE
    print(f"Input speech: {len(speech_pcm)} bytes (~{speech_duration_s:.2f}s)")

    silence_chunk = b"\x00" * CHUNK_BYTES
    speech_chunks = [speech_pcm[i:i + CHUNK_BYTES] for i in range(0, len(speech_pcm), CHUNK_BYTES)]
    n_speech_chunks = len(speech_chunks)
    chunks = speech_chunks + [silence_chunk] * TRAILING_SILENCE_CHUNKS

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print(f"Connected to {WS_URL}. Streaming {len(chunks)} chunks "
              f"({SEND_INTERVAL_S * 1000:.0f}ms apart, simulating real-time mic input)...")

        t_speech_end = None
        t_transcript = None
        t_response = None
        t_first_audio = None
        t_last_audio = None
        audio_bytes_received = 0
        transcript_text = None
        response_text = None

        async def sender():
            nonlocal t_speech_end
            for i, chunk in enumerate(chunks):
                await ws.send(chunk)
                await asyncio.sleep(SEND_INTERVAL_S)
                # "speech end" = the moment the last *real* speech chunk was
                # sent. Everything after is silence padding purely so the
                # server's VAD can detect end-of-utterance - not part of what
                # a real user would perceive as "when I stopped talking".
                if i == n_speech_chunks - 1:
                    t_speech_end = time.time()

        async def receiver():
            nonlocal t_transcript, t_response, t_first_audio, t_last_audio
            nonlocal audio_bytes_received, transcript_text, response_text
            try:
                async for msg in ws:
                    now = time.time()
                    if isinstance(msg, bytes):
                        if t_first_audio is None:
                            t_first_audio = now
                        t_last_audio = now
                        audio_bytes_received += len(msg)
                    else:
                        data = json.loads(msg)
                        if data.get("type") == "transcript":
                            t_transcript = now
                            transcript_text = data["transcript"]
                            print(f"  [transcript] '{transcript_text}'")
                        elif data.get("type") == "response":
                            t_response = now
                            response_text = data["response"]
                            print(f"  [response]   '{response_text}'")
                            break  # response text arrives after all TTS audio for this turn
            except websockets.exceptions.ConnectionClosed:
                pass

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await send_task
        # Give the receiver a bounded window to finish draining this turn's audio/response
        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
            print("  WARNING: timed out waiting for response")

    print("\n--- Results ---")
    print(f"Transcript: '{transcript_text}'")
    print(f"Response:   '{response_text}'")
    print(f"Audio bytes received: {audio_bytes_received}")
    if t_speech_end and t_transcript:
        print(f"Speech-end -> transcript:      {(t_transcript - t_speech_end) * 1000:.0f}ms")
    if t_transcript and t_first_audio:
        print(f"Transcript -> first TTS audio: {(t_first_audio - t_transcript) * 1000:.0f}ms")
    if t_speech_end and t_first_audio:
        print(f"Speech-end -> first TTS audio (perceived latency): {(t_first_audio - t_speech_end) * 1000:.0f}ms")
    if t_speech_end and t_last_audio:
        print(f"Speech-end -> last TTS audio:  {(t_last_audio - t_speech_end) * 1000:.0f}ms")
    if t_speech_end and t_response:
        print(f"Speech-end -> total turn:      {(t_response - t_speech_end) * 1000:.0f}ms")

    print("\nFetching server-side /metrics for the component breakdown...")
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/metrics/latency", timeout=5) as r:
            print(json.dumps(json.loads(r.read()), indent=2))
    except Exception as e:
        print(f"  (could not fetch /metrics/latency: {e})")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the capital of France?"
    asyncio.run(run(q))
