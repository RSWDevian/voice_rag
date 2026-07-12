"""
GPU-accelerated timing test for local Piper TTS (src/tts_v2.py) - runs the
ONNX model on CUDA via onnxruntime-gpu instead of CPU.
Compares against the < 150ms TTS first-audio target from CLAUDE.md, and
against tests/local_tests/piper_tts.test.py (CPU) for a direct A/B comparison.

Requires `onnxruntime-gpu` instead of the CPU-only `onnxruntime` (both
packages provide the same `onnxruntime` import name, so only one can be
installed at a time):
    pip uninstall onnxruntime
    pip install onnxruntime-gpu==1.20.1
(1.20.1 pinned deliberately - onnxruntime-gpu 1.21+ links against CUDA 13's
libcudart.so.13, but this project's CUDA libs come from torch==2.4.1+cu121's
bundled nvidia-*-cu12 packages, i.e. CUDA 12.1. 1.20.1 is the newest release
still built against CUDA 12.x/cuDNN 9.x, so it resolves those libs directly
with no LD_LIBRARY_PATH/system CUDA Toolkit install needed.)

Usage: PYTHONPATH=. venv/bin/python tests/local_tests/piper_tts_gpu.test.py
"""

import glob
import os
import sys


def _ensure_cuda_libs_on_path() -> None:
    """
    onnxruntime-gpu doesn't bundle cuBLAS/cuDNN itself - it dlopen's them at
    CUDAExecutionProvider init (InferenceSession creation), expecting them on
    LD_LIBRARY_PATH. torch already pulled compatible copies in via its
    nvidia-cublas-cu12/nvidia-cudnn-cu12 pip deps, so point at those instead
    of requiring a separate system CUDA Toolkit/cuDNN install.

    glibc's dynamic linker only reads LD_LIBRARY_PATH once at process
    startup, so mutating os.environ from within an already-running process
    has no effect on later dlopen() calls - the CUDA provider would silently
    fall back to CPU. So if the paths aren't already present, re-exec this
    script as a fresh process with the corrected environment instead.
    """
    import nvidia

    nvidia_dir = os.path.dirname(nvidia.__file__)
    lib_dirs = glob.glob(os.path.join(nvidia_dir, "*", "lib"))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_dirs = current.split(os.pathsep) if current else []

    if all(d in current_dirs for d in lib_dirs):
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + current_dirs)
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs_on_path()

import asyncio
import time
import wave
from pathlib import Path

import onnxruntime as ort

from src.config import config
from src.tts_v2 import PiperTTS, TARGET_SAMPLE_RATE

TEXT = "Hello! This is a test of the text to speech system."
OUTPUT_PATH = Path(__file__).parent / "output" / "piper_tts_gpu_output.wav"


async def test_tts_synthesis_latency_gpu():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        sys.exit(
            "ERROR: onnxruntime has no CUDAExecutionProvider available "
            f"(found: {ort.get_available_providers()}).\n"
            "Install the GPU build with: pip uninstall onnxruntime && pip install onnxruntime-gpu==1.20.1"
        )

    config.PIPER_USE_CUDA = True  # force GPU for this test, independent of .env
    tts = PiperTTS()

    active_providers = tts.voice.session.get_providers()
    if "CUDAExecutionProvider" not in active_providers:
        sys.exit(
            f"ERROR: Piper session fell back to CPU (active providers: {active_providers}).\n"
            "Check that the GPU is free (nvidia-smi) and cuDNN/cuBLAS libs are reachable."
        )

    # Warm-up call so model load / first-inference overhead (CUDA context
    # init, kernel autotuning) isn't counted against the measured latency.
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
    print(f"Active onnxruntime providers: {active_providers}")
    print(f"Saved audio: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print(f"Latency (full clip, post warm-up): {latency_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 150ms first audio (not directly comparable - see note above)")

    await tts.close()


if __name__ == "__main__":
    asyncio.run(test_tts_synthesis_latency_gpu())
