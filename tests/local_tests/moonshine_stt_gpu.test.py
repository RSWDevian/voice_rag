"""
GPU-support diagnostic + timing test for local Moonshine STT (src/stt_v2.py).

Short answer up front: Moonshine's Linux x86_64 wheel (moonshine-voice==0.0.65)
bundles its own private ONNX Runtime build inside libmoonshine.so - it does
NOT use the pip `onnxruntime`/`onnxruntime-gpu` package at all (unlike Piper
in tests/local_tests/piper_tts_gpu.test.py, which does). Evidence:

  1. `ldd libmoonshine.so` shows no libcudart/libcublas/libcudnn dependency -
     the bundled ONNX Runtime was built CPU-only.
  2. Passing `options={"use_cuda": "1"}` to Transcriber() - the one place a
     provider-like flag could plausibly be plumbed through the C API's
     generic key=value options dict - fails with
     "Unknown transcriber option: 'use_cuda'", proving the native library
     doesn't recognize any GPU-related option on this platform/build.

So there is currently no supported way to run Moonshine STT on GPU here.
This script proves both points programmatically, then falls back to
reporting the (CPU) transcription latency for reference, matching
tests/local_tests/moonshine_stt.test.py.

Usage: PYTHONPATH=. venv/bin/python tests/local_tests/moonshine_stt_gpu.test.py
"""

import subprocess
import time
from pathlib import Path

import moonshine_voice

from src.stt_v2 import MoonshineSTT
from src.config import config

SAMPLE_AUDIO = Path(moonshine_voice.__file__).parent / "assets" / "clone-test.wav"


def _check_bundled_onnxruntime_has_no_cuda() -> bool:
    """Returns True if libmoonshine.so has no CUDA runtime library linked."""
    lib_path = Path(moonshine_voice.__file__).parent / "libmoonshine.so"
    result = subprocess.run(["ldd", str(lib_path)], capture_output=True, text=True)
    linked_libs = result.stdout.lower()
    return not any(lib in linked_libs for lib in ("libcudart", "libcublas", "libcudnn"))


def _check_use_cuda_option_rejected() -> str:
    """Attempts Transcriber(options={"use_cuda": "1"}) and returns the error (or "" if it succeeded)."""
    from moonshine_voice import ModelArch, Transcriber, get_model_for_language

    model_path, model_arch = get_model_for_language(
        "en", ModelArch.SMALL_STREAMING, cache_root=config.MODELS_DIR
    )
    try:
        transcriber = Transcriber(
            model_path=model_path, model_arch=model_arch, options={"use_cuda": "1"}
        )
        transcriber.close()
        return ""
    except Exception as e:
        return str(e)


async def _measure_cpu_latency() -> tuple[str, float]:
    stt = MoonshineSTT()

    with open(SAMPLE_AUDIO, "rb") as f:
        f.seek(44)  # skip the WAV header - transcribe() expects raw PCM16
        audio_bytes = f.read()

    await stt.transcribe(audio_bytes)  # warm-up, excluded from the measured latency

    start = time.time()
    text = await stt.transcribe(audio_bytes)
    latency_ms = (time.time() - start) * 1000

    await stt.close()
    return text, latency_ms


def test_moonshine_gpu_support():
    no_cuda_linked = _check_bundled_onnxruntime_has_no_cuda()
    print(f"libmoonshine.so has no CUDA runtime linked: {no_cuda_linked}")

    use_cuda_error = _check_use_cuda_option_rejected()
    if use_cuda_error:
        print(f"Transcriber(options={{'use_cuda': '1'}}) rejected: {use_cuda_error}")
    else:
        print("Transcriber(options={'use_cuda': '1'}) was accepted (unexpected - re-check this script's assumptions)")

    gpu_supported = (not no_cuda_linked) or (not use_cuda_error)
    print(f"\nVerdict: GPU {'IS' if gpu_supported else 'is NOT'} usable for Moonshine STT on this build.")

    import asyncio
    text, latency_ms = asyncio.run(_measure_cpu_latency())
    print(f"\nFallback CPU benchmark (post warm-up):")
    print(f"Transcript: {text}")
    print(f"Latency: {latency_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 300ms ASR first token")


if __name__ == "__main__":
    test_moonshine_gpu_support()
