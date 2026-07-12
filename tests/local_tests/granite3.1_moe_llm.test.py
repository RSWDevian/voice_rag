"""
Ad hoc timing test for local Llama 3 served via Ollama (not part of src/ yet).
Compares against the < 200ms LLM first-token target from CLAUDE.md.

Assumes an Ollama server is already running locally (`ollama serve`, or the
desktop app) with the model pulled (`ollama pull llama3`). Exits with a clear
error if the model isn't present rather than silently failing on the request.

Usage: PYTHONPATH=. venv/bin/python tests/local_tests/llama3_llm.test.py
"""

import json
import sys
import time

import httpx

BASE_URL = "http://localhost:11434"
MODEL = "ibm/granite3.1-moe:1b"
PROMPT = "In one short sentence, what is retrieval-augmented generation?"


def _check_model_available(client: httpx.Client) -> None:
    try:
        response = client.get(f"{BASE_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
    except httpx.ConnectError:
        sys.exit(f"ERROR: could not reach Ollama at {BASE_URL} - is `ollama serve` running?")

    models = {m["name"] for m in response.json().get("models", [])}
    if MODEL not in models:
        sys.exit(
            f"ERROR: model '{MODEL}' not found in Ollama (available: {sorted(models) or 'none'}).\n"
            f"Run `ollama pull {MODEL.split(':')[0]}` first."
        )


def _generate_streaming(client: httpx.Client) -> tuple[str, float, float]:
    start = time.time()
    first_token_ms = None
    chunks = []

    with client.stream(
        "POST",
        f"{BASE_URL}/api/generate",
        json={"model": MODEL, "prompt": PROMPT, "stream": True},
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            content = data.get("response", "")
            if content and first_token_ms is None:
                first_token_ms = (time.time() - start) * 1000
            chunks.append(content)
            if data.get("done"):
                break

    total_ms = (time.time() - start) * 1000
    return "".join(chunks), first_token_ms, total_ms


def test_llm_streaming_latency():
    with httpx.Client() as client:
        _check_model_available(client)

        # Warm-up call: loads the model into VRAM so cold-load time isn't
        # counted against the measured latency.
        _generate_streaming(client)

        response, first_token_ms, total_ms = _generate_streaming(client)

    print(f"Model: {MODEL} (via Ollama at {BASE_URL})")
    print(f"Response: {response}")
    print(f"Time to first token (post warm-up): {first_token_ms:.0f}ms")
    print(f"Total latency (post warm-up): {total_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 200ms first token")


if __name__ == "__main__":
    test_llm_streaming_latency()
