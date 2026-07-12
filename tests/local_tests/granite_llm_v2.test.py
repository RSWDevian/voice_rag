"""
Timing test for src/llm_v2.py (local Ollama, ibm/granite3.1-moe:1b) going
through the actual StreamingLLM/get_llm() wrapper used by src/pipeline.py -
as opposed to tests/local_tests/granite3.1_moe_llm.test.py, which hits
Ollama's HTTP API directly and doesn't exercise src/llm_v2.py at all.
Compares against the < 200ms LLM first-token target from CLAUDE.md.

Usage: PYTHONPATH=. venv/bin/python tests/local_tests/granite_llm_v2.test.py
"""

import asyncio
import time

from src.llm_v2 import StreamingLLM

QUERY = {"text": "In one short sentence, what is retrieval-augmented generation?", "intent": "question"}
CONTEXT = []


async def test_llm_v2_streaming_latency():
    llm = StreamingLLM()

    # Warm-up call so Ollama's model-load-into-VRAM cost isn't counted
    # against the measured latency.
    async for _ in llm.stream_response(QUERY, CONTEXT):
        pass

    start = time.time()
    first_chunk_ms = None
    chunks = []

    async for chunk in llm.stream_response(QUERY, CONTEXT):
        if first_chunk_ms is None:
            first_chunk_ms = (time.time() - start) * 1000
        chunks.append(chunk)

    total_ms = (time.time() - start) * 1000
    response = "".join(chunks)

    print(f"Model: {llm.llm.model} (via Ollama at {llm.llm.base_url})")
    print(f"Response: {response}")
    print(f"Time to first chunk (post warm-up): {first_chunk_ms:.0f}ms")
    print(f"Total latency (post warm-up): {total_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 200ms first token")

    await llm.close()


if __name__ == "__main__":
    asyncio.run(test_llm_v2_streaming_latency())
