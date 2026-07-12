"""
Ad hoc timing test for Groq's chat completions API (not part of src/ yet).
Compares against the < 200ms LLM first-token target from CLAUDE.md.

Usage: venv/bin/python tests/groq_llm.test.py
Requires: pip install groq, GROQ_API_KEY set in .env or the shell.
"""

import os
import time

from src.config import config  # noqa: F401  (side effect: load_dotenv())
from groq import Groq

MODEL = "openai/gpt-oss-120b"
PROMPT = "In one short sentence, what is retrieval-augmented generation?"


def test_llm_streaming_latency():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    start = time.time()
    first_token_ms = None
    chunks = []

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_completion_tokens=150,
        stream=True,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        if content and first_token_ms is None:
            first_token_ms = (time.time() - start) * 1000
        chunks.append(content)

    total_ms = (time.time() - start) * 1000
    response = "".join(chunks)

    print(f"Model: {MODEL}")
    print(f"Response: {response}")
    print(f"Time to first token: {first_token_ms:.0f}ms")
    print(f"Total latency: {total_ms:.0f}ms")
    print(f"Target (CLAUDE.md): < 200ms first token")


if __name__ == "__main__":
    test_llm_streaming_latency()
