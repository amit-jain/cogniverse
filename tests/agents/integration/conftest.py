"""
Shared fixtures and utilities for agent integration tests.
"""

import httpx
import pytest


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is available."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def is_openai_api_available() -> bool:
    """Check if OpenAI API key is available."""
    import os

    return bool(os.getenv("OPENAI_API_KEY"))


# Skip markers for integration tests
skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama service not available at http://localhost:11434",
)

skip_if_no_openai = pytest.mark.skipif(
    not is_openai_api_available(), reason="OPENAI_API_KEY environment variable not set"
)
