"""
Configuration for real end-to-end integration tests.
"""

import os
from typing import Any, Dict

# Test environment configuration
E2E_CONFIG: Dict[str, Any] = {
    # LLM Configuration
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "smollm3:8b"),
    "openai_api_key": "fake-key",  # Ollama doesn't need real key
    # Backend Services
    "vespa_url": os.getenv("VESPA_URL", "http://localhost:8080"),
    "phoenix_url": os.getenv("PHOENIX_URL", "http://localhost:6006"),
    # Test Settings
    "test_timeout": int(os.getenv("E2E_TEST_TIMEOUT", "300")),  # 5 minutes default
    "optimization_rounds": int(
        os.getenv("DSPy_OPTIMIZATION_ROUNDS", "1")
    ),  # Keep quick for tests
    "max_training_examples": int(
        os.getenv("MAX_TRAINING_EXAMPLES", "3")
    ),  # Limit for speed
    # Test Data
    "test_video_dir": os.getenv(
        "TEST_VIDEO_DIR", "data/testset/evaluation/sample_videos"
    ),
    "test_queries_file": os.getenv(
        "TEST_QUERIES_FILE", "data/testset/evaluation/video_search_queries.csv"
    ),
    # Feature Flags
    "enable_vespa_tests": os.getenv("ENABLE_VESPA_TESTS", "false").lower() == "true",
    "enable_phoenix_tests": os.getenv("ENABLE_PHOENIX_TESTS", "false").lower()
    == "true",
    "enable_long_running_tests": os.getenv("ENABLE_LONG_RUNNING_TESTS", "false").lower()
    == "true",
}

# Model-specific configurations
MODEL_CONFIGS = {
    "smollm3:8b": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "timeout": 30,
    },
    "llama3.2:3b": {
        "max_tokens": 4096,
        "temperature": 0.5,
        "timeout": 45,
    },
    "qwen2.5:3b": {
        "max_tokens": 4096,
        "temperature": 0.6,
        "timeout": 40,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for specific model."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["smollm3:8b"])


def is_service_available(service_name: str) -> bool:
    """Check if a required service is available."""
    try:
        import requests

        if service_name == "ollama":
            url = E2E_CONFIG["ollama_base_url"].replace("/v1", "/api/tags")
            response = requests.get(url, timeout=5)
            return response.status_code == 200

        elif service_name == "vespa":
            response = requests.get(
                f"{E2E_CONFIG['vespa_url']}/ApplicationStatus", timeout=5
            )
            return response.status_code == 200

        elif service_name == "phoenix":
            response = requests.get(E2E_CONFIG["phoenix_url"], timeout=5)
            return response.status_code == 200

        return False

    except Exception:
        return False


def get_available_models() -> list:
    """Get list of available models from Ollama."""
    try:
        import requests

        url = E2E_CONFIG["ollama_base_url"].replace("/v1", "/api/tags")
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]

        return []

    except Exception:
        return []
