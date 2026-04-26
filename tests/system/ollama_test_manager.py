"""Spin up an Ollama container with a small multimodal model preloaded.

Used by integration tests that need a real LLM provider for the visual judge
end-to-end path. Models are persisted in a named Docker volume
(``cogniverse-test-ollama-models``) so subsequent runs reuse the pulled
weights instead of re-downloading on every CI/dev invocation.
"""

from __future__ import annotations

import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import requests

DEFAULT_MODEL = "moondream"
MODELS_VOLUME = "cogniverse-test-ollama-models"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Ollama never came up at {url}")


@dataclass
class OllamaInstance:
    container_name: str
    api_port: int
    model: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"


class OllamaTestManager:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._instance: OllamaInstance | None = None

    def start(self, name_prefix: str = "ollama-cogniverse-test") -> OllamaInstance:
        # Ensure the model volume exists so weights persist across runs.
        subprocess.run(
            ["docker", "volume", "create", MODELS_VOLUME],
            check=False,
            capture_output=True,
        )

        name = f"{name_prefix}-{uuid.uuid4().hex[:8]}"
        api_port = _free_port()

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "-p",
                f"{api_port}:11434",
                "-v",
                f"{MODELS_VOLUME}:/root/.ollama",
                "ollama/ollama:latest",
            ],
            check=True,
            capture_output=True,
        )

        try:
            _wait_for_http(f"http://127.0.0.1:{api_port}/api/tags", timeout=60)
            # Pull the requested model (idempotent, cached in the volume).
            subprocess.run(
                ["docker", "exec", name, "ollama", "pull", self.model],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception:
            self._stop(name)
            raise

        self._instance = OllamaInstance(
            container_name=name, api_port=api_port, model=self.model
        )
        return self._instance

    @staticmethod
    def _stop(container_name: str) -> None:
        subprocess.run(
            ["docker", "stop", container_name],
            check=False,
            capture_output=True,
            timeout=15,
        )

    def stop(self) -> None:
        if self._instance is None:
            return
        self._stop(self._instance.container_name)
        self._instance = None

    @contextmanager
    def lifecycle(self, name_prefix: str = "ollama-cogniverse-test"):
        try:
            yield self.start(name_prefix=name_prefix)
        finally:
            self.stop()
