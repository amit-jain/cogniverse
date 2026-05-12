"""Spin-up pylate sidecar fixtures for integration tests.

Companion to ``vllm_sidecar.py`` — builds and runs the
``deploy/pylate/`` Docker image with a configurable ColBERT or DenseOn
model, returning a base URL tests can point ``RemoteColBERTLoader`` /
``RemoteInferenceClient`` at.

Usage::

    def test_my_path(pylate_sidecar):
        url = pylate_sidecar.spawn(
            model="lightonai/Reason-ModernColBERT",
            mode="multi_vector",
        )
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_DIR = REPO_ROOT / "deploy" / "pylate"
DEFAULT_IMAGE_TAG = "cogniverse/pylate:inttest"
DEFAULT_HEALTH_DEADLINE_SECONDS = 300
HOST_HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, deadline_seconds: int, container: str) -> None:
    end = time.monotonic() + deadline_seconds
    last_err: Optional[str] = None
    while time.monotonic() < end:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(2)
    logs = subprocess.run(
        ["docker", "logs", "--tail", "200", container],
        capture_output=True,
        text=True,
        check=False,
    )
    raise AssertionError(
        f"pylate sidecar at {base_url} did not become healthy within "
        f"{deadline_seconds}s (last error: {last_err})\n"
        f"--- container logs ---\n{logs.stdout}\n{logs.stderr}"
    )


@dataclass
class _SpawnedSidecar:
    container: str
    base_url: str


@dataclass
class PylateSidecarFactory:
    """Per-session manager for pylate sidecar containers."""

    image_tag: str = DEFAULT_IMAGE_TAG
    health_deadline_seconds: int = DEFAULT_HEALTH_DEADLINE_SECONDS
    _image_built: bool = False
    _spawned: dict[tuple, _SpawnedSidecar] = field(default_factory=dict)

    def _build_image(self) -> None:
        if self._image_built:
            return
        subprocess.run(
            ["docker", "build", "-t", self.image_tag, str(SIDECAR_DIR)],
            check=True,
            timeout=1200,
        )
        self._image_built = True

    def spawn(
        self,
        model: str,
        *,
        mode: str = "multi_vector",
        device: str = "cpu",
    ) -> str:
        """Return base URL for a pylate sidecar serving ``model`` in ``mode``.

        Cached by (model, mode, device) so reuse across tests is free.
        """
        key = (model, mode, device)
        if key in self._spawned:
            return self._spawned[key].base_url

        self._build_image()

        container = f"cogniverse-pylate-test-{uuid.uuid4().hex[:8]}"
        port = _free_port()
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container,
            "-p",
            f"{port}:8080",
            "-e",
            f"MODEL_NAME={model}",
            "-e",
            f"MODE={mode}",
            "-e",
            f"DEVICE={device}",
            # Per-test sidecar — make this more attractive to the kernel
            # OOM-killer than session-scoped Vespa (which sets
            # oom-score-adj=-1000). Easier to restart this than to lose
            # Vespa's accumulated schema state mid-sweep.
            "--oom-score-adj=500",
        ]
        if os.path.isdir(HOST_HF_CACHE):
            cmd.extend(["-v", f"{HOST_HF_CACHE}:/root/.cache/huggingface"])
        cmd.append(self.image_tag)
        subprocess.run(cmd, check=True, timeout=60)

        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_health(base_url, self.health_deadline_seconds, container)
        except Exception:
            subprocess.run(
                ["docker", "rm", "-f", container],
                check=False,
                timeout=30,
                capture_output=True,
            )
            raise

        self._spawned[key] = _SpawnedSidecar(container=container, base_url=base_url)
        return base_url

    def teardown(self) -> None:
        for sidecar in self._spawned.values():
            subprocess.run(
                ["docker", "rm", "-f", sidecar.container],
                check=False,
                timeout=30,
                capture_output=True,
            )
        self._spawned.clear()
