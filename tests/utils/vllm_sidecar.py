"""Spin-up vLLM sidecar fixtures for integration tests.

Replaces ``ThreadingHTTPServer`` stubs with real vLLM containers serving
the same OpenAI-compatible contract production uses. Tests exercise the
``Remote*Loader`` + ``RemoteInferenceClient`` code paths against actual
inference behavior, not a mocked response shape.

Usage::

    def test_my_remote_path(vllm_sidecar):
        url = vllm_sidecar.spawn(
            model="openai/whisper-tiny",
            extra_args=["--max-model-len", "448"],
        )
        # url is http://127.0.0.1:<free port>; container auto-cleaned
        # when the session ends.

The factory caches sidecars by ``(model, image, extra_args)`` so a
second test asking for the same backend reuses the running container.
First-time pull of ``vllm/vllm-openai-cpu`` is ~5 GB (5-10 min); a
warmed cache responds in <2 min including model load.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests

DEFAULT_IMAGE = "vllm/vllm-openai-cpu:v0.23.0"
DEFAULT_HEALTH_DEADLINE_SECONDS = 600
HOST_HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def _merge_serve_args(model: str, extra_args: Optional[list[str]]) -> list[str]:
    """``extra_args`` plus serving defaults the deploy chart also applies.

    - ``--gpu-memory-utilization 0.10`` when unset (CPU vLLM budgets host RAM
      from this; the default 0.92 aborts on a loaded test host).
    - ``--limit-mm-per-prompt {"video":0,"image":1}`` for qwen3_vl (Tomoro
      ColQwen3): its ViT vision tower makes vLLM's startup profiler allocate a
      worst-case video attention buffer and OOM. Tomoro embeds image frames,
      never native video. Mirrors ``charts/.../values*.yaml`` so the sidecar
      exercises the real serving config.
    """
    merged = list(extra_args or [])
    if not any(a == "--gpu-memory-utilization" for a in merged):
        merged.extend(["--gpu-memory-utilization", "0.10"])
    if "colqwen3" in model.lower() and not any(
        a == "--limit-mm-per-prompt" for a in merged
    ):
        merged.extend(["--limit-mm-per-prompt", '{"video":0,"image":1}'])
    return merged


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_models(base_url: str, deadline_seconds: int, container: str) -> None:
    """Poll ``/v1/models`` until vLLM finishes loading the served model."""
    end = time.monotonic() + deadline_seconds
    last_err: Optional[str] = None
    while time.monotonic() < end:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if resp.status_code == 200 and resp.json().get("data"):
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
        f"vllm sidecar at {base_url} did not become healthy within "
        f"{deadline_seconds}s (last error: {last_err})\n"
        f"--- container logs ---\n{logs.stdout}\n{logs.stderr}"
    )


@dataclass
class _SpawnedSidecar:
    container: str
    base_url: str


@dataclass
class VllmSidecarFactory:
    """Per-session manager that spawns and reuses vLLM containers."""

    image: str = DEFAULT_IMAGE
    health_deadline_seconds: int = DEFAULT_HEALTH_DEADLINE_SECONDS
    _spawned: dict[tuple, _SpawnedSidecar] = field(default_factory=dict)

    def spawn(
        self,
        model: str,
        *,
        extra_args: Optional[list[str]] = None,
        image: Optional[str] = None,
        device: str = "cpu",
        env: Optional[dict[str, str]] = None,
    ) -> str:
        """Return a base URL serving ``model``. Cached by (model, image, args)."""
        image = image or self.image
        key = (
            model,
            image,
            tuple(extra_args or ()),
            device,
            tuple(sorted((env or {}).items())),
        )
        if key in self._spawned:
            return self._spawned[key].base_url

        container = f"cogniverse-vllm-test-{uuid.uuid4().hex[:8]}"
        port = _free_port()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "-p",
            f"{port}:8000",
            # CPU vllm reads this for its budget (default 0.1 of host RAM
            # = ~12 GiB on a 128 GiB box). Lower here so the sidecar can
            # start even when other test infra (Ollama 7b, Vespa, embedders)
            # has eaten most of the host's free RAM. Override per-test by
            # passing a different value via extra_args before this default.
            "-e",
            "VLLM_CPU_MEMORY_UTILIZATION=0.05",
            "-e",
            "VLLM_CPU_KVCACHE_SPACE=2",
            # Make this per-test sidecar more attractive to the kernel
            # OOM-killer than the session-scoped Vespa (which sets
            # oom-score-adj=-1000). vllm sidecars are short-lived and
            # easily restarted; losing one fails its own tests but
            # doesn't cascade. Losing Vespa breaks every memory test
            # downstream.
            "--oom-score-adj=500",
        ]
        for env_key, env_value in (env or {}).items():
            cmd.extend(["-e", f"{env_key}={env_value}"])
        if device == "rocm":
            # ROCm containers need the kernel fusion + render devices and
            # membership of the render group (gid varies per distro).
            cmd.extend(
                [
                    "--device",
                    "/dev/kfd",
                    "--device",
                    "/dev/dri",
                    "--group-add",
                    "video",
                    "--group-add",
                    "render",
                    "--security-opt",
                    "seccomp=unconfined",
                ]
            )
        if os.path.isdir(HOST_HF_CACHE):
            cmd.extend(["-v", f"{HOST_HF_CACHE}:/root/.cache/huggingface"])
        cmd.extend([image, "--model", model])
        cmd.extend(_merge_serve_args(model, extra_args))
        subprocess.run(cmd, check=True, timeout=60)

        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_models(base_url, self.health_deadline_seconds, container)
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
