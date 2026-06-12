"""Self-provisioned LLM for integration tests.

Integration tests must not depend on the k3d cluster (that's the e2e
tier). ``ensure_llm()`` provisions a vLLM Docker sidecar serving the
SAME model as the production student LM — so semantic assertions and
goldens keep their meaning — and points the config chain at it by
writing a session config and exporting ``COGNIVERSE_CONFIG``.

The container has a fixed name and is reused across pytest sessions:
first spawn pays the model load, every later session reattaches in
seconds. On a ROCm host the sidecar runs GPU-accelerated (detected via
``detect_torch_backend``); elsewhere it falls back to CPU vLLM.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER = "cogniverse-test-llm"
MODEL = "google/gemma-4-e4b-it"
HOST_PORT = 29110
HERMETIC_CONFIG = REPO_ROOT / "outputs" / ".hermetic" / "config.json"
_HF_CACHE = str(Path.home() / ".cache" / "huggingface")

_IMAGES = {
    "rocm": "vllm/vllm-openai-rocm:v0.20.0",
    "cpu": "vllm/vllm-openai-cpu:latest",
}


def _detect_device() -> str:
    try:
        from cogniverse_cli.images import detect_torch_backend

        backend = detect_torch_backend()
    except Exception:
        backend = "cpu"
    return "rocm" if backend == "rocm" else "cpu"


def _healthy(base_url: str, timeout: float = 3.0) -> bool:
    try:
        return requests.get(f"{base_url}/v1/models", timeout=timeout).status_code == 200
    except Exception:
        return False


def _container_state() -> Optional[str]:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def _spawn(device: str, gpu_utilization: float = 0.25) -> None:
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER,
        "-p",
        f"{HOST_PORT}:8000",
        "-v",
        f"{_HF_CACHE}:/root/.cache/huggingface",
        # Short-lived relative to the session Vespa — prefer killing this
        # over the shared containers under memory pressure.
        "--oom-score-adj=400",
    ]
    if device == "rocm":
        cmd += [
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
        engine_args = [
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            str(gpu_utilization),
        ]
    else:
        cmd += ["-e", "VLLM_CPU_KVCACHE_SPACE=4"]
        engine_args = ["--max-model-len", "4096"]
    cmd += [_IMAGES[device], "--model", MODEL, *engine_args]
    subprocess.run(cmd, check=True, timeout=120)


def _write_session_config(api_base: str) -> Path:
    config = json.loads((REPO_ROOT / "configs" / "config.json").read_text())
    llm = config.setdefault("llm_config", {})
    for key in ("primary", "teacher"):
        endpoint = llm.setdefault(key, {})
        endpoint["api_base"] = api_base
    # Legacy top-level fallback some readers still consult.
    config["base_url"] = api_base
    HERMETIC_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    HERMETIC_CONFIG.write_text(json.dumps(config, indent=2))
    return HERMETIC_CONFIG


def ensure_llm(deadline_s: float = 900.0) -> Optional[str]:
    """Provision (or reattach to) the test LM; return its OpenAI base URL.

    Side effect: exports ``COGNIVERSE_CONFIG`` pointing at a session
    config whose llm endpoints target the sidecar — every config-chain
    consumer in this process resolves to it. Returns None when Docker is
    unavailable or the model never becomes ready (callers skip).
    """
    base_url = f"http://127.0.0.1:{HOST_PORT}/v1"
    probe = f"http://127.0.0.1:{HOST_PORT}"

    state = _container_state()
    if state == "running" and _healthy(probe):
        os.environ["COGNIVERSE_CONFIG"] = str(_write_session_config(base_url))
        return base_url
    def _await_ready(budget_s: float) -> bool:
        deadline = time.time() + budget_s
        while time.time() < deadline:
            if _healthy(probe):
                return True
            if _container_state() != "running":
                return False  # crashed/exited — caller tries the next rung
            time.sleep(5)
        return False

    try:
        if state == "running":
            if _await_ready(deadline_s):
                os.environ["COGNIVERSE_CONFIG"] = str(_write_session_config(base_url))
                return base_url
            return None
        if state is not None:
            subprocess.run(["docker", "start", CONTAINER], check=True, timeout=60)
            if _await_ready(deadline_s):
                os.environ["COGNIVERSE_CONFIG"] = str(_write_session_config(base_url))
                return base_url
            subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)

        # Fresh spawns: on ROCm the cluster's vLLM pods may hold most of
        # the unified pool, so step the budget down before giving up on
        # the GPU; CPU vLLM is the always-works fallback.
        device = _detect_device()
        attempts = (
            [("rocm", 0.25), ("rocm", 0.12), ("cpu", 0.0)]
            if device == "rocm"
            else [("cpu", 0.0)]
        )
        for dev, util in attempts:
            subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
            _spawn(dev, gpu_utilization=util)
            if _await_ready(deadline_s):
                os.environ["COGNIVERSE_CONFIG"] = str(_write_session_config(base_url))
                return base_url
    except Exception:
        return None
    return None
