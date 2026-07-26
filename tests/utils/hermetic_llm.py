"""Self-provisioned exact LLMs for integration tests.

``ensure_llm()`` first reuses a configured endpoint only when its OpenAI
model-list contract names the requested production model exactly.
Otherwise it provisions that identical model in a local vLLM sidecar.
``activate_llms()`` then writes the selected exact production roles into
the session config.

Each model has a fixed container name and is reused across pytest
sessions. On a ROCm host the sidecar runs GPU-accelerated; elsewhere it
falls back to CPU vLLM.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from tests.utils.vllm_sidecar import (
    _configured_model_urls,
    _server_base,
    find_exact_model_endpoint,
    listed_model_ids,
    serves_exact_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = REPO_ROOT / "configs" / "config.json"
CONTAINER = "cogniverse-test-llm"
MODEL = "google/gemma-4-e4b-it"
HOST_PORT = 29110
TEACHER_CONTAINER = "cogniverse-test-llm-teacher"
TEACHER_MODEL = "google/gemma-4-26b-a4b-it"
TEACHER_HOST_PORT = 29111
HERMETIC_CONFIG_DIR = REPO_ROOT / "outputs" / ".hermetic"
_LOCK_PATH = (
    Path(tempfile.gettempdir()) / f"cogniverse-exact-llm-sidecars-{os.getuid()}.lock"
)
_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
_ENSURE_LOCK = threading.Lock()
_SIDECARS = {
    MODEL: (CONTAINER, HOST_PORT),
    TEACHER_MODEL: (TEACHER_CONTAINER, TEACHER_HOST_PORT),
}

_IMAGES = {
    "rocm": "vllm/vllm-openai-rocm:v0.23.0",
    "cpu": "vllm/vllm-openai-cpu:v0.23.0",
}


def _detect_device() -> str:
    try:
        from cogniverse_cli.images import detect_torch_backend

        backend = detect_torch_backend()
    except Exception:
        backend = "cpu"
    return "rocm" if backend == "rocm" else "cpu"


def _healthy(base_url: str, model: str, timeout: float = 3.0) -> bool:
    return serves_exact_model(base_url, model, timeout)


def _container_state(container: str) -> str | None:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def _container_logs(container: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "200", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unable to read container logs: {exc}"
    return "\n".join(part for part in (out.stdout, out.stderr) if part).strip()


def _wait_for_container_absence(container: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            result = subprocess.run(
                ["docker", "inspect", container],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"docker could not verify removal of {container!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        detail = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        ).strip()
        if result.returncode != 0 and "No such" in detail:
            return True
        if result.returncode != 0:
            raise RuntimeError(
                f"docker could not inspect exact-model container {container!r}: "
                f"{detail or f'exit {result.returncode}'}"
            )
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _remove_container(container: str) -> None:
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if _wait_for_container_absence(container):
            return
        raise RuntimeError(
            f"docker removal timed out and exact-model container "
            f"{container!r} still exists"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"docker could not remove exact-model container {container!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0 or "No such container" in detail:
        return
    if "removal" in detail and "in progress" in detail:
        if _wait_for_container_absence(container):
            return
        raise RuntimeError(
            f"docker removal remained in progress and exact-model container "
            f"{container!r} still exists"
        )
    raise RuntimeError(
        f"docker could not remove exact-model container {container!r}: "
        f"{detail or f'exit {result.returncode}'}"
    )


def _cleanup_container(container: str) -> str:
    try:
        _remove_container(container)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return f"cleanup failed: {type(exc).__name__}: {exc}"
    return "cleanup completed"


def _exception_detail(exc: Exception) -> str:
    details = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, subprocess.CalledProcessError):
        details.extend(
            str(part).strip()
            for part in (exc.stdout, exc.stderr)
            if part and str(part).strip()
        )
    return "\n".join(details)


@contextmanager
def _ensure_lock():
    """Serialize exact-model provisioning across threads and pytest processes."""
    with _ENSURE_LOCK:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK_PATH.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _spawn(
    model: str,
    container: str,
    host_port: int,
    device: str,
    gpu_utilization: float = 0.25,
) -> None:
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "-p",
        f"{host_port}:8000",
        "-v",
        f"{_HF_CACHE}:/root/.cache/huggingface",
        # The host resolv.conf can point at a dead resolver (k3d node DNS
        # breakage) — pin public resolvers so a fresh model download works.
        "--dns",
        "1.1.1.1",
        "--dns",
        "8.8.8.8",
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
        # 16384: the RLM long-doc tests carry a ~7400-token REPL prompt
        # plus an 800-token output budget — smaller windows overflow by
        # design of the tier's own consumers.
        engine_args = [
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            str(gpu_utilization),
        ]
    else:
        cmd += ["-e", "VLLM_CPU_KVCACHE_SPACE=4"]
        engine_args = ["--max-model-len", "16384"]
    cmd += [_IMAGES[device], "--model", model, *engine_args]
    subprocess.run(cmd, check=True, timeout=120)


def _write_session_config(
    primary_api_base: str,
    teacher_api_base: str | None,
    *,
    source_config: Path = SOURCE_CONFIG,
) -> Path:
    config = json.loads(source_config.read_text())
    llm = config.setdefault("llm_config", {})
    primary = llm.setdefault("primary", {})
    primary["model"] = f"openai/{MODEL}"
    primary["api_base"] = primary_api_base
    if teacher_api_base is None:
        llm.pop("teacher", None)
    else:
        teacher = llm.setdefault("teacher", {})
        teacher["model"] = f"openai/{TEACHER_MODEL}"
        teacher["api_base"] = teacher_api_base
    HERMETIC_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = HERMETIC_CONFIG_DIR / f"config-{os.getpid()}.json"
    pending = config_path.with_name(f".{config_path.name}.{threading.get_ident()}.tmp")
    try:
        pending.write_text(json.dumps(config, indent=2))
        os.replace(pending, config_path)
    finally:
        pending.unlink(missing_ok=True)
    return config_path


def activate_llms(
    primary_api_base: str,
    teacher_api_base: str | None = None,
    *,
    source_config: Path = SOURCE_CONFIG,
) -> Path:
    """Publish the verified exact LM roles selected for this process."""
    primary_api_base = f"{_server_base(primary_api_base)}/v1"
    if teacher_api_base is not None:
        teacher_api_base = f"{_server_base(teacher_api_base)}/v1"
    config_path = _write_session_config(
        primary_api_base,
        teacher_api_base,
        source_config=source_config,
    )
    os.environ["COGNIVERSE_CONFIG"] = str(config_path)
    os.environ["TEST_LLM_API_BASE"] = primary_api_base
    os.environ["TEST_LLM_MODEL"] = MODEL
    os.environ.setdefault("OPENAI_API_KEY", "not-required")
    return config_path


def ensure_llm(model: str = MODEL, deadline_s: float = 900.0) -> str:
    """Resolve or provision ``model`` exactly and return its OpenAI base URL."""
    try:
        container, host_port = _SIDECARS[model]
    except KeyError as exc:
        raise ValueError(f"No exact local sidecar is configured for {model!r}") from exc

    with _ensure_lock():
        configured = find_exact_model_endpoint(model, _configured_model_urls(model))
        if configured is not None:
            return f"{configured}/v1"

        local_base = f"http://127.0.0.1:{host_port}"

        def _await_ready(budget_s: float) -> bool:
            deadline = time.monotonic() + budget_s
            while time.monotonic() < deadline:
                model_ids = listed_model_ids(local_base)
                if model_ids is not None:
                    return model in model_ids
                if _container_state(container) != "running":
                    return False
                time.sleep(5)
            return False

        try:
            state = _container_state(container)
            if state == "running":
                model_ids = listed_model_ids(local_base)
                if model_ids is not None and model not in model_ids:
                    _remove_container(container)
                    state = None
                elif _healthy(local_base, model):
                    return f"{local_base}/v1"
            if state == "running":
                if _await_ready(deadline_s):
                    return f"{local_base}/v1"
                _remove_container(container)
                state = None
            if state is not None:
                subprocess.run(
                    ["docker", "start", container],
                    check=True,
                    timeout=60,
                    capture_output=True,
                    text=True,
                )
                if _await_ready(deadline_s):
                    return f"{local_base}/v1"
                _remove_container(container)

            device = _detect_device()
            attempts = (
                [("rocm", 0.25), ("rocm", 0.12), ("cpu", 0.0)]
                if device == "rocm"
                else [("cpu", 0.0)]
            )
            errors: list[str] = []
            for dev, util in attempts:
                _remove_container(container)
                try:
                    _spawn(
                        model,
                        container,
                        host_port,
                        dev,
                        gpu_utilization=util,
                    )
                    if _await_ready(deadline_s):
                        return f"{local_base}/v1"
                    errors.append(
                        f"{dev} sidecar did not serve {model!r}; "
                        f"container logs:\n{_container_logs(container)}"
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    errors.append(
                        f"{dev} sidecar failed: {exc}; "
                        f"container logs:\n{_container_logs(container)}"
                    )

            detail = "; ".join(errors) or "no local launch attempt completed"
            raise RuntimeError(
                f"No configured endpoint or local vLLM sidecar served exact model "
                f"{model!r}: {detail}"
            )
        except Exception as exc:
            logs = _container_logs(container)
            cleanup = _cleanup_container(container)
            raise RuntimeError(
                f"Failed to provision exact model {model!r} in container "
                f"{container!r}: {_exception_detail(exc)}\n"
                f"container logs:\n{logs}\n{cleanup}"
            ) from exc
