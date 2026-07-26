"""Cluster-first exact-model vLLM fixtures for integration tests.

The factory checks explicit test overrides, then dynamically discovers
host-published inference services in the isolated ``cogniverse-e2e``
k3d cluster, then the development ``cogniverse`` cluster. It reuses an
endpoint only when its OpenAI model-list response names the requested
model exactly. Otherwise it launches an identical local vLLM container.

Usage::

    def test_my_remote_path(vllm_sidecar):
        url = vllm_sidecar.spawn(
            model="openai/whisper-tiny",
            extra_args=["--max-model-len", "448"],
        )
        # url is the verified cluster service, or a local fallback
        # that is cleaned up when the session ends.

The factory caches resolution by model and serving arguments, and
serializes first use so concurrent consumers cannot launch duplicate
fallbacks.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests

DEFAULT_IMAGE = "vllm/vllm-openai-cpu:v0.23.0"
DEFAULT_HEALTH_DEADLINE_SECONDS = 600
HOST_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
E2E_CONTEXT = "k3d-cogniverse-e2e"
E2E_CLUSTER = "cogniverse-e2e"
DEV_CONTEXT = "k3d-cogniverse"
DEV_CLUSTER = "cogniverse"

# Containers are labelled with the spawning pytest pid so the next session
# can reap leftovers whose owner died without running fixture teardown
# (SIGKILL skips the finally). A dead sidecar holds model weights in host
# RAM — several of these plus a Vespa JVM starved the whole host once.
OWNER_LABEL = "cogniverse-test-owner-pid"


def reap_dead_owner_containers(label: str = OWNER_LABEL) -> None:
    """Remove containers labelled with an owner pid that no longer exists.

    Also removes already-Exited labelled containers (they only hold disk,
    but they accumulate forever otherwise). Containers belonging to LIVE
    pids — concurrent pytest sessions — are never touched.
    """
    listing = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={label}",
            "--format",
            '{{.ID}}\t{{.State}}\t{{.Label "' + label + '"}}',
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        container_id, state, owner_pid = parts
        owner_alive = owner_pid.isdigit() and os.path.exists(f"/proc/{owner_pid}")
        if state == "running" and owner_alive:
            continue
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            timeout=30,
            check=False,
        )


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


def _server_base(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def listed_model_ids(base_url: str, timeout: float = 2.0) -> set[str] | None:
    """Return exact model IDs from a valid OpenAI model-list response."""
    try:
        response = requests.get(f"{_server_base(base_url)}/v1/models", timeout=timeout)
        if response.status_code != 200:
            return None
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return None
    rows = payload.get("data")
    if not isinstance(rows, list) or not all(
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and row.get("object") == "model"
        for row in rows
    ):
        return None
    return {row["id"] for row in rows}


def serves_exact_model(base_url: str, model: str, timeout: float = 2.0) -> bool:
    """Return whether an OpenAI-compatible endpoint lists ``model`` exactly."""
    model_ids = listed_model_ids(base_url, timeout)
    return model_ids is not None and model in model_ids


def _bare_model(model: object) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    return model[len("openai/") :] if model.startswith("openai/") else model


def _command_json(command: list[str]) -> object | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError):
        return None


def _container_declares_model(container: object, model: str) -> bool:
    if not isinstance(container, dict):
        return False
    env = container.get("env")
    if isinstance(env, list):
        for entry in env:
            if (
                isinstance(entry, dict)
                and entry.get("name") == "MODEL_NAME"
                and entry.get("value") == model
            ):
                return True
    tokens: list[str] = []
    for command_field in ("command", "args"):
        values = container.get(command_field)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            try:
                tokens.extend(shlex.split(value))
            except ValueError:
                tokens.append(value)
    return any(
        token == model and index > 0 and tokens[index - 1] in {"serve", "--model"}
        for index, token in enumerate(tokens)
    )


def _discover_cluster_model_urls(
    model: str,
    *,
    context: str,
    cluster: str,
) -> tuple[str, ...]:
    """Map an exact cluster workload to its dynamically published host port."""
    resources = _command_json(
        [
            "kubectl",
            "--context",
            context,
            "get",
            "deployments,statefulsets,services",
            "--all-namespaces",
            "-o",
            "json",
        ]
    )
    if not isinstance(resources, dict) or not isinstance(resources.get("items"), list):
        return ()

    workload_labels: list[tuple[str, dict[str, str]]] = []
    for item in resources["items"]:
        if not isinstance(item, dict) or item.get("kind") not in {
            "Deployment",
            "StatefulSet",
        }:
            continue
        metadata = item.get("metadata")
        spec = item.get("spec")
        template = spec.get("template") if isinstance(spec, dict) else None
        template_metadata = (
            template.get("metadata") if isinstance(template, dict) else None
        )
        pod_spec = template.get("spec") if isinstance(template, dict) else None
        containers = pod_spec.get("containers") if isinstance(pod_spec, dict) else None
        labels = (
            template_metadata.get("labels")
            if isinstance(template_metadata, dict)
            else None
        )
        namespace = metadata.get("namespace") if isinstance(metadata, dict) else None
        if (
            isinstance(namespace, str)
            and isinstance(labels, dict)
            and labels
            and isinstance(containers, list)
            and any(
                _container_declares_model(container, model) for container in containers
            )
        ):
            workload_labels.append(
                (
                    namespace,
                    {
                        key: value
                        for key, value in labels.items()
                        if isinstance(key, str) and isinstance(value, str)
                    },
                )
            )

    node_ports: list[int] = []
    for item in resources["items"]:
        if not isinstance(item, dict) or item.get("kind") != "Service":
            continue
        metadata = item.get("metadata")
        spec = item.get("spec")
        namespace = metadata.get("namespace") if isinstance(metadata, dict) else None
        selector = spec.get("selector") if isinstance(spec, dict) else None
        ports = spec.get("ports") if isinstance(spec, dict) else None
        if (
            not isinstance(namespace, str)
            or not isinstance(selector, dict)
            or not selector
            or not isinstance(ports, list)
        ):
            continue
        if not any(
            workload_namespace == namespace
            and all(labels.get(key) == value for key, value in selector.items())
            for workload_namespace, labels in workload_labels
        ):
            continue
        node_ports.extend(
            port["nodePort"]
            for port in ports
            if isinstance(port, dict) and isinstance(port.get("nodePort"), int)
        )
    if not node_ports:
        return ()

    try:
        load_balancers = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"label=k3d.cluster={cluster}",
                "--filter",
                "label=k3d.role=loadbalancer",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if load_balancers.returncode != 0:
        return ()

    candidates: list[str] = []
    for container in load_balancers.stdout.splitlines():
        published = _command_json(
            [
                "docker",
                "inspect",
                container,
                "--format",
                "{{json .NetworkSettings.Ports}}",
            ]
        )
        if not isinstance(published, dict):
            continue
        for node_port in node_ports:
            bindings = published.get(f"{node_port}/tcp")
            if not isinstance(bindings, list):
                continue
            for binding in bindings:
                host_port = (
                    binding.get("HostPort") if isinstance(binding, dict) else None
                )
                if isinstance(host_port, str) and host_port.isdigit():
                    candidates.append(f"http://127.0.0.1:{host_port}")
    return tuple(dict.fromkeys(candidates))


def _discover_e2e_model_urls(model: str) -> tuple[str, ...]:
    return _discover_cluster_model_urls(
        model,
        context=E2E_CONTEXT,
        cluster=E2E_CLUSTER,
    )


def _discover_dev_model_urls(model: str) -> tuple[str, ...]:
    return _discover_cluster_model_urls(
        model,
        context=DEV_CONTEXT,
        cluster=DEV_CLUSTER,
    )


def _configured_model_urls(model: str) -> tuple[str, ...]:
    """Collect explicit, e2e-cluster, then dev-cluster candidates."""
    candidates: list[str] = []

    env_model = _bare_model(os.environ.get("TEST_LLM_MODEL"))
    env_api_base = os.environ.get("TEST_LLM_API_BASE")
    if env_model == model and env_api_base:
        candidates.append(env_api_base)

    raw_urls = os.environ.get("INFERENCE_SERVICE_URLS")
    if raw_urls:
        try:
            env_urls = json.loads(raw_urls)
        except ValueError:
            env_urls = {}
        if isinstance(env_urls, dict):
            candidates.extend(url for url in env_urls.values() if isinstance(url, str))

    candidates.extend(_discover_e2e_model_urls(model))
    candidates.extend(_discover_dev_model_urls(model))

    return tuple(dict.fromkeys(_server_base(url) for url in candidates if url))


def find_exact_model_endpoint(model: str, urls: tuple[str, ...]) -> str | None:
    """Return the first reachable URL with a valid exact-model API response."""
    for url in urls:
        base_url = _server_base(url)
        if serves_exact_model(base_url, model):
            return base_url
    return None


def _container_logs(container: str) -> str:
    try:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "200", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unable to read container logs: {exc}"
    return "\n".join(part for part in (logs.stdout, logs.stderr) if part).strip()


def _remove_sidecar_container(container: str) -> str | None:
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container],
            check=False,
            timeout=30,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"cleanup failed: {type(exc).__name__}: {exc}"
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        return f"cleanup failed with exit {result.returncode}: {detail}"
    return None


def _append_image_command(
    command: list[str],
    image: str,
    model: str,
    serve_args: list[str],
) -> None:
    if "whisper" not in model.lower():
        command.extend([image, "--model", model, *serve_args])
        return
    serve_command = shlex.join(["vllm", "serve", model, *serve_args])
    command.extend(
        [
            "--entrypoint",
            "sh",
            image,
            "-c",
            (
                "pip install --no-cache-dir --quiet soundfile librosa || exit 1; "
                f"exec {serve_command}"
            ),
        ]
    )


def _wait_for_models(
    base_url: str, model: str, deadline_seconds: int, container: str
) -> None:
    """Poll ``/v1/models`` until vLLM finishes loading the served model."""
    end = time.monotonic() + deadline_seconds
    last_err: Optional[str] = None
    while time.monotonic() < end:
        if serves_exact_model(base_url, model):
            return
        last_err = f"{model!r} absent from a valid /v1/models response"
        time.sleep(2)
    raise AssertionError(
        f"vllm sidecar at {base_url} did not become healthy within "
        f"{deadline_seconds}s (last error: {last_err})\n"
        f"--- container logs ---\n{_container_logs(container)}"
    )


@dataclass
class _SpawnedSidecar:
    container: str | None
    base_url: str


@dataclass
class VllmSidecarFactory:
    """Per-session manager for exact remote services and local sidecars."""

    image: str = DEFAULT_IMAGE
    health_deadline_seconds: int = DEFAULT_HEALTH_DEADLINE_SECONDS
    configured_urls: tuple[str, ...] | None = None
    _spawned: dict[tuple, _SpawnedSidecar] = field(default_factory=dict)
    _spawn_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def spawn(
        self,
        model: str,
        *,
        extra_args: Optional[list[str]] = None,
        image: Optional[str] = None,
        device: str = "cpu",
        env: Optional[dict[str, str]] = None,
    ) -> str:
        """Reuse an exact configured model or spawn its identical sidecar."""
        image = image or self.image
        key = (
            model,
            image,
            tuple(extra_args or ()),
            device,
            tuple(sorted((env or {}).items())),
        )
        with self._spawn_lock:
            if key in self._spawned:
                return self._spawned[key].base_url

            configured_urls = (
                self.configured_urls
                if self.configured_urls is not None
                else _configured_model_urls(model)
            )
            configured_url = find_exact_model_endpoint(model, configured_urls)
            if configured_url is not None:
                self._spawned[key] = _SpawnedSidecar(
                    container=None, base_url=configured_url
                )
                return configured_url

            # Reclaim RAM from sidecars whose owning session was SIGKILLed
            # before its teardown could run.
            reap_dead_owner_containers()

            container = f"cogniverse-vllm-test-{uuid.uuid4().hex[:8]}"
            port = _free_port()
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--label",
                f"{OWNER_LABEL}={os.getpid()}",
                "-p",
                f"{port}:8000",
                "-e",
                "VLLM_CPU_MEMORY_UTILIZATION=0.05",
                "-e",
                "VLLM_CPU_KVCACHE_SPACE=2",
                "--oom-score-adj=500",
            ]
            for env_key, env_value in (env or {}).items():
                cmd.extend(["-e", f"{env_key}={env_value}"])
            if device == "rocm":
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
            _append_image_command(
                cmd,
                image,
                model,
                _merge_serve_args(model, extra_args),
            )

            base_url = f"http://127.0.0.1:{port}"
            try:
                subprocess.run(cmd, check=True, timeout=60)
                _wait_for_models(
                    base_url, model, self.health_deadline_seconds, container
                )
            except Exception as exc:
                stderr = getattr(exc, "stderr", None)
                details = (
                    f"{type(exc).__name__}: {exc}"
                    + (f"\nstderr:\n{stderr}" if stderr else "")
                    + f"\ncontainer logs:\n{_container_logs(container)}"
                )
                cleanup_error = _remove_sidecar_container(container)
                if cleanup_error is not None:
                    details += f"\n{cleanup_error}"
                raise RuntimeError(
                    f"Failed to launch exact vLLM model {model!r} at "
                    f"{base_url}:\n{details}"
                ) from exc

            self._spawned[key] = _SpawnedSidecar(container=container, base_url=base_url)
            return base_url

    def teardown(self) -> None:
        with self._spawn_lock:
            cleanup_errors: list[str] = []
            try:
                for sidecar in self._spawned.values():
                    if sidecar.container is None:
                        continue
                    cleanup_error = _remove_sidecar_container(sidecar.container)
                    if cleanup_error is not None:
                        cleanup_errors.append(f"{sidecar.container}: {cleanup_error}")
            finally:
                self._spawned.clear()
            if cleanup_errors:
                raise RuntimeError(
                    "Failed to remove exact vLLM sidecars: " + "; ".join(cleanup_errors)
                )
