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

import fcntl
import json
import os
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

DEFAULT_IMAGE = "vllm/vllm-openai-cpu:v0.23.0"
DEFAULT_HEALTH_DEADLINE_SECONDS = 600
# Test-owned Hugging Face cache, deliberately separate from the user's
# ~/.cache/huggingface: containers previously ran as root and wrote
# root-owned entries into the personal cache, which breaks host-side
# (in-process) model loads with permission errors. Test containers mount
# this directory at /hf-cache and run as the invoking user, so every
# entry stays user-owned; host-side oracles share it via ``hub/``.
TEST_HF_CACHE = os.path.expanduser("~/.cache/cogniverse-tests/huggingface")
CONTAINER_HF_CACHE = "/hf-cache"


def writable_test_hf_cache() -> str:
    """Create the test-owned HF cache and prove it is writable.

    Raises with context instead of letting a model load fail later with an
    opaque permission error mid-download.
    """
    hub = Path(TEST_HF_CACHE) / "hub"
    try:
        hub.mkdir(parents=True, exist_ok=True)
        probe = hub / f".writable-probe-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"test HF cache {TEST_HF_CACHE} is not writable "
            f"({type(exc).__name__}: {exc}); remove foreign-owned entries or "
            "free the path"
        ) from exc
    return TEST_HF_CACHE


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


# Exact-model sidecars (see tests/utils/hermetic_llm.py) are reused across
# pytest sessions on purpose, because re-provisioning one reloads multi-GB
# weights. They therefore carry a marker of their own instead of OWNER_LABEL:
# the owner-pid reaper removes every container whose spawning session is gone,
# which is exactly the state a reusable sidecar sits in between sessions.
# Their lifetime is bounded by age instead.
EXACT_MODEL_LABEL = "cogniverse-test-exact-model"

# A resident sidecar keeps its model weights in host memory — on a ROCm host
# the GPU's GTT aperture backs them with system RAM (~26GB for gemma-4-e4b) —
# and the shared Vespa refuses to boot with less than 4GB available. Six hours
# covers one working stretch of repeated pytest runs against a warm container,
# so reuse still pays for itself, while a forgotten sidecar costs part of a day
# of RAM rather than holding it until the host reboots. Re-provisioning after
# the window is one load from the local Hugging Face cache.
EXACT_MODEL_MAX_AGE_SECONDS = 6 * 3600

EXACT_MODEL_LOCK_PATH = (
    Path(tempfile.gettempdir()) / f"cogniverse-exact-llm-sidecars-{os.getuid()}.lock"
)
_EXACT_MODEL_LEASE_DIR = (
    Path(tempfile.gettempdir()) / f"cogniverse-exact-llm-leases-{os.getuid()}"
)


@contextmanager
def exact_model_provisioning_lock(*, blocking: bool = True):
    """Serialize exact-model selection and age reclaim across pytest processes.

    Yields whether the lock was taken. With ``blocking=False`` a lock already
    held by another session yields ``False`` instead of waiting: that session is
    provisioning right now, so it is actively using its sidecars and a reclaim
    has nothing to do while it runs.
    """
    EXACT_MODEL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXACT_MODEL_LOCK_PATH.open("a+") as lock_file:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(lock_file.fileno(), flags)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def lease_exact_model_container(container: str) -> None:
    """Record that this pytest process serves tests from ``container``.

    A lease held by a live process keeps the age reclaim away from a sidecar a
    session has already resolved into its config; a lease whose process is gone
    is dropped by the reclaim itself.
    """
    _EXACT_MODEL_LEASE_DIR.mkdir(parents=True, exist_ok=True)
    (_EXACT_MODEL_LEASE_DIR / f"{container}.{os.getpid()}").touch()


def _live_leased_containers() -> set[str]:
    """Containers leased by a live process, dropping the leases of dead ones."""
    try:
        entries = sorted(_EXACT_MODEL_LEASE_DIR.iterdir())
    except FileNotFoundError:
        return set()
    leased: set[str] = set()
    for entry in entries:
        container, _, holder_pid = entry.name.rpartition(".")
        if not container or not holder_pid.isdigit():
            continue
        if os.path.exists(f"/proc/{holder_pid}"):
            leased.add(container)
        else:
            entry.unlink(missing_ok=True)
    return leased


def _container_age_seconds(container_id: str) -> float | None:
    """Age from docker's own creation time; ``None`` when it is already gone."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Created}}", container_id],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        if "No such" in detail:
            return None
        raise RuntimeError(
            f"docker could not read the creation time of exact-model container "
            f"{container_id}: {detail or f'exit {result.returncode}'}"
        )
    created = result.stdout.strip()
    try:
        created_at = datetime.fromisoformat(created)
    except ValueError as exc:
        raise RuntimeError(
            f"docker reported an unreadable creation time for exact-model "
            f"container {container_id}: {created!r}"
        ) from exc
    if created_at.tzinfo is None:
        raise RuntimeError(
            f"docker reported a creation time without a timezone for exact-model "
            f"container {container_id}: {created!r}"
        )
    return (datetime.now(timezone.utc) - created_at).total_seconds()


def reclaim_stale_exact_model_containers(
    max_age_seconds: float = EXACT_MODEL_MAX_AGE_SECONDS,
) -> None:
    """Remove exact-model sidecars older than ``max_age_seconds``.

    ``hermetic_llm`` reuses these sidecars across sessions, so they carry no
    owner pid and ``reap_dead_owner_containers`` never sees them: without an age
    bound nothing removes one, and a forgotten sidecar holds its weights in host
    RAM until the host reboots. Age comes from docker's creation timestamp, so
    neither a restart nor an in-process bookkeeping gap can make a container
    look fresher than it is.

    A sidecar leased by a live pytest process is kept whatever its age, and the
    whole pass is skipped while another session holds the provisioning lock, so
    a reclaim can never take a container a session has selected or is in the
    middle of selecting. ``ensure_llm`` re-provisions on demand afterwards.
    """
    with exact_model_provisioning_lock(blocking=False) as acquired:
        if not acquired:
            return
        listing = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label={EXACT_MODEL_LABEL}",
                "--format",
                "{{.ID}}\t{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if listing.returncode != 0:
            detail = "\n".join(
                part for part in (listing.stdout, listing.stderr) if part
            ).strip()
            raise RuntimeError(
                f"docker could not list exact-model containers: "
                f"{detail or f'exit {listing.returncode}'}"
            )
        candidates: list[tuple[str, str]] = []
        for line in listing.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            candidates.append((parts[0], parts[1]))
        if not candidates:
            return

        leased = _live_leased_containers()
        removal_errors: list[str] = []
        for container_id, name in candidates:
            if name in leased:
                continue
            age_seconds = _container_age_seconds(container_id)
            if age_seconds is None or age_seconds <= max_age_seconds:
                continue
            removed = subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            detail = "\n".join(
                part for part in (removed.stdout, removed.stderr) if part
            ).strip()
            if removed.returncode != 0 and "No such container" not in detail:
                removal_errors.append(
                    f"{name} ({container_id}): {detail or f'exit {removed.returncode}'}"
                )
        if removal_errors:
            raise RuntimeError(
                "docker could not remove stale exact-model containers: "
                + "; ".join(removal_errors)
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
    # The container runs as the invoking (non-root) user, so pip cannot
    # write into the image's root-owned venv — install the audio extras
    # into the writable cache mount and expose them via PYTHONPATH.
    extras_dir = f"{CONTAINER_HF_CACHE}/.pip-audio-extras"
    command.extend(
        [
            "--entrypoint",
            "sh",
            image,
            "-c",
            (
                f"pip install --no-cache-dir --quiet --target {extras_dir} "
                "soundfile librosa || exit 1; "
                f'export PYTHONPATH="{extras_dir}${{PYTHONPATH:+:$PYTHONPATH}}"; '
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


def _prepare_pinned_snapshot(
    model: str,
    revision: str,
    required_files: tuple[str, ...],
) -> None:
    if not required_files:
        raise ValueError("Pinned model snapshots require an explicit file contract")

    cache_dir = os.path.join(writable_test_hf_cache(), "hub")
    snapshot_path: str | None = None
    try:
        snapshot_path = snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    missing = (
        list(required_files)
        if snapshot_path is None
        else [
            relative_path
            for relative_path in required_files
            if not os.path.isfile(os.path.join(snapshot_path, relative_path))
        ]
    )
    if missing:
        try:
            snapshot_path = snapshot_download(
                repo_id=model,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Failed to provision pinned model {model!r} at {revision}: {exc}"
            ) from exc
        missing = [
            relative_path
            for relative_path in required_files
            if not os.path.isfile(os.path.join(snapshot_path, relative_path))
        ]
    if missing:
        raise RuntimeError(
            f"Pinned model {model!r} at {revision} is missing required files: "
            + ", ".join(missing)
        )


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
        model_revision: str | None = None,
        required_snapshot_files: tuple[str, ...] = (),
        extra_args: Optional[list[str]] = None,
        image: Optional[str] = None,
        device: str = "cpu",
        env: Optional[dict[str, str]] = None,
    ) -> str:
        """Reuse an exact configured model or spawn its identical sidecar."""
        image = image or self.image
        key = (
            model,
            model_revision,
            required_snapshot_files,
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

            resolved_env = dict(env or {})
            resolved_args = list(extra_args or ())
            if model_revision is not None:
                if "--revision" in resolved_args:
                    raise ValueError(
                        "Pass the pinned revision through model_revision only"
                    )
                _prepare_pinned_snapshot(
                    model,
                    model_revision,
                    required_snapshot_files,
                )
                resolved_args.extend(["--revision", model_revision])
                resolved_env["HF_HUB_OFFLINE"] = "1"

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
            for env_key, env_value in resolved_env.items():
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
            # Run as the invoking user against the test-owned cache so model
            # downloads and engine caches stay user-owned on the host. HOME
            # inside the mount keeps every ~-derived cache path writable;
            # LOGNAME/USER keep getpass.getuser() working for a uid with no
            # container passwd entry (torch inductor derives its cache dir
            # from it and crashes on KeyError otherwise).
            cmd.extend(
                [
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    f"HOME={CONTAINER_HF_CACHE}",
                    "-e",
                    f"HF_HOME={CONTAINER_HF_CACHE}",
                    "-e",
                    "LOGNAME=cogniverse",
                    "-e",
                    "USER=cogniverse",
                    "-v",
                    f"{writable_test_hf_cache()}:{CONTAINER_HF_CACHE}",
                ]
            )
            _append_image_command(
                cmd,
                image,
                model,
                _merge_serve_args(model, resolved_args),
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
