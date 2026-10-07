"""Remote model endpoints for tests: cluster discovery and model-list probes.

Discovery maps a workload in the isolated ``cogniverse-e2e`` k3d cluster, or
the development ``cogniverse`` cluster, that serves an exact model to the host
port its load balancer publishes. ``tests/fixtures/inference.py`` validates
what it finds; nothing here starts a model. The module also reaps containers
whose owning pytest process died before its teardown ran.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_LOOPBACK_PROBE_TIMEOUT_S = 2.0
_MEASURED_REMOTE_SCALE_UP_S = 48.87
_REMOTE_PROBE_TIMEOUT_S = 90.0
_PROBE_ATTEMPTS = 3
_PROBE_RETRY_PAUSE_S = 1.0


def _probe_timeout(base_url: str) -> float:
    """Return the model-list budget for an endpoint's locality."""
    host = urlparse(base_url).hostname or ""
    if host in _LOOPBACK_HOSTS:
        return _LOOPBACK_PROBE_TIMEOUT_S
    return _REMOTE_PROBE_TIMEOUT_S


# Test-owned Hugging Face cache for in-process reference models, kept apart
# from the user's ~/.cache/huggingface.
TEST_HF_CACHE = os.path.expanduser("~/.cache/cogniverse-tests/huggingface")


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


def _wait_until_container_gone(container_id: str, timeout: float = 30.0) -> bool:
    """Whether ``container_id`` stops existing within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"id={container_id}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if listed.returncode == 0 and not listed.stdout.strip():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def reap_dead_owner_containers(label: str = OWNER_LABEL) -> list[str]:
    """Remove containers labelled with an owner pid that no longer exists.

    Also removes already-Exited labelled containers (they only hold disk,
    but they accumulate forever otherwise). Containers belonging to LIVE
    pids — concurrent pytest sessions — are never touched. Returns the ids
    this call removed; a container another reaper removed first is not one of
    them. Raises when docker cannot list the containers or remove one.
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
    if listing.returncode != 0:
        detail = "\n".join(
            part for part in (listing.stdout, listing.stderr) if part
        ).strip()
        raise RuntimeError(
            f"docker could not list containers labelled {label}: "
            f"{detail or f'exit {listing.returncode}'}"
        )
    removed: list[str] = []
    failures: list[str] = []
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        container_id, state, owner_pid = parts
        owner_alive = owner_pid.isdigit() and os.path.exists(f"/proc/{owner_pid}")
        if owner_alive and state not in {"exited", "dead"}:
            continue
        result = subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            removed.append(container_id)
        elif not _wait_until_container_gone(container_id):
            detail = "\n".join(
                part for part in (result.stdout, result.stderr) if part
            ).strip()
            failures.append(f"{container_id}: {detail or f'exit {result.returncode}'}")
    if failures:
        raise RuntimeError(
            "docker could not remove dead-owner containers: " + "; ".join(failures)
        )
    return removed


def reap_dead_owner_networks(label: str = OWNER_LABEL) -> list[str]:
    """Remove networks labelled with an owner pid that no longer exists.

    A stack fixture creates a network per run and removes it in a ``finally``,
    where the removal races the endpoint release of the containers just torn
    down. The container reaper cannot see what is left: networks are a
    separate namespace. Networks belonging to LIVE pids are never touched.
    Returns the names this call removed. Raises when docker cannot list the
    networks or remove one that is still present afterwards.
    """
    listing = subprocess.run(
        [
            "docker",
            "network",
            "ls",
            "--filter",
            f"label={label}",
            "--format",
            '{{.Name}}\t{{.Label "' + label + '"}}',
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
            f"docker could not list networks labelled {label}: "
            f"{detail or f'exit {listing.returncode}'}"
        )
    removed: list[str] = []
    failures: list[str] = []
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        name, owner_pid = parts
        if owner_pid.isdigit() and os.path.exists(f"/proc/{owner_pid}"):
            continue
        result = subprocess.run(
            ["docker", "network", "rm", name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            removed.append(name)
            continue
        present = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if name in present.stdout.split():
            detail = "\n".join(
                part for part in (result.stdout, result.stderr) if part
            ).strip()
            failures.append(f"{name}: {detail or f'exit {result.returncode}'}")
    if failures:
        raise RuntimeError(
            "docker could not remove dead-owner networks: " + "; ".join(failures)
        )
    return removed


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_base(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


@dataclass(frozen=True, slots=True)
class ModelListProbe:
    """What one model-list probe of an endpoint established."""

    base_url: str
    model_ids: frozenset[str] | None
    failure: str | None = None

    def serves(self, model: str) -> bool:
        return self.model_ids is not None and model in self.model_ids

    def outcome(self) -> str:
        if self.model_ids is not None:
            return f"lists {sorted(self.model_ids)}"
        return self.failure or "no model list"


def probe_model_list(base_url: str, timeout: float | None = None) -> ModelListProbe:
    """Probe an OpenAI model-list endpoint and record why it failed, if it did."""
    if timeout is None:
        timeout = _probe_timeout(base_url)
    api_key = os.environ.get("COGNIVERSE_INFERENCE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    server = _server_base(base_url)
    url = f"{server}/v1/models"
    payload = None
    failure = "no probe attempted"
    for attempt in range(1, _PROBE_ATTEMPTS + 1):
        transient = False
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                payload = response.json()
                break
            transient = response.status_code >= 500
            failure = f"HTTP {response.status_code}"
            logger.warning(
                "Model listing at %s refused the probe: HTTP %s (attempt %s/%s, "
                "timeout=%ss, api key %s)",
                base_url,
                response.status_code,
                attempt,
                _PROBE_ATTEMPTS,
                timeout,
                "present" if api_key else "absent",
            )
        except (requests.RequestException, ValueError) as exc:
            transient = True
            failure = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Model listing at %s failed after %ss (attempt %s/%s): %s: %s",
                base_url,
                timeout,
                attempt,
                _PROBE_ATTEMPTS,
                type(exc).__name__,
                exc,
            )
        if not transient:
            return ModelListProbe(server, None, f"{failure} (not retried)")
        if attempt < _PROBE_ATTEMPTS:
            time.sleep(_PROBE_RETRY_PAUSE_S)
    if payload is None:
        return ModelListProbe(
            server,
            None,
            f"{failure} on {_PROBE_ATTEMPTS} of {_PROBE_ATTEMPTS} attempts",
        )
    invalid = ModelListProbe(
        server, None, "answered HTTP 200 without a valid OpenAI model list"
    )
    if not isinstance(payload, dict) or payload.get("object") != "list":
        return invalid
    rows = payload.get("data")
    if not isinstance(rows, list) or not all(
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and row.get("object") == "model"
        for row in rows
    ):
        return invalid
    return ModelListProbe(server, frozenset(row["id"] for row in rows))


def listed_model_ids(base_url: str, timeout: float | None = None) -> set[str] | None:
    """Return exact model IDs from a valid OpenAI model-list response."""
    probe = probe_model_list(base_url, timeout)
    return None if probe.model_ids is None else set(probe.model_ids)


def serves_exact_model(base_url: str, model: str, timeout: float | None = None) -> bool:
    """Return whether an OpenAI-compatible endpoint lists ``model`` exactly."""
    model_ids = listed_model_ids(base_url, timeout)
    return model_ids is not None and model in model_ids


class ModelEndpointDiscoveryError(RuntimeError):
    """A kube context that exists could not say which endpoints it publishes."""

    def __init__(self, context: str, detail: str) -> None:
        self.context = context
        self.detail = detail
        super().__init__(
            f"Could not discover the endpoints kube context {context!r} publishes, "
            f"so whether it serves the model remotely is unknown: {detail}"
        )


def _run_json(command: list[str]) -> tuple[object | None, str]:
    """Run ``command``; return its parsed JSON output or why there is none."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return None, detail.strip() or f"exit {result.returncode}"
    try:
        return json.loads(result.stdout), ""
    except (TypeError, ValueError) as exc:
        return None, f"unparseable output: {exc}"


def _command_json(command: list[str]) -> object | None:
    return _run_json(command)[0]


def _kube_context_exists(context: str) -> bool:
    """Whether the active kubeconfig defines ``context``.

    Without kubectl no context exists. A kubeconfig kubectl cannot read raises:
    it may define the context being asked about.
    """
    try:
        listed = subprocess.run(
            ["kubectl", "config", "get-contexts", "-o", "name"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        raise ModelEndpointDiscoveryError(
            context, f"kubectl config get-contexts: {type(exc).__name__}: {exc}"
        ) from exc
    if listed.returncode != 0:
        detail = "\n".join(part for part in (listed.stdout, listed.stderr) if part)
        raise ModelEndpointDiscoveryError(
            context,
            f"kubectl config get-contexts: "
            f"{detail.strip() or f'exit {listed.returncode}'}",
        )
    return context in listed.stdout.split()


def _kubectl_items(context: str, command: list[str]) -> list | None:
    """Run a kubectl list query against ``context`` and return its items.

    ``None`` when the kubeconfig does not define ``context``. A defined context
    whose query fails twice raises: its workloads may publish the endpoint
    being resolved.
    """
    resources = _command_json(command)
    if resources is None:
        if not _kube_context_exists(context):
            return None
        time.sleep(2)
        resources, detail = _run_json(command)
        if resources is None:
            raise ModelEndpointDiscoveryError(context, f"kubectl: {detail}")
    if not isinstance(resources, dict) or not isinstance(resources.get("items"), list):
        raise ModelEndpointDiscoveryError(
            context, f"kubectl returned no resource list: {str(resources)[:200]}"
        )
    return resources["items"]


@dataclass(frozen=True, slots=True)
class _DiscoveredClusterEndpoint:
    base_url: str
    model_revision: str | None = None


def _container_tokens(container: object) -> list[str]:
    if not isinstance(container, dict):
        return []
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
    return tokens


# The variables each model server reads its served model from
# (cogniverse_cli/modal_inference/servers).
_MODEL_ENV_NAMES = frozenset(
    {"MODEL_NAME", "CLAP_EMBED_MODEL", "VIDEO_EMBED_MODEL", "FACE_EMBED_MODEL"}
)


def _container_declares_model(container: object, model: str) -> bool:
    if not isinstance(container, dict):
        return False
    env = container.get("env")
    if isinstance(env, list):
        for entry in env:
            if (
                isinstance(entry, dict)
                and entry.get("name") in _MODEL_ENV_NAMES
                and entry.get("value") == model
            ):
                return True
    tokens = _container_tokens(container)
    return any(
        token == model and index > 0 and tokens[index - 1] in {"serve", "--model"}
        for index, token in enumerate(tokens)
    )


def _container_model_revision(container: object) -> str | None:
    tokens = _container_tokens(container)
    for index, token in enumerate(tokens):
        if token == "--revision" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _discover_cluster_model_urls(
    model: str,
    *,
    context: str,
    cluster: str,
) -> tuple[_DiscoveredClusterEndpoint, ...]:
    """Map an exact cluster workload to its dynamically published host port.

    A context the kubeconfig does not define publishes nothing. A defined one
    whose workloads or load balancer cannot be read raises.
    """
    items = _kubectl_items(
        context,
        [
            "kubectl",
            "--context",
            context,
            "get",
            "deployments,statefulsets,services",
            "--all-namespaces",
            "-o",
            "json",
        ],
    )
    if items is None:
        return ()

    workload_labels: list[tuple[str, dict[str, str], str | None]] = []
    for item in items:
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
        if not (
            isinstance(namespace, str)
            and isinstance(labels, dict)
            and labels
            and isinstance(containers, list)
        ):
            continue
        matching_container = next(
            (
                container
                for container in containers
                if _container_declares_model(container, model)
            ),
            None,
        )
        if matching_container is None:
            continue
        workload_labels.append(
            (
                namespace,
                {
                    key: value
                    for key, value in labels.items()
                    if isinstance(key, str) and isinstance(value, str)
                },
                _container_model_revision(matching_container),
            )
        )

    node_ports: list[tuple[int, str | None]] = []
    for item in items:
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
        matched_revision: str | None = None
        matched = False
        for workload_namespace, labels, revision in workload_labels:
            if workload_namespace != namespace:
                continue
            if not all(labels.get(key) == value for key, value in selector.items()):
                continue
            matched = True
            if revision is not None:
                matched_revision = revision
                break
            matched_revision = revision
        if not matched:
            continue
        node_ports.extend(
            (
                port["nodePort"],
                matched_revision,
            )
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise ModelEndpointDiscoveryError(
            context, f"docker ps: {type(exc).__name__}: {exc}"
        ) from exc
    if load_balancers.returncode != 0:
        detail = "\n".join(
            part for part in (load_balancers.stdout, load_balancers.stderr) if part
        ).strip()
        raise ModelEndpointDiscoveryError(
            context, f"docker ps: {detail or f'exit {load_balancers.returncode}'}"
        )

    candidates: dict[str, _DiscoveredClusterEndpoint] = {}
    for container in load_balancers.stdout.splitlines():
        published, detail = _run_json(
            [
                "docker",
                "inspect",
                container,
                "--format",
                "{{json .NetworkSettings.Ports}}",
            ]
        )
        if not isinstance(published, dict):
            raise ModelEndpointDiscoveryError(
                context, f"docker inspect {container}: {detail or 'no port map'}"
            )
        for node_port, revision in node_ports:
            bindings = published.get(f"{node_port}/tcp")
            if not isinstance(bindings, list):
                continue
            for binding in bindings:
                host_port = (
                    binding.get("HostPort") if isinstance(binding, dict) else None
                )
                if isinstance(host_port, str) and host_port.isdigit():
                    endpoint = _DiscoveredClusterEndpoint(
                        base_url=f"http://127.0.0.1:{host_port}",
                        model_revision=revision,
                    )
                    candidates.setdefault(endpoint.base_url, endpoint)
    return tuple(candidates.values())


def _discover_e2e_model_urls(
    model: str,
) -> tuple[_DiscoveredClusterEndpoint, ...]:
    return _discover_cluster_model_urls(
        model,
        context=E2E_CONTEXT,
        cluster=E2E_CLUSTER,
    )


def _discover_dev_model_urls(
    model: str,
) -> tuple[_DiscoveredClusterEndpoint, ...]:
    return _discover_cluster_model_urls(
        model,
        context=DEV_CONTEXT,
        cluster=DEV_CLUSTER,
    )
