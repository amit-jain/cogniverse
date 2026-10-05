"""Inference sidecars on the e2e cluster: deployment overrides and readiness probes."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from tests.e2e.cluster import E2E_CLUSTER_NAME

_E2E_TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"


_E2E_ASR_MODELS = {
    "cpu": "openai/whisper-tiny",
    "cuda": "openai/whisper-large-v3-turbo",
    "rocm": "openai/whisper-large-v3-turbo",
}


# The e2e stack's own OpenShell gateway: ``ensure_host_gateway`` runs
# ``openshell gateway start --port <OPENSHELL_GATEWAY_HOST_PORT>`` (28080 by
# default), which registers it under the CLI's default name.
E2E_OPENSHELL_GATEWAY = "openshell"
E2E_OPENSHELL_DEFAULT_PORT = 28080


class E2EGatewayError(RuntimeError):
    """The e2e stack's own OpenShell gateway is missing or not the active one."""


def e2e_gateway_metadata(config_root: Path | None = None) -> dict:
    """The e2e stack's own gateway's ``metadata.json``.

    The deploy and the cert sync read the host's active gateway, so this
    refuses unless the active gateway is the e2e one, registered with the
    port the e2e path starts it on. A test's private gateway, or any other one,
    left active on the host would otherwise be deployed into the cluster.
    """
    import json
    import os

    root = config_root or Path.home() / ".config" / "openshell"
    expected_port = int(
        os.environ.get("OPENSHELL_GATEWAY_HOST_PORT", E2E_OPENSHELL_DEFAULT_PORT)
    )
    active_file = root / "active_gateway"
    active = active_file.read_text().strip() if active_file.exists() else None
    metadata_file = root / "gateways" / E2E_OPENSHELL_GATEWAY / "metadata.json"
    if not metadata_file.exists():
        raise E2EGatewayError(
            f"the e2e OpenShell gateway {E2E_OPENSHELL_GATEWAY!r} is not "
            f"registered ({metadata_file} is missing); the active gateway is "
            f"{active!r}. Start it with `openshell gateway start --port "
            f"{expected_port}`."
        )
    if active != E2E_OPENSHELL_GATEWAY:
        raise E2EGatewayError(
            f"the active OpenShell gateway is {active!r}, not the e2e gateway "
            f"{E2E_OPENSHELL_GATEWAY!r}; refusing to deploy another gateway into "
            f"the cluster. Select it with `openshell gateway select "
            f"{E2E_OPENSHELL_GATEWAY}`."
        )
    metadata = json.loads(metadata_file.read_text())
    if (metadata.get("name"), metadata.get("gateway_port")) != (
        E2E_OPENSHELL_GATEWAY,
        expected_port,
    ):
        raise E2EGatewayError(
            f"the e2e OpenShell gateway's metadata names "
            f"{metadata.get('name')!r} on port {metadata.get('gateway_port')!r}, "
            f"expected {E2E_OPENSHELL_GATEWAY!r} on {expected_port}: {metadata!r}"
        )
    return metadata


def _e2e_docker_network_gateway_ip() -> str:
    network_name = f"k3d-{E2E_CLUSTER_NAME}"
    command = [
        "docker",
        "network",
        "inspect",
        network_name,
        "-f",
        "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "docker network gateway inspection failed: "
            f"{shlex.join(command)}\nstderr: {(result.stderr or '').strip()}"
        )
    gateway_ip = result.stdout.strip()
    if not gateway_ip:
        raise RuntimeError(
            "docker network gateway inspection returned an empty IP: "
            f"{shlex.join(command)}"
        )
    return gateway_ip


# A locally served teacher needs 20Gi of the node's 123.5Gi. Video embedding
# and transcription cannot make room for it -- the session fixture ingests the
# corpus, and the shipped profiles bind embedding to vllm_colpali and
# transcription to vllm_asr. Only the code retriever is unused here, so the
# rest of the room comes from right-sizing requests to measured usage. Served
# from Modal, neither chat model is on the node and the code retriever fits.
_E2E_DISABLED_INFERENCE_SERVICES = frozenset({"code_colbert_pylate"})


# Off in the chart and the k3s overlay, whose policy enables only the pods a
# profile's inference_service names. Video ingestion calls face_embed when it
# is deployed, so the e2e cluster serves it for the tests that exercise it.
_E2E_ENABLED_INFERENCE_SERVICES = frozenset({"face_embed"})


def _e2e_disabled_inference_services(llm_serving: str) -> frozenset[str]:
    from cogniverse_cli.config import LLM_SERVING_LOCAL

    if llm_serving == LLM_SERVING_LOCAL:
        return _E2E_DISABLED_INFERENCE_SERVICES
    return frozenset()


def _e2e_deployment_overrides() -> dict[str, str]:
    from cogniverse_cli.sandbox import pod_gateway_endpoint

    from tests.e2e.deployment.conftest import e2e_llm_serving_mode

    overrides = {
        **{
            f"inference.{service}.enabled": "false"
            for service in sorted(
                _e2e_disabled_inference_services(e2e_llm_serving_mode())
            )
        },
        **{
            f"inference.{service}.enabled": "true"
            for service in sorted(_E2E_ENABLED_INFERENCE_SERVICES)
        },
        "runtime.sandbox.enabled": "true",
        "runtime.sandbox.inCluster.enabled": "false",
        "runtime.sandbox.gatewayEndpoint": pod_gateway_endpoint(e2e_gateway_metadata()),
        "runtime.sandbox.hostGatewayIP": _e2e_docker_network_gateway_ip(),
    }
    for service in (
        "vllm_colpali",
        "vllm_asr",
        "vllm_llm_student",
        "vllm_llm_teacher",
    ):
        overrides[f"inference.{service}.livenessProbe.initialDelaySeconds"] = "1200"
        overrides[f"inference.{service}.livenessProbe.failureThreshold"] = "60"
    return overrides


def _e2e_required_model_probes(backend: str) -> list[tuple[str, str]]:
    """Model endpoints this cluster must serve, derived from the enabled set.

    A service switched off by the deployment overrides has no pod to probe, so
    gating on it would fail readiness for a model nothing deployed.
    """
    probes: list[tuple[str, str]] = []
    if (
        backend in {"cuda", "rocm"}
        and "vllm_colpali" not in _E2E_DISABLED_INFERENCE_SERVICES
    ):
        probes.append(("http://127.0.0.1:33901", _E2E_TOMORO_MODEL))
    if "vllm_asr" not in _E2E_DISABLED_INFERENCE_SERVICES:
        probes.append(
            (
                "http://127.0.0.1:33905",
                _E2E_ASR_MODELS.get(backend, "openai/whisper-large-v3-turbo"),
            )
        )
    return probes


# Sidecars the session fixture's own bootstrap exercises: video embedding
# (colpali), transcription (asr), document and text embeddings
# (colbert_pylate, denseon), audio embedding (clap_embed) and graph extraction
# (gliner). Only the first two speak OpenAI's /v1/models, so the rest were
# ungated -- on 2026-09-01 a run proceeded with gliner still starting and the
# seven selected tests ERRORed reporting "ingestion did not complete", which
# names the symptom rather than the missing model. Every sidecar serves
# /health, so all six are gateable.
_E2E_GATED_INFERENCE_SERVICES = (
    "vllm_colpali",
    "vllm_asr",
    "colbert_pylate",
    "denseon",
    "clap_embed",
    "gliner",
)


# The e2e loadbalancer maps NodePort 290NN to host port 339NN (E2E_HOST_PORTS).
_E2E_NODEPORT_TO_HOST = 33900 - 29000


def _chart_inference_node_ports() -> dict[str, int]:
    """service -> nodePort, read from the shipped chart rather than restated."""

    import re as _re

    values = Path(__file__).resolve().parents[2] / "charts/cogniverse/values.yaml"
    ports: dict[str, int] = {}
    current: str | None = None
    for line in values.read_text().splitlines():
        header = _re.match(r"^  ([a-z_]+):\s*$", line)
        if header:
            current = header.group(1)
        port = _re.search(r"nodePort:\s*(\d+)", line)
        if port and current:
            ports[current] = int(port.group(1))
    return ports


def e2e_required_health_probes(backend: str) -> list[tuple[str, str]]:
    """(service, base_url) for every enabled sidecar this cluster must serve.

    A service switched off by the deployment overrides has no pod to probe, so
    gating on it would fail readiness for something nothing deployed.
    """

    node_ports = _chart_inference_node_ports()
    probes: list[tuple[str, str]] = []
    for service in _E2E_GATED_INFERENCE_SERVICES:
        if service in _E2E_DISABLED_INFERENCE_SERVICES:
            continue
        node_port = node_ports.get(service)
        if node_port is None:
            raise RuntimeError(
                f"{service} is gated for e2e readiness but the shipped chart "
                "declares no nodePort for it, so the gate cannot reach it. "
                "Either the service was renamed or its Service lost nodePort."
            )
        probes.append(
            (service, f"http://localhost:{node_port + _E2E_NODEPORT_TO_HOST}")
        )
    return probes
