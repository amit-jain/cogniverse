"""The e2e cluster gate must cover every inference sidecar it depends on.

On 2026-09-01 a rerun proceeded with cogniverse-gliner still starting. The
session fixture ingests sample content, graph extraction calls GLiNER, and all
seven selected tests ERRORed in fixture setup five minutes later reporting
"Sample content ingestion did not complete within 300s" -- a message about
ingestion, not about a model that was never up.

The gate that should have caught it probed only the two services that speak
OpenAI's ``/v1/models``. Every sidecar serves ``/health``, so the remaining
ones are gateable; the docstring on the probe list already claimed it was
"derived from the enabled set".
"""

from __future__ import annotations

import pathlib
import re

import pytest

import tests.e2e.conftest as e2e_conftest

_VALUES = pathlib.Path(__file__).resolve().parents[3] / "charts/cogniverse/values.yaml"


def _chart_node_ports() -> dict[str, int]:
    """service -> nodePort, read from the shipped chart."""
    ports: dict[str, int] = {}
    current: str | None = None
    for line in _VALUES.read_text().splitlines():
        header = re.match(r"^  ([a-z_]+):\s*$", line)
        if header:
            current = header.group(1)
        port = re.search(r"nodePort:\s*(\d+)", line)
        if port and current:
            ports[current] = int(port.group(1))
    return ports


def test_the_chart_still_exposes_the_node_ports_the_gate_derives_from():
    ports = _chart_node_ports()
    assert ports["gliner"] == 29007, ports
    assert ports["vllm_colpali"] == 29001, ports
    assert ports["vllm_asr"] == 29005, ports


def test_every_gated_service_maps_to_the_forwarded_host_port():
    for service, url in e2e_conftest.e2e_required_health_probes("rocm"):
        host_port = int(url.rsplit(":", 1)[1])
        assert host_port in e2e_conftest.E2E_HOST_PORTS, (
            f"{service} is gated at {url}, which the e2e port-forward does not "
            f"expose; forwarded ports are {sorted(e2e_conftest.E2E_HOST_PORTS)}"
        )


def test_gliner_is_gated_because_sample_ingestion_calls_it():
    probed = dict(e2e_required := e2e_conftest.e2e_required_health_probes("rocm"))
    assert "gliner" in probed, e2e_required
    assert probed["gliner"] == e2e_conftest.GLINER_URL, probed


def test_a_service_disabled_by_the_deployment_is_not_gated():
    probed = dict(e2e_conftest.e2e_required_health_probes("rocm"))
    for disabled in e2e_conftest._E2E_DISABLED_INFERENCE_SERVICES:
        assert disabled not in probed, (
            f"{disabled} is switched off by the e2e overrides, so it has no pod "
            "to probe and gating on it would fail readiness for a model nothing "
            "deployed"
        )


def test_the_gated_set_is_exactly_the_sidecars_sample_ingestion_needs():
    # video embed (colpali), transcription (asr), document + text embeddings
    # (colbert_pylate, denseon), audio embed (clap_embed) and graph extraction
    # (gliner). These are the six the launcher warms and the six the session
    # fixture's own ingest exercises.
    assert sorted(dict(e2e_conftest.e2e_required_health_probes("rocm"))) == [
        "clap_embed",
        "colbert_pylate",
        "denseon",
        "gliner",
        "vllm_asr",
        "vllm_colpali",
    ]


@pytest.mark.parametrize("backend", ["cpu", "rocm", "cuda"])
def test_the_gate_is_never_empty_for_any_backend(backend):
    assert e2e_conftest.e2e_required_health_probes(backend), backend


def test_every_inference_sidecar_the_chart_exposes_is_port_forwarded():
    """A new sidecar is unreachable from e2e until its host port is forwarded.

    `video_embed` (X-CLIP) shipped a chart Service with nodePort 29012 while
    E2E_HOST_PORTS stopped at 33911, so nothing in the e2e suite could reach
    it -- silently, because no test named it yet. Deriving the requirement from
    the chart catches the next one at commit time rather than when someone
    first tries to gate or probe it.
    """

    missing = {
        service: node_port
        for service, node_port in _chart_node_ports().items()
        if 29000 <= node_port < 30000
        and (node_port + 4900) not in e2e_conftest.E2E_HOST_PORTS
    }
    assert missing == {}, (
        "these inference sidecars declare a nodePort the e2e loadbalancer does "
        f"not forward: {missing}. Add host port nodePort+4900 to E2E_HOST_PORTS "
        "(it takes effect on the next cluster create)."
    )
