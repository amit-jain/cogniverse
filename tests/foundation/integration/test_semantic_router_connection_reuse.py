"""Routed calls reuse Envoy's upstream connection, against the live stack.

Envoy pools upstream connections per worker thread. A caller's separate client
connections land on different workers, so with one worker per host core every
call arriving on a new client connection found no pooled upstream connection
and dialed the backend again - to Modal, a fresh TCP+TLS handshake per call.
This module runs its own stack so the pool starts empty: only the readiness
probe has reached the backend before the first test.
"""

from __future__ import annotations

import subprocess

import pytest
import requests
import yaml

from tests.utils.semantic_router_stack import CHART_VALUES

pytestmark = [pytest.mark.integration]

_CALLS = 6


def _upstream_stats(stack: dict) -> dict[str, int]:
    """``cluster.llm_upstream.*`` counters from the stack's Envoy admin port.

    Read from the router's container, which shares the stack network; the
    Envoy image carries no HTTP client of its own.
    """
    admin_port = yaml.safe_load(CHART_VALUES.read_text())["semanticRouter"]["envoy"][
        "adminPort"
    ]
    url = (
        f"http://{stack['envoy_container']}:{admin_port}/stats"
        "?filter=^cluster.llm_upstream.upstream_cx_"
    )
    scrape = subprocess.run(
        [
            "docker",
            "exec",
            stack["router_container"],
            "python3",
            "-c",
            "import urllib.request; print(urllib.request.urlopen("
            f"{url!r}, timeout=10).read().decode())",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    stats: dict[str, int] = {}
    for line in scrape.splitlines():
        name, _, value = line.partition(": ")
        if value.isdigit():
            stats[name.removeprefix("cluster.llm_upstream.")] = int(value)
    return stats


def test_calls_on_separate_client_connections_share_one_upstream_connection(
    semantic_router_stack,
):
    base_url = semantic_router_stack["base_url"].rstrip("/")

    statuses = [
        requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": "cogniverse-classification",
                "messages": [{"role": "user", "content": f"reuse probe {call}"}],
            },
            headers={
                "x-authz-user-id": "connection-reuse-tenant",
                "x-authz-user-groups": "free",
                "connection": "close",
            },
            timeout=30,
        ).status_code
        for call in range(_CALLS)
    ]
    stats = _upstream_stats(semantic_router_stack)

    assert statuses == [200] * _CALLS
    assert stats["upstream_cx_total"] == 1
    assert stats["upstream_cx_active"] == 1
