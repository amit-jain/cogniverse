"""The shipped ingress reaches the runtime API, and the Service still does.

The chart publishes the runtime under one path prefix and adds no rewrite
annotation: the runtime resolves the prefix from the path each request
carries, so the ingress entry point and the bare in-cluster Service are served
by one process. Every assertion here therefore compares the answer through the
ingress with the answer to the same call on the Service, and the ``/a2a``
sub-app — mounted rather than included — is exercised on both.

The k3s overlay enables Traefik on its own host and publishes no host port, so
the ingress is reached through a ``kubectl port-forward`` to the ingress
controller's Service with an explicit ``Host`` header.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest
import yaml

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from tests.e2e.conftest import (
    DASHBOARD,
    K3S_VALUES,
    KUBECTL_CONTEXT,
    RUNTIME,
    SAMPLE_VIDEO_CONTENT_ID,
    TENANT_DEPLOY_TIMEOUT_S,
    TENANT_ID,
    register_tenant_and_wait,
    unique_id,
)
from tests.e2e.test_api_e2e import PROFILE
from tests.e2e.test_pi_harness_e2e import harness_models

pytestmark = [pytest.mark.e2e]

INGRESS_NAMESPACE = "kube-system"
PORT_FORWARD_READY_TIMEOUT_S = 60.0


def _ingress_hosts() -> list[dict]:
    """The ingress rules the e2e overlay ships, read from the overlay."""
    values = yaml.safe_load(Path(K3S_VALUES).read_text())
    ingress = values["ingress"]
    if not ingress.get("enabled"):
        raise AssertionError(
            f"{K3S_VALUES} disables the ingress; this module pins the routing "
            "the deployed overlay publishes"
        )
    return ingress["hosts"]


def _rule_path(service: str) -> str:
    """The single path prefix the overlay publishes ``service`` under."""
    paths = [
        entry["path"]
        for host in _ingress_hosts()
        for entry in host["paths"]
        if entry["service"] == service
    ]
    if len(paths) != 1:
        raise AssertionError(
            f"{K3S_VALUES} publishes {service!r} under {paths}; the runtime's "
            "root path is derived from exactly one rule"
        )
    return paths[0].rstrip("/")


def _ingress_host() -> str:
    hosts = {host["host"] for host in _ingress_hosts()}
    if len(hosts) != 1:
        raise AssertionError(f"{K3S_VALUES} declares several ingress hosts: {hosts}")
    return hosts.pop()


API_PREFIX = _rule_path("runtime")
DASHBOARD_PREFIX = _rule_path("dashboard")
INGRESS_HOST = _ingress_host()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kubectl(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--context", KUBECTL_CONTEXT, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ingress_controller_service() -> str:
    """The Traefik Service the cluster's ingress class is served by."""
    listed = _kubectl(
        "-n",
        INGRESS_NAMESPACE,
        "get",
        "svc",
        "-l",
        "app.kubernetes.io/name=traefik",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    )
    names = listed.stdout.split()
    if len(names) != 1:
        inventory = _kubectl(
            "-n", INGRESS_NAMESPACE, "get", "svc", "-o", "name"
        ).stdout.strip()
        pytest.fail(
            f"expected exactly one Traefik Service in namespace "
            f"{INGRESS_NAMESPACE}, found {names}; namespace holds:\n{inventory}",
            pytrace=False,
        )
    return names[0]


@pytest.fixture(scope="module")
def ingress_url():
    """Port-forward to the ingress controller; yields its local base URL."""
    if shutil.which("kubectl") is None:
        pytest.fail("kubectl is required to reach the cluster ingress", pytrace=False)

    service = _ingress_controller_service()
    local_port = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            INGRESS_NAMESPACE,
            "port-forward",
            f"svc/{service}",
            f"{local_port}:80",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{local_port}"
    try:
        deadline = time.monotonic() + PORT_FORWARD_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                httpx.get(
                    f"{url}{API_PREFIX}/health/live",
                    headers={"Host": INGRESS_HOST},
                    timeout=5.0,
                )
                break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                time.sleep(1.0)
        else:
            proc.terminate()
            pytest.fail(
                f"port-forward to {INGRESS_NAMESPACE}/{service} did not accept "
                f"connections on {url} within {PORT_FORWARD_READY_TIMEOUT_S:.0f}s",
                pytrace=False,
            )
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def harness_key():
    """A harness key bound to a tenant this module mints and tears down."""
    tenant_id = canonical_tenant_id(unique_id("prode2eclients"))
    register_tenant_and_wait(tenant_id, created_by="e2e")
    with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        minted = client.post(
            "/admin/harness/keys",
            json={"tenant_id": tenant_id, "name": "ingress-routing-e2e"},
        )
        assert minted.status_code == 200, minted.text
        record = minted.json()
        try:
            yield record["key"]
        finally:
            revoked = client.delete(f"/admin/harness/keys/{record['key_hash']}")
            assert revoked.json() == {
                "revoked": True,
                "key_hash": record["key_hash"],
            }


def _through_ingress(ingress_url: str, path: str, **kwargs) -> httpx.Response:
    headers = {"Host": INGRESS_HOST, **kwargs.pop("headers", {})}
    return httpx.request(
        kwargs.pop("method", "GET"),
        f"{ingress_url}{path}",
        headers=headers,
        timeout=kwargs.pop("timeout", 120.0),
        **kwargs,
    )


def test_the_published_prefix_and_the_service_serve_one_liveness_document(ingress_url):
    """``/health/live`` answers identically under the prefix and on the Service.

    The runtime is published under the prefix with no rewrite, so the ingress
    answer is the Service answer byte for byte, and the unprefixed path through
    the same ingress belongs to the dashboard rule instead.
    """
    on_service = httpx.get(f"{RUNTIME}/health/live", timeout=30.0)
    assert (on_service.status_code, on_service.json()) == (200, {"status": "alive"})

    through_prefix = _through_ingress(ingress_url, f"{API_PREFIX}/health/live")
    assert (through_prefix.status_code, through_prefix.text) == (
        on_service.status_code,
        on_service.text,
    ), (
        f"GET {API_PREFIX}/health/live through the ingress answered "
        f"{through_prefix.status_code} {through_prefix.text[:300]!r}"
    )

    # The dashboard owns the unprefixed rule, so the runtime's liveness
    # document must not come back from it. Paired with the assertion above,
    # this cannot hold by the ingress being unreachable.
    unprefixed = _through_ingress(ingress_url, "/health/live")
    assert (unprefixed.status_code, unprefixed.text) != (
        on_service.status_code,
        on_service.text,
    ), (
        f"the unprefixed path reached the runtime through the ingress: "
        f"{unprefixed.text[:300]!r}"
    )


def test_the_published_prefix_serves_the_openapi_surface(ingress_url):
    """``/docs`` and ``/openapi.json`` resolve under the prefix and bare.

    Both are generated by the application itself rather than routed, so they
    are the surface a static root path breaks first.
    """
    for path, expected in (
        ("/openapi.json", "application/json"),
        ("/docs", "text/html"),
    ):
        through_prefix = _through_ingress(ingress_url, f"{API_PREFIX}{path}")
        on_service = httpx.get(f"{RUNTIME}{path}", timeout=30.0)
        assert through_prefix.status_code == 200, (
            f"GET {API_PREFIX}{path} through the ingress answered "
            f"{through_prefix.status_code}: {through_prefix.text[:300]!r}"
        )
        assert on_service.status_code == 200, on_service.text[:300]
        assert through_prefix.headers["content-type"].split(";")[0] == expected
        assert on_service.headers["content-type"].split(";")[0] == expected

    # The served document declares the runtime's own title, so a 200 from an
    # unrelated service cannot satisfy the pin.
    served = _through_ingress(ingress_url, f"{API_PREFIX}/openapi.json").json()
    assert (
        served["info"]["title"]
        == httpx.get(f"{RUNTIME}/openapi.json", timeout=30.0).json()["info"]["title"]
    )


def test_the_root_endpoint_reports_the_entry_point_it_was_reached_through(ingress_url):
    """``GET /`` on the runtime names the prefix the caller actually used."""
    through_prefix = _through_ingress(ingress_url, f"{API_PREFIX}/").json()
    on_service = httpx.get(f"{RUNTIME}/", timeout=30.0).json()

    assert through_prefix["docs"] == f"{API_PREFIX}/docs"
    assert through_prefix["health"] == f"{API_PREFIX}/health"
    assert on_service["docs"] == "/docs"
    assert on_service["health"] == "/health"
    assert through_prefix["service"] == on_service["service"] == "Cogniverse Runtime"
    assert through_prefix["version"] == on_service["version"]


def test_the_model_catalogue_is_the_same_through_both_entry_points(
    ingress_url, harness_key
):
    """``/v1/models`` lists the shipped catalogue under the prefix and bare."""
    authorization = {"Authorization": f"Bearer {harness_key}"}
    through_prefix = _through_ingress(
        ingress_url, f"{API_PREFIX}/v1/models", headers=authorization
    )
    on_service = httpx.get(f"{RUNTIME}/v1/models", headers=authorization, timeout=30.0)

    assert through_prefix.status_code == 200, (
        f"GET {API_PREFIX}/v1/models through the ingress answered "
        f"{through_prefix.status_code}: {through_prefix.text[:300]!r}"
    )
    assert on_service.status_code == 200, on_service.text[:300]

    catalogue = list(harness_models())
    assert [model["id"] for model in through_prefix.json()["data"]] == catalogue
    assert [model["id"] for model in on_service.json()["data"]] == catalogue
    assert through_prefix.json()["object"] == on_service.json()["object"] == "list"

    # A missing key is rejected by the route, not by the ingress: the same
    # refusal document comes back through both entry points, which is what
    # tells a resolved route from a 404 the prefix never reached.
    refused = _through_ingress(ingress_url, f"{API_PREFIX}/v1/models")
    refused_on_service = httpx.get(f"{RUNTIME}/v1/models", timeout=30.0)
    assert (refused.status_code, refused.text) == (
        refused_on_service.status_code,
        refused_on_service.text,
    )
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"] == "Bearer"
    assert refused.json()["error"]["code"] == "invalid_api_key"


def test_the_mounted_a2a_app_answers_on_the_service_and_under_the_prefix(ingress_url):
    """The ``/a2a`` sub-app resolves on the bare Service path and the prefix.

    The dashboard, the CLI and the rest of this suite address ``/a2a/`` on the
    Service with no prefix, while the ingress carries the prefix. A root path
    the process applies unconditionally resolves the mount against the prefix
    and answers 404 for the bare path.
    """
    card_path = "/a2a/.well-known/agent.json"
    on_service_card = httpx.get(f"{RUNTIME}{card_path}", timeout=30.0)
    assert on_service_card.status_code == 200, (
        f"GET {card_path} on the Service answered "
        f"{on_service_card.status_code}: {on_service_card.text[:300]!r}"
    )
    assert on_service_card.json()["capabilities"]["streaming"] is True

    through_prefix_card = _through_ingress(ingress_url, f"{API_PREFIX}{card_path}")
    assert through_prefix_card.status_code == 200, (
        f"GET {API_PREFIX}{card_path} through the ingress answered "
        f"{through_prefix_card.status_code}: {through_prefix_card.text[:300]!r}"
    )
    assert through_prefix_card.text == on_service_card.text

    def _jsonrpc(request_id: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "search for nature videos"}],
                    "messageId": str(uuid.uuid4()),
                },
                "configuration": {"acceptedOutputModes": ["text"]},
                "metadata": {"tenant_id": TENANT_ID},
            },
        }

    on_service_turn = httpx.post(
        f"{RUNTIME}/a2a/", json=_jsonrpc("ingress-e2e-service"), timeout=900.0
    )
    assert on_service_turn.status_code == 200, (
        f"POST /a2a/ on the Service answered {on_service_turn.status_code}: "
        f"{on_service_turn.text[:300]!r}"
    )
    service_result = on_service_turn.json()["result"]
    assert service_result["contextId"]

    through_prefix_turn = _through_ingress(
        ingress_url,
        f"{API_PREFIX}/a2a/",
        method="POST",
        json=_jsonrpc("ingress-e2e-prefix"),
        timeout=900.0,
    )
    assert through_prefix_turn.status_code == 200, (
        f"POST {API_PREFIX}/a2a/ through the ingress answered "
        f"{through_prefix_turn.status_code}: {through_prefix_turn.text[:300]!r}"
    )
    assert through_prefix_turn.json()["result"]["contextId"]


def test_the_seeded_search_answers_the_same_hits_through_the_ingress(ingress_url):
    """``POST /search/`` returns the seeded content identically on both paths."""
    payload = {
        "query": SAMPLE_VIDEO_CONTENT_ID,
        "profile": PROFILE,
        "top_k": 5,
        "tenant_id": TENANT_ID,
    }
    on_service = httpx.post(f"{RUNTIME}/search/", json=payload, timeout=900.0).json()
    through_prefix = _through_ingress(
        ingress_url, f"{API_PREFIX}/search/", method="POST", json=payload, timeout=900.0
    )
    assert through_prefix.status_code == 200, (
        f"POST {API_PREFIX}/search/ through the ingress answered "
        f"{through_prefix.status_code}: {through_prefix.text[:300]!r}"
    )
    served = through_prefix.json()

    # The tracked fixture is the only content whose id is the query, so the
    # equality below cannot be satisfied by two empty result lists.
    assert served["results"][0]["source_id"] == SAMPLE_VIDEO_CONTENT_ID
    assert (served["query"], served["profile"], served["results_count"]) == (
        on_service["query"],
        on_service["profile"],
        on_service["results_count"],
    )
    assert [row["document_id"] for row in served["results"]] == [
        row["document_id"] for row in on_service["results"]
    ]


def test_the_unprefixed_rule_serves_the_dashboard(ingress_url):
    """The overlay's other rule is the dashboard, and the prefix did not take it."""
    assert DASHBOARD_PREFIX == "", (
        f"{K3S_VALUES} publishes the dashboard under {DASHBOARD_PREFIX!r}; this "
        "test pins the root rule the overlay ships"
    )
    root = _through_ingress(ingress_url, "/")
    on_service = httpx.get(f"{DASHBOARD}/", timeout=30.0)
    assert (root.status_code, root.text) == (
        on_service.status_code,
        on_service.text,
    ), f"the root rule answered {root.status_code}: {root.text[:300]!r}"
    assert root.status_code == 200
