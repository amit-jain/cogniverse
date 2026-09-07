"""A deploy returns only when its generation runs on every Vespa service and
each schema new to the cluster has accepted a real feed."""

import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from cogniverse_core.registries.exceptions import (
    BackendDeploymentError,
    SchemaConvergenceError,
)
from cogniverse_vespa import backend as backend_module
from cogniverse_vespa.backend import VespaBackend

pytestmark = pytest.mark.ci_fast

TENANT = "conv_gate"
PEER_TENANT = "conv_gate_peer"
CONVERGE_PATH = (
    "/application/v2/tenant/default/application/default/environment/prod/"
    "region/default/instance/default/serviceconverge"
)
SERVICE_TYPES = {
    "container",
    "container-clustercontroller",
    "distributor",
    "logserver-container",
    "metricsproxy-container",
    "searchnode",
    "storagenode",
}


def _probe_path(schema: str) -> str:
    return f"/document/v1/{schema}/{schema}/docid/convergence_probe"


def _active_generation(vespa_instance) -> int:
    response = requests.get(
        f"http://localhost:{vespa_instance['config_port']}{CONVERGE_PATH}",
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()["currentGeneration"]


def _signal(container: str, process: str, signal: str) -> None:
    pid = subprocess.run(
        ["docker", "exec", container, "pgrep", "-x", process],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert pid.isdecimal() is True
    subprocess.run(
        ["docker", "exec", container, "kill", f"-{signal}", pid],
        check=True,
        timeout=10,
    )


def _wait_for_registry_to_see(
    backend, full_names: set[str], timeout: float = 90
) -> None:
    """Block until the registry's storage read lists every schema in
    ``full_names``. A content node that was frozen is marked down by the
    cluster controller; until it is back up and its buckets are re-reported,
    the config-store visit answers with fewer documents than exist."""
    deadline = time.monotonic() + timeout
    seen: set[str] = set()
    while time.monotonic() < deadline:
        seen = {
            info.full_schema_name for info in backend.schema_registry._get_all_schemas()
        }
        if full_names <= seen:
            return
        time.sleep(1)
    raise AssertionError(
        f"registry never listed {sorted(full_names - seen)} within {timeout}s; saw {sorted(seen)}"
    )


def _wipe_tenant(backend) -> None:
    for tenant in (TENANT, PEER_TENANT):
        for base in ("agent_memories", "provenance", "wiki_pages"):
            full = backend.get_tenant_schema_name(tenant, base)
            if full in backend.schema_manager.list_deployed_document_types():
                backend.schema_manager.delete_schema(tenant, base)


def _intent_record(backend, full_name: str) -> dict:
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    entry = backend.schema_registry._config_manager.store.get_config(
        tenant_id=SYSTEM_TENANT_ID,
        scope=ConfigScope.SCHEMA,
        service="schema_deployment_intents",
        config_key=full_name,
    )
    if entry is None:
        raise AssertionError(f"no deployment intent journaled for {full_name}")
    return entry.config_value


@pytest.fixture(scope="module")
def backend(get_backend):
    backend = get_backend(TENANT)
    _wipe_tenant(backend)
    yield backend
    _wipe_tenant(backend)


@pytest.fixture
def http_trace(monkeypatch):
    """Every request the gate sends, in order, with its response."""
    trace = []
    send = requests.Session.send

    def record(session, request, **kwargs):
        response = send(session, request, **kwargs)
        url = urlsplit(request.url)
        trace.append(
            {
                "method": request.method,
                "path": url.path,
                "query": parse_qs(url.query),
                "status": response.status_code,
                "body": response.text,
            }
        )
        return response

    monkeypatch.setattr(requests.Session, "send", record)
    return trace


def _probes(trace):
    return [
        (entry["method"], entry["path"], entry["status"])
        for entry in trace
        if entry["path"].endswith("/docid/convergence_probe")
    ]


def _converge_calls(trace):
    return [entry for entry in trace if entry["path"] == CONVERGE_PATH]


def test_deploy_waits_for_generation_then_feeds_only_the_new_schema(
    backend, http_trace, vespa_instance
):
    full = backend.schema_registry.deploy_schema(TENANT, "agent_memories")
    assert full == "agent_memories_conv_gate_conv_gate"

    activations = [
        entry
        for entry in http_trace
        if entry["path"] == "/application/v2/tenant/default/prepareandactivate"
    ]
    assert [entry["status"] for entry in activations] == [200]
    generation = int(json.loads(activations[0]["body"])["session-id"])

    converge = _converge_calls(http_trace)
    assert {entry["status"] for entry in converge} == {200}
    last = json.loads(converge[-1]["body"])
    assert {service["type"] for service in last["services"]} == SERVICE_TYPES
    assert {
        service["type"]
        for service in last["services"]
        if service["currentGeneration"] < generation
    } == set()

    assert _probes(http_trace) == [
        ("POST", _probe_path(full), 200),
        ("DELETE", _probe_path(full), 200),
    ]
    order = [id(entry) for entry in http_trace]
    first_probe = next(
        entry for entry in http_trace if entry["path"] == _probe_path(full)
    )
    assert order.index(id(converge[-1])) < order.index(id(first_probe))

    base_url = vespa_instance["base_url"]
    leftover = requests.get(f"{base_url}{_probe_path(full)}", timeout=10)
    assert leftover.status_code == 404
    assert leftover.json() == {
        "pathId": _probe_path(full),
        "id": f"id:{full}:{full}::convergence_probe",
    }

    fed = requests.post(
        f"{base_url}/document/v1/{full}/{full}/docid/after_gate",
        json={"fields": {}},
        timeout=10,
    )
    assert fed.status_code == 200
    assert fed.json() == {
        "pathId": f"/document/v1/{full}/{full}/docid/after_gate",
        "id": f"id:{full}:{full}::after_gate",
    }
    removed = requests.delete(
        f"{base_url}/document/v1/{full}/{full}/docid/after_gate", timeout=10
    )
    assert removed.status_code == 200

    http_trace.clear()
    second = backend.schema_registry.deploy_schema(TENANT, "provenance")
    assert second == "provenance_conv_gate_conv_gate"
    assert _probes(http_trace) == [
        ("POST", _probe_path(second), 200),
        ("DELETE", _probe_path(second), 200),
    ]


@pytest.mark.parametrize("process", ["vespa-distribut", "vespa-proton-bi"])
def test_deploy_rejects_a_service_that_never_runs_the_generation(
    backend, vespa_instance, monkeypatch, process, http_trace
):
    """A frozen service keeps its old generation; the deploy must not
    report the schema live while a feed would be refused."""
    import cogniverse_core.registries.schema_deployment_intents as intent_module
    import cogniverse_core.registries.schema_registry as registry_module

    registered = {
        backend.schema_registry.deploy_schema(TENANT, base)
        for base in ("agent_memories", "provenance")
    }
    backend._wait_for_schema_convergence(_active_generation(vespa_instance), [])
    monkeypatch.setattr(backend_module, "SCHEMA_CONVERGENCE_TIMEOUT_S", 12)
    monkeypatch.setattr(registry_module, "_SCHEMA_INTENT_GRACE_S", 600)
    container = vespa_instance["container_name"]
    full = backend.get_tenant_schema_name(TENANT, "wiki_pages")
    deployment_owner = backend.schema_registry._backend
    activate = deployment_owner._deploy_package
    activated = []

    def activate_then_freeze(*args, **kwargs):
        generation = activate(*args, **kwargs)
        activated.append(generation)
        _signal(container, process, "STOP")
        return generation

    monkeypatch.setattr(deployment_owner, "_deploy_package", activate_then_freeze)
    try:
        with pytest.raises(BackendDeploymentError) as exc_info:
            backend.schema_registry.deploy_schema(TENANT, "wiki_pages")
    finally:
        if activated:
            _signal(container, process, "CONT")
    monkeypatch.setattr(deployment_owner, "_deploy_package", activate)

    [generation] = activated
    assert type(exc_info.value.__cause__) is SchemaConvergenceError
    assert exc_info.value.__cause__.generation == generation
    prefix = (
        f"Backend deployment failed for schema '{full}': Schema convergence "
        f"not confirmed after 12s — generation {generation} was activated by "
        "the config server but is not live on every service: services behind "
        f"generation {generation}: "
    )
    message = str(exc_info.value)
    assert message.startswith(prefix), message
    suffix = ". The schema is live; its registration completes by recovery."
    assert message.endswith(suffix), message
    lagging = json.loads(message[len(prefix) : -len(suffix)].replace("'", '"'))
    frozen = {"vespa-distribut": "distributor", "vespa-proton-bi": "searchnode"}[
        process
    ]
    by_type = {entry.split("@")[0]: entry.rsplit("=", 1)[1] for entry in lagging}
    assert by_type[frozen] == "-1"
    last_convergence = json.loads(_converge_calls(http_trace)[-1]["body"])
    assert lagging == sorted(
        f"{service['type']}@{service['host']}:{service['port']}"
        f"={service['currentGeneration']}"
        for service in last_convergence["services"]
        if service["currentGeneration"] < generation
    )

    # The generation was activated: the schema is live and its registration
    # is still owed. The intent stays pending so every package built inside
    # the grace carries the schema, and recovery completes it afterwards.
    _wait_for_registry_to_see(backend, registered)
    intent = _intent_record(backend, full)
    assert (intent["state"], intent["attempts"]) == ("pending", 0)
    live = set(backend.schema_manager.list_deployed_document_types())
    assert full in live
    assert set(backend.schema_registry.reserved_schemas(live)) == {full}
    peer = backend.get_tenant_schema_name(PEER_TENANT, "wiki_pages")
    assert backend.schema_registry.deploy_schema(PEER_TENANT, "wiki_pages") == peer
    assert set(backend.schema_manager.list_deployed_document_types()) & {
        full,
        peer,
    } == {full, peer}
    assert _intent_record(backend, full)["state"] == "pending"
    backend.schema_manager.delete_schema(PEER_TENANT, "wiki_pages")
    journal_now = intent_module._now
    monkeypatch.setattr(intent_module, "_now", lambda: journal_now() + 601)
    backend.deploy_schemas([])
    assert _intent_record(backend, full)["state"] == "complete"
    assert {
        schema.full_schema_name
        for schema in backend.schema_registry.get_tenant_schemas(TENANT)
    } == {
        "agent_memories_conv_gate_conv_gate",
        "provenance_conv_gate_conv_gate",
        "wiki_pages_conv_gate_conv_gate",
    }
    backend.schema_manager.delete_schema(TENANT, "wiki_pages")
    assert full not in backend.schema_manager.list_deployed_document_types()
    assert backend.schema_registry.deploy_schema(TENANT, "wiki_pages") == full
    fed = requests.post(
        f"{vespa_instance['base_url']}/document/v1/{full}/{full}/docid/after_thaw",
        json={"fields": {}},
        timeout=10,
    )
    assert fed.status_code == 200, fed.text
    requests.delete(
        f"{vespa_instance['base_url']}/document/v1/{full}/{full}/docid/after_thaw",
        timeout=10,
    )
    backend.schema_manager.delete_schema(TENANT, "wiki_pages")


@pytest.fixture
def gate_backend(vespa_instance):
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = vespa_instance["http_port"]
    backend._config_port = vespa_instance["config_port"]
    return backend


def test_gate_rejects_a_schema_the_feed_path_does_not_know(
    gate_backend, vespa_instance, http_trace
):
    generation = _active_generation(vespa_instance)
    with pytest.raises(RuntimeError) as exc_info:
        gate_backend._wait_for_schema_convergence(
            generation, ["convergence_missing"], timeout=3
        )
    assert str(exc_info.value) == (
        f"Schema convergence not confirmed after 3s — generation {generation} "
        "is live on every service but these schemas never accepted a feed: "
        "{'convergence_missing': 'feed HTTP 400: {\"pathId\":\"/document/v1/"
        'convergence_missing/convergence_missing/docid/convergence_probe",'
        '"message":"Document type convergence_missing does not exist"}\'}'
    )
    assert set(_probes(http_trace)) == {
        ("POST", _probe_path("convergence_missing"), 400)
    }


def test_gate_reports_an_unreachable_config_server(gate_backend):
    sock = __import__("socket").socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()
    gate_backend._config_port = dead_port
    with pytest.raises(RuntimeError) as exc_info:
        gate_backend._wait_for_schema_convergence(7, ["never_probed"], timeout=2)
    message = str(exc_info.value)
    assert re.fullmatch(
        "Schema convergence not confirmed after 2s — generation 7 was activated "
        "by the config server but is not live on every service: serviceconverge "
        f"request failed: HTTPConnectionPool\\(host='localhost', port={dead_port}\\): "
        "Max retries exceeded with url: " + re.escape(CONVERGE_PATH) + r"\?timeout=1 "
        r"\(Caused by NewConnectionError\(.*Connection refused.*\)\)",
        message,
        re.DOTALL,
    ), message


def test_concurrent_gates_share_one_probe_document(gate_backend, vespa_instance):
    generation = _active_generation(vespa_instance)
    barrier = threading.Barrier(2)
    send = requests.Session.send
    statuses = []

    def simultaneous_send(session, request, **kwargs):
        is_probe = urlsplit(request.url).path == _probe_path("config_metadata")
        if is_probe:
            barrier.wait(timeout=15)
        response = send(session, request, **kwargs)
        if is_probe:
            statuses.append((request.method, response.status_code))
        return response

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests.Session, "send", simultaneous_send)
        with ThreadPoolExecutor(max_workers=2) as pool:
            waits = [
                pool.submit(
                    gate_backend._wait_for_schema_convergence,
                    generation,
                    ["config_metadata"],
                )
                for _ in range(2)
            ]
            assert [wait.result(timeout=60) for wait in waits] == [None, None]

    assert sorted(statuses) == [
        ("DELETE", 200),
        ("DELETE", 200),
        ("POST", 200),
        ("POST", 200),
    ]
    leftover = requests.get(
        f"{vespa_instance['base_url']}{_probe_path('config_metadata')}", timeout=10
    )
    assert leftover.status_code == 404


def test_convergence_budget_covers_a_config_proxy_restart():
    """A configproxy restart re-converges the jdisc containers in 77.7s
    (measured); the budget must outlast it."""
    assert backend_module.SCHEMA_CONVERGENCE_TIMEOUT_S == 120
