"""Tenant provisioning through the installed command and fixture-owned Vespa."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from vespa.application import Vespa

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.conftest import (
    _shared_vespa_application_package,
    _shared_vespa_run_args,
    _vespa_wait_for_config_ready,
    _vespa_wait_for_query_ready,
)
from tests.utils.docker_utils import start_docker_container_with_port_retry
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def provisioning_store():
    from cogniverse_vespa.metadata_schemas import (
        create_adapter_registry_schema,
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    name, port, config_port = start_docker_container_with_port_retry(
        "tests.conftest",
        name_prefix="provisioning-tests",
        image="vespaengine/vespa:8.668.5",
        container_ports=(8080, 19071),
        extra_run_args=_shared_vespa_run_args(
            owner_pid=os.getpid(), docker_platform="linux/amd64"
        ),
        max_attempts=3,
    )
    try:
        assert _vespa_wait_for_config_ready(config_port) is True
        schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=config_port
        )._deploy_package(_shared_vespa_application_package(schemas))
        assert _vespa_wait_for_query_ready(port) is True
        endpoint = {"http_port": port, "config_port": config_port}
        cm = make_config_manager(endpoint)
        for tenant in (
            "provisionpeer:production",
            "provisiona:production",
            "provisionb:production",
            "provisionc:production",
            "provisionfailure:production",
        ):
            cm.add_backend_profile(
                BackendProfileConfig(
                    profile_name="graph_profile",
                    type="document",
                    schema_name="knowledge_graph",
                    embedding_model="lightonai/LateOn",
                ),
                tenant_id=tenant,
            )
        backend = BackendRegistry.get_ingestion_backend(
            "vespa",
            tenant_id="provisionpeer:production",
            config={
                "url": "http://localhost",
                "port": port,
                "config_port": config_port,
            },
            config_manager=cm,
            schema_loader=FilesystemSchemaLoader(ROOT / "configs/schemas"),
        )
        peer_schema = backend.schema_registry.deploy_schema(
            "provisionpeer:production", "knowledge_graph"
        )
        app = Vespa(url=f"http://localhost:{port}")
        feed_status = []
        app.feed_iterable(
            [
                {
                    "id": "peer-marker",
                    "fields": {
                        "doc_id": "peer-marker",
                        "name": "peer survives provisioning",
                    },
                }
            ],
            schema=peer_schema,
            callback=lambda response, doc_id: feed_status.append(
                (doc_id, response.status_code)
            ),
        )
        assert feed_status == [("peer-marker", 200)]
        yield endpoint, cm, app, peer_schema
    finally:
        BackendRegistry.clear_instances()
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


def _command(endpoint, tenant, step, *, extra_env=None):
    env = (
        os.environ
        | {
            "BACKEND_URL": "http://localhost",
            "BACKEND_PORT": str(endpoint["http_port"]),
            "VESPA_CONFIG_PORT": str(endpoint["config_port"]),
        }
        | (extra_env or {})
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.provision_tenant",
            "--tenant-id",
            tenant,
            "--step",
            step,
            "--profiles",
            "graph_profile",
        ],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )


def _peer(app, schema):
    response = app.get_data(schema=schema, data_id="peer-marker")
    assert response.status_code == 200
    assert response.json["fields"]["doc_id"] == "peer-marker"
    assert response.json["fields"]["name"] == "peer survives provisioning"


def test_installed_schema_command_round_trips_canonical_registration(
    provisioning_store,
):
    endpoint, cm, app, peer = provisioning_store
    result = _command(endpoint, "provisiona:production", "schemas")
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.strip()
        == "Provisioned schemas for tenant provisiona:production: knowledge_graph_provisiona_production"
    )
    row = cm.store.get_config(
        tenant_id="provisiona:production",
        scope=ConfigScope.SCHEMA,
        service="schema_registry",
        config_key="schema_knowledge_graph",
    )
    assert (
        row.config_value["full_schema_name"] == "knowledge_graph_provisiona_production"
    )
    verified = _command(endpoint, "provisiona:production", "verify")
    assert verified.returncode == 0, verified.stderr
    assert (
        verified.stdout.strip()
        == "Provisioned verify for tenant provisiona:production: knowledge_graph_provisiona_production"
    )
    again = _command(endpoint, "provisiona:production", "schemas")
    assert (again.returncode, again.stdout) == (0, result.stdout)
    _peer(app, peer)


def test_concurrent_provisioning_keeps_both_tenants_and_peer(provisioning_store):
    endpoint, cm, app, peer = provisioning_store
    from cogniverse_runtime import provision_tenant

    barrier = threading.Barrier(2)
    arrivals = []
    arrival_lock = threading.Lock()

    class ConfigProxy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with arrival_lock:
                arrivals.append(self.path)
                initial = len(arrivals) <= 2
            if initial:
                barrier.wait(timeout=60)
            response = requests.post(
                f"http://localhost:{endpoint['config_port']}{self.path}",
                data=body,
                headers={
                    "Content-Type": self.headers.get("Content-Type", "application/zip")
                },
                timeout=180,
            )
            self.send_response(response.status_code)
            self.end_headers()
            self.wfile.write(response.content)

        def do_GET(self):
            response = requests.get(
                f"http://localhost:{endpoint['config_port']}{self.path}", timeout=30
            )
            self.send_response(response.status_code)
            self.end_headers()
            self.wfile.write(response.content)

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ConfigProxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()

    def deploy(tenant):
        return _command(
            endpoint,
            tenant,
            "schemas",
            extra_env={"VESPA_CONFIG_PORT": str(proxy.server_port)},
        )

    assert provision_tenant.__name__ == "cogniverse_runtime.provision_tenant"
    tenants = ("provisionb:production", "provisionc:production")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(deploy, tenants))
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)
    assert arrivals[:2] == ["/application/v2/tenant/default/prepareandactivate"] * 2
    assert [result.returncode for result in results] == [0, 0], [
        result.stderr for result in results
    ]
    for tenant in tenants:
        row = cm.store.get_config(
            tenant_id=tenant,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key="schema_knowledge_graph",
        )
        assert (
            row.config_value["full_schema_name"]
            == f"knowledge_graph_{tenant.replace(':', '_')}"
        )
    _peer(app, peer)


def test_schema_boundary_failure_cannot_report_success_and_retry_recovers(
    provisioning_store, monkeypatch
):
    endpoint, cm, app, peer = provisioning_store
    from cogniverse_runtime import provision_tenant
    from cogniverse_vespa.backend import VespaBackend

    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(endpoint["http_port"]))
    monkeypatch.setenv("VESPA_CONFIG_PORT", str(endpoint["config_port"]))
    original = VespaBackend._deploy_package
    attempts = []

    def fail_at_activation(backend, package, *args, **kwargs):
        attempts.append(backend._config_port)
        # Real transport failure after registry preparation, before activation.
        requests.post(
            "http://127.0.0.1:1/application/v2/tenant/default/prepareandactivate",
            timeout=1,
        )

    monkeypatch.setattr(VespaBackend, "_deploy_package", fail_at_activation)
    with pytest.raises(
        RuntimeError, match="provisionfailure:production.*knowledge_graph"
    ):
        provision_tenant.deploy_schemas(
            "provisionfailure:production", ["graph_profile"]
        )
    assert attempts == [endpoint["config_port"]]
    assert (
        cm.store.get_config(
            tenant_id="provisionfailure:production",
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key="schema_knowledge_graph",
        )
        is None
    )
    _peer(app, peer)
    monkeypatch.setattr(VespaBackend, "_deploy_package", original)
    assert provision_tenant.deploy_schemas(
        "provisionfailure:production", ["graph_profile"]
    ) == ["knowledge_graph_provisionfailure_production"]
    _peer(app, peer)


def test_concurrent_telemetry_steps_create_only_their_canonical_projects(
    provisioning_store, phoenix_container
):
    from phoenix.client import Client

    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TENANT_ID_ATTRIBUTE

    endpoint, _, _, _ = provisioning_store
    tenants = ("provisiontelemetrya", "provisiontelemetryb")
    start = threading.Barrier(2)

    def emit(tenant):
        start.wait(timeout=5)
        return _command(
            endpoint,
            tenant,
            "telemetry",
            extra_env={"TELEMETRY_OTLP_ENDPOINT": phoenix_container["grpc_endpoint"]},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(emit, tenants))
    assert [result.returncode for result in results] == [0, 0], [
        result.stderr for result in results
    ]
    client = Client(base_url=phoenix_container["http_endpoint"])
    for tenant in tenants:
        canonical = f"{tenant}:{tenant}"
        project = TelemetryConfig().get_project_name(canonical)
        # Phoenix indexes an accepted export asynchronously; the step already
        # blocked on the exporter, so this only covers the indexing lag.
        deadline = time.monotonic() + 60
        # Phoenix nests a dotted attribute key, so "tenant.id" arrives as
        # {"id": ...} under the "tenant" column. Both halves come from the
        # production constant so a rename fails here.
        namespace, leaf = TENANT_ID_ATTRIBUTE.split(".", 1)
        column = f"attributes.{namespace}"
        frame = client.spans.get_spans_dataframe(project_identifier=project)
        while (
            frame.empty or column not in frame.columns
        ) and time.monotonic() < deadline:
            time.sleep(0.5)
            frame = client.spans.get_spans_dataframe(project_identifier=project)
        assert column in frame.columns, sorted(frame.columns)
        assert frame[["name", column]].to_dict("records") == [
            {"name": "provision.probe", column: {leaf: canonical}}
        ]


def test_telemetry_boundary_failure_exits_without_completion(
    provisioning_store, phoenix_container
):
    endpoint, _, app, peer = provisioning_store
    result = _command(
        endpoint,
        "provisiontelemetryfailure",
        "telemetry",
        extra_env={"TELEMETRY_OTLP_ENDPOINT": "http://127.0.0.1:1"},
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "Provisioning telemetry failed for tenant provisiontelemetryfailure:provisiontelemetryfailure"
        in result.stderr
    )
    assert "127.0.0.1:1" in result.stderr
    _peer(app, peer)
