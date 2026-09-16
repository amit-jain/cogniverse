"""Tenant provisioning through the installed command and fixture-owned Vespa.

Each step runs the argv and the environment the tenant-provisioning
WorkflowTemplate declares for it, with only the endpoint values redirected at
the fixture. A step the template launches without the variables it needs
therefore fails here the way it fails in the cluster.
"""

from __future__ import annotations

import json
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
import yaml
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
        )._deploy_package(lambda: _shared_vespa_application_package(schemas))
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


WORKFLOW = ROOT / "workflows" / "tenant-provisioning.yaml"
# The template that runs each ``--step``.
STEP_TEMPLATES = {
    "schemas": "deploy-schemas",
    "verify": "verify-tenant",
    "telemetry": "create-phoenix-project",
    "memory": "initialize-memory",
    "tier": "set-tier",
}


def _workflow_template(name):
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    (template,) = [t for t in document["spec"]["templates"] if t["name"] == name]
    return template["container"]


def _command(endpoint, tenant, step, *, profiles="graph_profile", extra_env=None):
    """Run ``step`` exactly as its workflow template declares it.

    Only the endpoint values are redirected at the fixture; the variable
    names are the template's own, so a template that omits one runs the step
    without it here too.
    """
    container = _workflow_template(STEP_TEMPLATES[step])
    parameters = {
        "{{workflow.parameters.tenant-id}}": tenant,
        "{{workflow.parameters.profiles}}": profiles,
        "{{workflow.parameters.tier}}": "pro",
    }
    argv = [parameters.get(token, token) for token in container["args"]]

    redirect = {
        "BACKEND_URL": "http://localhost",
        "BACKEND_PORT": str(endpoint["http_port"]),
        "VESPA_CONFIG_PORT": str(endpoint["config_port"]),
    }
    declared = {entry["name"]: entry["value"] for entry in container["env"]}
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BACKEND_", "VESPA_", "TELEMETRY_"))
    }
    env |= {name: redirect.get(name, value) for name, value in declared.items()}
    env |= extra_env or {}
    return subprocess.run(
        [sys.executable, "-m", "cogniverse_runtime.provision_tenant", *argv],
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
    """Two tenants provision at once. A prepare-and-activate replaces the
    whole application package, so the deploy lease must hand the config
    server one package at a time even while both steps are in flight; both
    tenants end up registered and the peer's documents survive."""
    endpoint, cm, app, peer = provisioning_store
    from cogniverse_runtime import provision_tenant

    record_lock = threading.Lock()
    arrivals = []
    packages = []
    arrival_lock = threading.Lock()
    inflight = {"now": 0, "max": 0}

    class ConfigProxy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            activating = self.path.endswith("/prepareandactivate")
            with arrival_lock:
                arrivals.append(self.path)
                first = activating and len(arrivals) == 1
                if activating:
                    inflight["now"] += 1
                    inflight["max"] = max(inflight["max"], inflight["now"])
            if first:
                # Hold the first activation open long enough that a second
                # deployer unserialised by the lease would overlap it.
                time.sleep(3)
            entered = time.monotonic()
            try:
                response = requests.post(
                    f"http://localhost:{endpoint['config_port']}{self.path}",
                    data=body,
                    headers={
                        "Content-Type": self.headers.get(
                            "Content-Type", "application/zip"
                        )
                    },
                    timeout=180,
                )
            finally:
                with record_lock:
                    packages.append((entered, time.monotonic()))
                if activating:
                    with arrival_lock:
                        inflight["now"] -= 1
            self.send_response(response.status_code)
            self.end_headers()
            self.wfile.write(response.content)

        def do_GET(self):
            response = requests.get(
                f"http://localhost:{endpoint['config_port']}{self.path}", timeout=60
            )
            self.send_response(response.status_code)
            self.end_headers()
            self.wfile.write(response.content)

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ConfigProxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()

    start = threading.Barrier(2)
    steps = []

    def deploy(tenant):
        start.wait(timeout=60)
        began = time.monotonic()
        result = _command(
            endpoint,
            tenant,
            "schemas",
            extra_env={"VESPA_CONFIG_PORT": str(proxy.server_port)},
        )
        with record_lock:
            steps.append((began, time.monotonic()))
        return result

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
    # The deployment lease admits one application replacement at a time, so
    # the second deployer's activation starts only after the first returns.
    assert inflight["max"] == 1
    assert [result.returncode for result in results] == [0, 0], [
        result.stderr for result in results
    ]
    assert arrivals == ["/application/v2/tenant/default/prepareandactivate"] * 2
    # The two steps really ran at the same time...
    (first_step, second_step) = sorted(steps)
    assert second_step[0] < first_step[1]
    # ...while their application deploys did not overlap.
    (first_package, second_package) = sorted(packages)
    assert first_package[1] <= second_package[0]
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


SHIPPED_PROFILES = ["video_colpali_smol500_mv_frame", "video_xclip_sv_chunk_6s"]


def _workflow_default_profiles():
    (document,) = [
        d
        for d in yaml.safe_load_all(WORKFLOW.read_text())
        if d["kind"] == "WorkflowTemplate"
    ]
    parameters = {
        entry["name"]: entry["value"]
        for entry in document["spec"]["arguments"]["parameters"]
    }
    return parameters["profiles"].split(",")


def _registry_row(cm, tenant, base_schema_name):
    return cm.store.get_config(
        tenant_id=tenant,
        scope=ConfigScope.SCHEMA,
        service="schema_registry",
        config_key=f"schema_{base_schema_name}",
    )


def _tenant_backend(cm, endpoint, tenant):
    return BackendRegistry.get_ingestion_backend(
        "vespa",
        tenant_id=tenant,
        config={
            "url": "http://localhost",
            "port": endpoint["http_port"],
            "config_port": endpoint["config_port"],
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(ROOT / "configs/schemas"),
    )


def test_a_tenant_with_no_stored_rows_provisions_the_shipped_profiles(
    provisioning_store,
):
    """A tenant registered a minute ago owns no backend rows. The workflow's
    own default profiles must still resolve, deploy and verify."""
    endpoint, cm, app, peer = provisioning_store
    tenant = "provisionfresh"
    canonical = "provisionfresh:provisionfresh"
    profiles = _workflow_default_profiles()
    assert profiles == SHIPPED_PROFILES
    assert cm.get_backend_config(tenant_id=canonical).profiles == {}

    deployed = _command(endpoint, tenant, "schemas", profiles=",".join(profiles))
    assert deployed.returncode == 0, deployed.stderr
    assert deployed.stdout.strip() == (
        f"Provisioned schemas for tenant {canonical}: "
        "video_colpali_smol500_mv_frame_provisionfresh_provisionfresh, "
        "video_xclip_sv_chunk_6s_provisionfresh_provisionfresh"
    )
    for base in profiles:
        row = _registry_row(cm, canonical, base)
        assert row is not None, base
        assert (
            row.config_value["full_schema_name"]
            == f"{base}_provisionfresh_provisionfresh"
        )

    verified = _command(endpoint, tenant, "verify", profiles=",".join(profiles))
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.strip() == (
        f"Provisioned verify for tenant {canonical}: "
        "video_colpali_smol500_mv_frame_provisionfresh_provisionfresh, "
        "video_xclip_sv_chunk_6s_provisionfresh_provisionfresh"
    )
    _peer(app, peer)


def test_an_unknown_profile_is_refused_with_the_catalog_it_searched(
    provisioning_store,
):
    endpoint, cm, app, peer = provisioning_store
    result = _command(endpoint, "provisionfresh", "schemas", profiles="no_such_profile")
    assert result.returncode == 1
    assert result.stdout == ""
    known = sorted(
        json.loads((ROOT / "configs/config.json").read_text())["backend"]["profiles"]
    )
    assert result.stderr.strip() == (
        "Provisioning failed for tenant provisionfresh:provisionfresh: profile "
        f"'no_such_profile' is not configured. Configured profiles: {known}"
    )
    _peer(app, peer)


def test_a_profile_whose_schema_is_absent_registers_none_of_the_batch(
    provisioning_store, monkeypatch
):
    """The profiles of one run land as one application package: a profile
    that cannot be loaded leaves the tenant with no schema at all rather than
    a half-provisioned set."""
    endpoint, cm, app, peer = provisioning_store
    from cogniverse_runtime import provision_tenant

    monkeypatch.setenv("BACKEND_URL", "http://localhost")
    monkeypatch.setenv("BACKEND_PORT", str(endpoint["http_port"]))
    monkeypatch.setenv("VESPA_CONFIG_PORT", str(endpoint["config_port"]))
    tenant = "provisionpartial:production"
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="graph_profile",
            type="document",
            schema_name="knowledge_graph",
            embedding_model="lightonai/LateOn",
        ),
        tenant_id=tenant,
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="absent_schema_profile",
            type="document",
            schema_name="no_such_base_schema",
            embedding_model="lightonai/LateOn",
        ),
        tenant_id=tenant,
    )

    with pytest.raises(RuntimeError) as raised:
        provision_tenant.deploy_schemas(
            tenant, ["graph_profile", "absent_schema_profile"]
        )
    assert str(raised.value).startswith(
        "Provisioning schemas failed for tenant provisionpartial:production: "
        "knowledge_graph, no_such_base_schema: "
    )
    assert _registry_row(cm, tenant, "knowledge_graph") is None
    assert _registry_row(cm, tenant, "no_such_base_schema") is None
    _peer(app, peer)


def test_verify_fails_when_a_registered_schema_is_not_queryable(provisioning_store):
    """The registry row says deployed; the YQL probe says the live
    application package has never heard of it. The step must fail."""
    endpoint, cm, app, peer = provisioning_store
    tenant = "provisionphantom:production"
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="graph_profile",
            type="document",
            schema_name="knowledge_graph",
            embedding_model="lightonai/LateOn",
        ),
        tenant_id=tenant,
    )
    backend = _tenant_backend(cm, endpoint, tenant)
    definition = (ROOT / "configs/schemas/knowledge_graph_schema.json").read_text()
    backend.schema_registry.register_schema(
        tenant_id=tenant,
        base_schema_name="knowledge_graph",
        full_schema_name="knowledge_graph_provisionphantom_production",
        schema_definition=definition,
    )
    assert backend.schema_exists(schema_name="knowledge_graph", tenant_id=tenant)

    result = _command(endpoint, tenant, "verify")
    assert result.returncode == 1
    assert result.stdout == ""
    reported = result.stderr.strip().splitlines()[-1]
    assert reported.startswith(
        "Provisioning verify failed for tenant provisionphantom:production: "
        "knowledge_graph is not queryable: "
    )
    assert (
        "Could not resolve source ref 'knowledge_graph_provisionphantom_production'"
        in reported
    )
    _peer(app, peer)


def test_concurrent_resolution_keeps_each_tenants_override(provisioning_store):
    """Two tenants resolve the same profile name at once through the shared
    parsed-config cache: the one with an override gets it, the one without
    gets the shipped schema."""
    endpoint, cm, app, peer = provisioning_store
    from cogniverse_runtime import provision_tenant

    overridden = "provisionmergea:production"
    plain = "provisionmergeb:production"
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="wiki_semantic",
            type="document",
            schema_name="knowledge_graph",
            embedding_model="lightonai/LateOn",
        ),
        tenant_id=overridden,
    )
    assert cm.get_backend_config(tenant_id=plain).profiles == {}

    start = threading.Barrier(2)
    arrivals = []
    arrival_lock = threading.Lock()

    def resolve(tenant):
        start.wait(timeout=30)
        with arrival_lock:
            arrivals.append(tenant)
        return provision_tenant.resolve_schema_names(cm, tenant, ["wiki_semantic"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, (overridden, plain)))

    assert sorted(arrivals) == [overridden, plain]
    assert results == [["knowledge_graph"], ["wiki_pages"]]
    _peer(app, peer)


def test_resolution_against_a_dead_store_fails_instead_of_reporting_success(
    provisioning_store,
):
    endpoint, _, app, peer = provisioning_store
    result = _command(
        endpoint,
        "provisiondeadstore",
        "schemas",
        extra_env={"BACKEND_PORT": "1", "VESPA_CONFIG_PORT": "1"},
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert "provisiondeadstore:provisiondeadstore" in result.stderr
    _peer(app, peer)
