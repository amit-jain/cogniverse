"""Proton flush tuning shipped by the deploy funnels, observed on a real Vespa."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest
import requests

import cogniverse_vespa.vespa_schema_manager as vespa_schema_manager
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import BackendConfig
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import FLUSH_COMPONENT_MAXAGE_S
from tests.utils.vespa_test_helpers import make_config_manager, schema_full_name

pytestmark = pytest.mark.integration

_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"
_BASE_SCHEMA = "provenance"
_PROTON_CONFIG_ID = "cogniverse_content/search/cluster.cogniverse_content/0"
_PROTON_STATE_URL = "http://localhost:19113/state/v1/config"
_SEARCH_DIR = "/opt/vespa/var/db/vespa/search/cluster.cogniverse_content/n0"
_TEST_FLUSH_MAXAGE_S = 20
_TENANTS = {
    "test_metadata_bootstrap_sets_production_flush_maxage": "bootstrap",
    "test_tenant_schema_deploy_sets_production_flush_maxage": "tenant",
    "test_config_generations_of_document_less_db_are_pruned_after_maxage": "prune",
}
_GENERATIONS = 3
_PRUNE_BUDGET_S = 120

_APPLIED_CONFIG = re.compile(
    r"DocumentDB\((?P<db>[^)]+)\): Applied config, .*saved=(?P<saved>yes|no), "
    r"serialNum=(?P<serial>\d+)"
)
_EVENT = re.compile(r'name="(?P<name>[^"]+)" value="(?P<value>\{.*\})"$')


def _docker_exec(container: str, *argv: str) -> str:
    return subprocess.run(
        ["docker", "exec", container, *argv],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _active_generation(config_port: int) -> int:
    resp = requests.get(
        f"http://localhost:{config_port}/application/v2/tenant/default/application/default",
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["generation"]


def _wait_for_proton_generation(vespa: dict) -> None:
    generation = _active_generation(vespa["config_port"])
    deadline = time.monotonic() + 120
    seen = None
    while time.monotonic() < deadline:
        state = json.loads(
            _docker_exec(vespa["container_name"], "curl", "-s", _PROTON_STATE_URL)
        )
        seen = state["config"]["proton"]["generation"]
        if seen == generation:
            return
        time.sleep(1)
    raise AssertionError(
        f"proton stayed on generation {seen}; config server activated {generation}"
    )


def _effective_maxage(vespa: dict) -> int:
    _wait_for_proton_generation(vespa)
    raw = _docker_exec(
        vespa["container_name"],
        "vespa-get-config",
        "-j",
        "-n",
        "vespa.config.search.core.proton",
        "-i",
        _PROTON_CONFIG_ID,
    )
    return json.loads(raw)["flush"]["memory"]["maxage"]["time"]


def _proton_log(container: str) -> list[str]:
    return _docker_exec(
        container, "vespa-logfmt", "-l", "all", "-s", "component,message"
    ).splitlines()


def _applied_serials(log: list[str], db: str) -> dict[int, bool]:
    serials: dict[int, bool] = {}
    for line in log:
        match = _APPLIED_CONFIG.search(line)
        if match and match.group("db") == db:
            serials[int(match.group("serial"))] = match.group("saved") == "yes"
    return serials


def _prune_events(log: list[str], domain: str) -> list[dict]:
    found = []
    for line in log:
        match = _EVENT.search(line)
        if match and match.group("name") == "transactionlog.prune.complete":
            value = json.loads(match.group("value"))
            if value["domain"] == domain:
                found.append(value)
    return found


def _config_snapshots(container: str, db: str) -> set[str]:
    listing = _docker_exec(container, "ls", f"{_SEARCH_DIR}/documents/{db}/config")
    return {entry for entry in listing.split() if entry.startswith("config-")}


@pytest.fixture
def tenant_backend(shared_vespa, request, tmp_path):
    """Backend + registry for a fresh tenant whose base schema lives in a test-owned directory."""
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    source = _SCHEMAS_DIR / f"{_BASE_SCHEMA}_schema.json"
    (schemas_dir / source.name).write_text(source.read_text())

    tenant_id = f"flushmaxage_{_TENANTS[request.node.name]}"
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(schemas_dir)
    backend = VespaBackend(
        backend_config=BackendConfig(
            tenant_id=tenant_id,
            backend_type="vespa",
            url="http://localhost",
            port=shared_vespa["http_port"],
        ),
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.schema_registry = SchemaRegistry(
        config_manager=config_manager, backend=backend, schema_loader=schema_loader
    )
    backend.initialize({"tenant_id": tenant_id})
    yield {"backend": backend, "tenant_id": tenant_id, "schemas_dir": schemas_dir}
    backend.close()


def _add_field(schemas_dir: Path, name: str) -> None:
    path = schemas_dir / f"{_BASE_SCHEMA}_schema.json"
    schema = json.loads(path.read_text())
    schema["document"]["fields"].append(
        {"name": name, "type": "string", "indexing": ["summary", "attribute"]}
    )
    path.write_text(json.dumps(schema))


def test_metadata_bootstrap_sets_production_flush_maxage(shared_vespa, tenant_backend):
    assert _effective_maxage(shared_vespa) == FLUSH_COMPONENT_MAXAGE_S


def test_tenant_schema_deploy_sets_production_flush_maxage(
    shared_vespa, tenant_backend
):
    tenant_backend["backend"].schema_registry.deploy_schema(
        tenant_id=tenant_backend["tenant_id"], base_schema_name=_BASE_SCHEMA
    )
    assert _effective_maxage(shared_vespa) == FLUSH_COMPONENT_MAXAGE_S


def test_config_generations_of_document_less_db_are_pruned_after_maxage(
    shared_vespa, tenant_backend, monkeypatch
):
    container = shared_vespa["container_name"]
    registry = tenant_backend["backend"].schema_registry
    tenant_id = tenant_backend["tenant_id"]
    db = schema_full_name(_BASE_SCHEMA, tenant_id)

    registry.deploy_schema(tenant_id=tenant_id, base_schema_name=_BASE_SCHEMA)
    for generation in range(1, _GENERATIONS + 1):
        _add_field(tenant_backend["schemas_dir"], f"generation_{generation}")
        registry.deploy_schema(
            tenant_id=tenant_id, base_schema_name=_BASE_SCHEMA, force=True
        )
    assert _effective_maxage(shared_vespa) == FLUSH_COMPONENT_MAXAGE_S

    applied = _applied_serials(_proton_log(container), db)
    saved = [serial for serial in sorted(applied) if applied[serial]]
    assert sorted(applied) == [2, *saved]
    assert applied[2] is False
    assert len(saved) == _GENERATIONS
    assert _config_snapshots(container, db) == {"config-1"} | {
        f"config-{serial}" for serial in saved
    }
    assert _prune_events(_proton_log(container), db) == []

    monkeypatch.setattr(
        vespa_schema_manager, "FLUSH_COMPONENT_MAXAGE_S", _TEST_FLUSH_MAXAGE_S
    )
    _add_field(tenant_backend["schemas_dir"], "generation_tuned")
    registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name=_BASE_SCHEMA, force=True
    )
    assert _effective_maxage(shared_vespa) == _TEST_FLUSH_MAXAGE_S
    applied_after = _applied_serials(_proton_log(container), db)
    current = max(applied_after)
    assert applied_after == {**applied, current: True}
    assert current > saved[-1]

    deadline = time.monotonic() + _PRUNE_BUDGET_S
    while time.monotonic() < deadline and _config_snapshots(container, db) != {
        f"config-{current}"
    }:
        time.sleep(2)
    assert _config_snapshots(container, db) == {f"config-{current}"}
    pruned = [
        event["serialnum"]["pruned"]
        for event in _prune_events(_proton_log(container), db)
    ]
    assert pruned == sorted(set(pruned))
    assert pruned[-1:] == [max([current, *pruned])]

    monkeypatch.undo()
    _add_field(tenant_backend["schemas_dir"], "generation_restored")
    registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name=_BASE_SCHEMA, force=True
    )
    assert _effective_maxage(shared_vespa) == FLUSH_COMPONENT_MAXAGE_S
