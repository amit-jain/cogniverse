"""Two processes activating a first schema never drop each other's work.

Every Vespa activation replaces the whole application package, so two
deployers that each enumerate the live schemas and then post their own
package activate packages missing the other's schemas — and on the delete
paths, with the removal override, that deletes the peer's documents too.

This cannot run against the shared e2e cluster: a regression removes schemas
belonging to the seeded tenant and to every other tenant a concurrent test
minted, and that corpus is seeded once per session. It lives here, in the
sub-suite that stands up its own cluster and is invoked as its own run.

Both deployers are host processes contending through the cluster's own config
store over the forwarded Vespa ports, which is what makes them a real
cross-process race rather than two threads.
"""

from __future__ import annotations

import multiprocessing
import time
import uuid
from pathlib import Path

import httpx
import pytest
from vespa.application import Vespa

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore

from .conftest import PORTS

BASE_SCHEMA = "agent_memories"
REPO_ROOT = Path(__file__).resolve().parents[3]

_SCHEMAS_LIST_PATH = (
    "/application/v2/tenant/default/application/default/"
    "environment/prod/region/default/instance/default/content/schemas/"
)


def _backend():
    BackendRegistry.clear_instances()
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=PORTS["vespa_http"]
    )
    return BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "port": PORTS["vespa_http"],
                "config_port": PORTS["vespa_config"],
            }
        },
        config_manager=ConfigManager(store=store),
        schema_loader=FilesystemSchemaLoader(REPO_ROOT / "configs" / "schemas"),
    )


def _deployed_schema_names() -> set[str]:
    resp = httpx.get(
        f"http://localhost:{PORTS['vespa_config']}{_SCHEMAS_LIST_PATH}", timeout=30.0
    )
    resp.raise_for_status()
    return {
        entry.rsplit("/", 1)[-1][: -len(".sd")]
        for entry in resp.json()
        if entry.endswith(".sd")
    }


def _feed_marker(schema: str, marker: str) -> None:
    app = Vespa(url=f"http://localhost:{PORTS['vespa_http']}")
    deadline = time.monotonic() + 120
    while True:
        statuses: list[int] = []
        app.feed_iterable(
            [{"id": "marker", "fields": {"id": "marker", "text": marker}}],
            schema=schema,
            namespace=schema,
            callback=lambda response, doc_id: statuses.append(response.status_code),
        )
        if statuses == [200]:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{schema} did not accept its marker: {statuses}")
        time.sleep(0.5)


def _stored_marker(schema: str) -> dict:
    response = Vespa(url=f"http://localhost:{PORTS['vespa_http']}").get_data(
        schema=schema, namespace=schema, data_id="marker"
    )
    assert response.status_code == 200, response.json
    return response.json["fields"]


def _activate_first_schema(tenant: str, marker: str, result) -> None:
    """Deploy this tenant's first schema, recording each activation window."""
    backend = _backend()
    manager = backend.schema_manager
    original = manager._post_package
    windows: list[tuple[float, float]] = []

    def timed(*args, **kwargs):
        started = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            windows.append((started, time.time()))

    manager._post_package = timed
    try:
        schema = backend.schema_registry.deploy_schema(tenant, BASE_SCHEMA)
        _feed_marker(schema, marker)
        result.put({"tenant": tenant, "schema": schema, "windows": windows})
    except Exception as exc:  # surfaced by the parent's assertions
        result.put({"tenant": tenant, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        manager._post_package = original


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_docker
class TestConcurrentFirstSchemaActivation:
    """A first-schema activation is serialized and loses nothing."""

    def test_two_processes_keep_both_schemas_and_every_peer(
        self, deployed_stack, request
    ):
        """Two concurrent first activations leave both new schemas, every
        pre-existing schema, and both tenants' documents in place.

        Unserialized, the loser rebuilt its package from a snapshot taken
        before the winner activated and reposted it, so the winner's schema —
        and with the removal override every document under it — disappeared.
        """
        run = uuid.uuid4().hex[:8]
        tenants = [f"prode2epipe{run}a", f"prode2epipe{run}b"]
        markers = {tenant: f"{tenant} document" for tenant in tenants}

        before = _deployed_schema_names()
        assert before, (
            "the deployment cluster reports no deployed schemas, so the "
            "survivor set this race must preserve would be empty"
        )

        cleanup_backend = _backend()

        def release() -> None:
            cleanup_backend.schema_manager.delete_tenant_schemas_bulk(tenants)

        request.addfinalizer(release)

        ctx = multiprocessing.get_context("spawn")
        result = ctx.Queue()
        processes = [
            ctx.Process(
                target=_activate_first_schema, args=(tenant, markers[tenant], result)
            )
            for tenant in tenants
        ]
        for process in processes:
            process.start()
        outcomes = {}
        for _ in processes:
            payload = result.get(timeout=1800)
            outcomes[payload["tenant"]] = payload
        for process in processes:
            process.join(timeout=300)
            assert process.exitcode == 0, process.exitcode

        assert sorted(outcomes) == sorted(tenants)
        for tenant in tenants:
            assert "error" not in outcomes[tenant], outcomes[tenant]

        expected_names = {outcomes[tenant]["schema"] for tenant in tenants}
        assert expected_names == {
            cleanup_backend.schema_manager.get_tenant_schema_name(tenant, BASE_SCHEMA)
            for tenant in tenants
        }
        assert _deployed_schema_names() == before | expected_names

        for tenant in tenants:
            assert _stored_marker(outcomes[tenant]["schema"]) == {
                "id": "marker",
                "text": markers[tenant],
            }

        # The lease serialized the activations: no window of one deployer
        # overlaps a window of the other.
        first, second = (outcomes[tenant]["windows"] for tenant in tenants)
        assert [bool(first), bool(second)] == [True, True]
        assert [
            (a, b) for a in first for b in second if a[0] < b[1] and b[0] < a[1]
        ] == []
