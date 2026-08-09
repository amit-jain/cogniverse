"""The memory suite's schema deploy merges instead of replacing.

Vespa reads a deployment package as the cluster's complete schema list, so a
package carrying only the memory schemas either drops every peer tenant's
schema or is refused outright. These tests drive the real shared Vespa and
pin that a peer tenant's schema and its documents survive the memory deploy,
that two tenants deploying at the same time both end up live, and that a
deploy against a dead config port raises instead of quietly changing nothing.
"""

from __future__ import annotations

import threading
import uuid

import pytest
import requests

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.memory.conftest import (
    MEM0_ROUNDTRIP_SCHEMA,
    MEMORY_SCHEMA,
    PROVENANCE_SCHEMA,
    WIKI_SCHEMA,
    deploy_memory_schemas,
)
from tests.utils.tenant_helpers import MEM0_ROUNDTRIP_TENANT_ID, MEMORY_TENANT_ID
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    load_raw_schema_json,
    schema_full_name,
)

pytestmark = pytest.mark.integration

_PEER_TENANT = "memdeploy_peer"
_PEER_SCHEMA = schema_full_name("wiki_pages", _PEER_TENANT)


def _wiki_embedding_dim() -> int:
    """The width declared for wiki_pages.embedding in configs/schemas/."""
    import re

    schema = load_raw_schema_json("wiki_pages")
    for field in schema["document"]["fields"]:
        if field["name"] == "embedding":
            match = re.search(r"\[(\d+)\]", field["type"])
            assert match, f"embedding declares no dimension: {field['type']!r}"
            return int(match.group(1))
    raise KeyError("wiki_pages has no embedding field")


_EXPECTED_MEMORY_SCHEMAS = {
    (MEMORY_TENANT_ID, "agent_memories"): MEMORY_SCHEMA,
    (MEMORY_TENANT_ID, "wiki_pages"): WIKI_SCHEMA,
    (MEMORY_TENANT_ID, "provenance"): PROVENANCE_SCHEMA,
    (MEM0_ROUNDTRIP_TENANT_ID, "agent_memories"): MEM0_ROUNDTRIP_SCHEMA,
}


def _schema_manager(config_port: int) -> VespaSchemaManager:
    return VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=config_port,
    )


def _deployed(config_port: int) -> list[str]:
    """Vespa's own schema listing, raising rather than masking a probe failure."""
    return _schema_manager(config_port).list_deployed_document_types(
        raise_on_failure=True
    )


def _peer_document_fields(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "tenant_id": _PEER_TENANT,
        "page_type": "topic",
        "title": "Peer tenant survives the memory deploy",
        "content": "A wiki page owned by a tenant the memory suite never touches.",
        "slug": "peer-tenant-survives",
        "entities": "[]",
        "sources": "[]",
        "cross_references": "[]",
        "update_count": 7,
        "created_at": "2026-01-02T03:04:05+00:00",
        "updated_at": "2026-01-02T03:04:06+00:00",
        "embedding": [0.125] * _wiki_embedding_dim(),
    }


def _put_peer_document(http_port: int, doc_id: str, fields: dict) -> None:
    url = (
        f"http://localhost:{http_port}/document/v1/wiki_content/"
        f"{_PEER_SCHEMA}/docid/{doc_id}"
    )
    resp = requests.post(url, json={"fields": fields}, timeout=30)
    assert resp.status_code == 200, f"peer feed failed {resp.status_code}: {resp.text}"


def _get_peer_document(http_port: int, doc_id: str) -> dict:
    url = (
        f"http://localhost:{http_port}/document/v1/wiki_content/"
        f"{_PEER_SCHEMA}/docid/{doc_id}"
    )
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200, f"peer read failed {resp.status_code}: {resp.text}"
    return resp.json()["fields"]


@pytest.fixture
def clean_registry_singletons():
    """Drop the process-wide backend/registry singletons around a test.

    ``deploy_tenant_schema`` builds a backend through BackendRegistry and
    leaves it cached with the endpoints it was given; a test that points one
    at a dead port must not hand that backend to the next test.
    """
    yield
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None


def test_memory_deploy_preserves_a_peer_tenants_schema_and_documents(
    shared_memory_vespa, clean_registry_singletons
):
    """A peer tenant's schema and its document survive the memory deploy."""
    http_port = shared_memory_vespa["http_port"]
    config_port = shared_memory_vespa["config_port"]

    peer_deployed = deploy_tenant_schema(
        shared_memory_vespa,
        tenant_id=_PEER_TENANT,
        base_schema_name="wiki_pages",
        config_manager=shared_memory_vespa["config_manager"],
    )
    assert peer_deployed == _PEER_SCHEMA

    doc_id = f"peer_{uuid.uuid4().hex[:10]}"
    fields = _peer_document_fields(doc_id)
    _put_peer_document(http_port, doc_id, fields)

    redeployed = deploy_memory_schemas(
        shared_memory_vespa,
        config_manager=shared_memory_vespa["config_manager"],
        force=True,
    )
    assert redeployed == _EXPECTED_MEMORY_SCHEMAS

    live = _deployed(config_port)
    assert _PEER_SCHEMA in live, (
        f"the memory deploy dropped the peer schema {_PEER_SCHEMA!r}; "
        f"Vespa now serves {sorted(live)}"
    )
    for schema_name in _EXPECTED_MEMORY_SCHEMAS.values():
        assert schema_name in live

    read_back = _get_peer_document(http_port, doc_id)
    assert read_back["doc_id"] == fields["doc_id"]
    assert read_back["tenant_id"] == _PEER_TENANT
    assert read_back["title"] == fields["title"]
    assert read_back["content"] == fields["content"]
    assert read_back["slug"] == fields["slug"]
    assert read_back["update_count"] == 7
    assert read_back["created_at"] == fields["created_at"]
    assert read_back["updated_at"] == fields["updated_at"]


def test_two_tenants_deploying_concurrently_both_end_up_live(
    shared_memory_vespa, clean_registry_singletons
):
    """Barrier-gated deploys of two tenants leave both schemas serving.

    Each deploy ships the complete schema list, so a lost merge shows up as
    the other thread's schema missing from Vespa's listing.
    """
    config_port = shared_memory_vespa["config_port"]
    suffix = uuid.uuid4().hex[:8]
    tenants = (f"memdeploy_a_{suffix}", f"memdeploy_b_{suffix}")
    expected = {t: schema_full_name("wiki_pages", t) for t in tenants}

    barrier = threading.Barrier(len(tenants))
    returned: dict[str, str] = {}
    errors: dict[str, BaseException] = {}
    lock = threading.Lock()

    def _deploy(tenant_id: str) -> None:
        try:
            barrier.wait(timeout=60)
            name = deploy_tenant_schema(
                shared_memory_vespa,
                tenant_id=tenant_id,
                base_schema_name="wiki_pages",
                config_manager=shared_memory_vespa["config_manager"],
            )
            with lock:
                returned[tenant_id] = name
        except BaseException as exc:  # noqa: BLE001 - reported below
            with lock:
                errors[tenant_id] = exc

    threads = [
        threading.Thread(target=_deploy, args=(tenant_id,), name=f"deploy-{tenant_id}")
        for tenant_id in tenants
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
    assert [t.is_alive() for t in threads] == [False, False]

    assert errors == {}, f"concurrent deploys raised: {errors}"
    assert returned == expected

    live = _deployed(config_port)
    for tenant_id, schema_name in expected.items():
        assert schema_name in live, (
            f"{tenant_id}'s schema {schema_name!r} is not live after the "
            f"concurrent deploy; Vespa serves {sorted(live)}"
        )
    for schema_name in _EXPECTED_MEMORY_SCHEMAS.values():
        assert schema_name in live


def test_deploy_against_dead_config_port_raises_and_changes_nothing(
    shared_memory_vespa, clean_registry_singletons
):
    """A config-server outage surfaces as an error naming the schema."""
    import socket

    from cogniverse_core.registries.exceptions import BackendDeploymentError

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    dead_config_port = sock.getsockname()[1]
    sock.close()

    config_port = shared_memory_vespa["config_port"]
    before = _deployed(config_port)

    dead_endpoint = dict(shared_memory_vespa)
    dead_endpoint["config_port"] = dead_config_port

    # The process-wide registry is bound to the live backend and is reused
    # whenever the data endpoint matches, which would route the deploy back
    # to the live config server instead of the dead one.
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None

    tenant_id = f"memdeploy_dead_{uuid.uuid4().hex[:8]}"
    expected_schema = schema_full_name("wiki_pages", tenant_id)

    with pytest.raises(BackendDeploymentError) as excinfo:
        deploy_tenant_schema(
            dead_endpoint,
            tenant_id=tenant_id,
            base_schema_name="wiki_pages",
            config_manager=shared_memory_vespa["config_manager"],
        )
    assert expected_schema in str(excinfo.value), (
        f"the failure must name the schema it could not deploy; got {excinfo.value}"
    )

    after = _deployed(config_port)
    assert after == before
    assert expected_schema not in after
