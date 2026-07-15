"""Namespace-aware raw-fields document primitives against real Vespa.

These are the sanctioned Document v1 surface for callers that own their own
field shapes (wiki pages, knowledge-graph nodes/edges, content back-refs) —
replacing hand-built ``/document/v1`` HTTP. Pins: full-put → get round-trip,
NAMESPACE ISOLATION (same id under another namespace is absent), partial
update with assign semantics, delete, and get→None for missing docs.
"""

from __future__ import annotations

import time
import uuid

import pytest

from cogniverse_vespa.backend import VespaBackend

pytestmark = [pytest.mark.integration]

SCHEMA = "config_metadata"  # deployed by the shared fixture's metadata deploy
NAMESPACE = "nsprobe"


@pytest.fixture(scope="module")
def doc_backend(vespa_instance):
    """A VespaBackend slice with only what the document primitives need."""
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = vespa_instance["http_port"]
    backend._metadata_app = None
    backend._metadata_app_key = None
    backend.config = {}
    yield backend
    if backend._metadata_app is not None:
        backend._metadata_app.close()


def _fields(doc_id: str, value: str) -> dict:
    now = int(time.time())
    return {
        "config_id": doc_id,
        "tenant_id": "nsprobe_tenant",
        "scope": "system",
        "service": "nsprobe",
        "config_key": "probe",
        "config_value": value,
        "version": 1,
        "created_at": str(now),
        "updated_at": str(now),
    }


def test_put_get_round_trip_and_namespace_isolation(doc_backend):
    doc_id = f"ns-{uuid.uuid4().hex[:8]}"
    written = _fields(doc_id, '{"beta": true}')

    doc_backend.put_document_fields(
        doc_id, written, schema_name=SCHEMA, namespace=NAMESPACE
    )

    got = doc_backend.get_document_fields(
        doc_id, schema_name=SCHEMA, namespace=NAMESPACE
    )
    assert got is not None
    for key, value in written.items():
        assert got[key] == value, f"field {key} did not round-trip"

    # The SAME id under the default namespace must be absent — namespaces
    # genuinely partition the document space.
    assert doc_backend.get_document_fields(doc_id, schema_name=SCHEMA) is None, (
        "namespace isolation violated: default namespace sees the nsprobe doc"
    )


def test_partial_update_assigns_only_named_fields(doc_backend):
    doc_id = f"ns-{uuid.uuid4().hex[:8]}"
    doc_backend.put_document_fields(
        doc_id, _fields(doc_id, "before"), schema_name=SCHEMA, namespace=NAMESPACE
    )

    doc_backend.update_document_fields(
        doc_id,
        {"config_value": "after"},
        schema_name=SCHEMA,
        namespace=NAMESPACE,
    )

    got = doc_backend.get_document_fields(
        doc_id, schema_name=SCHEMA, namespace=NAMESPACE
    )
    assert got["config_value"] == "after"
    assert got["tenant_id"] == "nsprobe_tenant"  # untouched fields survive


def test_delete_removes_the_document(doc_backend):
    doc_id = f"ns-{uuid.uuid4().hex[:8]}"
    doc_backend.put_document_fields(
        doc_id, _fields(doc_id, "x"), schema_name=SCHEMA, namespace=NAMESPACE
    )
    assert (
        doc_backend.get_document_fields(doc_id, schema_name=SCHEMA, namespace=NAMESPACE)
        is not None
    )

    doc_backend.delete_document_fields(doc_id, schema_name=SCHEMA, namespace=NAMESPACE)

    assert (
        doc_backend.get_document_fields(doc_id, schema_name=SCHEMA, namespace=NAMESPACE)
        is None
    )


def test_get_missing_document_returns_none(doc_backend):
    assert (
        doc_backend.get_document_fields(
            f"never-{uuid.uuid4().hex[:8]}", schema_name=SCHEMA, namespace=NAMESPACE
        )
        is None
    )


def test_update_with_create_makes_a_missing_document(doc_backend):
    """update_document_fields(create=True) upserts an absent doc — the create
    branch the ingestion path deliberately does NOT use, pinned here."""
    doc_id = f"ns-{uuid.uuid4().hex[:8]}"

    doc_backend.update_document_fields(
        doc_id,
        _fields(doc_id, "created-via-update"),
        schema_name=SCHEMA,
        namespace=NAMESPACE,
        create=True,
    )

    got = doc_backend.get_document_fields(
        doc_id, schema_name=SCHEMA, namespace=NAMESPACE
    )
    assert got is not None
    assert got["config_value"] == "created-via-update"
