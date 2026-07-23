"""VespaSearchBackend.batch_get_documents must read the schema it is told to,
not the shared per-request ``self.schema_name``.

The search backend is a single process-global instance shared across every
tenant, and each search request rewrites ``self.schema_name``. A document fetch
that read that shared attribute would, after tenant A's search, read tenant A's
schema for a tenant B fetch that interleaved between the write and the read.
Passing the schema explicitly closes that cross-tenant window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_vespa.search_backend import VespaSearchBackend

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingHandle:
    """Stands in for a pyvespa sync session; records the schema of each read."""

    def __init__(self):
        self.schemas_seen = []

    def get_data(self, schema, data_id, namespace, raise_on_not_found):
        self.schemas_seen.append(schema)
        # 404 → the fetch loop skips this id; we only care about the schema arg.
        return SimpleNamespace(status_code=404, json={})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _backend_with_shared_schema(shared_schema: str, handle: _RecordingHandle):
    backend = VespaSearchBackend.__new__(VespaSearchBackend)
    backend.pool = None
    backend.schema_name = shared_schema
    backend.vespa = SimpleNamespace(syncio=lambda: handle)
    return backend


def test_batch_get_uses_explicit_schema_not_shared_attr():
    handle = _RecordingHandle()
    # The shared attribute holds tenant A's schema (A searched last)...
    backend = _backend_with_shared_schema("video_tenant_a", handle)

    # ...but tenant B fetches with its own schema explicitly.
    backend.batch_get_documents(["doc-1"], schema_name="video_tenant_b")

    assert handle.schemas_seen == ["video_tenant_b"]
    assert "video_tenant_a" not in handle.schemas_seen


def test_get_document_threads_schema_through():
    handle = _RecordingHandle()
    backend = _backend_with_shared_schema("video_tenant_a", handle)

    backend.get_document("doc-9", schema_name="video_tenant_b")

    assert handle.schemas_seen == ["video_tenant_b"]


def test_legacy_call_without_schema_falls_back_to_shared():
    # Backward compatibility: a caller that omits the schema still resolves to
    # the instance attribute (the pre-existing behavior, kept for legacy paths).
    handle = _RecordingHandle()
    backend = _backend_with_shared_schema("video_default", handle)

    backend.batch_get_documents(["doc-1"])

    assert handle.schemas_seen == ["video_default"]
