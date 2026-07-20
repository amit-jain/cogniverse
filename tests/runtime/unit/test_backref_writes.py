"""Content back-ref writes surface missing target docs instead of no-op'ing.

Vespa answers an update with ``create=false`` on an ABSENT document with a
plain 200 no-op — so a drifted video_id/segment derivation silently dropped
every entity/relation/claim back-ref with nothing in the logs. The writer must
check the target exists and WARN when it doesn't, while still writing the
others.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.routers.ingestion import _write_backrefs_to_content

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _config_manager():
    cm = MagicMock()
    cm.get_system_config.return_value = MagicMock(
        backend_url="http://localhost", backend_port=1
    )
    return cm


@pytest.mark.asyncio
async def test_missing_target_warns_and_others_still_write(caplog):
    backend = MagicMock()
    backend.get_document_fields.side_effect = lambda doc_id, **k: (
        {} if doc_id.endswith("seg_0") else None
    )

    backrefs = {
        "0": {"entity_ids": ["e1"], "relation_ids": [], "claim_ids": []},
        "1": {"entity_ids": ["e2"], "relation_ids": ["r1"], "claim_ids": []},
    }
    processing_results = {
        "__schema_name__": "video_colpali_acme",
        "video_id": "vid",
        "fed_documents": [
            {"schema": "video_colpali_acme", "doc_id": "vid_seg_0", "segment_id": "0"},
            {"schema": "video_colpali_acme", "doc_id": "vid_seg_1", "segment_id": "1"},
        ],
    }

    with caplog.at_level(logging.WARNING):
        await _write_backrefs_to_content(
            backrefs_by_segment=backrefs,
            processing_results=processing_results,
            source_doc_id="vid",
            tenant_id="acme:acme",
            config_manager=_config_manager(),
            backend=backend,
        )

    # The existing doc got its back-refs...
    update_ids = [c.args[0] for c in backend.update_document_fields.call_args_list]
    assert update_ids == ["vid_seg_0"]
    kwargs = backend.update_document_fields.call_args_list[0]
    assert kwargs.args[1]["entity_ids"] == ["e1"]
    assert kwargs.kwargs["namespace"] == "content"

    # ...and the missing one was WARNED about, not silently no-op'd.
    warned = [r.message for r in caplog.records if "vid_seg_1" in r.getMessage()]
    assert warned, "missing back-ref target must produce a warning naming the doc"


@pytest.mark.asyncio
async def test_all_targets_present_writes_all_without_warnings(caplog):
    backend = MagicMock()
    backend.get_document_fields.return_value = {}

    backrefs = {"0": {"entity_ids": ["e1"], "relation_ids": [], "claim_ids": []}}
    processing_results = {
        "__schema_name__": "video_colpali_acme",
        "video_id": "vid",
        "fed_documents": [
            {"schema": "video_colpali_acme", "doc_id": "vid_seg_0", "segment_id": "0"}
        ],
    }

    with caplog.at_level(logging.WARNING):
        await _write_backrefs_to_content(
            backrefs_by_segment=backrefs,
            processing_results=processing_results,
            source_doc_id="vid",
            tenant_id="acme:acme",
            config_manager=_config_manager(),
            backend=backend,
        )

    backend.update_document_fields.assert_called_once()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _two_segment_results():
    return {
        "__schema_name__": "video_colpali_acme",
        "video_id": "vid",
        "fed_documents": [
            {"schema": "video_colpali_acme", "doc_id": "vid_seg_0", "segment_id": "0"},
            {"schema": "video_colpali_acme", "doc_id": "vid_seg_1", "segment_id": "1"},
        ],
    }


_BACKREFS = {
    "0": {"entity_ids": ["e1"], "relation_ids": [], "claim_ids": []},
    "1": {"entity_ids": ["e2"], "relation_ids": ["r1"], "claim_ids": []},
}


@pytest.mark.asyncio
async def test_total_backend_outage_raises_not_silent_success():
    """When EVERY back-ref update fails (a Vespa outage during the back-ref
    phase), the writer must raise — otherwise the KG nodes/edges persist while
    no content doc receives its entity/relation/claim ids and the ingest is
    recorded successful, indistinguishable from a clean run."""
    backend = MagicMock()
    backend.get_document_fields.side_effect = lambda doc_id, **k: {}  # target exists
    backend.update_document_fields.side_effect = ConnectionError("vespa down")

    with pytest.raises(RuntimeError, match="all 2 content back-ref updates failed"):
        await _write_backrefs_to_content(
            backrefs_by_segment=_BACKREFS,
            processing_results=_two_segment_results(),
            source_doc_id="vid",
            tenant_id="acme:acme",
            config_manager=_config_manager(),
            backend=backend,
        )


@pytest.mark.asyncio
async def test_partial_backend_failure_is_tolerated():
    """One doc's failure must not abort the rest — a partial outage still
    persists the back-refs it can and does NOT raise."""
    backend = MagicMock()
    backend.get_document_fields.side_effect = lambda doc_id, **k: {}

    def _update(doc_id, *a, **k):
        if doc_id.endswith("seg_1"):
            raise ConnectionError("one target down")
        return None

    backend.update_document_fields.side_effect = _update

    # Must NOT raise (only one of two failed).
    await _write_backrefs_to_content(
        backrefs_by_segment=_BACKREFS,
        processing_results=_two_segment_results(),
        source_doc_id="vid",
        tenant_id="acme:acme",
        config_manager=_config_manager(),
        backend=backend,
    )
    updated = [c.args[0] for c in backend.update_document_fields.call_args_list]
    assert "vid_seg_0" in updated, "the healthy doc must still get its back-refs"
