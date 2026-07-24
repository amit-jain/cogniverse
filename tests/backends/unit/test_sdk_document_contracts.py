"""Document / SearchResult / ConfigEntry / workflow-record serialization
contracts.

These pin the sdk's deserialization boundary: corrupt payloads raise with
the offending field named (never a silently mistyped object that detonates
inside the backend later), lifecycle setters mutate exactly the fields
they name, and serializers tolerate the shapes real payloads carry.
"""

from __future__ import annotations

import time

import pytest

from cogniverse_sdk.document import (
    ContentType,
    Document,
    ProcessingStatus,
    SearchResult,
)
from cogniverse_sdk.interfaces.config_store import ConfigEntry, ConfigScope
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowTemplate,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class TestDocumentLifecycleSetters:
    def test_mark_completed_sets_status_time_and_touch(self):
        doc = Document(title="t")
        before = doc.updated_at
        time.sleep(0)  # same-second updates are fine; assert >= not >
        doc.mark_completed(processing_time=1.5)
        assert doc.status is ProcessingStatus.COMPLETED
        assert doc.processing_time == 1.5
        assert doc.updated_at >= before

    def test_mark_failed_records_error(self):
        doc = Document(title="t")
        doc.mark_failed("boom")
        assert doc.status is ProcessingStatus.FAILED
        assert doc.error_message == "boom"

    def test_set_processing_status_clears_stale_error(self):
        doc = Document(title="t")
        doc.mark_failed("boom")
        doc.set_processing_status(ProcessingStatus.PROCESSING)
        assert doc.status is ProcessingStatus.PROCESSING
        assert doc.error_message is None


class TestDocumentFromDictContract:
    def test_round_trip_preserves_all_fields(self):
        doc = Document(
            title="t", text_content="body", metadata={"k": "v"}, content_id="c1"
        )
        doc.add_embedding("e", [1.0])
        assert Document.from_dict(doc.to_dict()) == doc

    def test_digit_string_timestamp_coerced(self):
        doc = Document.from_dict({"created_at": "1700000000"})
        assert doc.created_at == 1700000000

    def test_millisecond_timestamp_converted_to_seconds(self):
        doc = Document.from_dict({"created_at": 1700000000000})
        assert doc.created_at == 1700000000

    def test_non_numeric_timestamp_raises_with_field_name(self):
        with pytest.raises(TypeError, match="created_at"):
            Document.from_dict({"created_at": "yesterday"})

    def test_unknown_content_type_raises_with_value(self):
        with pytest.raises(ValueError, match="hologram"):
            Document.from_dict({"content_type": "hologram"})

    def test_unknown_status_raises_with_value(self):
        with pytest.raises(ValueError, match="exploded"):
            Document.from_dict({"status": "exploded"})

    def test_scalar_embeddings_raises(self):
        with pytest.raises(TypeError, match="embeddings"):
            Document.from_dict({"embeddings": "corrupt"})

    def test_null_metadata_and_embeddings_become_empty_dicts(self):
        doc = Document.from_dict({"metadata": None, "embeddings": None})
        assert doc.metadata == {}
        assert doc.embeddings == {}

    def test_auto_detect_still_works(self):
        doc = Document.from_dict({"content_path": "/x/clip.mp4"})
        assert doc.content_type is ContentType.VIDEO


class TestSearchResultToDict:
    def test_numeric_bounds_include_duration(self):
        r = SearchResult(
            Document(metadata={"start_time": 10, "end_time": 25}), score=0.9
        )
        out = r.to_dict()
        assert out["temporal_info"] == {
            "start_time": 10,
            "end_time": 25,
            "duration": 15,
        }
        assert out["score"] == 0.9

    def test_string_timecodes_serialize_without_duration(self):
        r = SearchResult(
            Document(metadata={"start_time": "00:10", "end_time": "00:25"}),
            score=0.9,
        )
        out = r.to_dict()
        assert out["temporal_info"]["start_time"] == "00:10"
        assert out["temporal_info"]["end_time"] == "00:25"
        assert "duration" not in out["temporal_info"]

    def test_source_id_surfaces(self):
        r = SearchResult(Document(metadata={"source_id": "vid_1"}), score=0.5)
        assert r.to_dict()["source_id"] == "vid_1"


class TestConfigEntryFromDictContract:
    def _payload(self):
        return {
            "tenant_id": "acme:acme",
            "scope": "system",
            "service": "svc",
            "config_key": "k",
            "config_value": {"a": 1},
            "version": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        }

    def test_round_trip(self):
        entry = ConfigEntry.from_dict(self._payload())
        assert entry.scope is ConfigScope.SYSTEM
        assert ConfigEntry.from_dict(entry.to_dict()) == entry

    def test_missing_field_named_in_error(self):
        payload = self._payload()
        del payload["tenant_id"]
        with pytest.raises(ValueError, match="tenant_id"):
            ConfigEntry.from_dict(payload)

    def test_bad_scope_named_in_error(self):
        payload = self._payload()
        payload["scope"] = "bogus"
        with pytest.raises(ValueError, match="ConfigEntry.from_dict"):
            ConfigEntry.from_dict(payload)

    def test_bad_datetime_named_in_error(self):
        payload = self._payload()
        payload["created_at"] = "not-a-date"
        with pytest.raises(ValueError, match="ConfigEntry.from_dict"):
            ConfigEntry.from_dict(payload)


class TestWorkflowRecordsTolerateSchemaDrift:
    def test_workflow_execution_ignores_extra_keys(self):
        we = WorkflowExecution(
            workflow_id="w1",
            query="q",
            query_type="search",
            execution_time=1.0,
            success=True,
            agent_sequence=["a"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )
        payload = {**we.to_dict(), "added_by_newer_writer": True}
        assert WorkflowExecution.from_dict(payload) == we

    def test_agent_performance_ignores_extra_keys(self):
        ap = AgentPerformance(agent_name="a")
        payload = {**ap.to_dict(), "future_field": 1}
        assert AgentPerformance.from_dict(payload) == ap

    def test_workflow_template_ignores_extra_keys(self):
        wt = WorkflowTemplate(
            template_id="t1",
            name="n",
            description="d",
            query_patterns=["p"],
            task_sequence=["a"],
            expected_execution_time=1.0,
            success_rate=0.8,
        )
        payload = {**wt.to_dict(), "future_field": "x"}
        assert WorkflowTemplate.from_dict(payload) == wt


class TestDocumentSchemaFieldMapping:
    """The write-side translation layer: a generic Document serializes into
    one schema's declared field names via DocumentFieldMapping. Unmapped
    generic fields are omitted — schemas only ever receive fields they
    declare, which is what makes a generic Document feedable at all."""

    def _mapping(self, **overrides):
        from cogniverse_sdk.document import DocumentFieldMapping

        base = dict(
            id="document_id",
            title="document_title",
            text_content="full_text",
            content_type="document_type",
            created_at="creation_timestamp",
            created_at_format="epoch",
            embeddings={"embedding": "embedding"},
        )
        base.update(overrides)
        return DocumentFieldMapping(**base)

    def test_maps_core_fields_to_schema_names_exactly(self):
        doc = Document(
            id="doc-1",
            content_type=ContentType.TEXT,
            title="My Title",
            text_content="body text",
            created_at=1700000000,
            updated_at=1700000000,
        )
        out = doc.to_schema_fields(self._mapping())
        assert out == {
            "document_id": "doc-1",
            "document_title": "My Title",
            "full_text": "body text",
            "document_type": "text",
            "creation_timestamp": 1700000000,
        }

    def test_unmapped_generic_fields_are_omitted(self):
        doc = Document(
            id="doc-1",
            description="never fed",
            content_id="also never fed",
            title="t",
        )
        out = doc.to_schema_fields(self._mapping(description=None, content_id=None))
        assert "never fed" not in out.values()
        assert "also never fed" not in out.values()

    def test_iso_created_at_format(self):
        doc = Document(id="d", created_at=1700000000, updated_at=1700000000)
        out = doc.to_schema_fields(
            self._mapping(created_at="created_at", created_at_format="iso")
        )
        assert out["created_at"] == "2023-11-14T22:13:20+00:00"

    def test_embeddings_map_wrapped_and_raw(self):
        doc = Document(id="d")
        doc.add_embedding("embedding", [1.0, 2.0])
        doc.embeddings["raw"] = [3.0]
        out = doc.to_schema_fields(
            self._mapping(embeddings={"embedding": "vec_a", "raw": "vec_b"})
        )
        assert out["vec_a"] == [1.0, 2.0]
        assert out["vec_b"] == [3.0]

    def test_metadata_passes_through_and_core_fields_win(self):
        doc = Document(
            id="d",
            title="real title",
            metadata={"page_count": 3, "document_title": "stale metadata copy"},
        )
        out = doc.to_schema_fields(self._mapping())
        assert out["page_count"] == 3
        assert out["document_title"] == "real title"

    def test_include_metadata_false_drops_metadata(self):
        doc = Document(id="d", metadata={"page_count": 3})
        out = doc.to_schema_fields(self._mapping(include_metadata=False))
        assert "page_count" not in out

    def test_mapping_from_dict_rejects_unknown_keys(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="typo_key"):
            DocumentFieldMapping.from_dict({"typo_key": "x"})

    def test_mapping_rejects_bad_created_at_format(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="created_at_format"):
            DocumentFieldMapping(created_at_format="unix")
