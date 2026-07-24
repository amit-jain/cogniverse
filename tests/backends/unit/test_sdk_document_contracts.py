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
