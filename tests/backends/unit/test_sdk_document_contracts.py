"""Document / SearchResult / ConfigEntry / workflow-record serialization
contracts.

These pin the sdk's deserialization boundary: corrupt payloads raise with
the offending field named (never a silently mistyped object that detonates
inside the backend later), lifecycle setters mutate exactly the fields
they name, and deserializers accept only their serializers' canonical shape.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

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
    WorkflowStore,
    WorkflowTemplate,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _document_payload(**overrides):
    payload = Document(
        id="doc-1",
        created_at=1700000000,
        updated_at=1700000001,
    ).to_dict()
    payload.update(overrides)
    return payload


def _workflow_execution_payload(**overrides):
    payload = WorkflowExecution(
        workflow_id="w1",
        query="q",
        query_type="search",
        execution_time=1.0,
        success=True,
        agent_sequence=["a"],
        task_count=1,
        parallel_efficiency=1.0,
        confidence_score=0.9,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ).to_dict()
    payload.update(overrides)
    return payload


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

    @pytest.mark.parametrize(
        "value",
        [
            "1700000000",
            1700000000000,
            -1700000000000,
            1700000000.0,
            True,
            float("nan"),
            float("inf"),
        ],
    )
    def test_noncanonical_timestamp_raises_with_field_name(self, value):
        with pytest.raises(TypeError, match="created_at"):
            Document.from_dict(_document_payload(created_at=value))

    def test_unknown_content_type_raises_with_value(self):
        with pytest.raises(ValueError, match="hologram"):
            Document.from_dict(_document_payload(content_type="hologram"))

    def test_unknown_status_raises_with_value(self):
        with pytest.raises(ValueError, match="exploded"):
            Document.from_dict(_document_payload(status="exploded"))

    def test_scalar_embeddings_raises(self):
        with pytest.raises(TypeError, match="embeddings"):
            Document.from_dict(_document_payload(embeddings="corrupt"))

    @pytest.mark.parametrize("field_name", ["metadata", "embeddings"])
    def test_null_mapping_is_rejected(self, field_name):
        with pytest.raises(TypeError, match=field_name):
            Document.from_dict(_document_payload(**{field_name: None}))

    def test_auto_detect_still_works(self):
        doc = Document.from_dict(_document_payload(content_path="/x/clip.mp4"))
        assert doc.content_type is ContentType.VIDEO

    @pytest.mark.parametrize("scalar_type", ["int64", "float64", "bool_"])
    def test_numpy_timestamp_is_rejected(self, scalar_type):
        import numpy as np

        with pytest.raises(TypeError, match="created_at"):
            Document.from_dict(
                _document_payload(created_at=getattr(np, scalar_type)(1))
            )

    def test_auto_detect_covers_every_extension_branch(self):
        cases = {
            "/x/p.png": ContentType.IMAGE,
            "/x/a.wav": ContentType.AUDIO,
            "/x/n.md": ContentType.TEXT,
            "/x/d.csv": ContentType.DATAFRAME,
            "/x/unknown.xyz": ContentType.DOCUMENT,  # default, unchanged
        }
        for path, expected in cases.items():
            assert (
                Document.from_dict(_document_payload(content_path=path)).content_type
                is expected
            )

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError, match="unknown fields.*obsolete_field"):
            Document.from_dict(_document_payload(obsolete_field=True))

    def test_missing_fields_are_rejected(self):
        payload = _document_payload()
        del payload["updated_at"]

        with pytest.raises(ValueError, match="missing fields.*updated_at"):
            Document.from_dict(payload)


class TestDocumentEmbeddingAccess:
    @pytest.mark.parametrize(
        ("embeddings", "match"),
        [
            ({"raw": [1.0, 2.0]}, r"embeddings\['raw'\].*wrapper"),
            ({"missing": {"data": [1.0], "metadata": {}}}, "missing fields"),
            (
                {
                    "extra": {
                        "data": [1.0],
                        "metadata": {},
                        "created_at": 1700000000,
                        "obsolete": True,
                    }
                },
                "unknown fields",
            ),
            (
                {
                    "metadata": {
                        "data": [1.0],
                        "metadata": None,
                        "created_at": 1700000000,
                    }
                },
                r"metadata.*dict",
            ),
            (
                {
                    "timestamp": {
                        "data": [1.0],
                        "metadata": {},
                        "created_at": "1700000000",
                    }
                },
                r"created_at.*integer timestamp",
            ),
            (
                {
                    1: {
                        "data": [1.0],
                        "metadata": {},
                        "created_at": 1700000000,
                    }
                },
                "embedding name.*str",
            ),
        ],
    )
    def test_noncanonical_embedding_wrapper_is_rejected(self, embeddings, match):
        with pytest.raises((TypeError, ValueError), match=match):
            Document(embeddings=embeddings)

    def test_add_embedding_rejects_non_mapping_metadata(self):
        doc = Document()

        with pytest.raises(TypeError, match="metadata.*dict"):
            doc.add_embedding("wrapped", [3.0], [])

    def test_wrapped_embedding_returns_data_and_metadata(self):
        doc = Document()
        doc.add_embedding("wrapped", [3.0], {"model": "clip"})

        assert doc.get_embedding("wrapped") == [3.0]
        assert doc.get_embedding_metadata("wrapped") == {"model": "clip"}


class TestSearchResultToDict:
    @pytest.mark.parametrize("score", [1, True, float("nan"), float("inf")])
    def test_noncanonical_score_is_rejected(self, score):
        with pytest.raises(TypeError, match="score.*finite float"):
            SearchResult(Document(), score=score)

    @pytest.mark.parametrize("highlights", [[], "", 0, False])
    def test_non_mapping_highlights_are_rejected(self, highlights):
        with pytest.raises(TypeError, match="highlights.*dict"):
            SearchResult(Document(), score=0.5, highlights=highlights)

    def test_non_document_is_rejected(self):
        with pytest.raises(TypeError, match="document.*Document"):
            SearchResult({"id": "d"}, score=0.5)

    def test_none_highlights_uses_empty_mapping(self):
        assert SearchResult(Document(), score=0.5, highlights=None).highlights == {}

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

    def test_boolean_bounds_do_not_create_numeric_duration(self):
        result = SearchResult(
            Document(metadata={"start_time": True, "end_time": False}),
            score=0.5,
        ).to_dict()

        assert result["temporal_info"] == {
            "start_time": True,
            "end_time": False,
        }


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

    @pytest.mark.parametrize("value", [None, 1700000000])
    def test_non_string_datetime_names_the_field(self, value):
        payload = self._payload()
        payload["created_at"] = value

        with pytest.raises(ValueError, match="created_at"):
            ConfigEntry.from_dict(payload)

    def test_naive_datetime_is_rejected(self):
        payload = self._payload()
        payload["created_at"] = "2026-01-01T00:00:00"

        with pytest.raises(ValueError, match="created_at.*timezone"):
            ConfigEntry.from_dict(payload)

    def test_non_utc_datetime_is_rejected(self):
        payload = self._payload()
        payload["created_at"] = "2026-01-01T05:30:00+05:30"

        with pytest.raises(ValueError, match="created_at.*canonical UTC"):
            ConfigEntry.from_dict(payload)

    def test_unknown_fields_are_rejected(self):
        payload = self._payload()
        payload["obsolete_field"] = True

        with pytest.raises(ValueError, match="unknown fields.*obsolete_field"):
            ConfigEntry.from_dict(payload)

    def test_version_is_not_defaulted(self):
        payload = self._payload()
        del payload["version"]

        with pytest.raises(ValueError, match="missing fields.*version"):
            ConfigEntry.from_dict(payload)

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("tenant_id", 1, "tenant_id.*str"),
            ("scope", "system", "scope.*ConfigScope"),
            ("service", None, "service.*str"),
            ("config_key", 1, "config_key.*str"),
            ("config_value", [], "config_value.*dict"),
            ("version", True, "version.*positive integer"),
            ("version", 0, "version.*positive integer"),
        ],
    )
    def test_constructor_rejects_noncanonical_fields(self, field_name, value, match):
        kwargs = {
            "tenant_id": "acme:acme",
            "scope": ConfigScope.SYSTEM,
            "service": "svc",
            "config_key": "k",
            "config_value": {"a": 1},
            "version": 1,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        }
        kwargs[field_name] = value

        with pytest.raises((TypeError, ValueError), match=match):
            ConfigEntry(**kwargs)

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("tenant_id", 1, "tenant_id.*str"),
            ("service", None, "service.*str"),
            ("config_key", 1, "config_key.*str"),
            ("config_value", [], "config_value.*dict"),
            ("version", True, "version.*positive integer"),
        ],
    )
    def test_payload_rejects_noncanonical_fields(self, field_name, value, match):
        payload = self._payload()
        payload[field_name] = value

        with pytest.raises(ValueError, match=match):
            ConfigEntry.from_dict(payload)


class TestWorkflowRecordsRequireCanonicalFields:
    def test_workflow_execution_rejects_extra_keys(self):
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
        with pytest.raises(ValueError, match="unknown fields.*added_by_newer_writer"):
            WorkflowExecution.from_dict(payload)

    def test_agent_performance_rejects_extra_keys(self):
        ap = AgentPerformance(agent_name="a")
        payload = {**ap.to_dict(), "future_field": 1}
        with pytest.raises(ValueError, match="unknown fields.*future_field"):
            AgentPerformance.from_dict(payload)

    def test_workflow_template_rejects_extra_keys(self):
        wt = WorkflowTemplate(
            template_id="t1",
            name="n",
            description="d",
            query_patterns=["p"],
            task_sequence=[{"agent": "a"}],
            expected_execution_time=1.0,
            success_rate=0.8,
        )
        payload = {**wt.to_dict(), "future_field": "x"}
        with pytest.raises(ValueError, match="unknown fields.*future_field"):
            WorkflowTemplate.from_dict(payload)

    @pytest.mark.parametrize(
        ("record_type", "payload", "field_name"),
        [
            (
                WorkflowExecution,
                _workflow_execution_payload(),
                "metadata",
            ),
            (
                AgentPerformance,
                AgentPerformance(
                    agent_name="a",
                    last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ).to_dict(),
                "performance_trend",
            ),
            (
                WorkflowTemplate,
                WorkflowTemplate(
                    template_id="t1",
                    name="n",
                    description="d",
                    query_patterns=["p"],
                    task_sequence=[],
                    expected_execution_time=1.0,
                    success_rate=0.8,
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ).to_dict(),
                "usage_count",
            ),
        ],
    )
    def test_missing_fields_are_rejected(self, record_type, payload, field_name):
        del payload[field_name]

        with pytest.raises(ValueError, match=f"missing fields.*{field_name}"):
            record_type.from_dict(payload)


class TestWorkflowRecordValueContract:
    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("workflow_id", 1, "workflow_id.*str"),
            ("query", None, "query.*str"),
            ("query_type", 1, "query_type.*str"),
            ("execution_time", 1, "execution_time.*finite float"),
            ("execution_time", -1.0, "execution_time.*at least 0"),
            ("success", 1, "success.*bool"),
            ("agent_sequence", ["a", 1], "agent_sequence.*list of str"),
            ("task_count", True, "task_count.*non-negative integer"),
            ("parallel_efficiency", 1.1, "parallel_efficiency.*between 0 and 1"),
            ("confidence_score", float("nan"), "confidence_score.*finite float"),
            ("user_satisfaction", 1.1, "user_satisfaction.*between 0 and 1"),
            ("error_details", 1, "error_details.*str or None"),
            ("metadata", [], "metadata.*dict"),
        ],
    )
    def test_workflow_execution_rejects_invalid_values(self, field_name, value, match):
        kwargs = {
            "workflow_id": "w1",
            "query": "q",
            "query_type": "search",
            "execution_time": 1.0,
            "success": True,
            "agent_sequence": ["a"],
            "task_count": 1,
            "parallel_efficiency": 1.0,
            "confidence_score": 0.9,
        }
        kwargs[field_name] = value

        with pytest.raises((TypeError, ValueError), match=match):
            WorkflowExecution(**kwargs)

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("agent_name", 1, "agent_name.*str"),
            ("total_executions", True, "total_executions.*non-negative integer"),
            ("successful_executions", -1, "successful_executions.*non-negative"),
            ("average_execution_time", 1, "average_execution_time.*finite float"),
            ("average_confidence", 1.1, "average_confidence.*between 0 and 1"),
            ("error_rate", float("inf"), "error_rate.*finite float"),
            (
                "preferred_query_types",
                ["search", 1],
                "preferred_query_types.*list of str",
            ),
            ("performance_trend", "unknown", "performance_trend.*improving"),
        ],
    )
    def test_agent_performance_rejects_invalid_values(self, field_name, value, match):
        kwargs = {"agent_name": "agent"}
        kwargs[field_name] = value

        with pytest.raises((TypeError, ValueError), match=match):
            AgentPerformance(**kwargs)

    def test_agent_performance_rejects_success_count_above_total(self):
        with pytest.raises(ValueError, match="successful_executions.*total_executions"):
            AgentPerformance(
                agent_name="agent",
                total_executions=1,
                successful_executions=2,
            )

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("template_id", 1, "template_id.*str"),
            ("name", None, "name.*str"),
            ("description", 1, "description.*str"),
            ("query_patterns", ["p", 1], "query_patterns.*list of str"),
            ("task_sequence", ["agent"], "task_sequence.*list of dict"),
            (
                "expected_execution_time",
                -1.0,
                "expected_execution_time.*at least 0",
            ),
            ("success_rate", 1.1, "success_rate.*between 0 and 1"),
            ("usage_count", True, "usage_count.*non-negative integer"),
        ],
    )
    def test_workflow_template_rejects_invalid_values(self, field_name, value, match):
        kwargs = {
            "template_id": "t1",
            "name": "n",
            "description": "d",
            "query_patterns": ["p"],
            "task_sequence": [{"agent": "a"}],
            "expected_execution_time": 1.0,
            "success_rate": 0.8,
        }
        kwargs[field_name] = value

        with pytest.raises((TypeError, ValueError), match=match):
            WorkflowTemplate(**kwargs)


class TestWorkflowRecordDatetimeContract:
    def test_defaults_are_utc(self):
        execution = WorkflowExecution(
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
        profile = AgentPerformance(agent_name="a")
        template = WorkflowTemplate(
            template_id="t1",
            name="n",
            description="d",
            query_patterns=["p"],
            task_sequence=[],
            expected_execution_time=1.0,
            success_rate=0.8,
        )

        assert execution.timestamp.utcoffset() == timedelta(0)
        assert profile.last_updated.utcoffset() == timedelta(0)
        assert template.created_at.utcoffset() == timedelta(0)

    @pytest.mark.parametrize(
        ("record_type", "payload", "field_name"),
        [
            (
                WorkflowExecution,
                _workflow_execution_payload(timestamp=1700000000),
                "timestamp",
            ),
            (
                AgentPerformance,
                AgentPerformance(
                    agent_name="a",
                    last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ).to_dict()
                | {"last_updated": "2026-01-01T00:00:00"},
                "last_updated",
            ),
            (
                WorkflowTemplate,
                WorkflowTemplate(
                    template_id="t1",
                    name="n",
                    description="d",
                    query_patterns=["p"],
                    task_sequence=[],
                    expected_execution_time=1.0,
                    success_rate=0.8,
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ).to_dict()
                | {"last_used": 1700000000},
                "last_used",
            ),
        ],
    )
    def test_invalid_datetime_names_the_field(self, record_type, payload, field_name):
        with pytest.raises(ValueError, match=field_name):
            record_type.from_dict(payload)

    def test_non_utc_workflow_timestamp_is_rejected(self):
        payload = _workflow_execution_payload(timestamp="2026-01-01T05:30:00+05:30")

        with pytest.raises(ValueError, match="timestamp.*canonical UTC"):
            WorkflowExecution.from_dict(payload)


class _CorpusStore(WorkflowStore):
    def __init__(self):
        self.executions = {}
        self.profiles = {}
        self.patterns = {}
        self.a_profile_written = asyncio.Event()
        self.release_a = asyncio.Event()
        self.b_complete = asyncio.Event()
        self.profile_save_calls = 0
        self.pattern_save_calls = 0
        self.execution_save_calls = 0
        self.fail_forward_execution = False
        self.fail_profile_restore = False
        self.profile_barrier = None

    async def save_executions(self, tenant_id, executions):
        self.execution_save_calls += 1
        if self.fail_forward_execution and self.execution_save_calls == 1:
            raise ConnectionError("forward executions failed")
        self.executions[tenant_id] = list(executions)
        if executions and executions[0].workflow_id == "b":
            self.b_complete.set()

    async def load_executions(self, tenant_id):
        return list(self.executions.get(tenant_id, []))

    async def save_agent_profiles(self, tenant_id, profiles):
        self.profile_save_calls += 1
        if self.fail_profile_restore and self.profile_save_calls == 2:
            raise RuntimeError("profile restore failed")
        self.profiles[tenant_id] = list(profiles)
        if self.profile_barrier is not None:
            await self.profile_barrier.wait()
        if profiles and profiles[0].agent_name == "agent_a":
            self.a_profile_written.set()
            await self.release_a.wait()

    async def load_agent_profiles(self, tenant_id):
        return list(self.profiles.get(tenant_id, []))

    async def save_query_patterns(self, tenant_id, patterns):
        self.pattern_save_calls += 1
        self.patterns[tenant_id] = dict(patterns)

    async def load_query_patterns(self, tenant_id):
        return dict(self.patterns.get(tenant_id, {}))

    async def save_template(self, tenant_id, template):
        raise NotImplementedError

    async def load_templates(self, tenant_id):
        raise NotImplementedError

    async def delete_template(self, tenant_id, template_id):
        raise NotImplementedError

    def health_check(self):
        return True

    def get_stats(self):
        return {}


def _corpus_execution(workflow_id):
    return WorkflowExecution(
        workflow_id=workflow_id,
        query=workflow_id,
        query_type="search",
        execution_time=1.0,
        success=True,
        agent_sequence=[f"agent_{workflow_id}"],
        task_count=1,
        parallel_efficiency=1.0,
        confidence_score=0.9,
    )


class TestWorkflowLearningCorpusContract:
    async def test_empty_patterns_replace_stale_patterns(self):
        store = _CorpusStore()
        store.patterns["acme:prod"] = {"stale": ["old"]}

        await store.save_learning_corpus(
            "acme:prod",
            [_corpus_execution("new")],
            [AgentPerformance(agent_name="agent_new")],
            {},
        )

        assert store.patterns["acme:prod"] == {}

    async def test_same_tenant_concurrent_saves_remain_coherent(self):
        store = _CorpusStore()

        save_a = asyncio.create_task(
            store.save_learning_corpus(
                "acme:prod",
                [_corpus_execution("a")],
                [AgentPerformance(agent_name="agent_a")],
                {"a": ["pattern_a"]},
            )
        )
        await asyncio.wait_for(store.a_profile_written.wait(), timeout=1)

        save_b = asyncio.create_task(
            store.save_learning_corpus(
                "acme:prod",
                [_corpus_execution("b")],
                [AgentPerformance(agent_name="agent_b")],
                {"b": ["pattern_b"]},
            )
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(store.b_complete.wait(), timeout=0.2)
        store.release_a.set()
        await asyncio.gather(save_a, save_b)

        assert [item.workflow_id for item in store.executions["acme:prod"]] == ["b"]
        assert [item.agent_name for item in store.profiles["acme:prod"]] == ["agent_b"]
        assert store.patterns["acme:prod"] == {"b": ["pattern_b"]}

    async def test_different_tenants_save_independently(self):
        store = _CorpusStore()
        store.profile_barrier = asyncio.Barrier(2)

        await asyncio.wait_for(
            asyncio.gather(
                store.save_learning_corpus(
                    "acme:a",
                    [_corpus_execution("a")],
                    [AgentPerformance(agent_name="tenant_a")],
                    {"a": ["pattern_a"]},
                ),
                store.save_learning_corpus(
                    "acme:b",
                    [_corpus_execution("b")],
                    [AgentPerformance(agent_name="tenant_b")],
                    {"b": ["pattern_b"]},
                ),
            ),
            timeout=1,
        )

        assert [item.agent_name for item in store.profiles["acme:a"]] == ["tenant_a"]
        assert [item.agent_name for item in store.profiles["acme:b"]] == ["tenant_b"]

    async def test_restore_failure_surfaces_both_errors_and_attempts_all_restores(self):
        store = _CorpusStore()
        tenant = "acme:prod"
        store.executions[tenant] = [_corpus_execution("old")]
        store.profiles[tenant] = [AgentPerformance(agent_name="agent_old")]
        store.patterns[tenant] = {"old": ["pattern_old"]}
        store.fail_forward_execution = True
        store.fail_profile_restore = True

        with pytest.raises(ExceptionGroup) as exc_info:
            await store.save_learning_corpus(
                tenant,
                [_corpus_execution("new")],
                [AgentPerformance(agent_name="agent_new")],
                {"new": ["pattern_new"]},
            )

        assert [type(error) for error in exc_info.value.exceptions] == [
            ConnectionError,
            RuntimeError,
        ]
        assert [str(error) for error in exc_info.value.exceptions] == [
            "forward executions failed",
            "profile restore failed",
        ]
        assert store.profile_save_calls == 2
        assert store.pattern_save_calls == 2
        assert store.execution_save_calls == 2
        assert store.patterns[tenant] == {"old": ["pattern_old"]}
        assert [item.workflow_id for item in store.executions[tenant]] == ["old"]


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

    def test_embedding_wrapper_maps_exact_vector(self):
        doc = Document(id="d")
        doc.add_embedding("embedding", [1.0, 2.0])
        out = doc.to_schema_fields(self._mapping(embeddings={"embedding": "vec"}))

        assert out["vec"] == [1.0, 2.0]

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

    @pytest.mark.parametrize(
        ("payload", "field_name"),
        [
            ({"id": 42}, "id"),
            ({"include_metadata": "false"}, "include_metadata"),
        ],
    )
    def test_mapping_from_dict_rejects_mistyped_scalar_fields(
        self, payload, field_name
    ):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(TypeError, match=field_name):
            DocumentFieldMapping.from_dict(payload)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"id": 42}, "id.*str or None"),
            ({"include_metadata": "false"}, "include_metadata.*bool"),
            ({"metadata_fields": []}, "metadata_fields.*dict"),
            ({"metadata_fields": {"source": 5}}, "metadata_fields.*str to str"),
            ({"embeddings": []}, "embeddings.*dict"),
            ({"embeddings": {"embedding": 123}}, "embeddings.*str to str"),
        ],
    )
    def test_mapping_constructor_rejects_noncanonical_fields(self, kwargs, match):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises((TypeError, ValueError), match=match):
            DocumentFieldMapping(**kwargs)

    def test_mapping_mutation_is_rejected_at_serialization(self):
        mapping = self._mapping()
        mapping.id = 42

        with pytest.raises(TypeError, match="id.*str or None"):
            Document(id="d").to_schema_fields(mapping)

    def test_non_mapping_object_is_rejected_at_serialization(self):
        with pytest.raises(TypeError, match="mapping.*DocumentFieldMapping"):
            Document(id="d").to_schema_fields({})

    def test_mapping_from_schema_json_rejects_non_dict_schema(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(TypeError, match="schema_json"):
            DocumentFieldMapping.from_schema_json(["not-a-schema"])

    def test_mapping_rejects_bad_created_at_format(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="created_at_format"):
            DocumentFieldMapping(created_at_format="unix")

    def test_mapping_from_dict_rejects_list_embeddings(self):
        """A list-typed embeddings block fails at load with a clear message,
        not later at ``mapping.embeddings.items()`` during the feed."""
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="embeddings must be a dict"):
            DocumentFieldMapping.from_dict({"embeddings": ["embedding"]})

    def test_mapping_from_dict_rejects_non_str_embedding_values(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="str to str"):
            DocumentFieldMapping.from_dict({"embeddings": {"embedding": 123}})

    def test_content_type_as_raw_string_is_rejected(self):
        doc = Document(id="d", title="t")
        doc.content_type = "video"
        with pytest.raises(TypeError, match="content_type.*ContentType"):
            doc.to_schema_fields(self._mapping())

    @pytest.mark.parametrize(
        ("field_name", "value", "expected_type"),
        [
            ("content_type", "video", "ContentType"),
            ("status", "pending", "ProcessingStatus"),
            ("metadata", None, "dict"),
            ("embeddings", None, "dict"),
        ],
    )
    def test_constructor_rejects_noncanonical_fields(
        self, field_name, value, expected_type
    ):
        with pytest.raises(TypeError, match=f"{field_name}.*{expected_type}"):
            Document(**{field_name: value})

    def test_float_created_at_is_rejected(self):
        with pytest.raises(TypeError, match="created_at.*integer timestamp"):
            Document(id="d", created_at=1700000000.75, updated_at=1700000000.75)

    def test_epoch_ms_format_multiplies_to_milliseconds(self):
        """creation_timestamp is a millisecond field on video/audio schemas;
        epoch_ms scales the seconds epoch so it lands as ms, not 1970."""
        doc = Document(id="d", created_at=1700000000, updated_at=1700000000)
        out = doc.to_schema_fields(self._mapping(created_at_format="epoch_ms"))
        assert out["creation_timestamp"] == 1700000000000
        assert type(out["creation_timestamp"]) is int

    def test_metadata_fields_rename_to_schema_field(self):
        """A value carried in metadata is renamed to its schema field name."""
        doc = Document(id="d")
        doc.metadata = {"segment_index": 3}
        out = doc.to_schema_fields(
            self._mapping(
                id=None,
                title=None,
                text_content=None,
                content_type=None,
                created_at=None,
                embeddings={},
                metadata_fields={"segment_index": "segment_id"},
            )
        )
        assert out == {"segment_id": 3}

    def test_include_metadata_false_feeds_only_declared_renames(self):
        """With include_metadata off, only the explicitly renamed metadata keys
        reach the output — no blanket passthrough of unknown metadata keys."""
        doc = Document(id="d")
        doc.metadata = {"segment_index": 3, "junk_key": "should not feed"}
        out = doc.to_schema_fields(
            self._mapping(
                include_metadata=False,
                metadata_fields={"segment_index": "segment_id"},
            )
        )
        assert out["segment_id"] == 3
        assert "junk_key" not in out
        assert "segment_index" not in out

    def test_metadata_fields_from_dict_rejects_non_str_values(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        with pytest.raises(ValueError, match="metadata_fields must map str to str"):
            DocumentFieldMapping.from_dict({"metadata_fields": {"segment_index": 5}})


class TestMappingPathAndUpdatedAt:
    def test_content_path_and_updated_at_map(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        doc = Document(
            id="d",
            content_path="/data/report.txt",
            created_at=1700000000,
            updated_at=1700000600,
        )
        out = doc.to_schema_fields(
            DocumentFieldMapping(
                id="doc_id",
                content_path="file_path",
                created_at="created_at",
                updated_at="updated_at",
                created_at_format="iso",
            )
        )
        assert out["file_path"] == "/data/report.txt"
        assert out["created_at"] == "2023-11-14T22:13:20+00:00"
        assert out["updated_at"] == "2023-11-14T22:23:20+00:00"


class TestDeclaredSchemaMappings:
    """Every document_mapping block in configs/schemas must parse and refer
    only to fields its schema actually declares, with timestamp formats
    matching the field types — a typo here means feeds 400 at runtime."""

    def _schemas_with_mappings(self):
        import json
        from pathlib import Path

        for path in sorted(Path("configs/schemas").glob("*_schema.json")):
            data = json.loads(path.read_text())
            if "document_mapping" in data:
                yield path.name, data

    def _field_types(self, schema_json):
        types = {}

        def walk(node):
            if isinstance(node, dict):
                name = node.get("name")
                if name and "type" in node and not str(name).startswith("query("):
                    types[name] = node["type"]
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(schema_json.get("document", {}))
        return types

    def test_all_declared_mappings_are_valid(self):
        from cogniverse_sdk.document import DocumentFieldMapping

        checked = 0
        for name, schema_json in self._schemas_with_mappings():
            mapping = DocumentFieldMapping.from_dict(schema_json["document_mapping"])
            types = self._field_types(schema_json)

            targets = {
                key: getattr(mapping, key)
                for key in (
                    "id",
                    "title",
                    "text_content",
                    "description",
                    "content_type",
                    "content_id",
                    "content_path",
                    "created_at",
                    "updated_at",
                )
                if getattr(mapping, key)
            }
            for generic, target in targets.items():
                assert target in types, f"{name}: {generic} -> {target} not in schema"
            for target in mapping.embeddings.values():
                assert target in types, f"{name}: embedding target {target} missing"

            for stamp_field in ("created_at", "updated_at"):
                target = getattr(mapping, stamp_field)
                if not target:
                    continue
                if mapping.created_at_format == "epoch":
                    assert types[target] in ("long", "int"), (
                        f"{name}: {target} is {types[target]}, epoch needs long/int"
                    )
                else:
                    assert types[target] == "string", (
                        f"{name}: {target} is {types[target]}, iso needs string"
                    )
            checked += 1

        assert checked >= 13, f"expected all declared mappings, found {checked}"
