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

    def test_numpy_int_timestamp_coerced(self):
        import numpy as np

        doc = Document.from_dict({"created_at": np.int64(1700000000)})
        assert doc.created_at == 1700000000
        assert type(doc.created_at) is int

    def test_numpy_float_timestamp_coerced(self):
        import numpy as np

        doc = Document.from_dict({"created_at": np.float64(1700000000.0)})
        assert doc.created_at == 1700000000
        assert type(doc.created_at) is int

    def test_numpy_bool_timestamp_rejected(self):
        import numpy as np

        with pytest.raises(TypeError, match="created_at"):
            Document.from_dict({"created_at": np.bool_(True)})

    def test_python_bool_timestamp_rejected(self):
        with pytest.raises(TypeError, match="created_at"):
            Document.from_dict({"created_at": True})

    def test_auto_detect_covers_every_extension_branch(self):
        cases = {
            "/x/p.png": ContentType.IMAGE,
            "/x/a.wav": ContentType.AUDIO,
            "/x/n.md": ContentType.TEXT,
            "/x/d.csv": ContentType.DATAFRAME,
            "/x/unknown.xyz": ContentType.DOCUMENT,  # default, unchanged
        }
        for path, expected in cases.items():
            assert Document.from_dict({"content_path": path}).content_type is expected


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

    def test_content_type_as_raw_string_is_coerced(self):
        """A Document whose content_type was set to a raw string still
        serializes (coerced through the enum) instead of raising AttributeError."""
        doc = Document(id="d", title="t")
        doc.content_type = "video"  # raw string, not the enum
        out = doc.to_schema_fields(self._mapping())
        assert out["document_type"] == "video"

    def test_content_type_garbage_string_raises_valueerror(self):
        doc = Document(id="d", title="t")
        doc.content_type = "hologram"
        with pytest.raises(ValueError, match="hologram"):
            doc.to_schema_fields(self._mapping())

    def test_float_created_at_truncated_to_int_in_epoch(self):
        """A float epoch must land as an int in the schema's long field."""
        doc = Document(id="d", created_at=1700000000.75, updated_at=1700000000.75)
        out = doc.to_schema_fields(self._mapping())
        assert out["creation_timestamp"] == 1700000000
        assert type(out["creation_timestamp"]) is int

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
            self._mapping(metadata_fields={"segment_index": "segment_id"})
        )
        assert out["segment_id"] == 3

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
