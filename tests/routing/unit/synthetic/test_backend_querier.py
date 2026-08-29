"""BackendQuerier._query_profile grounds samples in real backend content."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import pytest

import cogniverse_synthetic.backend_querier as backend_querier_module
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
    FieldMappingConfig,
)
from cogniverse_sdk.document import DocumentFieldMapping
from cogniverse_synthetic.backend_querier import BackendQuerier

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCHEMAS_DIR = _REPO_ROOT / "configs" / "schemas"
_SHIPPED_BACKEND_PROFILES = json.loads(
    (_REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8")
)["backend"]["profiles"]


def _shipped_backend_profile(profile_name: str) -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(
        profile_name, _SHIPPED_BACKEND_PROFILES[profile_name]
    )


def _shipped_backend_profile_dict(profile_name: str) -> dict:
    profile = _shipped_backend_profile(profile_name).to_dict()
    profile["profile_name"] = profile_name
    return profile


class _RecordingBackend:
    """Same signature as the real ``Backend.query_metadata_documents``."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.calls: list[dict] = []
        self.schema_resolutions: list[tuple[str, str]] = []

    def get_tenant_schema_name(self, tenant_id, base_schema_name):
        self.schema_resolutions.append((tenant_id, base_schema_name))
        return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.calls.append({"schema": schema, "yql": yql, "kwargs": kwargs})
        return self.docs


class _SchemaRecordingBackend:
    """Record per-schema calls and return schema-specific documents."""

    def __init__(self, docs_by_schema: dict[str, list[dict]]) -> None:
        self.docs_by_schema = docs_by_schema
        self.calls: list[dict] = []

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.calls.append({"schema": schema, "yql": yql, "kwargs": kwargs})
        return self.docs_by_schema[schema]


def _querier(backend, *, profiles=None) -> BackendQuerier:
    return BackendQuerier(
        backend=backend,
        backend_config=BackendConfig(
            profiles=profiles or {},
            tenant_id="test:unit",
        ),
        field_mappings=FieldMappingConfig(),
    )


def test_backend_querier_requires_a_real_backend() -> None:
    with pytest.raises(ValueError, match="^backend is required$"):
        _querier(None)


def test_empty_field_mapping_config_hydrates_canonical_defaults() -> None:
    hydrated = FieldMappingConfig.from_dict({})

    assert hydrated.to_dict() == FieldMappingConfig().to_dict()
    assert hydrated.topic_fields == [
        "video_title",
        "audio_title",
        "image_title",
        "document_title",
        "chunk_name",
        "title",
    ]
    assert hydrated.description_fields == [
        "segment_description",
        "image_description",
        "full_text",
        "source_code",
        "content",
        "description",
    ]


def test_field_mapping_config_rejects_unknown_fields() -> None:
    with pytest.raises(
        ValueError,
        match="^FieldMappingConfig contains unsupported fields: legacy_topics$",
    ):
        FieldMappingConfig.from_dict({"legacy_topics": ["title"]})


def test_default_field_mappings_preserve_supported_modality_content() -> None:
    querier = _querier(_RecordingBackend([]))

    samples = querier._extract_fields_from_results(
        [
            {
                "audio_title": "Apollo mission control",
                "audio_transcript": "The Eagle has landed.",
            },
            {
                "image_title": "Saturn V launch",
                "image_description": "The rocket clears the launch tower.",
            },
            {
                "document_title": "Apollo 11 flight plan",
                "full_text": "The plan specifies the lunar landing sequence.",
            },
        ],
        {
            "profile_name": "configured_profile",
            "schema_name": "configured_profile",
            "embedding_type": "multi_vector",
            "type": "document",
        },
    )

    assert [sample.get("topic") for sample in samples] == [
        "Apollo mission control",
        "Saturn V launch",
        "Apollo 11 flight plan",
    ]
    assert [sample.get("description") for sample in samples] == [
        None,
        "The rocket clears the launch tower.",
        "The plan specifies the lunar landing sequence.",
    ]
    assert [sample.get("transcript") for sample in samples] == [
        "The Eagle has landed.",
        None,
        None,
    ]
    assert {(sample["profile_type"], sample["modality"]) for sample in samples} == {
        ("document", "DOCUMENT")
    }


def test_backend_samples_propagate_exact_keyword_free_profile_modality() -> None:
    samples = _querier(_RecordingBackend([]))._extract_fields_from_results(
        [{"title": "Alpha", "description": "opaque corpus item"}],
        {
            "profile_name": "opaque_document",
            "schema_name": "alpha",
            "embedding_type": "dense",
            "type": "document",
        },
    )

    assert samples == [
        {
            "topic": "Alpha",
            "description": "opaque corpus item",
            "start_time": 0.0,
            "end_time": 0.0,
            "video_id": "",
            "source_id": "",
            "segment_id": 0,
            "creation_timestamp": None,
            "schema_name": "alpha",
            "profile_name": "opaque_document",
            "embedding_type": "dense",
            "profile_type": "document",
            "modality": "DOCUMENT",
            "profile_metadata": {
                "schema_name": "alpha",
                "embedding_model": None,
                "embedding_type": "dense",
                "type": "document",
            },
        }
    ]


def test_source_identity_ignores_fields_the_schema_does_not_declare() -> None:
    """The sampled source id comes from the schema's declared
    document_mapping.id and nothing else. A guessed fallback list would pick
    a plausible-looking field from another schema's vocabulary and emit a
    confident, wrong identity - and it would silently mask a schema that
    declares no id at all."""
    document = {
        "code_id": "the_declared_one",
        "video_id": "a_different_schemas_field",
        "document_id": "another_one",
        "source_id": "and_another",
        "id": "and_another_still",
    }
    assert (
        backend_querier_module.BackendQuerier._source_identity_value(
            document, "code_id"
        )
        == "the_declared_one"
    )
    assert (
        backend_querier_module.BackendQuerier._source_identity_value(document, None)
        == ""
    )
    assert (
        backend_querier_module.BackendQuerier._source_identity_value(
            {"video_id": "present"}, "code_id"
        )
        == ""
    )


def test_schema_source_identity_field_matches_document_mapping_id() -> None:
    for schema_path in sorted(_SCHEMAS_DIR.glob("*_schema.json")):
        schema_name = schema_path.stem.removesuffix("_schema")
        schema_json = json.loads(schema_path.read_text(encoding="utf-8"))
        mapping = DocumentFieldMapping.from_schema_json(
            schema_json, schema_name=schema_name, required=False
        )
        expected_source_id = None if mapping is None else mapping.id

        assert (
            backend_querier_module._schema_source_identity_field(schema_name)
            == expected_source_id
        )


@pytest.mark.asyncio
async def test_diverse_sampling_round_robins_blank_topics_by_schema_identity() -> None:
    backend = _RecordingBackend(
        [
            {
                "code_id": "source-a",
                "segment_id": 1,
                "chunk_name": "",
                "source_code": "",
            },
            {
                "code_id": "source-a",
                "segment_id": 2,
                "chunk_name": "",
                "source_code": "",
            },
            {
                "code_id": "source-b",
                "segment_id": 3,
                "chunk_name": "",
                "source_code": "",
            },
        ]
    )
    querier = _querier(backend)

    samples = await querier._query_profile(
        {
            "profile_name": "code_lateon_mv",
            "schema_name": "code_lateon_mv",
            "type": "code",
        },
        sample_size=3,
        strategy="diverse",
        tenant_id="acme:media",
    )

    assert [sample["source_id"] for sample in samples] == [
        "source-a",
        "source-b",
        "source-a",
    ]
    assert [sample["segment_id"] for sample in samples] == [1, 3, 2]


async def test_query_profile_grounds_samples_without_duplicate_kwarg() -> None:
    backend = _RecordingBackend(
        [{"title": "Robots", "description": "bots play soccer"}]
    )
    querier = _querier(backend)

    samples = await querier._query_profile(
        {
            "profile_name": "video_frame",
            "schema_name": "video_frame",
            "type": "video",
        },
        sample_size=5,
        strategy="diverse",
        tenant_id="acme:media",
    )

    assert len(samples) == 1
    assert samples[0]["topic"] == "Robots"
    assert samples[0]["description"] == "bots play soccer"

    call = backend.calls[0]
    assert call["schema"] == "video_frame"
    assert call["yql"] == "select * from sources video_frame where true limit 25"
    assert "yql" not in call["kwargs"], "yql must not be duplicated into kwargs"
    assert call["kwargs"] == {"hits": 25, "tenant_id": "acme:media"}
    assert backend.schema_resolutions == []


async def test_temporal_recent_strategy_builds_exact_cutoff_and_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cogniverse_synthetic.backend_querier.time.time",
        lambda: 2_000_000_000,
    )
    backend = _RecordingBackend([{"title": "T"}])
    querier = _querier(backend)

    await querier._query_profile(
        {
            "profile_name": "video_frame",
            "schema_name": "video_frame",
            "type": "video",
        },
        sample_size=3,
        strategy="temporal_recent",
        tenant_id="acme:media",
    )

    assert backend.calls[0]["yql"] == (
        "select * from sources video_frame "
        "where creation_timestamp >= 1992224000000 "
        "order by creation_timestamp desc limit 3"
    )
    assert backend.calls[0]["kwargs"] == {
        "hits": 3,
        "tenant_id": "acme:media",
    }


async def test_entity_rich_uses_only_text_fields_produced_by_profile() -> None:
    backend = _RecordingBackend(
        [
            {"audio_transcript": ""},
            {"audio_transcript": "Curie discovered radium"},
        ]
    )

    samples = await _querier(backend)._query_profile(
        {
            "profile_name": "audio_content",
            "schema_name": "audio_content",
            "type": "audio",
            "pipeline_config": {
                "generate_descriptions": False,
                "transcribe_audio": True,
            },
        },
        sample_size=4,
        strategy="entity_rich",
        tenant_id="acme:media",
    )

    assert [sample["transcript"] for sample in samples] == ["Curie discovered radium"]
    assert backend.calls[0]["yql"] == (
        "select * from sources audio_content where true "
        "order by creation_timestamp desc limit 10"
    )
    assert backend.calls[0]["kwargs"] == {"hits": 10, "tenant_id": "acme:media"}


async def test_entity_rich_pages_past_blank_rows_to_fill_requested_sample() -> None:
    class _PagedBackend(_RecordingBackend):
        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            self.calls.append({"schema": schema, "yql": yql, "kwargs": kwargs})
            offset = kwargs.get("offset", 0)
            return self.docs[offset : offset + kwargs["hits"]]

    backend = _PagedBackend(
        [{"audio_id": f"blank-{index}", "audio_transcript": ""} for index in range(10)]
        + [
            {
                "audio_id": "grounded-result",
                "audio_transcript": "Marie Curie discovered radium",
            }
        ]
    )

    samples = await _querier(backend)._query_profile(
        {
            "profile_name": "audio_content",
            "schema_name": "audio_content",
            "type": "audio",
            "pipeline_config": {
                "generate_descriptions": False,
                "transcribe_audio": True,
            },
        },
        sample_size=1,
        strategy="entity_rich",
        tenant_id="acme:media",
    )

    assert [sample["transcript"] for sample in samples] == [
        "Marie Curie discovered radium"
    ]
    assert [call["kwargs"] for call in backend.calls] == [
        {"hits": 10, "tenant_id": "acme:media"},
        {"hits": 10, "tenant_id": "acme:media", "offset": 10},
    ]
    assert [call["yql"] for call in backend.calls] == [
        "select * from sources audio_content where true "
        "order by creation_timestamp desc limit 10",
        "select * from sources audio_content where true "
        "order by creation_timestamp desc limit 20 offset 10",
    ]


async def test_entity_rich_rejects_a_repeated_backend_page() -> None:
    backend = _RecordingBackend(
        [{"audio_id": f"blank-{index}", "audio_transcript": ""} for index in range(10)]
    )

    with pytest.raises(
        RuntimeError,
        match=("^Vespa repeated the entity-rich page for audio_content at offset 10$"),
    ):
        await _querier(backend)._query_profile(
            {
                "profile_name": "audio_content",
                "schema_name": "audio_content",
                "type": "audio",
                "pipeline_config": {
                    "generate_descriptions": False,
                    "transcribe_audio": True,
                },
            },
            sample_size=1,
            strategy="entity_rich",
            tenant_id="acme:media",
        )


async def test_concurrent_entity_rich_queries_keep_profile_fields_isolated() -> None:
    backend = _RecordingBackend(
        [
            {
                "title": "audio-rich",
                "audio_transcript": "spoken facts",
                "segment_description": "",
            },
            {
                "title": "video-rich",
                "audio_transcript": "",
                "segment_description": "visible entities",
            },
        ]
    )
    querier = _querier(backend)
    requests = [
        (
            f"tenant_{index}:media",
            {
                "profile_name": "audio_content",
                "schema_name": "audio_content",
                "type": "audio",
                "pipeline_config": {
                    "generate_descriptions": False,
                    "transcribe_audio": True,
                },
            }
            if index % 2 == 0
            else {
                "profile_name": "video_frame",
                "schema_name": "video_frame",
                "type": "video",
                "pipeline_config": {
                    "generate_descriptions": True,
                    "transcribe_audio": False,
                },
            },
        )
        for index in range(8)
    ]

    results = await asyncio.gather(
        *[
            querier._query_profile(
                profile_config,
                sample_size=2,
                strategy="entity_rich",
                tenant_id=tenant_id,
            )
            for tenant_id, profile_config in requests
        ]
    )

    for (tenant_id, profile_config), samples in zip(requests, results, strict=True):
        expected_topic = (
            "audio-rich"
            if profile_config["schema_name"] == "audio_content"
            else "video-rich"
        )
        assert [sample["topic"] for sample in samples] == [expected_topic]
        schema = profile_config["schema_name"]
        assert any(
            call["schema"] == schema
            and call["yql"]
            == (
                f"select * from sources {schema} where true "
                "order by creation_timestamp desc limit 10"
            )
            and call["kwargs"] == {"hits": 10, "tenant_id": tenant_id}
            for call in backend.calls
        )


async def test_entity_rich_rejects_missing_required_transcript_mapping() -> None:
    backend = _RecordingBackend([])
    querier = BackendQuerier(
        backend=backend,
        backend_config=BackendConfig(profiles={}, tenant_id="test:unit"),
        field_mappings=FieldMappingConfig(transcript_fields=[]),
    )

    with pytest.raises(ValueError, match="requires a transcript field"):
        await querier._query_profile(
            {
                "schema_name": "audio_content",
                "pipeline_config": {"transcribe_audio": True},
            },
            sample_size=2,
            strategy="entity_rich",
            tenant_id="acme:media",
        )

    assert backend.calls == []


async def test_entity_rich_rejects_profile_without_text_producing_pipeline() -> None:
    backend = _RecordingBackend([{"title": "unfiltered result"}])

    with pytest.raises(
        ValueError,
        match=(
            "^entity_rich requires the profile pipeline to generate descriptions "
            "or transcribe audio$"
        ),
    ):
        await _querier(backend)._query_profile(
            {
                "schema_name": "video_frame",
                "pipeline_config": {
                    "generate_descriptions": False,
                    "transcribe_audio": False,
                },
            },
            sample_size=1,
            strategy="entity_rich",
            tenant_id="acme:media",
        )

    assert backend.schema_resolutions == []
    assert backend.calls == []


@pytest.mark.asyncio
async def test_entity_rich_query_profiles_skips_non_qualifying_shipped_profiles(
    caplog,
) -> None:
    backend = _SchemaRecordingBackend(
        {
            "video_colpali_smol500_mv_frame": [
                {
                    "video_title": "Saturn V launch",
                    "segment_description": (
                        "The Saturn V rocket clears the launch tower."
                    ),
                    "audio_transcript": "Saturn V ignition is confirmed.",
                }
            ],
            "audio_content": [
                {
                    "audio_title": "Curie lecture",
                    "audio_transcript": (
                        "Marie Curie and Pierre Curie discovered radium."
                    ),
                }
            ],
        }
    )
    querier = _querier(
        backend,
        profiles={
            name: _shipped_backend_profile_dict(name)
            for name in (
                "video_videoprism_large_mv_chunk_30s",
                "video_colpali_smol500_mv_frame",
                "audio_clap_semantic",
            )
        },
    )

    with caplog.at_level(logging.WARNING):
        samples = await querier.query_profiles(
            [
                _shipped_backend_profile_dict("video_videoprism_large_mv_chunk_30s"),
                _shipped_backend_profile_dict("video_colpali_smol500_mv_frame"),
                _shipped_backend_profile_dict("audio_clap_semantic"),
            ],
            sample_size=2,
            strategy="entity_rich",
            tenant_id="acme:media",
        )

    assert [sample["profile_name"] for sample in samples] == [
        "video_colpali_smol500_mv_frame",
        "audio_clap_semantic",
    ]
    assert [sample["schema_name"] for sample in samples] == [
        "video_colpali_smol500_mv_frame",
        "audio_content",
    ]
    assert [sample["topic"] for sample in samples] == [
        "Saturn V launch",
        "Curie lecture",
    ]
    assert [sample.get("description") for sample in samples] == [
        "The Saturn V rocket clears the launch tower.",
        None,
    ]
    assert [sample["transcript"] for sample in samples] == [
        "Saturn V ignition is confirmed.",
        "Marie Curie and Pierre Curie discovered radium.",
    ]
    assert backend.calls == [
        {
            "schema": "video_colpali_smol500_mv_frame",
            "yql": (
                "select * from sources video_colpali_smol500_mv_frame "
                "where true order by creation_timestamp desc limit 10"
            ),
            "kwargs": {"hits": 10, "tenant_id": "acme:media"},
        },
        {
            "schema": "audio_content",
            "yql": (
                "select * from sources audio_content "
                "where true order by creation_timestamp desc limit 10"
            ),
            "kwargs": {"hits": 10, "tenant_id": "acme:media"},
        },
    ]
    assert [
        record.message for record in caplog.records if record.levelno == logging.WARNING
    ] == [
        (
            "entity_rich skips non-qualifying backend profiles for tenant "
            "'acme:media': video_videoprism_large_mv_chunk_30s (entity_rich "
            "requires the profile pipeline to generate descriptions or "
            "transcribe audio)"
        )
    ]


@pytest.mark.asyncio
async def test_entity_rich_query_profiles_raises_when_no_qualifying_profiles_remain(
    caplog,
) -> None:
    backend = _SchemaRecordingBackend({})
    querier = _querier(
        backend,
        profiles={
            name: _shipped_backend_profile_dict(name)
            for name in (
                "video_videoprism_large_mv_chunk_30s",
                "video_videoprism_base_mv_chunk_30s",
            )
        },
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            ValueError,
            match=(
                "^entity_rich requires at least one qualifying backend profile "
                "for tenant 'acme:media'; excluded profiles: "
                "video_videoprism_large_mv_chunk_30s \\("
                "entity_rich requires the profile pipeline to generate "
                "descriptions or transcribe audio\\), "
                "video_videoprism_base_mv_chunk_30s \\("
                "entity_rich requires the profile pipeline to generate "
                "descriptions or transcribe audio\\)$"
            ),
        ):
            await querier.query_profiles(
                [
                    _shipped_backend_profile_dict(
                        "video_videoprism_large_mv_chunk_30s"
                    ),
                    _shipped_backend_profile_dict("video_videoprism_base_mv_chunk_30s"),
                ],
                sample_size=1,
                strategy="entity_rich",
                tenant_id="acme:media",
            )

    assert backend.calls == []
    assert caplog.records == []


async def test_query_profile_raises_on_backend_runtime_error() -> None:
    """A backend outage propagates — flattening it to [] reads as "no
    matching documents" and silently produces empty synthetic datasets."""

    class _Boom(_RecordingBackend):
        def __init__(self):
            super().__init__([])

        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            raise RuntimeError("vespa unreachable")

    with pytest.raises(RuntimeError, match="vespa unreachable"):
        await _querier(_Boom())._query_profile(
            {"profile_name": "s", "schema_name": "s", "type": "video"},
            5,
            "diverse",
            tenant_id="acme:media",
        )


async def test_query_profile_propagates_signature_typeerror() -> None:
    """A real argument-mismatch (programming bug) must surface, not be masked
    as an empty result."""

    class _BadSignature(_RecordingBackend):
        def __init__(self):
            super().__init__([])

        def query_metadata_documents(self, schema, yql=None):
            return []

    with pytest.raises(TypeError):
        await _querier(_BadSignature())._query_profile(
            {"profile_name": "s", "schema_name": "s", "type": "video"},
            5,
            "diverse",
            tenant_id="acme:media",
        )


async def test_query_profile_propagates_schema_resolution_failure() -> None:
    class _BrokenResolver(_RecordingBackend):
        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            raise RuntimeError("tenant schema registry unavailable")

    with pytest.raises(RuntimeError, match="tenant schema registry unavailable"):
        await _querier(_BrokenResolver([]))._query_profile(
            {"profile_name": "video_frame", "schema_name": "video_frame"},
            5,
            "diverse",
            tenant_id="acme:media",
        )


@pytest.mark.parametrize(
    "profile_config", [{}, {"schema_name": None}, {"schema_name": " "}]
)
async def test_query_profile_rejects_missing_schema_before_backend_calls(
    profile_config,
) -> None:
    backend = _RecordingBackend([])

    with pytest.raises(
        ValueError,
        match="^profile_config requires a non-empty schema_name$",
    ):
        await _querier(backend)._query_profile(
            profile_config,
            1,
            "diverse",
            tenant_id="acme:media",
        )

    assert backend.schema_resolutions == []
    assert backend.calls == []


async def test_query_profiles_returns_exact_requested_total() -> None:
    backend = _RecordingBackend(
        [
            {"title": "first"},
            {"title": "second"},
            {"title": "third"},
        ]
    )
    querier = _querier(backend)

    samples = await querier.query_profiles(
        [
            {"profile_name": "image", "schema_name": "image", "type": "image"},
            {"profile_name": "audio", "schema_name": "audio", "type": "audio"},
            {
                "profile_name": "text",
                "schema_name": "text",
                "type": "document",
            },
        ],
        sample_size=1,
        tenant_id="acme:media",
    )

    assert [sample["topic"] for sample in samples] == ["first"]
    assert len(backend.calls) == 1
    assert backend.calls[0]["schema"] == "image"
    assert backend.calls[0]["kwargs"]["hits"] == 5


async def test_query_profiles_does_not_return_partial_data_after_outage() -> None:
    class _FailsSecondProfile(_RecordingBackend):
        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            if schema == "audio":
                raise RuntimeError("audio schema unavailable")
            return super().query_metadata_documents(
                schema=schema, query=query, yql=yql, **kwargs
            )

    querier = _querier(_FailsSecondProfile([{"title": "image result"}]))

    with pytest.raises(RuntimeError, match="audio schema unavailable"):
        await querier.query_profiles(
            [
                {
                    "profile_name": "image",
                    "schema_name": "image",
                    "type": "image",
                },
                {
                    "profile_name": "audio",
                    "schema_name": "audio",
                    "type": "audio",
                },
            ],
            sample_size=2,
            tenant_id="acme:media",
        )


async def test_query_profiles_rejects_unknown_strategy_before_backend_branch() -> None:
    backend = _RecordingBackend([])

    with pytest.raises(ValueError, match="Unsupported sampling strategy 'random'"):
        await _querier(backend).query_profiles(
            [{"profile_name": "image", "schema_name": "image", "type": "image"}],
            sample_size=1,
            strategy="random",
            tenant_id="acme:media",
        )

    assert backend.schema_resolutions == []
    assert backend.calls == []


def test_build_yql_rejects_unknown_strategy() -> None:
    backend = _RecordingBackend([])
    with pytest.raises(ValueError, match="Unsupported sampling strategy 'random'"):
        _querier(backend)._build_yql(
            "image",
            sample_size=1,
            strategy="random",
            profile_config={"schema_name": "image", "type": "image"},
        )
    assert backend.schema_resolutions == []
    assert backend.calls == []


async def test_query_profiles_keeps_event_loop_responsive_during_sync_backend_read() -> (
    None
):
    release = threading.Event()

    class _BlockingBackend(_RecordingBackend):
        def __init__(self) -> None:
            super().__init__([{"title": "released"}])
            self.released_by_event_loop = False

        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            self.released_by_event_loop = release.wait(timeout=0.2)
            return super().query_metadata_documents(
                schema=schema, query=query, yql=yql, **kwargs
            )

    async def release_backend() -> None:
        await asyncio.sleep(0)
        release.set()

    backend = _BlockingBackend()
    samples, _ = await asyncio.gather(
        _querier(backend).query_profiles(
            [{"profile_name": "image", "schema_name": "image", "type": "image"}],
            sample_size=1,
            tenant_id="acme:media",
        ),
        release_backend(),
    )

    assert backend.released_by_event_loop is True
    assert [sample["topic"] for sample in samples] == ["released"]


async def test_concurrent_queries_forward_each_request_tenant_without_bleed() -> None:
    backend = _RecordingBackend([{"title": "tenant result"}])
    querier = _querier(backend)
    tenants = [f"tenant-{index}:media" for index in range(8)]

    results = await asyncio.gather(
        *[
            querier.query_profiles(
                [
                    {
                        "profile_name": "image",
                        "schema_name": "image",
                        "type": "image",
                    }
                ],
                sample_size=1,
                tenant_id=tenant_id,
            )
            for tenant_id in tenants
        ]
    )

    assert [[sample["topic"] for sample in samples] for samples in results] == [
        ["tenant result"]
    ] * 8
    assert sorted(call["kwargs"]["tenant_id"] for call in backend.calls) == tenants
    assert [call["schema"] for call in backend.calls] == ["image"] * 8
    assert [
        call["yql"].split(" sources ", 1)[1].split(" where ", 1)[0]
        for call in backend.calls
    ] == ["image"] * 8


async def test_query_by_modality_reads_only_matching_tenant_profile() -> None:
    backend = _RecordingBackend([{"title": "video result"}])
    profiles = {
        "video_profile": BackendProfileConfig(
            profile_name="video_profile",
            type="video",
            schema_name="video_segments",
        ),
        "document_profile": BackendProfileConfig(
            profile_name="document_profile",
            type="document",
            schema_name="document_segments",
        ),
    }

    samples = await _querier(backend, profiles=profiles).query_by_modality(
        "VIDEO",
        sample_size=1,
        tenant_id="acme:media",
    )

    assert [sample["topic"] for sample in samples] == ["video result"]
    assert len(backend.calls) == 1
    assert backend.calls[0]["schema"] == "video_segments"
    assert backend.calls[0]["kwargs"]["tenant_id"] == "acme:media"


@pytest.mark.parametrize("modality", ["", "unknown", "video", "Video"])
async def test_query_by_modality_rejects_unknown_modality(modality) -> None:
    backend = _RecordingBackend([])
    with pytest.raises(ValueError, match="Unsupported modality"):
        await _querier(backend).query_by_modality(
            modality,
            sample_size=1,
            tenant_id="acme:media",
        )


@pytest.mark.parametrize("tenant_id", ["", "bad tenant!", "__system__"])
async def test_query_by_modality_rejects_invalid_tenant_before_backend_branch(
    tenant_id,
) -> None:
    backend = _RecordingBackend([])

    with pytest.raises(ValueError, match="tenant_id"):
        await _querier(backend).query_by_modality(
            "VIDEO",
            sample_size=1,
            tenant_id=tenant_id,
        )

    assert backend.schema_resolutions == []
    assert backend.calls == []


async def test_query_by_modality_rejects_missing_profile_before_backend_branch() -> (
    None
):
    backend = _RecordingBackend([])

    with pytest.raises(ValueError, match="No backend profiles configured for AUDIO"):
        await _querier(backend).query_by_modality(
            "AUDIO",
            sample_size=1,
            tenant_id="acme:media",
        )

    assert backend.schema_resolutions == []
    assert backend.calls == []


class _PagingBackend:
    """Returns a fixed corpus, honouring the querier's hits/offset paging."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.calls: list[dict] = []

    def get_tenant_schema_name(self, tenant_id, base_schema_name):
        return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.calls.append({"schema": schema, "yql": yql, "kwargs": kwargs})
        offset = kwargs.get("offset", 0)
        return self.docs[offset : offset + kwargs["hits"]]


_VIDEO_PROFILE = {
    "profile_name": "video_frames_mv",
    "schema_name": "video_frames",
    "type": "video",
    "embedding_type": "multi_vector",
    "pipeline_config": {"extract_keyframes": True, "generate_descriptions": True},
}
_TWO_VIDEO_CORPUS = [
    {"video_title": "video_a", "segment_description": "frame a1", "video_id": "a"},
    {"video_title": "video_a", "segment_description": "frame a2", "video_id": "a"},
    {"video_title": "video_a", "segment_description": "frame a3", "video_id": "a"},
    {"video_title": "video_b", "segment_description": "frame b1", "video_id": "b"},
    {"video_title": "video_b", "segment_description": "frame b2", "video_id": "b"},
    {"video_title": "video_b", "segment_description": "frame b3", "video_id": "b"},
]


def test_diverse_spreads_samples_across_distinct_sources() -> None:
    backend = _PagingBackend(_TWO_VIDEO_CORPUS)
    querier = _querier(backend)

    samples = asyncio.run(
        querier._query_profile(
            _VIDEO_PROFILE, 2, "diverse", tenant_id="flywheel_org:production"
        )
    )

    assert [sample["topic"] for sample in samples] == ["video_a", "video_b"]
    assert [sample["description"] for sample in samples] == ["frame a1", "frame b1"]


def test_diverse_and_multi_modal_sequences_do_not_share_one_query() -> None:
    emitted = {}
    sampled = {}
    for strategy in ("diverse", "multi_modal_sequences"):
        backend = _PagingBackend(_TWO_VIDEO_CORPUS)
        querier = _querier(backend)
        samples = asyncio.run(
            querier._query_profile(
                _VIDEO_PROFILE, 2, strategy, tenant_id="flywheel_org:production"
            )
        )
        emitted[strategy] = [call["yql"] for call in backend.calls]
        sampled[strategy] = [sample["description"] for sample in samples]

    assert emitted["diverse"] == [
        "select * from sources video_frames where true limit 10"
    ]
    assert emitted["multi_modal_sequences"] == [
        "select * from sources video_frames where true limit 2"
    ]
    assert sampled["diverse"] == ["frame a1", "frame b1"]
    assert sampled["multi_modal_sequences"] == ["frame a1", "frame a2"]
