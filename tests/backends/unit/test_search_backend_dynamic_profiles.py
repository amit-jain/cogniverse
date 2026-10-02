"""Unit tests for VespaSearchBackend profile resolution and query handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.common.utils.retry import RetryConfig
from cogniverse_core.query.encoders import EncoderNotConfiguredError
from cogniverse_vespa.search_backend import (
    VespaSearchBackend,
    _source_collapse_fetch_limit,
    _source_collapse_oversample,
)


def _make_backend(profiles: dict | None = None) -> VespaSearchBackend:
    """Build a backend without touching real Vespa / pool / metrics.

    Every schema its profiles name reads as deployed.
    """
    built: list[VespaSearchBackend] = []

    def is_schema_deployed(_tenant_id, base_schema_name):
        return base_schema_name in {
            config.get("schema_name", name)
            for name, config in built[0].profiles.items()
        }

    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
                "profiles": profiles or {},
            },
            is_schema_deployed=is_schema_deployed,
        )
    built.append(backend)
    return backend


def test_initialize_populates_profiles_from_top_level_config():
    """Reinitialize-in-place must actually re-read the profiles dict.

    Regression guard: previously ``initialize()`` re-assigned url/port/etc.
    but silently skipped ``profiles``, leaving any caller who re-init'd
    the backend with an empty profiles dict at query time.
    """
    backend = _make_backend()
    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend.initialize(
            {
                "url": "http://localhost",
                "port": 8080,
                "profiles": {"alpha": {"type": "memory"}},
                "default_profiles": {"memory": {"profile": "alpha"}},
            }
        )
    assert backend.profiles == {"alpha": {"type": "memory"}}
    assert backend.default_profiles == {"memory": {"profile": "alpha"}}


def test_initialize_preserves_constructor_retry_configuration():
    retry_config = RetryConfig(max_attempts=7, initial_delay=0, jitter=False)
    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={"url": "http://localhost", "port": 8080},
            retry_config=retry_config,
            is_schema_deployed=lambda _tenant_id, _base: False,
        )
        backend.initialize({"url": "http://localhost", "port": 8080})

    assert backend.retry_config is retry_config


def test_initialize_reads_profiles_from_nested_backend_section():
    """``get_search_backend`` passes profiles under config['backend'] in some
    call paths (see ``backend_registry.get_search_backend``)."""
    backend = _make_backend()
    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend.initialize(
            {
                "url": "http://localhost",
                "port": 8080,
                "backend": {"profiles": {"beta": {"type": "video"}}},
            }
        )
    assert backend.profiles == {"beta": {"type": "video"}}


def test_hybrid_audio_encoder_uses_semantic_model():
    config_manager = MagicMock()
    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
            },
            config_manager=config_manager,
            is_schema_deployed=lambda _tenant_id, _base: False,
        )
    profile = {
        "embedding_model": "laion/clap-htsat-unfused",
        "semantic_model": "lightonai/LateOn",
        "embedding_type": "multi_vector",
    }

    with (
        patch(
            "cogniverse_foundation.config.utils.get_config",
            return_value={"backend": {"profiles": {"audio": profile}}},
        ),
        patch(
            "cogniverse_core.query.encoders.QueryEncoderFactory.create_encoder",
            return_value=MagicMock(),
        ) as create_encoder,
    ):
        backend._resolve_encoder_for_profile("audio", profile, "acme:content")

    create_encoder.assert_called_once_with(
        "audio",
        "lightonai/LateOn",
        config={"backend": {"profiles": {"audio": profile}}},
    )


def test_search_raises_when_profile_not_found():
    backend = _make_backend({"known": {"type": "video"}})
    with pytest.raises(ValueError) as caught:
        backend.search(
            query_dict={
                "query": "hi",
                "type": "video",
                "profile": "does_not_exist",
                "tenant_id": "acme",
            }
        )
    assert str(caught.value) == (
        "Requested profile 'does_not_exist' not found. Available profiles: ['known']"
    )


def test_search_accepts_empty_query_text_when_embeddings_provided():
    """Mem0 invokes search with pre-computed embeddings and empty query text.

    Regression guard: search_backend used to reject ``query=""`` with
    ``"query_dict must contain 'query' key with text query"``. Mem0 retried
    3× per call, and the orchestrator's detailed_report path hung past the
    300s client timeout. The ``test_gateway_confidence_in_range`` e2e
    test was the visible failure.
    """
    import numpy as np

    backend = _make_backend({"known": {"type": "memory"}})
    with (
        patch(
            "cogniverse_vespa.search_backend._RANKING_STRATEGIES_CACHE",
            {},
        ),
        pytest.raises(
            ValueError,
            match="No ranking strategies found for schema 'known'",
        ),
    ):
        backend.search(
            query_dict={
                "query": "",
                "type": "memory",
                "profile": "known",
                "tenant_id": "acme",
                "query_embeddings": np.zeros(768, dtype=np.float32),
            }
        )


def test_search_still_rejects_missing_text_and_embeddings():
    """If caller supplies neither text nor embeddings, we must still fail loudly."""
    backend = _make_backend({"known": {"type": "memory"}})
    with pytest.raises(ValueError, match="query_embeddings"):
        backend.search(
            query_dict={
                "query": "",
                "type": "memory",
                "profile": "known",
                "tenant_id": "acme",
            }
        )


def test_search_raises_naming_the_missing_model_when_embeddings_are_needed():
    """A strategy that declares it needs float/binary embeddings must fail
    loudly when neither query_embeddings nor an encoder is available, naming
    what the profile is missing. Querying Vespa without the embedding tensor
    silently returns wrong (or 0) results. Uses the real ranking-strategy
    definitions via FilesystemSchemaLoader; the raise fires before any Vespa
    call, so no live backend is needed.
    """
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
                "profiles": {
                    "vcolpali": {
                        "type": "video",
                        "schema_name": "video_colpali_smol500_mv_frame",
                    }
                },
            },
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            is_schema_deployed=lambda _tenant_id, base: (
                base in {"video_colpali_smol500_mv_frame"}
            ),
        )

    with pytest.raises(EncoderNotConfiguredError) as excinfo:
        backend.search(
            query_dict={
                "query": "ocean waves",
                "type": "video",
                "profile": "vcolpali",
                "strategy": "float_float",
                "tenant_id": "acme",
            }
        )

    assert str(excinfo.value) == (
        "Profile 'vcolpali' declares neither 'semantic_model' nor "
        "'embedding_model', so no query encoder can be built. Add one to the "
        "profile or pass 'query_embeddings'."
    )


@pytest.mark.parametrize(
    ("top_k", "profile_config", "expected"),
    [
        (10, {}, 40),
        # The ceiling clamps the oversampling, not the request: a top_k above
        # the ceiling still fetches top_k, because collapsing to top_k distinct
        # sources cannot be done from fewer than top_k documents.
        (1000, {}, 1000),
        (300, {}, 300),
        (64, {}, 256),
        (10, {"source_collapse_oversample": 12}, 120),
    ],
)
def test_source_collapse_fetch_limit_uses_bounded_multiplier(
    top_k, profile_config, expected
):
    assert (
        _source_collapse_fetch_limit(top_k, _source_collapse_oversample(profile_config))
        == expected
    )


@pytest.mark.parametrize(
    ("profile_config", "message"),
    [
        (
            {"source_collapse_oversample": 0},
            "source_collapse_oversample must be >= 1",
        ),
        (
            {"source_collapse_oversample": 1.5},
            "source_collapse_oversample must be an integer",
        ),
        (
            {"source_collapse_oversample": 65},
            "source_collapse_oversample must be <= 64",
        ),
    ],
)
def test_source_collapse_oversample_rejects_invalid_values(profile_config, message):
    with pytest.raises(ValueError) as exc_info:
        _source_collapse_oversample(profile_config)

    assert str(exc_info.value) == message


def test_source_search_reads_the_profile_oversample_once(monkeypatch):
    """A source search derives its fetch limit and its per-source window from
    one read of the profile's oversample."""
    from cogniverse_vespa import search_backend

    reads = []
    read = search_backend._source_collapse_oversample

    def counting_read(profile_config):
        reads.append(dict(profile_config))
        return read(profile_config)

    monkeypatch.setattr(search_backend, "_source_collapse_oversample", counting_read)
    profile = {
        "type": "video",
        "schema_name": "video_colpali_smol500_mv_frame",
        "source_collapse_oversample": 3,
    }
    backend = _make_backend({"vcolpali": profile})
    with patch(
        "cogniverse_vespa.search_backend._RANKING_STRATEGIES_CACHE",
        {
            "video_colpali_smol500_mv_frame": {
                "bm25_only": {
                    "needs_float_embeddings": False,
                    "needs_binary_embeddings": False,
                }
            }
        },
    ):
        with pytest.raises(ValueError, match="source granularity requires"):
            backend.search(
                query_dict={
                    "query": "ocean waves",
                    "type": "video",
                    "profile": "vcolpali",
                    "tenant_id": "acme",
                    "result_granularity": "source",
                }
            )

    assert reads == [profile]


@pytest.mark.parametrize(
    "query_dict",
    [
        {
            "query": "ocean waves",
            "type": "video",
            "profile": "vcolpali",
            "tenant_id": "acme",
        },
        {
            "query": "ocean waves",
            "type": "video",
            "profile": "vcolpali",
            "tenant_id": "acme",
            "result_granularity": "source",
        },
    ],
    ids=["default_source", "requested_source"],
)
def test_search_raises_when_source_granularity_has_no_schema_loader(query_dict):
    backend = _make_backend(
        {
            "vcolpali": {
                "type": "video",
                "schema_name": "video_colpali_smol500_mv_frame",
            }
        }
    )
    with patch(
        "cogniverse_vespa.search_backend._RANKING_STRATEGIES_CACHE",
        {
            "video_colpali_smol500_mv_frame": {
                "bm25_only": {
                    "needs_float_embeddings": False,
                    "needs_binary_embeddings": False,
                }
            }
        },
    ):
        with pytest.raises(ValueError) as exc_info:
            backend.search(query_dict=query_dict)

    assert str(exc_info.value) == (
        "Profile 'vcolpali' (schema 'video_colpali_smol500_mv_frame') "
        "source granularity requires schema_loader"
    )


def test_search_does_not_retry_value_errors():
    """``ValueError`` signals a permanent config problem (profile missing,
    type unknown, bad inputs). The retry wrapper used to re-fire these 3×
    per call with exponential backoff; under the orchestrator's wiki path
    ('No profiles found for type wiki') that burned ~5 seconds per call
    and accumulated to push the complex-query test past its client
    timeout. Verify the retry wrapper no longer loops on ValueError.
    """
    backend = _make_backend({"known": {"type": "memory"}})

    # Fire a search that will fail at profile-resolution (ValueError).
    import time

    start = time.monotonic()
    try:
        backend.search(
            query_dict={
                "query": "any",
                "type": "video",  # no "video" profile registered
                "tenant_id": "acme",
            }
        )
    except ValueError:
        pass
    elapsed = time.monotonic() - start

    # Retry defaults: 3 attempts × ≥1s initial_delay ≈ >3s total.
    # Without retrying ValueError, the call returns immediately.
    assert elapsed < 1.0, (
        f"Search took {elapsed:.2f}s — retry wrapper is still looping on "
        "ValueError when it should fail fast for permanent config errors."
    )


@pytest.mark.unit
class TestFilterConditions:
    """_build_filter_conditions builds the Vespa YQL where-clause from a
    filters dict, including numeric/epoch range filters (used for date
    filtering on creation_timestamp)."""

    @staticmethod
    def _build(filters):
        # The method doesn't touch instance state, so a bare instance is fine.
        return VespaSearchBackend._build_filter_conditions(
            object.__new__(VespaSearchBackend), filters
        )

    def test_range_filter_emits_gte_and_lte(self):
        assert (
            self._build({"creation_timestamp": {"gte": 100, "lte": 200}})
            == "creation_timestamp >= 100 AND creation_timestamp <= 200"
        )

    def test_range_filter_gt_and_lt(self):
        assert self._build({"ts": {"gt": 5, "lt": 9}}) == "ts > 5 AND ts < 9"

    def test_range_combined_with_string_equality(self):
        assert (
            self._build({"tenant_id": "acme", "creation_timestamp": {"gte": 100}})
            == 'tenant_id contains "acme" AND creation_timestamp >= 100'
        )

    def test_empty_filters(self):
        assert self._build({}) == ""


class TestVespaConnectionSessionReuse:
    """Pooled connections must query over their persistent VespaSync client —
    ``Vespa.query()`` builds and tears down a fresh HTTP client per call, so
    routing through it gives the pool zero socket reuse."""

    def test_query_uses_persistent_sync_client(self):
        from unittest.mock import MagicMock

        from cogniverse_vespa.search_backend import VespaConnection

        with patch("cogniverse_vespa.search_backend.make_vespa_app") as mk:
            app = MagicMock()
            sync = MagicMock()
            app.syncio.return_value = sync
            mk.return_value = app

            conn = VespaConnection("http://unused:1", "c1")
            sync._open_http_client.assert_called_once()

            conn.query(yql="select 1")
            sync.query.assert_called_once_with(yql="select 1")
            app.query.assert_not_called()

            conn.close()
            sync._close_http_client.assert_called_once()


def test_initialize_honors_enable_metrics_false():
    """initialize() must not override enable_metrics=False from __init__.

    The registry construct-then-initialize path passes enable_metrics through
    __init__; initialize() used to overwrite self.metrics with an unconditional
    SearchMetrics(), silently re-enabling metrics for a backend that asked for
    them off.
    """
    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics") as metrics_cls,
    ):
        backend = VespaSearchBackend(
            enable_metrics=False, is_schema_deployed=lambda _tenant_id, _base: False
        )
        assert backend.metrics is None  # __init__ respected the knob
        backend.initialize(
            {"url": "http://localhost", "port": 8080, "schema_name": "s"}
        )
        assert backend.metrics is None  # initialize() still respects it

        default_backend = VespaSearchBackend(
            enable_metrics=True, is_schema_deployed=lambda _tenant_id, _base: False
        )
        default_backend.initialize(
            {"url": "http://localhost", "port": 8080, "schema_name": "s"}
        )
        # Default path keeps metrics on: the built SearchMetrics survives.
        assert default_backend.metrics is metrics_cls.return_value


def test_search_types_hits_by_resolved_profile_not_by_query_type():
    """The profile owns the schema, so it owns the content type of every hit.

    ``query_dict["type"]`` only selects a profile when none is named; the
    search agent sends its inferred modality there alongside its active
    profile, and the two can disagree.
    """
    from pathlib import Path

    from vespa.io import VespaQueryResponse

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_sdk.document import ContentType

    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
                "profiles": {
                    "audio_clap_semantic": {
                        "type": "audio",
                        "schema_name": "audio_content",
                    }
                },
            },
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            is_schema_deployed=lambda _tenant_id, base: base in {"audio_content"},
        )
    backend.pool = None
    backend.vespa = MagicMock()
    backend.vespa.query.return_value = VespaQueryResponse(
        json={
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 2},
                "coverage": {"coverage": 100, "documents": 2},
                "children": [
                    {
                        "id": "id:content:audio_content_acme_acme::clip-1",
                        "relevance": 0.9,
                        "fields": {"audio_id": "clip-1", "transcript": "fire"},
                    },
                    {
                        "id": "id:content:audio_content_acme_acme::clip-2",
                        "relevance": 0.8,
                        "fields": {"audio_id": "clip-2", "transcript": "rain"},
                    },
                ],
            }
        },
        status_code=200,
        url="http://localhost:8080/search/",
    )

    results = backend.search(
        query_dict={
            "query": "podcasts about deep learning",
            "type": "video",
            "profile": "audio_clap_semantic",
            "strategy": "transcript_search",
            "tenant_id": "acme:acme",
        }
    )

    assert [(r.document.id, r.document.content_type) for r in results] == [
        ("clip-1", ContentType.AUDIO),
        ("clip-2", ContentType.AUDIO),
    ]


def _profile_resolution_backend(profiles, default_profiles=None, config_manager=None):
    """A backend over the shipped schemas whose Vespa answers every query empty."""
    from pathlib import Path

    from vespa.io import VespaQueryResponse

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    with (
        patch("cogniverse_vespa.search_backend.ConnectionPool"),
        patch("cogniverse_vespa.search_backend.SearchMetrics"),
    ):
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": 8080,
                "profiles": profiles,
                "default_profiles": default_profiles or {},
            },
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            is_schema_deployed=lambda _tenant_id, _base: True,
        )
    backend.pool = None
    backend.vespa = MagicMock()
    backend.vespa.query.return_value = VespaQueryResponse(
        json={
            "root": {
                "id": "toplevel",
                "relevance": 1.0,
                "fields": {"totalCount": 0},
                "coverage": {"coverage": 100, "documents": 0},
            }
        },
        status_code=200,
        url="http://localhost:8080/search/",
    )
    return backend


def _queried_schemas(backend) -> list:
    return [
        call.kwargs["body"]["model.restrict"]
        for call in backend.vespa.query.call_args_list
    ]


VIDEO_PROFILES = {
    "vcolpali": {"type": "video", "schema_name": "video_colpali_smol500_mv_frame"},
}
TWO_VIDEO_PROFILES = {
    **VIDEO_PROFILES,
    "vqwen": {"type": "video", "schema_name": "video_colqwen_omni_mv_chunk_30s"},
}


def _video_query(**extra) -> dict:
    return {
        "query": "ocean waves",
        "type": "video",
        "strategy": "bm25_only",
        "tenant_id": "acme",
        "result_granularity": "segment",
        **extra,
    }


def test_video_search_naming_no_profile_and_no_selected_default_is_refused():
    backend = _profile_resolution_backend(VIDEO_PROFILES)

    with pytest.raises(ValueError) as excinfo:
        backend.search(_video_query())

    assert str(excinfo.value) == (
        "No profile specified on the request and tenant 'acme' has no "
        "configured default video profile."
    )
    assert _queried_schemas(backend) == []


def test_video_search_naming_no_profile_uses_the_selected_default():
    backend = _profile_resolution_backend(
        TWO_VIDEO_PROFILES, default_profiles={"video": {"profile": "vqwen"}}
    )

    backend.search(_video_query())

    assert _queried_schemas(backend) == ["video_colqwen_omni_mv_chunk_30s_acme_acme"]


def test_video_search_naming_no_profile_uses_the_active_video_profile(monkeypatch):
    import cogniverse_foundation.config.utils as config_utils

    read = []

    def get_config(tenant_id, config_manager):
        read.append(tenant_id)
        return {"active_video_profile": "vcolpali"}

    monkeypatch.setattr(config_utils, "get_config", get_config)
    backend = _profile_resolution_backend(
        TWO_VIDEO_PROFILES, config_manager=MagicMock()
    )

    backend.search(_video_query())

    # One tenant config read serves the profile merge and the default.
    assert read == ["acme"]
    assert _queried_schemas(backend) == ["video_colpali_smol500_mv_frame_acme_acme"]


def test_video_search_selecting_a_non_video_default_is_refused():
    backend = _profile_resolution_backend(
        {**VIDEO_PROFILES, "wiki": {"type": "wiki", "schema_name": "wiki_pages"}},
        default_profiles={"video": {"profile": "wiki"}},
    )

    with pytest.raises(ValueError) as excinfo:
        backend.search(_video_query())

    assert str(excinfo.value) == (
        "Default profile 'wiki' for type 'video' not found in available "
        "profiles: ['vcolpali']"
    )
    assert _queried_schemas(backend) == []


@pytest.mark.parametrize(
    ("profiles", "query", "schema"),
    [
        (
            {"wiki_semantic": {"type": "wiki", "schema_name": "wiki_pages"}},
            {"type": "wiki", "strategy": "bm25"},
            "wiki_pages_acme_acme",
        ),
        (
            {"audio_clap": {"type": "audio", "schema_name": "audio_content"}},
            {"type": "audio", "strategy": "transcript_search"},
            "audio_content_acme_acme",
        ),
    ],
    ids=["wiki", "audio"],
)
def test_types_without_a_default_selection_still_resolve_their_only_profile(
    profiles, query, schema
):
    backend = _profile_resolution_backend(profiles)

    backend.search({"query": "ocean waves", "tenant_id": "acme", **query})

    assert _queried_schemas(backend) == [schema]


def _stored_manager(store=None):
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    return ConfigManager(store=store or InMemoryConfigStore())


def _stored_profile(name: str, schema_name: str, profile_type: str = "wiki"):
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    return BackendProfileConfig(
        profile_name=name,
        type=profile_type,
        schema_name=schema_name,
        embedding_model="lightonai/DenseOn",
    )


_PROFILE_QUERIES = {
    "wiki": {"type": "wiki", "strategy": "bm25"},
    "audio": {"type": "audio", "strategy": "transcript_search"},
}


def _stored_profile_query(tenant_id: str, profile: str, profile_type="wiki") -> dict:
    return {
        "query": "ocean waves",
        "tenant_id": tenant_id,
        "profile": profile,
        **_PROFILE_QUERIES[profile_type],
    }


def _tenant_profiles(config_manager, tenant_id: str) -> dict:
    from cogniverse_foundation.config.utils import get_config

    return get_config(tenant_id=tenant_id, config_manager=config_manager).get(
        "backend"
    )["profiles"]


def test_a_profile_the_tenant_stored_after_the_backend_was_built_resolves_on_it():
    manager = _stored_manager()
    backend = _profile_resolution_backend(VIDEO_PROFILES, config_manager=manager)

    manager.add_backend_profile(
        _stored_profile("late_wiki", "wiki_pages"), tenant_id="acme"
    )
    backend.search(_stored_profile_query("acme", "late_wiki"))

    assert _queried_schemas(backend) == ["wiki_pages_acme_acme"]
    # The shared backend's own profiles stay what it was built with.
    assert backend.profiles == VIDEO_PROFILES


def test_another_tenants_stored_profile_is_neither_resolved_nor_listed():
    manager = _stored_manager()
    backend = _profile_resolution_backend(VIDEO_PROFILES, config_manager=manager)
    manager.add_backend_profile(
        _stored_profile("late_wiki", "wiki_pages"), tenant_id="acme"
    )
    visible_to_globex = list({**VIDEO_PROFILES, **_tenant_profiles(manager, "globex")})

    with pytest.raises(ValueError) as refused:
        backend.search(_stored_profile_query("globex", "late_wiki"))

    assert "late_wiki" not in visible_to_globex
    assert str(refused.value) == (
        "Requested profile 'late_wiki' not found. Available profiles: "
        f"{visible_to_globex}"
    )
    assert _queried_schemas(backend) == []


def test_a_profile_deleted_from_the_tenants_store_stops_resolving():
    manager = _stored_manager()
    backend = _profile_resolution_backend(VIDEO_PROFILES, config_manager=manager)
    manager.add_backend_profile(
        _stored_profile("late_wiki", "wiki_pages"), tenant_id="acme"
    )
    backend.search(_stored_profile_query("acme", "late_wiki"))

    assert manager.delete_backend_profile("late_wiki", tenant_id="acme") is True
    with pytest.raises(ValueError) as refused:
        backend.search(_stored_profile_query("acme", "late_wiki"))

    assert str(refused.value) == (
        "Requested profile 'late_wiki' not found. Available profiles: "
        f"{list({**VIDEO_PROFILES, **_tenant_profiles(manager, 'acme')})}"
    )
    assert _queried_schemas(backend) == ["wiki_pages_acme_acme"]


def test_concurrent_searches_by_tenants_sharing_a_profile_name_resolve_their_own():
    """Each tenant stores its own ``mine``; searches racing on the shared
    backend each query the schema of the searching tenant's profile."""
    import threading

    manager = _stored_manager()
    backend = _profile_resolution_backend(VIDEO_PROFILES, config_manager=manager)
    tenants = {
        "t0": ("wiki_pages", "wiki"),
        "t1": ("audio_content", "audio"),
        "t2": ("wiki_pages", "wiki"),
        "t3": ("audio_content", "audio"),
    }
    for tenant_id, (schema_name, profile_type) in tenants.items():
        manager.add_backend_profile(
            _stored_profile("mine", schema_name, profile_type), tenant_id=tenant_id
        )
    searches = [tenant_id for tenant_id in tenants for _ in range(2)]
    barrier = threading.Barrier(len(searches))
    errors = []

    def search(tenant_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            backend.search(
                _stored_profile_query(tenant_id, "mine", tenants[tenant_id][1])
            )
        except Exception as exc:
            errors.append(f"{tenant_id}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=search, args=(t,)) for t in searches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert [thread.is_alive() for thread in threads] == [False] * len(searches)
    assert errors == []
    assert sorted(_queried_schemas(backend)) == sorted(
        f"{tenants[tenant_id][0]}_{tenant_id}_{tenant_id}" for tenant_id in searches
    )
    assert backend.profiles == VIDEO_PROFILES


def test_a_search_whose_tenant_profiles_cannot_be_read_raises_and_queries_nothing():
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
    from tests.utils.memory_store import InMemoryConfigStore

    class UnreachableStore(InMemoryConfigStore):
        def get_config(self, *args, **kwargs):
            raise ConfigStoreUnavailableError("config store unreachable")

    backend = _profile_resolution_backend(
        VIDEO_PROFILES, config_manager=_stored_manager(UnreachableStore())
    )

    with pytest.raises(ConfigStoreUnavailableError) as refused:
        backend.search(_video_query(profile="vcolpali"))

    assert str(refused.value) == "config store unreachable"
    assert _queried_schemas(backend) == []
