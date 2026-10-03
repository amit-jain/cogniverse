"""RankingStrategyExtractor: schema-name source and single-vector detection.

Two regressions guarded here:

* ``_parse_ranking_profile`` recomputed ``schema_name`` via
  ``schema_json.get("schema", "")`` — dropping the ``name`` fallback that
  ``extract_from_schema`` uses, so a schema keyed by ``name`` persisted an
  empty ``schema_name``.
* Single-vector detection used ``"_sv_" in name.lower()`` only, missing the
  ``_lvt_`` single-vector schemas the authoritative
  ``_is_single_vector_schema`` helper recognises — so an LVT schema never
  enabled nearestNeighbor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogniverse_vespa.ranking_strategy_extractor import (
    RankingStrategyExtractor,
    extract_all_ranking_strategies,
)

_FLOAT_PROFILE = {
    "name": "float_float",
    "inputs": [{"name": "query(qt)", "type": "tensor<float>(x[128])"}],
}

_VIDEO_PHASED_DEFAULTS = {
    "configs/schemas/video_colpali_smol500_mv_frame_schema.json": {
        "qtb": "tensor<int8>(querytoken{}, v[40])",
        "qt": "tensor<float>(querytoken{}, v[320])",
    },
    "configs/schemas/video_colqwen_omni_mv_chunk_30s_schema.json": {
        "qtb": "tensor<int8>(querytoken{}, v[40])",
        "qt": "tensor<float>(querytoken{}, v[320])",
    },
}


def _write_schema(tmp_path, schema_dict):
    path = tmp_path / "s_schema.json"
    path.write_text(json.dumps(schema_dict))
    return path


def test_schema_name_populated_when_keyed_by_name(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "name": "video_colpali_sv_test",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank_profiles": [_FLOAT_PROFILE],
        },
    )

    strategies = RankingStrategyExtractor().extract_from_schema(path)

    assert strategies["float_float"].schema_name == "video_colpali_sv_test"


def test_lvt_schema_enables_nearest_neighbor(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "name": "video_xclip_lvt_global",
            "document": {
                "fields": [{"name": "embedding", "type": "tensor<float>(x[128])"}]
            },
            "rank_profiles": [
                {**_FLOAT_PROFILE, "first_phase": "closeness(field, embedding)"}
            ],
        },
    )

    strategy = RankingStrategyExtractor().extract_from_schema(path)["float_float"]

    assert strategy.use_nearestneighbor is True
    assert strategy.nearestneighbor_field == "embedding"
    assert strategy.nearestneighbor_tensor == "qt"


def test_sv_schema_still_enables_nearest_neighbor(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "schema": "video_colpali_sv_frame",
            "document": {
                "fields": [{"name": "embedding", "type": "tensor<float>(x[128])"}]
            },
            "rank_profiles": [
                {**_FLOAT_PROFILE, "first_phase": "closeness(field, embedding)"}
            ],
        },
    )

    strategy = RankingStrategyExtractor().extract_from_schema(path)["float_float"]

    assert strategy.use_nearestneighbor is True


def test_extract_all_strips_only_trailing_schema_suffix(tmp_path):
    (tmp_path / "code_schema_index_schema.json").write_text(
        json.dumps(
            {
                "schema": "code_schema_index",
                "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
                "rank_profiles": [_FLOAT_PROFILE],
            }
        )
    )

    all_strategies = extract_all_ranking_strategies(tmp_path)

    assert "code_schema_index" in all_strategies
    assert "code_index" not in all_strategies


def test_extract_all_ranking_strategies_memoized_and_invalidates(tmp_path):
    import os
    from unittest.mock import patch

    import cogniverse_vespa.ranking_strategy_extractor as rse

    _write_schema(
        tmp_path,
        {
            "schema": "memo_probe",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank_profiles": [_FLOAT_PROFILE],
        },
    )

    parsed = []
    real = rse.RankingStrategyExtractor.extract_from_schema

    def spy(self, path):
        parsed.append(path.name)
        return real(self, path)

    with patch.object(rse.RankingStrategyExtractor, "extract_from_schema", spy):
        first = extract_all_ranking_strategies(tmp_path)
        assert parsed == ["s_schema.json"]  # cold call parses the file
        assert "float_float" in first["s"]

        parsed.clear()
        second = extract_all_ranking_strategies(tmp_path)
        assert parsed == []  # unchanged dir -> cache hit, no re-parse
        assert second == first

        # A schema edit (new mtime) invalidates the memo and re-parses.
        parsed.clear()
        schema_file = tmp_path / "s_schema.json"
        st = schema_file.stat()
        os.utime(schema_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        extract_all_ranking_strategies(tmp_path)
        assert parsed == ["s_schema.json"]  # re-parsed after the edit


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HYBRID_SCHEMAS = [
    "configs/schemas/video_colpali_smol500_mv_frame_schema.json",
    "configs/schemas/video_colqwen_omni_mv_chunk_30s_schema.json",
]


def _rank_profiles(path: Path) -> dict:
    data = json.loads(path.read_text())

    def find(o):
        if isinstance(o, dict):
            if "rank_profiles" in o:
                return o["rank_profiles"]
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        return None

    return {r["name"]: r for r in (find(data) or []) if "name" in r}


def _normalized_profile(profile: dict) -> dict:
    return {
        key: value for key, value in profile.items() if key not in {"name", "inherits"}
    }


@pytest.mark.unit
@pytest.mark.parametrize("schema", _HYBRID_SCHEMAS)
def test_hybrid_rank_profiles_honor_phase_order_naming(schema):
    """A ``hybrid_binary_bm25*`` profile ranks every segment by binary MaxSim
    plus text in its first phase and a ``hybrid_bm25_binary*`` profile
    text-first, BM25 picking the candidates its second phase ranks by the same
    sum; the ``_no_description`` pair were once byte-identical (both
    text-first), silently giving hybrid_binary_bm25_no_description the wrong
    phase order."""
    profiles = _rank_profiles(_REPO_ROOT / schema)
    for suffix, text, bm25 in (
        ("", "text_sim", "text_bm25"),
        ("_no_description", "text_sim_no_desc", "text_bm25_no_desc"),
    ):
        binary_first = profiles[f"hybrid_binary_bm25{suffix}"]
        text_first = profiles[f"hybrid_bm25_binary{suffix}"]
        assert binary_first["first_phase"] == f"visual_sim_binary + {text}", (
            f"hybrid_binary_bm25{suffix} must rank binary MaxSim plus text first"
        )
        assert text_first["first_phase"] == bm25, (
            f"hybrid_bm25_binary{suffix} must rank text/bm25 first"
        )
        assert text_first["second_phase"] == {
            "expression": f"visual_sim_binary + {text}",
            "rerank_count": 100,
        }, f"hybrid_bm25_binary{suffix} must rerank by binary MaxSim plus text"
        assert binary_first["first_phase"] != text_first["first_phase"], (
            f"opposite-named hybrid profiles must differ (suffix={suffix!r})"
        )


@pytest.mark.unit
@pytest.mark.parametrize("schema_path", sorted(_VIDEO_PHASED_DEFAULTS))
def test_video_default_profile_matches_phased_schema(schema_path):
    profiles = _rank_profiles(_REPO_ROOT / schema_path)
    assert profiles["default"]["inherits"] == "phased"
    assert _normalized_profile(profiles["default"]) == _normalized_profile(
        profiles["phased"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("schema_path", "expected_inputs"),
    sorted(_VIDEO_PHASED_DEFAULTS.items()),
)
def test_video_default_profile_needs_both_query_tensors(schema_path, expected_inputs):
    strategy = RankingStrategyExtractor().extract_from_schema(_REPO_ROOT / schema_path)[
        "default"
    ]

    assert strategy.inputs == expected_inputs
    assert strategy.needs_float_embeddings is True
    assert strategy.needs_binary_embeddings is True
    assert strategy.query_tensors_needed == ["qtb", "qt"]


def test_vanished_schema_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A schema file that disappears between the directory glob and the
    signature stat() must be skipped — previously the unguarded stat in the
    memo-signature comprehension raised FileNotFoundError and 500'd every
    strategy listing during a concurrent schema rewrite."""
    from pathlib import Path

    _write_schema(
        tmp_path,
        {
            "schema": "survivor",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank_profiles": [_FLOAT_PROFILE],
        },
    )
    ghost = tmp_path / "ghost_schema.json"  # in the listing, never on disk

    real_glob = Path.glob

    def glob_with_ghost(self, pattern):
        results = list(real_glob(self, pattern))
        if self == tmp_path:
            results.append(ghost)
        return results

    monkeypatch.setattr(Path, "glob", glob_with_ghost)

    out = extract_all_ranking_strategies(tmp_path)

    assert "s" in out  # the survivor parsed
    assert "ghost" not in out


def test_memo_keeps_a_single_entry_per_directory(tmp_path):
    """Schema edits must REPLACE the dir's memo entry, not accrete new ones —
    the signature-keyed dict otherwise grows by one full strategies map per
    edit for the life of the process."""
    import os
    import time as _time

    import cogniverse_vespa.ranking_strategy_extractor as rse

    _write_schema(
        tmp_path,
        {
            "schema": "memo_bound",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank_profiles": [_FLOAT_PROFILE],
        },
    )
    schema_file = tmp_path / "s_schema.json"
    dir_key = str(tmp_path.resolve())

    for i in range(3):
        st = schema_file.stat()
        os.utime(
            schema_file,
            ns=(st.st_atime_ns, st.st_mtime_ns + (i + 1) * 1_000_000_000),
        )
        extract_all_ranking_strategies(tmp_path)
        _time.sleep(0.01)

    entries = [k for k in rse._ALL_STRATEGIES_CACHE if k[0] == dir_key]
    assert len(entries) == 1, f"{len(entries)} memo entries accreted for one directory"


def test_memo_hit_returns_a_fresh_dict_not_the_shared_one(tmp_path):
    """A caller mutating the returned mapping must not poison the memo for
    every later caller."""
    _write_schema(
        tmp_path,
        {
            "schema": "poison_probe",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank_profiles": [_FLOAT_PROFILE],
        },
    )

    first = extract_all_ranking_strategies(tmp_path)
    assert "s" in first
    first["INJECTED_SCHEMA"] = {}
    del first["s"]

    second = extract_all_ranking_strategies(tmp_path)
    assert "INJECTED_SCHEMA" not in second, "caller mutation poisoned the memo"
    assert "s" in second


def test_memo_hit_returns_fresh_nested_strategy_values(tmp_path):
    _write_schema(
        tmp_path,
        {
            "schema": "poison_probe",
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank-profiles": [_FLOAT_PROFILE],
        },
    )

    first = extract_all_ranking_strategies(tmp_path)
    first["s"]["float_float"].inputs["injected"] = "tensor<float>(x[1])"
    first["s"]["float_float"].query_tensors_needed.append("injected")
    first["s"]["injected_strategy"] = first["s"]["float_float"]

    second = extract_all_ranking_strategies(tmp_path)

    assert set(second["s"]) == {"float_float"}
    assert second["s"]["float_float"].inputs == {"qt": "tensor<float>(x[128])"}
    assert second["s"]["float_float"].query_tensors_needed == ["qt"]


# nearestNeighbor is derived structurally from the schema: the profile's first
# phase must score against a dense 1-d embedding attribute. The previous
# profile-NAME allowlist silently dropped ANN for any profile named outside it
# (the wiki hybrid/semantic_search profiles ranked BM25-only with a dead
# closeness term), while text-first profiles must stay off ANN.

_WIKI_FIELDS = {
    "fields": [
        {"name": "title", "type": "string"},
        {"name": "content", "type": "string"},
        {"name": "embedding", "type": "tensor<float>(d0[768])"},
    ]
}


def test_wiki_hybrid_profile_gets_nearestneighbor(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "name": "wiki_pages",
            "document": _WIKI_FIELDS,
            "rank_profiles": [
                {
                    "name": "hybrid",
                    "inputs": [{"name": "query(q)", "type": "tensor<float>(d0[768])"}],
                    "first_phase": {
                        "expression": (
                            "0.6 * closeness(field, embedding) "
                            "+ 0.2 * bm25(title) + 0.2 * bm25(content)"
                        )
                    },
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["hybrid"]
    assert s.use_nearestneighbor is True
    assert s.nearestneighbor_field == "embedding"
    assert s.nearestneighbor_tensor == "q"
    assert s.needs_text_query is True


def test_wiki_semantic_search_profile_gets_nearestneighbor(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "name": "wiki_pages",
            "document": _WIKI_FIELDS,
            "rank_profiles": [
                {
                    "name": "semantic_search",
                    "inputs": [{"name": "query(q)", "type": "tensor<float>(d0[768])"}],
                    "first_phase": {"expression": "closeness(field, embedding)"},
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["semantic_search"]
    assert s.use_nearestneighbor is True
    assert s.nearestneighbor_field == "embedding"
    assert s.nearestneighbor_tensor == "q"


_SV_FIELDS = {
    "fields": [
        {"name": "video_title", "type": "string"},
        {"name": "embedding", "type": "tensor<float>(v[768])"},
        {"name": "embedding_binary", "type": "tensor<int8>(v[96])"},
    ]
}


def test_function_indirection_resolves_to_nearestneighbor(tmp_path):
    """hybrid_float_bm25 hides closeness behind the visual_sim function."""
    path = _write_schema(
        tmp_path,
        {
            "name": "video_test_sv_chunk",
            "document": _SV_FIELDS,
            "rank_profiles": [
                {
                    "name": "hybrid_float_bm25",
                    "inputs": [{"name": "query(qt)", "type": "tensor<float>(v[768])"}],
                    "functions": [
                        {
                            "name": "visual_sim",
                            "expression": "closeness(field, embedding)",
                        },
                        {"name": "text_sim", "expression": "bm25(video_title)"},
                    ],
                    "first_phase": "visual_sim",
                    "second_phase": {"expression": "text_sim", "rerank_count": 100},
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["hybrid_float_bm25"]
    assert s.use_nearestneighbor is True
    assert s.nearestneighbor_field == "embedding"
    assert s.nearestneighbor_tensor == "qt"


def test_attribute_first_phase_uses_binary_pair(tmp_path):
    """float_binary scores query(qt) against the unpacked binary attribute —
    ANN retrieval must target the binary field with the binary tensor."""
    path = _write_schema(
        tmp_path,
        {
            "name": "video_test_sv_chunk",
            "document": _SV_FIELDS,
            "rank_profiles": [
                {
                    "name": "float_binary",
                    "inputs": [
                        {"name": "query(qtb)", "type": "tensor<int8>(v[96])"},
                        {"name": "query(qt)", "type": "tensor<float>(v[768])"},
                    ],
                    "functions": [
                        {
                            "name": "unpack_binary_representation",
                            "expression": "2*unpack_bits(attribute(embedding_binary)) - 1",
                        }
                    ],
                    "first_phase": "sum(query(qt) * unpack_binary_representation, v)",
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["float_binary"]
    assert s.use_nearestneighbor is True
    assert s.nearestneighbor_field == "embedding_binary"
    assert s.nearestneighbor_tensor == "qtb"


def test_binary_first_phase_prefers_binary_tensor(tmp_path):
    """phased retrieves by binary closeness and reranks float — ANN must pair
    the binary field with qtb even though qt is also declared."""
    path = _write_schema(
        tmp_path,
        {
            "name": "video_test_sv_chunk",
            "document": _SV_FIELDS,
            "rank_profiles": [
                {
                    "name": "phased",
                    "inputs": [
                        {"name": "query(qtb)", "type": "tensor<int8>(v[96])"},
                        {"name": "query(qt)", "type": "tensor<float>(v[768])"},
                    ],
                    "first_phase": "closeness(field, embedding_binary)",
                    "second_phase": {
                        "expression": "sum(query(qt) * unpack, v)",
                        "rerank_count": 100,
                    },
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["phased"]
    assert s.use_nearestneighbor is True
    assert s.nearestneighbor_field == "embedding_binary"
    assert s.nearestneighbor_tensor == "qtb"


def test_text_first_phase_stays_off_ann(tmp_path):
    """hybrid_bm25_binary retrieves by text and reranks by vector — retrieval
    must stay text-driven, not switch to ANN."""
    path = _write_schema(
        tmp_path,
        {
            "name": "video_test_sv_chunk",
            "document": _SV_FIELDS,
            "rank_profiles": [
                {
                    "name": "hybrid_bm25_binary",
                    "inputs": [{"name": "query(qtb)", "type": "tensor<int8>(v[96])"}],
                    "functions": [
                        {
                            "name": "visual_sim",
                            "expression": "closeness(field, embedding_binary)",
                        },
                        {"name": "text_sim", "expression": "bm25(video_title)"},
                    ],
                    "first_phase": "text_sim",
                    "second_phase": {"expression": "visual_sim", "rerank_count": 100},
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["hybrid_bm25_binary"]
    assert s.use_nearestneighbor is False


def test_mapped_tensor_field_never_ann(tmp_path):
    """Multi-vector (mapped) embedding fields cannot be ANN targets, whatever
    the profile is named."""
    path = _write_schema(
        tmp_path,
        {
            "name": "code_lateon_mv",
            "document": {
                "fields": [
                    {
                        "name": "embedding",
                        "type": "tensor<float>(patch{}, v[48])",
                    }
                ]
            },
            "rank_profiles": [
                {
                    "name": "float_float",
                    "inputs": [
                        {
                            "name": "query(qt)",
                            "type": "tensor<float>(querytoken{}, v[48])",
                        }
                    ],
                    "functions": [
                        {
                            "name": "max_sim",
                            "expression": (
                                "sum(reduce(sum(query(qt) * attribute(embedding), v),"
                                " max, patch), querytoken)"
                            ),
                        }
                    ],
                    "first_phase": "max_sim",
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["float_float"]
    assert s.use_nearestneighbor is False


def test_substring_text_in_name_does_not_classify_text(tmp_path):
    """A profile whose NAME merely embeds the letters 'text' (context_boost)
    but ranks purely by closeness must stay PURE_VISUAL."""
    from cogniverse_vespa.ranking_strategy_extractor import SearchStrategyType

    path = _write_schema(
        tmp_path,
        {
            "name": "video_test_sv_chunk",
            "document": _SV_FIELDS,
            "rank_profiles": [
                {
                    "name": "context_boost",
                    "inputs": [{"name": "query(qt)", "type": "tensor<float>(v[768])"}],
                    "first_phase": "closeness(field, embedding)",
                }
            ],
        },
    )
    s = RankingStrategyExtractor().extract_from_schema(path)["context_boost"]
    assert s.strategy_type is SearchStrategyType.PURE_VISUAL


# Visual-first hybrids score every document by an embedding in their first
# phase, so retrieval must not be narrowed to text matches; text-first
# hybrids and text strategies retrieve by text.
_FIRST_PHASE_EMBEDDING_FIELDS = {
    "video_colpali_smol500_mv_frame": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
        "hybrid_float_bm25_no_description": "embedding",
        "hybrid_binary_bm25_no_description": "embedding_binary",
        "hybrid_bm25_float": None,
        "hybrid_bm25_binary": None,
        "bm25_only": None,
        "float_float": "embedding",
    },
    "video_colqwen_omni_mv_chunk_30s": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
        "hybrid_float_bm25_no_description": "embedding",
        "hybrid_binary_bm25_no_description": "embedding_binary",
        "hybrid_bm25_float": None,
        "hybrid_bm25_binary": None,
    },
    "image_colpali_mv": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
    },
    "document_visual": {
        "hybrid_float_bm25": "colpali_embedding",
        "hybrid_binary_bm25": "colpali_embedding_binary",
    },
    "document_text": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
    },
    "lateon_mv": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
    },
    "code_lateon_mv": {"hybrid_float_bm25": "embedding"},
    "knowledge_graph": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
    },
    "audio_content": {
        "hybrid_semantic_bm25": "semantic_embedding_binary",
        "hybrid_acoustic_bm25": "acoustic_embedding",
        "transcript_search": None,
    },
    "video_xclip_sv_chunk_6s": {
        "hybrid_float_bm25": "embedding",
        "hybrid_binary_bm25": "embedding_binary",
        "hybrid_bm25_float": None,
        "hybrid_bm25_binary": None,
    },
}


@pytest.mark.unit
def test_hybrid_descriptions_name_what_ranks_them():
    strategies = RankingStrategyExtractor().extract_from_schema(
        _REPO_ROOT
        / "configs"
        / "schemas"
        / "video_colpali_smol500_mv_frame_schema.json"
    )

    assert {
        name: strategies[name].description
        for name in (
            "hybrid_float_bm25",
            "hybrid_binary_bm25",
            "hybrid_bm25_float",
            "hybrid_bm25_binary",
            "hybrid_bm25_binary_no_description",
        )
    } == {
        "hybrid_float_bm25": "Combined visual (float) and text search",
        "hybrid_binary_bm25": "Combined visual (binary) and text search",
        "hybrid_bm25_float": "Text-first search reranked by visual and text",
        "hybrid_bm25_binary": "Text-first search reranked by visual and text",
        "hybrid_bm25_binary_no_description": (
            "Text-first search reranked by visual and text (excluding descriptions)"
        ),
    }


@pytest.mark.unit
def test_single_vector_hybrids_retrieve_as_their_first_phase_scores():
    """The fused dense hybrids keep nearestNeighbor retrieval on the field
    their first phase scores; the text-first ones retrieve by text."""
    schemas = _REPO_ROOT / "configs" / "schemas"
    xclip = RankingStrategyExtractor().extract_from_schema(
        schemas / "video_xclip_sv_chunk_6s_schema.json"
    )
    audio = RankingStrategyExtractor().extract_from_schema(
        schemas / "audio_content_schema.json"
    )

    assert {
        name: (
            info.use_nearestneighbor,
            info.nearestneighbor_field,
            info.nearestneighbor_tensor,
            info.needs_text_query,
        )
        for name, info in (
            ("hybrid_float_bm25", xclip["hybrid_float_bm25"]),
            ("hybrid_binary_bm25", xclip["hybrid_binary_bm25"]),
            ("hybrid_bm25_float", xclip["hybrid_bm25_float"]),
            ("hybrid_bm25_binary", xclip["hybrid_bm25_binary"]),
            ("hybrid_acoustic_bm25", audio["hybrid_acoustic_bm25"]),
        )
    } == {
        "hybrid_float_bm25": (True, "embedding", "qt", True),
        "hybrid_binary_bm25": (True, "embedding_binary", "qtb", True),
        "hybrid_bm25_float": (False, None, None, True),
        "hybrid_bm25_binary": (False, None, None, True),
        "hybrid_acoustic_bm25": (True, "acoustic_embedding", "acoustic_query", True),
    }


@pytest.mark.unit
@pytest.mark.parametrize("schema", sorted(_FIRST_PHASE_EMBEDDING_FIELDS))
def test_first_phase_embedding_field_of_shipped_hybrids(schema):
    strategies = RankingStrategyExtractor().extract_from_schema(
        _REPO_ROOT / "configs" / "schemas" / f"{schema}_schema.json"
    )

    assert {
        name: strategies[name].first_phase_embedding_field
        for name in _FIRST_PHASE_EMBEDDING_FIELDS[schema]
    } == _FIRST_PHASE_EMBEDDING_FIELDS[schema]


@pytest.mark.unit
def test_first_phase_embedding_field_reaches_the_search_backend(tmp_path):
    """The search backend builds its YQL from the serialized strategy dict."""
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_vespa.search_backend import VespaSearchBackend

    for schema in ("video_colqwen_omni_mv_chunk_30s", "document_text"):
        source = _REPO_ROOT / "configs" / "schemas" / f"{schema}_schema.json"
        (tmp_path / source.name).write_text(source.read_text())
    backend = object.__new__(VespaSearchBackend)
    backend._schema_loader = FilesystemSchemaLoader(tmp_path)

    loaded = backend._load_ranking_strategies()

    assert {
        (schema, name): loaded[schema][name]["first_phase_embedding_field"]
        for schema in ("video_colqwen_omni_mv_chunk_30s", "document_text")
        for name in ("hybrid_float_bm25", "hybrid_binary_bm25", "bm25_only")
    } == {
        ("video_colqwen_omni_mv_chunk_30s", "hybrid_float_bm25"): "embedding",
        ("video_colqwen_omni_mv_chunk_30s", "hybrid_binary_bm25"): "embedding_binary",
        ("video_colqwen_omni_mv_chunk_30s", "bm25_only"): None,
        ("document_text", "hybrid_float_bm25"): "embedding",
        ("document_text", "hybrid_binary_bm25"): "embedding_binary",
        ("document_text", "bm25_only"): None,
    }


@pytest.mark.unit
def test_numeric_first_phase_attribute_is_not_an_embedding_field(tmp_path):
    """A hybrid whose first phase boosts text by a numeric attribute and
    reranks by an embedding retrieves by text: the attribute is no
    first-phase embedding field."""
    path = _write_schema(
        tmp_path,
        {
            "name": "boosted",
            "document": {
                "fields": [
                    {"name": "title", "type": "string"},
                    {"name": "popularity", "type": "double"},
                    {"name": "embedding", "type": "tensor<float>(token{}, v[4])"},
                ]
            },
            "rank_profiles": [
                {
                    "name": "hybrid_boosted_bm25",
                    "inputs": [
                        {
                            "name": "query(qt)",
                            "type": "tensor<float>(querytoken{}, v[4])",
                        }
                    ],
                    "first_phase": "bm25(title) + attribute(popularity)",
                    "second_phase": {
                        "expression": "sum(query(qt) * attribute(embedding))",
                        "rerank_count": 10,
                    },
                }
            ],
        },
    )

    strategy = RankingStrategyExtractor().extract_from_schema(path)[
        "hybrid_boosted_bm25"
    ]

    assert (strategy.needs_text_query, strategy.first_phase_embedding_field) == (
        True,
        None,
    )
