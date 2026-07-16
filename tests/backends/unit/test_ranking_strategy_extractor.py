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

from cogniverse_vespa.ranking_strategy_extractor import (
    RankingStrategyExtractor,
    extract_all_ranking_strategies,
)

_FLOAT_PROFILE = {
    "name": "float_float",
    "inputs": [{"name": "query(qt)", "type": "tensor<float>(x[128])"}],
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
            "rank-profiles": [_FLOAT_PROFILE],
        },
    )

    strategies = RankingStrategyExtractor().extract_from_schema(path)

    assert strategies["float_float"].schema_name == "video_colpali_sv_test"


def test_lvt_schema_enables_nearest_neighbor(tmp_path):
    path = _write_schema(
        tmp_path,
        {
            "name": "video_videoprism_lvt_global",
            "document": {
                "fields": [{"name": "embedding", "type": "tensor<float>(x[128])"}]
            },
            "rank-profiles": [
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
            "rank-profiles": [
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
                "rank-profiles": [_FLOAT_PROFILE],
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
            "rank-profiles": [_FLOAT_PROFILE],
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


from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HYBRID_SCHEMAS = [
    "configs/schemas/video_colpali_smol500_mv_frame_schema.json",
    "configs/schemas/video_colqwen_omni_mv_chunk_30s_schema.json",
    "tests/system/resources/schemas/video_colpali_smol500_mv_frame_schema.json",
    "tests/system/resources/schemas/video_colqwen_omni_mv_chunk_30s_schema.json",
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


@pytest.mark.unit
@pytest.mark.parametrize("schema", _HYBRID_SCHEMAS)
def test_hybrid_rank_profiles_honor_phase_order_naming(schema):
    """A ``hybrid_binary_bm25*`` profile must be binary-first and a
    ``hybrid_bm25_binary*`` profile text-first; the ``_no_description`` pair
    were once byte-identical (both text-first), silently giving
    hybrid_binary_bm25_no_description the wrong phase order."""
    profiles = _rank_profiles(_REPO_ROOT / schema)
    for suffix in ("", "_no_description"):
        binary_first = profiles[f"hybrid_binary_bm25{suffix}"]
        text_first = profiles[f"hybrid_bm25_binary{suffix}"]
        assert binary_first["first_phase"] == "visual_sim_binary", (
            f"hybrid_binary_bm25{suffix} must rank binary first"
        )
        assert "text_sim" in json.dumps(text_first["first_phase"]), (
            f"hybrid_bm25_binary{suffix} must rank text/bm25 first"
        )
        assert binary_first["first_phase"] != text_first["first_phase"], (
            f"opposite-named hybrid profiles must differ (suffix={suffix!r})"
        )


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
            "rank-profiles": [_FLOAT_PROFILE],
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
            "rank-profiles": [_FLOAT_PROFILE],
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
            "rank-profiles": [_FLOAT_PROFILE],
        },
    )

    first = extract_all_ranking_strategies(tmp_path)
    assert "s" in first
    first["INJECTED_SCHEMA"] = {}
    del first["s"]

    second = extract_all_ranking_strategies(tmp_path)
    assert "INJECTED_SCHEMA" not in second, "caller mutation poisoned the memo"
    assert "s" in second


# --------------------------------------------------------------------------- #
# nearestNeighbor is derived structurally from the schema: the profile's first
# phase must score against a dense 1-d embedding attribute. The previous
# profile-NAME allowlist silently dropped ANN for any profile named outside it
# (the wiki hybrid/semantic_search profiles ranked BM25-only with a dead
# closeness term), while text-first profiles must stay off ANN.
# --------------------------------------------------------------------------- #

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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
            "rank-profiles": [
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
