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
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank-profiles": [_FLOAT_PROFILE],
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
            "document": {"fields": [{"name": "embedding", "type": "tensor"}]},
            "rank-profiles": [_FLOAT_PROFILE],
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
