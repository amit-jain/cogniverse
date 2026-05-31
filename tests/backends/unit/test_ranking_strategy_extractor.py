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
