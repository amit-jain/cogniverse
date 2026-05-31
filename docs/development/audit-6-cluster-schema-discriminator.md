# Audit Cycle 6 — Cluster: schema-name discriminator divergence

Review summary for the `ranking_strategy_extractor` half of the Class-C
"schema-name substring discriminator" cluster. Single-vector vs multi-vector
schema routing must use the one authoritative helper, and the persisted
`schema_name` must come from one source.

Authoritative helper: `cogniverse_vespa.embedding_processor._is_single_vector_schema`
— lower-cases and checks the token-bracketed `_sv_` / `_lvt_` tokens (covered by
`tests/backends/unit/test_schema_name_matching.py`, incl. the `audio_alvtree_index`
substring-rejection case).

All fixes local on `main` (unpushed); each ships a test failing on pre-fix code.

## Findings & fixes (`libs/vespa/cogniverse_vespa/ranking_strategy_extractor.py`)

| # | Site | Class | Failure on happy path | Fix |
|---|------|-------|-----------------------|-----|
| 1 | `_parse_ranking_profile:126` | E | recomputed `schema_name = schema_json.get("schema", "")`, dropping the `name` fallback used at `extract_from_schema:57` → schemas keyed by `name` persisted an empty `schema_name` | pass the already-resolved `schema_name` down (single source); drop the recompute + the now-unused `schema_json` param |
| 2 | `extract_from_schema:58` | C | `is_single_vector = "_sv_" in schema_name.lower()` missed `_lvt_` single-vector schemas (VideoPrism global) → they never enabled nearestNeighbor ANN | `is_single_vector = _is_single_vector_schema(schema_name)` — converge on the authoritative helper (also covers uppercase + rejects `lvt` substrings without token bounds) |

## Tests (`tests/backends/unit/test_ranking_strategy_extractor.py`, fail on pre-fix code)

| Finding | Test | Strong assertion |
|---------|------|------------------|
| 1 | `test_schema_name_populated_when_keyed_by_name` | a schema keyed by `name` yields `strategy.schema_name == "video_colpali_sv_test"` (was `""`) |
| 2 | `test_lvt_schema_enables_nearest_neighbor` | an `_lvt_` schema with a `float_float` profile → `use_nearestneighbor is True`, field `embedding`, tensor `qt` (was `False`) |
| — | `test_sv_schema_still_enables_nearest_neighbor` | regression guard: `_sv_` path unchanged (passes pre- and post-fix) |

## Not in this cluster (related, deferred)

- `vespa/config/config_store.py:341` raw YQL interpolation in `get_config(version=N)`
  (Class C, needs a real-Vespa round-trip test) — tracked separately.
- `core/query/encoders.py:194` VideoPrism `"lvt" in model_name.lower()` global-vs-patch
  routing — same discriminator family, needs the model-loader boundary.
