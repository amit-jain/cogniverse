# Audit Cycle 6 — Cluster: `_schema` suffix strip via `str.replace`

Review summary for the Class-D "`str.replace("_schema", "")` to drop a
trailing suffix" cluster. `replace` removes **every** occurrence, so a schema
file whose logical name itself contains `_schema` (e.g. `code_schema_index`)
was mangled. Fixed with `str.removesuffix("_schema")` (Python 3.9+; repo runs
3.12).

Repo-wide sweep (`grep -rnP "\.stem\.replace\("` and
`\.replace\(['"]_schema['"]`) found exactly the two sites below.

## Findings & fixes

| # | Site | Class | Failure on happy path | Fix |
|---|------|-------|-----------------------|-----|
| 1 | `core/schemas/filesystem_loader.py:100` | D | `list_available_schemas` returned `f.stem.replace("_schema", "")` → `code_schema_index_schema.json` listed as `code_index`, which then fails `schema_exists` (looks for `code_index_schema.json`) | `f.stem.removesuffix("_schema")` |
| 2 | `vespa/ranking_strategy_extractor.py:312` | D | `extract_all_ranking_strategies` keyed strategies by `schema_file.stem.replace("_schema", "")` → wrong top-level key for any `*_schema*_schema.json` | `schema_file.stem.removesuffix("_schema")` |

## Tests (fail on pre-fix code)

| Finding | Test | Strong assertion |
|---------|------|------------------|
| 1 | `tests/core/unit/test_filesystem_loader.py` | `{video_schema.json, code_schema_index_schema.json}` → names `{"video", "code_schema_index"}`; the listed name round-trips through `schema_exists` |
| 2 | `tests/backends/unit/test_ranking_strategy_extractor.py::test_extract_all_strips_only_trailing_schema_suffix` | `code_schema_index_schema.json` → key `"code_schema_index"`, not `"code_index"` |
