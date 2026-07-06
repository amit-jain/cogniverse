# Ranking Strategy Testing Guide

This guide covers testing the production Vespa search path
(`VespaSearchBackend.search(query_dict)`) with all available ranking
strategies. Coverage lives in
`tests/runtime/integration/test_ranking_strategies_real.py` — a
parametrized integration test that drives every ranking strategy (each a
rank-profile-name string) against a real Vespa container with seeded
ColPali embeddings.

## Run the test

```bash
uv run pytest tests/runtime/integration/test_ranking_strategies_real.py -v
```

## What it covers

The test fixture chain builds the full real backend on demand:

- **Real Vespa**: `vespa_instance` fixture is a compatibility shim backed by
  the session-scoped `shared_vespa` container (see `tests/conftest.py`), with
  `video_colpali_smol500_mv_frame_test_unit` deployed at test setup.
- **Real vLLM ColPali**: `vllm_sidecar` fixture spawns
  `vllm/vllm-openai-cpu:v0.23.0` serving `TomoroAI/tomoro-colqwen3-embed-4b`
  (pooling runner, `embed` convert mode, 320-dim per-token embeddings) and
  binds `RemoteColPaliLoader` against it.
- **Real seed corpus**: three documents with real per-token ColPali
  embeddings (sunset / ocean / forest scenes with matching transcripts)
  fed into Vespa via `seeded_ranking_corpus`.

There is also `TestAutoSelectDefaultRanking`, which omits `strategy`
entirely and asserts the backend falls back to the schema's `default`
rank profile (the contract `SearchAgent` relies on after dropping its
hardcoded `binary_binary` default).

Each ranking strategy is a plain rank-profile-name string passed as the
`strategy` key of the `query_dict`. Every strategy is exercised
end-to-end through `VespaSearchBackend.search`:

| Class | Strategies | Inputs |
|---|---|---|
| Text-only | `bm25_only`, `bm25_no_description` | Text query |
| Visual | `float_float`, `binary_binary`, `float_binary`, `phased` | `query_embeddings` via `RemoteColPaliLoader(...).load_model()` client's `process_queries` |
| Hybrid | `hybrid_float_bm25`, `hybrid_binary_bm25`, `hybrid_bm25_binary`, `hybrid_bm25_float`, plus `_no_description` variants | Text + `query_embeddings` |

Each case asserts on the returned `List[SearchResult]`:
- Non-empty results from the seeded corpus.
- Descending relevance order by `result.score`.
- `result.document.metadata["source_id"]` falls within the seeded corpus.

## Prerequisites

- `docker` available (vLLM and Vespa run in containers).
- HuggingFace cache at `~/.cache/huggingface` (mounted into the vLLM
  container so `tomoro-colqwen3-embed-4b` weights are reused across runs).
- Host RAM headroom for the vLLM CPU sidecar: `tests/utils/vllm_sidecar.py`
  sets `VLLM_CPU_MEMORY_UTILIZATION=0.05` and `VLLM_CPU_KVCACHE_SPACE=2`
  (GiB) on the container, plus a merged `--gpu-memory-utilization 0.10`
  and (for colqwen3 models) `--limit-mm-per-prompt {"video":0,"image":1}`
  to avoid a vision-tower startup OOM.
- Environment variables `BACKEND_URL` / `BACKEND_PORT` set (or a
  `configs/config.json` with `backend.url`/`backend.port`) if constructing
  a `ConfigManager` via `create_default_config_manager()` outside the test
  fixtures, e.g. for the manual REPL usage below.

## Strategy reference

| Strategy | Type | Speed | Accuracy | Requirements |
|----------|------|--------|----------|-------------|
| `bm25_only` | Text | Fast | Good | Text query |
| `bm25_no_description` | Text | Fast | Good | Text query |
| `float_float` | Visual | Slow | Highest | Embeddings |
| `binary_binary` | Visual | Fastest | Good | Embeddings |
| `float_binary` | Visual | Fast | Very Good | Embeddings |
| `phased` | Visual | Fast | High | Embeddings |
| `hybrid_float_bm25` | Hybrid | Slow | Highest | Text + embeddings |
| `hybrid_binary_bm25` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_binary` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_float` | Hybrid | Medium | Very Good | Text + embeddings |
| `hybrid_float_bm25_no_description` | Hybrid | Slow | High | Text + embeddings |
| `hybrid_binary_bm25_no_description` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_binary_no_description` | Hybrid | Fast | Good | Text + embeddings |
| `hybrid_bm25_float_no_description` | Hybrid | Medium | Very Good | Text + embeddings |

The schema also defines a `default` rank profile (not in the table above,
since no test parametrizes it by name) that `TestAutoSelectDefaultRanking`
exercises indirectly by omitting `strategy` from the query dict.

## Manual usage from a Python REPL

```python
from cogniverse_vespa.search_backend import VespaSearchBackend
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

backend = VespaSearchBackend(
    config={
        "url": "http://localhost",
        "port": 8080,
        "profiles": {
            "test_colpali": {
                "type": "video",
                "schema_name": "video_colpali_smol500_mv_frame",
            }
        },
        # default_profiles[type] is a dict with "profile" (and optional
        # "strategy") keys — only consulted when >1 profile exists for a
        # type and query_dict omits "profile"/"strategy".
        "default_profiles": {"video": {"profile": "test_colpali"}},
    },
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Text-only — strategy is a rank-profile-name string. "schema_name" above
# is the base name; search() appends the tenant suffix internally, so
# tenant_id="test:unit" resolves to "video_colpali_smol500_mv_frame_test_unit".
results = backend.search({
    "query": "buck",
    "type": "video",
    "profile": "test_colpali",
    "strategy": "bm25_only",
    "top_k": 3,
    "tenant_id": "test:unit",
})

for result in results:
    print(result.score, result.document.metadata["source_id"])

# Visual / hybrid — pass pre-computed embeddings via `query_embeddings`
# (see test_ranking_strategies_real.py for how to encode a query through
# the vLLM sidecar: RemoteColPaliLoader(...).load_model() returns a client
# whose .process_queries is bound to RemoteInferenceClient.process_queries_vllm).
```

## Query tensor format

`VespaSearchBackend` formats query embeddings for Vespa via `_format_query_vector_param`
(defined in `cogniverse_vespa/search_backend.py`):

- **Single-vector schemas** (LVT sv_chunk): numpy array flattened to a dense list —
  `tensor(v[dim])` binding (e.g. `tensor<float>(v[768])`). A `(1, dim)` array is
  flattened before serialisation.
- **Multi-vector schemas** (ColPali, VideoPrism mv_chunk): numpy array converted to a
  `{str(token_index): vector_list}` dict — `tensor<float>(querytoken{}, v[dim])` or
  `tensor<int8>(querytoken{}, v[dim])` binding (the query-side input tensor; the
  *stored* document embedding field uses `tensor<bfloat16>(patch{}, v[dim])`,
  which is a different tensor from what queries bind to).

Schema arity is determined by `_is_single_vector_schema(schema_name)` from
`cogniverse_vespa.embedding_processor`. Pass the raw numpy array from your encoder;
the formatting is handled internally.

The bound input name depends on the rank profile's declared inputs (see
`configs/schemas/ranking_strategies.json`):

| Input name | Type | Used by |
|---|---|---|
| `qt` | float | ColPali/VideoPrism `float_float`, `float_binary`, `phased`, `hybrid_*float*` strategies |
| `qtb` | int8 (binary) | ColPali/VideoPrism `binary_binary`, `float_binary`, `phased`, `hybrid_*binary*` strategies |
| `acoustic_query` | float | audio schema strategies only (e.g. `audio_content`); same code branch as `qt` |
| `q` | generic | schemas whose rank profiles declare a bare `q` input (e.g. `wiki_pages`, `agent_memories`) |

Unrecognised input names are logged and skipped rather than raising, so a
schema/strategy mismatch surfaces as missing ranking signal, not an error.

## Related `VespaSearchBackend` coverage

This guide's real-Vespa/real-vLLM sweep in `test_ranking_strategies_real.py`
covers rank-profile correctness only. Other aspects of `VespaSearchBackend`
are covered by separate test files:

| File | Covers |
|---|---|
| `tests/backends/unit/test_build_query_inputs.py` | `_build_query` binds every declared float input (`qt`/`qtb`/`q`/`acoustic_query`), not just the common ones |
| `tests/backends/unit/test_filter_condition_quoting.py` | `_build_filter_conditions` produces well-formed, correctly quoted YQL for every filter shape |
| `tests/backends/unit/test_search_backend_dynamic_profiles.py` | Runtime `add_profile` / `remove_profile` on a live backend instance |
| `tests/backends/unit/test_profile_change_listener_chain.py` | `ConfigManager.add_backend_profile` → `profile_change_listener` → `BackendRegistry.add_profile_to_backends` → `VespaSearchBackend.add_profile` wiring |
| `tests/backends/unit/test_search_metrics.py` | `SearchMetrics` latency window stays bounded |
| `tests/backends/integration/test_dynamic_profile_search_visibility.py` | A profile added at runtime is immediately searchable (real Vespa) |
| `tests/runtime/integration/test_dynamic_profile_visibility.py` | `POST /admin/profiles` → `backend.search()` sees the new profile end-to-end |
| `tests/runtime/integration/test_search_integration.py` | Full router → ConfigManager → BackendRegistry → SchemaLoader wiring with a real ColPali query encoder |
| `tests/runtime/integration/test_export_embeddings_real_vespa.py` | `export_embeddings` filtering and Document-v1 selection escaping |
| `tests/runtime/integration/test_tenant_extensibility.py` | Tenant-scoped instructions/memory round-tripped through the real Vespa config store |
