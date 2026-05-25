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
  `vllm/vllm-openai-cpu` serving `vidore/colpali-v1.3-hf` and binds
  `RemoteColPaliLoader` against it.
- **Real seed corpus**: three documents with real per-token ColPali
  embeddings (sunset / ocean / forest scenes with matching transcripts)
  fed into Vespa via `seeded_ranking_corpus`.

Each ranking strategy is a plain rank-profile-name string passed as the
`strategy` key of the `query_dict`. Every strategy is exercised
end-to-end through `VespaSearchBackend.search`:

| Class | Strategies | Inputs |
|---|---|---|
| Text-only | `bm25_only`, `bm25_no_description` | Text query |
| Visual | `float_float`, `binary_binary`, `float_binary`, `phased` | `query_embeddings` via `RemoteColPaliLoader.client.process_queries` |
| Hybrid | `hybrid_float_bm25`, `hybrid_binary_bm25`, `hybrid_bm25_binary`, `hybrid_bm25_float`, plus `_no_description` variants | Text + `query_embeddings` |

Each case asserts on the returned `List[SearchResult]`:
- Non-empty results from the seeded corpus.
- Descending relevance order by `result.score`.
- `result.document.metadata["source_id"]` falls within the seeded corpus.

## Prerequisites

- `docker` available (vLLM and Vespa run in containers).
- HuggingFace cache at `~/.cache/huggingface` (mounted into the vLLM
  container so `colpali-v1.3-hf` weights are reused across runs).
- ~12 GiB host RAM headroom (vLLM CPU sets
  `--gpu-memory-utilization 0.10` against a 123 GiB host by default).

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
        "profiles": {"test_colpali": {"schema_name": "video_colpali_smol500_mv_frame"}},
        "default_profiles": {"video": "test_colpali"},
    },
    config_manager=config_manager,
    schema_loader=schema_loader,
)

# Text-only — strategy is a rank-profile-name string
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
# (see test_ranking_strategies_real.py for how to encode a query
# through the vLLM sidecar via RemoteColPaliLoader.process_queries).
```
