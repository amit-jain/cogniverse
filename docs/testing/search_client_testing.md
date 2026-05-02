# Search Client Testing Guide

This guide covers testing the Vespa search client with all available
ranking strategies. Coverage lives in
`tests/runtime/integration/test_ranking_strategies_real.py` — a
parametrized integration test that drives every `RankingStrategy`
variant against a real Vespa container with seeded ColPali embeddings.

## Run the test

```bash
uv run pytest tests/runtime/integration/test_ranking_strategies_real.py -v
```

## What it covers

The test fixture chain builds the full real backend on demand:

- **Real Vespa**: `vespa_instance` fixture spawns a Docker Vespa with
  `video_colpali_smol500_mv_frame_test_unit` deployed.
- **Real vLLM ColPali**: `vllm_sidecar` fixture spawns
  `vllm/vllm-openai-cpu` serving `vidore/colpali-v1.3-hf` and binds
  `RemoteColPaliLoader` against it.
- **Real seed corpus**: three documents with real per-token ColPali
  embeddings (sunset / ocean / forest scenes with matching transcripts)
  fed into Vespa via `seeded_ranking_corpus`.

Each `RankingStrategy` enum variant is exercised end-to-end:

| Class | Strategies | Inputs |
|---|---|---|
| Text-only | `bm25_only`, `bm25_no_description` | Text query |
| Visual | `float_float`, `binary_binary`, `float_binary`, `phased` | Query embeddings via `RemoteColPaliLoader.client.process_queries` |
| Hybrid | `hybrid_float_bm25`, `hybrid_binary_bm25`, `hybrid_bm25_binary`, `hybrid_bm25_float`, plus `_no_description` variants | Text + query embeddings |

Each case asserts:
- Non-empty results from the seeded corpus.
- Descending relevance order.
- Result `video_id` falls within the seeded corpus.

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
from cogniverse_vespa.vespa_search_client import VespaVideoSearchClient
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
client = VespaVideoSearchClient(
    backend_url="http://localhost",
    backend_port=8080,
    tenant_id="test_tenant",
    config_manager=config_manager,
)

# Text-only
results = client.search({
    "query": "buck",
    "ranking": "bm25_only",
    "top_k": 3,
    "schema": "video_colpali_smol500_mv_frame",
})

# Visual / hybrid — pass pre-computed embeddings via `embeddings=`
# (see test_ranking_strategies_real.py for how to encode a query
# through the vLLM sidecar via RemoteColPaliLoader.process_queries).
```
