# TODO: Pipeline artifact cache — multi-pod / shared backend

**Status:** Tracked in [project audit remediation](../../libs/core/cogniverse_core/common/cache/pipeline_cache.py). Two pieces, ship independently. Approved scope: **A + B**.

## Why this exists

`PipelineArtifactCache` (`libs/core/cogniverse_core/common/cache/pipeline_cache.py`) is configured `enabled: true` in `configs/config.json:395` with the only registered backend, `structured_filesystem` (local disk at `~/.cache/cogniverse/structured_pipeline/`). Two compounding problems:

1. **Multi-pod blind spot.** Local-disk cache is per-pod-ephemeral. Cache hits only happen when *the same* worker pod re-processes *the same* video. With N pods, hit rate trends to 1/N; pod restart wipes it; useless for a worker pool.
2. **Live path bypasses it anyway.** `process_video_async_with_strategies` delegates to `ProcessingStrategySet.process()`, which never calls the cache. The comment at `libs/runtime/cogniverse_runtime/ingestion/processing_strategy_set.py:484` documents it: *"Don't pass async cache to sync transcribe_audio — the cache methods are async coroutines and can't be called from sync code."* So even on a single pod, re-ingest re-extracts keyframes / re-transcribes audio / re-runs VLM.

The infrastructure for a shared backend already exists — it just hasn't been built.

## Piece A — S3 / MinIO cache backend

MinIO is already deployed for media uploads (`libs/runtime/cogniverse_runtime/ingestion_worker/minio_client.py`, `MINIO_ENDPOINT`). Reuse the endpoint, credentials, and bucket convention for cache storage.

**Files to add / change:**

| Change | Where | Notes |
|---|---|---|
| New `S3CacheBackend(CacheBackend)` | `libs/core/cogniverse_core/common/cache/backends/s3.py` | Implements the 6 abstract methods (`get`/`set`/`delete`/`exists`/`clear`/`get_stats`) via boto3. Key format `pipeline-cache/{tenant}/{key}`. TTL via S3 object metadata + a periodic cleanup task, or via the bucket's lifecycle policy. |
| New `S3BackendConfig` | same file | Endpoint, access key, secret key, bucket, key prefix, region — read from `MINIO_*` env vars the way `minio_client._client()` does, with config-file overrides. |
| Register the backend | `libs/core/cogniverse_core/common/cache/__init__.py` (or wherever `structured_filesystem` is registered) | `CacheBackendRegistry.register("s3", S3CacheBackend)` |
| Drop the hardcoded `if backend_type == "structured_filesystem"` | `libs/core/cogniverse_core/common/cache/registry.py:create()` | Replace with a registry-driven dispatch: each backend class advertises its config dataclass via a class attribute (e.g. `CONFIG_CLASS`), and `create()` looks it up. |
| Update config | `configs/config.json` (`pipeline_cache.backends`) | See "Layering" below. |

**Layering** (the `CacheConfig.backends: List[BackendConfig]` machinery already iterates by priority):

```jsonc
"pipeline_cache": {
  "enabled": true,
  "backends": [
    { "backend_type": "structured_filesystem",  // L1: hot, per-pod, microsecond hits
      "base_path": "~/.cache/cogniverse/structured_pipeline",
      "priority": 0,
      ...
    },
    { "backend_type": "s3",                     // L2: shared, durable, survives pod restart
      "endpoint": "${MINIO_ENDPOINT}",
      "bucket": "cogniverse-pipeline-cache",
      "key_prefix": "pipeline/",
      "priority": 1
    }
  ]
}
```

**Tests** (the lint+tighten-assertions discipline from this audit):
- Unit: `S3CacheBackend.set` → `get` round-trips bytes / numpy / dict / image payloads against a fake boto3 client (assert exact key shape, exact bytes round-trip — no `is not None` checks).
- Real-service integration: a stand-up of MinIO (the same one used for media upload tests, if there is one) + a real round-trip via `S3CacheBackend` with each artifact type that flows through `set_keyframes` / `set_transcript` / `set_descriptions`. Assert hit rate after a re-ingest is 100% for keyframes/transcripts/descriptions when L2 is configured and L1 is empty (mimics pod restart on the same logical video).

## Piece B — wire the cache into the live strategy path

The bypass at `processing_strategy_set.py:484` was about an async-vs-sync impedance mismatch. The fix is not to thread async cache into sync extractors — it's to do the cache work in the async caller, around the sync extraction, so the extractors stay sync.

Pattern:

```python
async def _process_segmentation(self, strategy, video_path, processor_manager, pipeline_context):
    cache = pipeline_context.cache    # already on the pipeline
    if cache:
        cached = await cache.get_keyframes(str(video_path), ...)
        if cached:
            return cached

    # existing sync extraction (unchanged)
    result = await asyncio.to_thread(processor.extract_keyframes, video_path, ...)

    if cache:
        await cache.set_keyframes(str(video_path), result, ...)
    return result
```

Same pattern for `_process_transcription` and `_process_description`. Each call site:
- `keyframes`: `_process_segmentation` (around `processing_strategy_set.py:464`)
- `transcript`: `_process_transcription` (around line 484 — replace the `None` cache arg with the wrap-around pattern above)
- `descriptions`: `_process_description` (around line 510+, find the VLM dispatch)

**Tests**:
- Real-service: ingest the same video twice; second pass must hit cache for every cached artifact (assert exact counts via `get_stats()` on each backend). On multi-pod simulation (L1 cleared between calls), L2 still hits.
- Behaviour: a cache miss followed by a successful extraction must persist to **both** backends in a layered config (assert via `get` against each backend directly).

## Order of operations

1. Land **A** first (new backend + registration + config). No behavioural change yet — but `CacheManager` now has a shared backend available. Ship behind config so production opts in.
2. Land **B** (strategy-set wiring). Behaviour flips: re-ingest now hits cache on the live path. On multi-pod, hit rate is now meaningful.
3. After both are in production for one cycle: revisit TTL policy and lifecycle rules on the MinIO bucket.

## What this is NOT

- Not a Redis cache (transcripts are small JSON, fine in S3 as L2; if hot-path access patterns later show the need, Redis fits the same `CacheBackend` registry).
- Not a replacement for the existing media-input MinIO bucket. **Different bucket / different prefix**. Confusing them was the original misunderstanding worth recording.
- Not in scope for the current 5th-audit remediation cycle. Tracked here for the next cycle.
