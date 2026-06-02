# Audit Cycle 6 — Cluster: with_concurrency() was a silent no-op

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `runtime/ingestion/pipeline_builder.py:104` `with_concurrency` | E | set `self._max_concurrent` but `build()` never passed it to `VideoIngestionPipeline`; the pipeline's concurrency was a per-call `process_videos_concurrent(max_concurrent=3)` default, so the builder setting was dropped | `VideoIngestionPipeline.__init__` accepts+stores `max_concurrent`; `build()` threads `_max_concurrent`; `process_videos_concurrent(max_concurrent=None)` falls back to `self.max_concurrent` (per-call override still wins) |

## Test (`tests/ingestion/unit/test_pipeline_concurrency_wiring.py`, fails on pre-fix)

Patches `asyncio.Semaphore` to capture its size: with `max_concurrent=None` the
pipeline-configured 7 is used (pre-fix passed `None`); a per-call `2` overrides.
