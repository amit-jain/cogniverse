# Audit Cycle 6 — Cluster: SingleVectorVideoProcessor.process arg mismap

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `runtime/ingestion/processors/single_vector_processor.py:480` `process` | D | `process(video_path, output_dir=None, **kwargs)` called `process_video(video_path, output_dir)`, but `process_video`'s 2nd positional is `transcript_data` — so `output_dir` (a Path) landed in `transcript_data`, and real `transcript_data`/`metadata` in `**kwargs` were dropped | forward `transcript_data`/`metadata` by keyword from kwargs; `output_dir` kept only for BaseProcessor signature compat (unused) |

No production caller (the live path calls `process_video` directly) — this was
a latent landmine on the `BaseProcessor.process` interface.

## Test (`tests/ingestion/unit/test_single_vector_process_adapter.py`, fails on pre-fix)

`process(..., output_dir=..., transcript_data={...}, metadata={...})` → the spy
`process_video` receives the dict transcript_data/metadata. Pre-fix
`transcript_data` was the output_dir Path.
