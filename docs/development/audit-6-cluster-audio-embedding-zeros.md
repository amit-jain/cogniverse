# Audit Cycle 6 — Cluster: acoustic embedding silently zeroed on failure

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `runtime/ingestion/processors/audio_embedding_generator.py` `generate_acoustic_embedding` | D | caught ANY exception (librosa decode, CLAP load, corrupt audio) and returned `np.zeros(512)`, which was then fed into Vespa — silently indexing a meaningless acoustic embedding | re-raise on failure; the ingestion path (`embedding_generator_impl.py:592-648`) already wraps each segment in `try/except`, so it now skips and records the failure instead of indexing zeros |

## Test (`tests/ingestion/unit/test_audio_embedding_failure.py`)

`test_clap_failure_raises_instead_of_zero_vector` injects a raising CLAP
processor and asserts `generate_acoustic_embedding` raises. Pre-fix: `DID NOT
RAISE` (returned the zero vector).
