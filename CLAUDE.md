# Multi-Agent RAG System

Video content analysis and search system with configurable processing pipelines.

## Key Components
- **Composing Agent**: Central orchestrator (ADK-based)
- **Video Search Agent**: ColPali/VideoPrism retrieval via Vespa
- **Video Processing Pipeline**: Configurable keyframe extraction, transcription, embeddings

## Development Guidelines

### Always use `uv run` for Python scripts in this project
```bash
# Ingestion
uv run python scripts/run_ingestion.py --video_dir data/testset/evaluation/sample_videos --backend vespa

# Comprehensive test with VideoPrism (requires JAX platform fix)
JAX_PLATFORM_NAME=cpu uv run python tests/comprehensive_video_query_test_v2.py --profiles direct_video_global direct_video_global_large frame_based_colpali --test-multiple-strategies

# Phoenix experiments with visualization (creates datasets and runs experiments)
uv run python scripts/run_experiments_with_visualization.py --dataset-path data/testset/evaluation/video_search_queries.csv --dataset-name golden_eval_v1 --profiles frame_based_colpali --test-multiple-strategies

# Phoenix dashboard (Analytics + Evaluation)
uv run streamlit run scripts/phoenix_dashboard_standalone.py --server.port 8501
```

### Vespa Schema Validation
- Check embedding dimensions: 128 (ColPali/ColQwen patch), 768 (base), 1024 (large)
- Binary format uses hex strings, float format uses actual floats
- Use pyvespa feed_iterable for ingestion (not raw HTTP)

### Common Error Patterns
- "Expected X values, got Y" → Format mismatch (hex vs float)
- "Connection aborted" → Batch size too large
- HTTP 400 → Schema/data format mismatch

### Testing Best Practices

**Core Principles**:
- Fix implementation to satisfy tests, never weaken tests
- 100% pass rate required before commit — 0 failed AND 0 skipped (infra skips = bugs)
- No shortcuts: no mocking away failures, no disabling, no hardcoding
- Always use `--tb=long` — short tracebacks hide root causes and force re-runs
- Tests must manage their own isolated infrastructure (unique ports, own Docker containers)

**Test Completeness for Wiring Changes**:
- Every change that wires components together (A saves, B loads) MUST have a round-trip integration test that exercises the full save-then-load path
- Constructor-acceptance tests ("it initializes without error") do NOT count as coverage for wiring correctness
- If the change fixes a specific bug (e.g., filename mismatch), write a test that would have caught the original bug — if it passes on the old broken code, it tests nothing
- Integration tests MUST exercise the real system boundary (e.g., actual Phoenix Docker instance) — mocking the boundary only proves internal wiring, not that the real system works. No exceptions.

**Tests Are Part of Implementation, Not a Separate Phase**:
- When implementing code that wires components together, write the round-trip integration test IN THE SAME PHASE as the code change — not "after all phases are done"
- Never ask permission to write tests that the rules already require. If the rules say "every wiring change needs a round-trip test", write the test. Period.
- Never defer tests to a later step, a "testing phase", or "Step N". The code is not done until its tests exist and pass.
- If you write a constructor that accepts a new dependency, the test that exercises that dependency's save→load path ships in the same commit.

**Documentation Updates Are Part of Implementation**:
- When changing a constructor signature, storage backend, config key, or public API, update the corresponding documentation IN THE SAME PHASE — not as a separate "documentation phase"
- Grep `docs/` for references to the old API/parameter/pattern and update them before moving to the next file
- If you change `storage_dir` to `telemetry_provider` in a class, find every doc that mentions `storage_dir` for that class and fix it immediately

**Development Testing**:
- Test with single video first: `--max-frames 1`
- Check logs: `tail -f outputs/logs/*.log`
- Never add base, simple, final, full, generic, comprehensive to class/file names

### Mandatory Pre-Commit Workflow

Route based on change type (see `.claude/rules/strict-commit.md` for full decision tree):

```
CODE ONLY  → lint-and-quality → quality-enforcer → commit-enforcer
DOCS ONLY  → doc-verifier → commit-enforcer
CODE+DOCS  → lint-and-quality → doc-verifier → quality-enforcer → commit-enforcer
```

**Agents**:
- **lint-and-quality** — ruff autofix + banned pattern scan + convention check (fast, static)
- **doc-verifier** — verify docs match actual code (imports, classes, configs, examples)
- **quality-enforcer** — test discovery + execution, `--tb=long`, 0 failed + 0 skipped
- **commit-enforcer** — imperative mood, no AI attribution, max 72 chars

**On demand**: `feature-dev:code-reviewer` (deep review), `codebase-integrity-auditor` (full audit)

## Project Status

**Architecture**: 11-package UV workspace (Foundation → Core → Implementation → Application)
**Status**: Production ready
**Documentation**: See `docs/` for comprehensive guides

## Development Rules

- Never try to shortcut the actual code by creating something on the side
- DO NOT EVER ASK TO COMMIT WITHOUT FIXING ANY KNOWN ISSUE THAT HAS COME ABOUT. EVEN IF THAT ISSUE HAS NOT BEEN INTRODUCED NOW
- Always run the tests will full logging output to a test log file
- Never ask permission to do what these instructions already require. If the rules say to write tests, write them. If the rules say to update docs, update them. Follow the instructions — do not ask "should I also do X?" when X is already mandated here.
- Each implementation step is not complete until: (1) code compiles, (2) tests for that code exist and pass, (3) docs referencing the changed API are updated. All three in the same step.