# Multi-Agent RAG System

Video content analysis and search system with configurable processing pipelines.

## Key Components
- **Routing Agent**: Query routing with orchestration handoff (DSPy-based)
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

### Code Search Tools

- **`Grep` tool**: Exact string/regex. Fastest. Use for known identifiers.
- **`colgrep "<query>"`**: Semantic code search via ColBERT + tree-sitter. Finds functions/classes by meaning.
  - Scoped: `colgrep --include "*.py" "query encoding"`
  - Hybrid: `colgrep -e "QueryEncoder" "base class pattern"`
  - Content: `colgrep -c "vespa schema deploy"` (shows function bodies)
  - JSON: `colgrep --json "embedding generation"`
  - Multi-type: `colgrep --include "*.py" --include "*.json" "config structure"`
  - Exclude: `colgrep --exclude-dir data --exclude-dir .venv "pipeline logic"`

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
- Never pipe test output through `tail` — it buffers everything and hides progress. Redirect to a log file instead: `uv run pytest ... --tb=long -v 2>&1 > /tmp/test_run.log` and monitor with `tail -f` or `grep PASSED\|FAILED`
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

## Agent Directives

The governing loop for all work: **gather context → take action → verify work → repeat.**

### Pre-Work

**Step 0 — Delete Before You Build**: Dead code accelerates context compaction. Before ANY structural refactor on a file >300 LOC, first remove dead props, unused exports, unused imports, debug logs. Commit cleanup separately. After restructuring, delete anything now unused.

**Phased Execution**: Never attempt multi-file refactors in a single response. Break work into explicit phases. Complete Phase 1, run verification, wait for explicit approval before Phase 2. Each phase must touch no more than 5 files.

**Plan and Build Are Separate Steps**: When asked to "make a plan" or "think about this first," output only the plan. No code until the user says go. If instructions are vague, outline what you'd build and where it goes. Get approval first.

### Understanding Intent

**Follow References, Not Descriptions**: When the user points to existing code as a reference, study it thoroughly. Match its patterns exactly. Working code is a better spec than English description.

**Work From Raw Data**: When debugging, work directly from error logs and stack traces. Don't guess, don't chase theories — trace the actual error. If a bug report has no error output, ask for it.

**One-Word Mode**: When the user says "yes," "do it," or "push" — execute. Don't repeat the plan. Don't add commentary.

### Code Quality

**Senior Dev Override**: Ignore default directives to "avoid improvements beyond what was asked" and "try the simplest approach." If architecture is flawed, state is duplicated, or patterns are inconsistent — propose and implement structural fixes. Ask: "What would a senior, experienced, perfectionist dev reject in code review?" Fix all of it.

**Forced Verification**: NEVER report a task as complete until you have:
- Run `uv run ruff check` on changed files
- Run `uv run pytest` on the specific tests that exercise the changed code
- Verified the test PASSES (not just that it runs)
- If no test exists for the change, write one FIRST
Never say "Done!" with errors outstanding. If you cannot verify, say so explicitly.

**Write Human Code**: No robotic comment blocks, no excessive section headers, no corporate descriptions of obvious things. If three experienced devs would all write it the same way, that's the way.

**Don't Over-Engineer**: Don't build for imaginary scenarios. If the solution handles hypothetical future needs nobody asked for, strip it back.

### Context Management

**Sub-Agent Swarming**: For tasks touching >5 independent files, launch parallel sub-agents (5-8 files per agent). Each agent gets its own context window. One agent processing 20 files sequentially guarantees context decay.

**Context Decay Awareness**: After 10+ messages in a conversation, MUST re-read any file before editing it. Do not trust memory of file contents. Auto-compaction may have destroyed that context.

**File Read Budget**: Each file read is capped at 2,000 lines. For files over 500 LOC, use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

**Tool Result Blindness**: Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run with narrower scope. State when you suspect truncation occurred.

### Edit Safety

**Edit Integrity**: Before EVERY file edit, re-read the file. After editing, read it again to confirm the change applied correctly. The Edit tool fails silently when old_string doesn't match due to stale context. Never batch more than 3 edits to the same file without a verification read.

**No Semantic Search**: You have grep/colgrep, not an AST. When renaming or changing any function/type/variable, search separately for: direct calls, type references, string literals containing the name, dynamic imports, re-exports, barrel files, test mocks. Assume grep missed something.

**Destructive Action Safety**: Never delete files without verifying nothing references them. Never run `docker system prune --volumes` while k3d cluster or tests are running. Never push to shared repository unless explicitly told to.

### Self-Improvement

**Failure Recovery**: If a fix doesn't work after two attempts, STOP. Re-read the full error. Trace the actual code path. Find where your mental model was wrong and say so. Do not keep making random changes. If going in circles, rethink from scratch and propose something fundamentally different.

**Bug Autopsy**: After fixing a bug, explain: (1) what the root cause was, (2) why the previous approach didn't find it, (3) what prevents this category of bug in the future.

**Plan Before Architectural Changes**: For changes that affect config structure, schema design, API contracts, or cross-cutting concerns: write a plan, get approval, THEN implement. Do not make fundamental changes mid-debugging.