# MANDATORY PRE-COMMIT PROTOCOL

After making code changes, run agents IN ORDER based on what changed. Never skip any. Never ask permission.

---

## Change Type Detection

Check `git diff --name-only HEAD` and route to the appropriate workflow:

```
CODE ONLY (*.py, configs/, *.toml) — no API/interface changes
  → lint-and-quality → quality-enforcer → commit-enforcer

CODE WITH API CHANGES (*.py that adds/removes/renames public classes,
  methods, config keys, storage backends, or constructor signatures)
  → lint-and-quality → doc-verifier → quality-enforcer → commit-enforcer

DOCS ONLY (docs/*.md)
  → doc-verifier → commit-enforcer

CODE + DOCS
  → lint-and-quality → doc-verifier → quality-enforcer → commit-enforcer

ON DEMAND (user request or periodic)
  → feature-dev:code-reviewer   (deep static analysis)
  → codebase-integrity-auditor  (full codebase audit)
```

**How to detect API changes**: After lint-and-quality passes, grep the diff for:
- New/removed class definitions, public method signatures, or `__init__` parameter changes
- Changed storage backends (filesystem → telemetry, local → remote)
- Renamed or removed config keys
If ANY match, route through doc-verifier even if no docs/*.md files changed.

### Out of scope for this protocol

The routing tree above covers `libs/`, `docs/`, `configs/`, `tests/`, `scripts/`, `*.toml`. The following are explicitly **out of scope** — only `commit-enforcer` (commit message format) applies, no other agent runs:

| Path | Reason |
|---|---|
| `.claude/rules/*.md` (this file, `audit.md`, etc.) | Operational protocols, not project documentation. `doc-verifier` checks `docs/` against `libs/` — `.claude/rules/` describes *process*, not code APIs, so the verifier has nothing to verify against. |
| `.claude/agents/*.md` | Agent definitions; same reason as above. |
| Root-level `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `MEMORY.md` | Project-level meta-files, not module documentation. |
| `~/.claude/projects/.../memory/*.md` | User memory, owned by the harness; never committed. |

A change to any of the above passes through `commit-enforcer` only. **If you want a structural gate on operational rule consistency** (rule files don't contradict each other or CLAUDE.md; referenced paths resolve; hunt-list regexes still match what they claim to), write a new agent — `doc-verifier` is the wrong tool for it.

---

## Agent 1: lint-and-quality

**What**: Fast static analysis — ruff lint + ruff format + banned pattern scan + convention check.
**When**: Any code change (*.py, configs/, *.toml).
**Must pass**:
- `uv run ruff check <paths>` — zero errors (autofix allowed)
- `uv run ruff format --check <paths>` — zero reformatting required
- Zero banned patterns, zero convention violations.
**If fails**: Fix all issues before proceeding. Never suppress with `# noqa`.

**Why both**: CI runs `ruff check` and `ruff format --check` as separate steps per
module (e.g., `.github/workflows/vespa-tests.yml` → `lint` job). A file can pass
`ruff check` but fail `ruff format --check` — they enforce different things (lint
vs. formatter). Running only one locally lets the other slip into CI.

---

## Agent 2: doc-verifier

**What**: Verifies documentation against actual codebase — imports, classes, methods, configs, examples.
**When**: Any docs change (docs/*.md), or code changes that affect documented APIs.
**Must pass**: 100% of documentation synchronized with code AND every touched subpackage has at least one module-guide reference (see "Doc Coverage Check" below).
**If fails**: Fix documentation to match actual code (or fix code if docs are correct).

### Doc Coverage Check (mandatory, every code-touching commit)

`doc-verifier` only audits docs that **exist**. A subpackage with no module guide at all silently passes — that's the gap that let `ingestion_worker/` (formerly `ingestion_v2/`) ship 9+ modules with zero documentation until a user flagged it.

After lint-and-quality passes, run this check **before invoking doc-verifier**:

```bash
# For every libs/<pkg>/<subpkg>/ directory touched in the diff,
# verify the subpkg name appears in at least one docs/modules/*.md.
for subpkg in $(git diff --name-only HEAD~1 HEAD -- 'libs/*/cogniverse_*/' \
                  | awk -F/ '/^libs\// {print $3"/"$4}' | sort -u); do
  if ! grep -rln "$subpkg" docs/modules/ >/dev/null 2>&1; then
    echo "MISSING DOC COVERAGE: $subpkg has no docs/modules/ reference"
  fi
done
```

When the check reports MISSING:
1. If the touched directory is a NEW subpackage → add a Package Structure section in the most relevant `docs/modules/*.md` (or create one if the parent module is also new).
2. If the touched directory is an EXISTING subpackage that was never documented → same fix, in the same commit. "It was always undocumented" is not an exemption.
3. Only after the doc exists may `doc-verifier` proceed to validate its contents.

This check is mandatory. Skipping it is a violation, not a triage step.

---

## Agent 3: quality-enforcer

**What**: Test discovery and execution — grep for affected tests, run with timeouts, verify 100% pass.
**When**: Any code change.
**Must pass**: 0 failed AND 0 skipped. Uses `--tb=long` ALWAYS.
**If fails**: Fix implementation (never weaken tests). Re-run until 100% passing.
**Wiring coverage**: For changes that connect components (A writes, B reads), verify at least one test exercises the full round-trip (save → load → assert equality). Flag if only constructor-acceptance tests exist.
**Reject if tests are missing**: If code changes wire components together but no round-trip integration test exists, quality-enforcer MUST fail the check — even if all existing tests pass. Missing tests = incomplete implementation.

**Self-audit the diff (mandatory — moves audit findings to commit-time)**: Before approving, run these against `git diff HEAD` for the change, so a class of bug that a future audit would catch is caught here instead:
- **Class C hunt-list** — run the relevant regexes from `.claude/rules/audit.md` (Class C hunt list) over the diff's added lines. Any new hit is a blocker, fixed in THIS commit — a syntactic footgun never ships to wait for an audit. (This is the local half of CI-gating the hunt-list; the CI job is the other half.)
- **Concurrency + fault tests present** — if the diff adds shared/cached state, an async path, a new store/manager/backend read, or a boundary call, REJECT unless it ships a concurrency (F) and a fault-injection (G) regression test that executes the interleaving / failure path (per CLAUDE.md → "Assert Concurrency and Failure at Feature-Time"). Happy-path tests alone fail the check, even when green.
- **Remediation is not exempt** — a fix commit self-audits its own diff by the same two rules. "The next audit will catch it" is a banned deferral.

**Baseline-aware invocation**: When you have already run tests manually during implementation, pass a `BASELINE ALREADY VERIFIED:` block in the prompt with the specific test files you've verified. The agent runs only import-graph-reachable tests NOT in that list, then reports baseline vs delta. Skip the baseline directive and let it run the full scope when the diff touches cross-cutting concerns (config, backends, registries, telemetry, memory, schema) or is large (20+ files) — transitive regressions via those layers can invalidate the baseline.

---

## Agent 4: commit-enforcer

**What**: Formulates commit message and creates the git commit.
**When**: After all validation agents pass.
**Must pass**: Imperative mood, max 72 chars, no AI attribution, no meta-commentary.
**If fails**: Adjust commit message format.

---

## On-Demand Agents

### feature-dev:code-reviewer
**What**: Deep static analysis — bugs, logic errors, security vulnerabilities, code quality.
**When**: User requests a code review, or before merging complex changes.

### codebase-integrity-auditor
**What**: Full codebase audit — broken implementations, misleading tests, outdated docs.
**When**: User requests periodic health check, or after major refactors.

---

## Commit Message Rules

FORBIDDEN:
- "Claude", "Assistant", "AI", "Generated"
- Any AI attribution or co-author lines
- Test counts ("fixes 5 tests", "all tests pass")
- Phase markers ("Phase 1", "Step 2")
- Meta-commentary ("as discussed", "per request")

REQUIRED format:
- Imperative mood verb first: Add, Fix, Update, Refactor, Remove
- Technical description of WHAT changed (not WHY or HOW)
- Max 72 chars first line

Example: `Fix embedding dimension mismatch in VideoPrism processor`

---

## Assertions Before Code (prerequisite for every feature)

Before writing production code for a feature, write down the strongest
assertions a real-service integration test could make against it —
exact values, exact set/dict shapes, exact substrings with surrounding
context, exact length range, exact persisted row in the backing store.
These are the contract the implementation owes.

Forbidden weak assertions: `assert x is not None`,
`assert isinstance(out, str) and out.strip()`,
`assert "kw" in output` without bounds, `assert len(hits) >= 1`. If the
strongest assertion you can write is "non-empty string," the contract
is undefined — refine it before coding.

For any feature with shared/cached state, an async path, or a boundary
call, the assertion list MUST also name a **concurrency invariant** (what
holds under N concurrent requests/threads) and a **fault contract** (what
happens when the boundary is down/hung/failing mid-op), each shipped as an
executable test in the same commit — see CLAUDE.md → "Assert Concurrency
and Failure at Feature-Time, Not Audit-Time". These belong in the plan's
assertion list next to the happy-path assertions.

`quality-enforcer` REJECTS a change whose tests use only weak
assertions, even when they pass, and a change that needed the concurrency
/ fault tests above but shipped only happy-path coverage. Audit checklist
in `docs/modules/agents.md` enumerates the failure patterns assertions
must catch.

## Implementation Completeness

NEVER declare "done" or "complete" when:
- Using placeholder values or hardcoded test data
- Adding TODO/FIXME comments in new code
- Creating stub methods that raise NotImplementedError
- Using fallback logic that masks missing implementation
- Adding backward compatibility shims instead of actual implementation
- Integration tests for wiring changes have not been written yet
- Documentation for changed APIs/constructors/storage backends has not been updated yet
- The touched subpackage has no `docs/modules/*.md` reference at all (see "Doc Coverage Check" above)
- The test contract was not defined before the code (see "Assertions Before Code" above)

If a condition cannot be met, RAISE an exception with a clear message.

## Atomic Implementation Rule

Each code change MUST ship with its tests and doc updates in the same step. Never split implementation into:
- "Step N: write code" → "Step N+1: write tests" → "Step N+2: update docs"

Instead, for each component changed:
1. Edit the code
2. Run the Doc Coverage Check (above) — if the touched subpackage has no `docs/modules/` reference, add one in this commit before continuing.
3. Grep `docs/` for references to changed APIs and update them
4. Write or update the integration test that exercises the new wiring
5. Run the test, fix any failures
6. THEN move to the next component

This is not optional. Do not ask permission. Do not defer. The rules in CLAUDE.md already mandate this — follow them.

## File Discipline

NEVER create new files when you can edit existing ones.
NEVER create:
- Demonstration or analysis scripts
- Standalone .md documentation files
- Files with "base", "simple", "final", "v2", "new_" prefixes
- Test files outside existing test directories
