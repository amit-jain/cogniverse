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

`quality-enforcer` REJECTS a change whose tests use only weak
assertions, even when they pass. Audit checklist in
`docs/modules/agents.md` enumerates the failure patterns assertions
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
