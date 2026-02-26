# MANDATORY PRE-COMMIT PROTOCOL

After making code changes, run agents IN ORDER based on what changed. Never skip any. Never ask permission.

---

## Change Type Detection

Check `git diff --name-only HEAD` and route to the appropriate workflow:

```
CODE ONLY (*.py, configs/, *.toml)
  → lint-and-quality → quality-enforcer → commit-enforcer

DOCS ONLY (docs/*.md)
  → doc-verifier → commit-enforcer

CODE + DOCS
  → lint-and-quality → doc-verifier → quality-enforcer → commit-enforcer

ON DEMAND (user request or periodic)
  → feature-dev:code-reviewer   (deep static analysis)
  → codebase-integrity-auditor  (full codebase audit)
```

---

## Agent 1: lint-and-quality

**What**: Fast static analysis — ruff lint + autofix, banned pattern scan, convention check.
**When**: Any code change (*.py, configs/, *.toml).
**Must pass**: Zero ruff errors, zero banned patterns, zero convention violations.
**If fails**: Fix all issues before proceeding. Never suppress with `# noqa`.

---

## Agent 2: doc-verifier

**What**: Verifies documentation against actual codebase — imports, classes, methods, configs, examples.
**When**: Any docs change (docs/*.md), or code changes that affect documented APIs.
**Must pass**: 100% of documentation synchronized with code.
**If fails**: Fix documentation to match actual code (or fix code if docs are correct).

---

## Agent 3: quality-enforcer

**What**: Test discovery and execution — grep for affected tests, run with timeouts, verify 100% pass.
**When**: Any code change.
**Must pass**: 0 failed AND 0 skipped. Uses `--tb=long` ALWAYS.
**If fails**: Fix implementation (never weaken tests). Re-run until 100% passing.

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

## Implementation Completeness

NEVER declare "done" or "complete" when:
- Using placeholder values or hardcoded test data
- Adding TODO/FIXME comments in new code
- Creating stub methods that raise NotImplementedError
- Using fallback logic that masks missing implementation
- Adding backward compatibility shims instead of actual implementation

If a condition cannot be met, RAISE an exception with a clear message.

## File Discipline

NEVER create new files when you can edit existing ones.
NEVER create:
- Demonstration or analysis scripts
- Standalone .md documentation files
- Files with "base", "simple", "final", "v2", "new_" prefixes
- Test files outside existing test directories
