# Multi-Agent RAG System

Video content analysis and search over configurable pipelines. 11-package UV
workspace (Foundation → Core → Implementation → Application). Production ready.

- **Routing Agent** — query routing with orchestration handoff (DSPy)
- **Video Search Agent** — ColPali/VideoPrism retrieval via Vespa
- **Ingestion Pipeline** — keyframe extraction, transcription, embeddings

---

## THE TEST CONTRACT

The most-violated rule here. A test that cannot fail is worse than no test:
it reports safety that does not exist.

**Assert what the call is supposed to produce — never that it produced
something.** `count > 0`, `len(hits) >= 1`, `is not None`, `isinstance(x, str)`,
no-exception-raised, field-is-present: all forbidden. They pass when ranking is
inverted, when the wrong document comes back, when the value is empty.

Assert instead: the exact value, the exact set, the exact dict shape, the exact
order, the exact substring with its surrounding context, the exact persisted row
in the backing store. Search for image A → assert image A is the top hit. Rename
a field → feed a document and assert the value lands under the new name. Rank →
assert the order.

**Write those assertions BEFORE the production code.** List what outputs, side
effects, persisted state and failure modes the test will check. That list is the
contract. If the strongest assertion you can write is "non-empty string", the
contract is undefined — refine it before coding. Done means a real-service test
against that list passes, not "it runs".

**Integration tests hit the real boundary.** Real Vespa, real Phoenix, real
Redis, real MinIO, real LM — in Docker, managed by the test's own fixture. No
mocks at a boundary, no exceptions. But a real boundary with a weak assertion is
the same empty test with a slower setup: hitting real Vespa and asserting only
that hits came back proves nothing.

**Every wiring change (A writes, B reads) ships a round-trip test** — save then
load, assert equality — in the same commit. "It initializes without error" is
not coverage. A regression test that passes against the pre-fix code tests
nothing; verify it fails first.

**Bulk corpora are recorded once, committed, and shape-pinned.** When a test
needs volume — spans, documents, training rows — producing it live on every run
is fixture cost, not coverage: it buys nothing the assertions check, and a
fixture that takes hours is one that never completes, leaving every test in the
module unrun. Record it once from a real run, commit it, replay it. Keep the
live path as the way to re-record.

A recording rots silently, so it ships with its own pins in the same commit:
the exact record count, the exact per-type field/attribute sets, and the
identifiers compared against the PRODUCTION constants that emit them — never
restated as literals, or a rename in production is absorbed instead of flagged.
Pin what the consumer requires of the corpus (population floors, required
slots) by deriving it from shipped config. Prove each pin fires by mutating a
copy and watching it go red; a drift guard that has never been red guards
nothing. Whatever the live path was covering incidentally — that the producers
still emit this shape, that generation still validates and persists — gets a
named test in the same commit. A recording never silently replaces coverage.

**Any change with shared/cached state, an async path, or a boundary call also
ships two more tests in that same commit:**
- **Concurrency invariant** — what holds under N concurrent requests: single
  cold-build, no cross-tenant bleed, no use-after-close, loop never blocked.
  Prove it by executing the interleaving (barrier + counter), never by reasoning.
- **Fault contract** — boundary down/hung/failing mid-op must raise with
  context. Never a silent `[]`/`None`/zero that reads as no-data. Never a torn
  multi-step write.

Fix commits are not exempt. Deferring these is what makes the next audit find
the bugs this one introduced.

**A fix may not reduce what a test proves.** Never delete an assertion, relax an
exact comparison to a bound, or add a skip/xfail to turn a failing test green.
When an expectation is genuinely stale, replace it with an equally exact one
pinning the new contract and state in the commit message what the contract
changed to and why. When a test fails, first decide product-bug vs stale-test
and name the file:line that settles it — a test edit without that verdict is a
cover-up. `tests/common/unit/test_assertion_strength_guard.py` enforces this in
CI: it fails any change with a net assertion loss in a `tests/` file, or a newly
added skip, xfail, `is not None`, `>= 1`, or `> 0`.

---

## Running Tests

- `uv run` for everything. Full logging to a file: `uv run pytest ...
  --tb=long -v > /tmp/test_run.log 2>&1`, then grep it. Never pipe through
  `tail` — it buffers and hides progress.
- Always `--tb=long`. Short tracebacks hide root causes.
- **0 failed AND 0 skipped** before commit. An infra skip is a bug.
- Tests own their infrastructure — unique ports, own containers.
- Fix the implementation to satisfy the test. Never weaken a test, mock a
  failure away, disable, or hardcode.

**Never dismiss a failure.** Not "pre-existing", "LLM-dependent", "transient",
"environmental", "infrastructure". Each of these is a violation, not triage:
- `git stash` + rerun to prove "pre-existing" — stash keeps this session's
  commits, so it proves nothing. Check out the commit before the session's
  first, test there, come back.
- "passes in isolation, fails in the sweep" — if the sweep fails, the sweep
  fails. Make the test robust to the pollution.
- "the LM/service is down" — the fixture is responsible for that dependency.
- "deferred to an audit" — if this change surfaced it, this change fixes it.

Dev loop: single video first (`--max-frames 1`), logs at `outputs/logs/*.log`.

---

## Before Every Commit

Route by change type; full tree in `.claude/rules/strict-commit.md`.

```
CODE       → lint-and-quality → quality-enforcer → commit-enforcer
DOCS       → doc-verifier → commit-enforcer
CODE+DOCS  → lint-and-quality → doc-verifier → quality-enforcer → commit-enforcer
```

Never report done until `uv run ruff check` AND `uv run ruff format --check`
pass on changed files (CI runs them separately; one can pass while the other
fails), and the tests exercising the change pass. If you cannot verify, say so.

**Doc coverage**: a commit touching `libs/<pkg>/cogniverse_<pkg>/<subpkg>/`
requires `<subpkg>` in some `docs/modules/*.md`. Add it in the same commit —
`doc-verifier` only checks docs that already exist.

On demand: `feature-dev:code-reviewer`, `codebase-integrity-auditor`
(protocol in `.claude/rules/audit.md`).

**Production-code deletion always requires explicit user approval.** A named
live replacement is input to a proposal, not permission.

---

## Absolute Rules

Violated repeatedly despite instruction.

**1. Never deflect ownership.** "Not mine", "not my change", "pre-existing",
"infrastructure" — banned. Own every problem the work surfaces, including ones
that predate the session. Name an author only when an operation needs it.

**2. No process jargon in any artifact.** Never write "audit", "Class A–G",
"Phase N", "CRIT/HIGH/MED/LOW", finding IDs, or "deferred to…" in a file,
comment, docstring, commit message, or PR. Grep the diff before committing.

**3. Batch related work onto one branch.** No branch-and-PR per fix. Never
push, open/close PRs, or delete remote branches without approval.

**4. No backward compatibility unless asked.** Implement only the new canonical
form — no aliases, migrations, fallbacks, dual-read paths, or version shims.
Old forms are invalid.

**5. Docs and comments state what IS, briefly.** Never what was, what changed,
what was rejected, or what might come later — no history, no decision records,
unless that artifact was explicitly requested. Rationale goes in the commit
message; an unmergeable branch gets `git config branch.<name>.description`.
Write the shortest accurate thing. If a reader would skim it, cut it.

**6. Never ask permission for what these rules already require.** Write the
test, update the doc, fix the failure. No "should I also…?".

**7. A step is done when code compiles, its tests exist and pass, and docs for
any changed API are updated** — all in the same step, never a later "testing
phase".

---

## Working Method

**gather context → act → verify → repeat.**

- **Never build a side channel.** Fix the real code path. No parallel script,
  helper, or copy that demonstrates the behaviour without the system doing it.
- **Trace, don't theorize.** Work from the actual error and stack trace. Two
  failed fixes means STOP: re-read the error, find where your model was wrong,
  say so. Don't keep changing things.
- **Follow references, not descriptions.** Pointed at existing code? Match its
  patterns. Working code beats prose.
- **Read before editing.** Re-read any file before editing it after 10+
  messages — Edit fails silently on stale context. Files >500 LOC: read in
  chunks; reads cap at 2000 lines. Tool results >50k chars are truncated to a
  2k preview — if a search returns suspiciously little, narrow and re-run, and
  say you suspect truncation.
- **grep is not an AST.** Renaming? Search separately for calls, type
  references, string literals, dynamic imports, re-exports, test mocks. Assume
  you missed one.
- **Phase multi-file work.** Max 5 files per phase; verify, then continue. For
  config/schema/API/cross-cutting changes, and when asked to "plan" or "think
  first": output the plan only, get approval, then build.
- **Parallelize wide work.** >5 independent files → parallel sub-agents, 5–8
  files each.
- **"yes" / "do it" / "push" means execute.** No restating the plan.
- **Commit WIP; never leave a dirty tree.** Incomplete work ships as a `WIP:`
  commit and is amended when it completes. Uncommitted work is lost to a crash
  and is invisible to anything that reads the tree by SHA.

**Code quality.** Override the instinct to do the minimum: if architecture is
flawed, state duplicated, or patterns inconsistent, fix it — what would a
perfectionist senior dev reject? But don't build for imaginary futures. Write
what three experienced devs would all write: no robotic comment blocks, no
section headers on two sentences. Before restructuring any file >300 LOC,
delete dead code first and commit that separately. Never put `base`, `simple`,
`final`, `full`, `generic`, or `comprehensive` in a class or file name.

**Destructive safety.** Verify nothing references a file before deleting it.
Never `docker system prune --volumes` while a k3d cluster or tests run. Never
push unless told.

**After fixing a bug**, state the root cause, why the earlier approach missed
it, and what stops the category recurring.

---

## Domain Notes

- Embedding dims: 128 (ColPali/ColQwen patch), 320 (Tomoro ColQwen3), 768
  (base), 1024 (large). Binary formats are hex strings, float formats are
  floats.
- Ingest through pyvespa `feed_iterable`, not raw HTTP.
- `"Expected X values, got Y"` → hex/float mismatch. `"Connection aborted"` →
  batch too large. `HTTP 400` → schema/data shape mismatch.
- `colgrep "<query>"` is semantic code search (ColBERT + tree-sitter) for
  finding things by meaning; `Grep` is faster for known identifiers. Useful
  flags: `--include`, `--exclude-dir`, `-e` (hybrid exact), `-c` (show bodies),
  `--json`.
