# SINGLE-PASS CODEBASE AUDIT PROTOCOL

One audit, all detection methods in parallel. **Reject the assumption that each audit only catches the class of bug its method was designed to find.**

---

## Why prior audits kept missing things

Across five audits the methodology grew strictly by accretion:

| Audit | Trusted | Blind spot the next audit exposed |
|---|---|---|
| 1st | grep + imports | "Imports cleanly, runs the wrong logic on real input" |
| 2nd | + tests pass | Tests that pass by mocking the broken thing away |
| 3rd | + AST / call-graph | Tests that mock the SYSTEM BOUNDARY (Phoenix / Vespa / HTTP) |
| 4th | + execution + introspection on real objects | Tests that ASSERT THE BROKEN SHAPE — execution against a self-confirming mock proves nothing |
| 5th | + audit-the-tests + pattern hunts + untested-surface scan | (whatever this protocol still presumes) |

Each step **trusted the artifacts the previous step trusted**. Each next audit had to drop one more trust assumption. The reason we needed five passes is that we kept discovering blind spots one at a time instead of enumerating them up front.

**This protocol enumerates them up front.** The five detection methods below are orthogonal; run all five in one audit cycle.

---

## The five bug classes and their detection methods

Every finding in audits 1–5 falls into one of these five classes. Hunt every class in every audit. No class is optional.

### Class A — Tests that lock in the broken contract

Detection: walk every unit test that uses `MagicMock`, `AsyncMock`, `monkeypatch`, or `patch.object` on a SYSTEM BOUNDARY (HTTP route, telemetry provider, registry, LM, storage, message queue). For each, ask:

1. Does the test assert the *payload the code builds* (suspect — encodes whatever the code does, not what the boundary accepts)?
2. Does the test patch a class method on the SUT to force a passing path that wouldn't pass against the real implementation?
3. Does the test docstring or comment hedge ("flagged separately for review", "for now use mock data")?

If any answer is yes, replace that test with a real-boundary round-trip via `httpx.ASGITransport`, a real Vespa Docker, a real Phoenix Docker, or a real subprocess for the suspect contract.

Concrete sweep:

```bash
# tests that mock the agents-router payload and never hit the real route
grep -rn "AsyncMock\|MagicMock" tests/ | grep -iE "tenant_id|payload|route|process" | head -50

# tests that monkeypatch the registry to fake authorisation
grep -rn "monkeypatch.setattr.*build_default_registry\|patch.*KnowledgeRegistry" tests/

# tests asserting on the buggy fallback (the prior-bug-encoded-in-asserts pattern)
grep -rn "test_returns_mock\|assert.*mock_spans\|mock_provider" tests/
```

### Class B — Untested surface

Detection: enumerate every entry point — REST routes, CLI commands, A2A endpoints, Streamlit tabs, Argo workflow steps, `__main__` blocks — and confirm at least one test exercises each. Untested surface is invisible to every other detection method, because every other method walks code that tests exist for.

```bash
# enumerate routes
grep -rn "@router.post\|@router.get\|@app.post\|@app.get" libs/ | awk -F: '{print $1":"$3}' | sort -u

# enumerate CLI commands
grep -rn "@click.command\|argparse.ArgumentParser\|sys.argv" scripts/ libs/

# enumerate dashboard tabs
grep -rn "def _render.*_tab\|st\.tabs\|register_tab" libs/dashboard/

# for each, grep tests/ for a test that hits it
```

A2A `cancel()` and the `_summarise` worker function were both Class B in the 5th audit — neither had a test, so no execution-based audit found them.

### Class C — Cross-file syntactic pattern hunts

Detection: maintain a list of footgun patterns and grep for each across the whole repo. Per-symbol introspection sees one function at a time; pattern audits sweep the repo.

Hunt list (extend it whenever a new pattern shows up):

```bash
# Unescaped YQL value interpolation (5th audit, 18 sites)
grep -rn 'contains "\{[^}]*\}"' libs/

# dataclasses.asdict on a non-dataclass (4th audit — broke dispatcher)
grep -rn "dataclasses.asdict\|asdict(" libs/ | grep -v "@dataclass"  # then verify each

# Naive datetime.now() on a query window (4th audit — IST host bug)
grep -rn "datetime\.now()" libs/ | grep -v "timezone\.utc\|datetime\.now(tz="

# DSPy module called without per-tenant LM context (5th audit)
grep -rn "self\._dspy_module(\|self\._module(\|module(text=" libs/ | head
# then for each, confirm a surrounding `with dspy.context(lm=...)`

# Pyvespa response without status_code check (5th audit)
grep -rn "vespa_client\..*_data\|self.app\..*_data" libs/ | head
# then for each, verify the next 5 lines check response.status_code

# Sort by version as plain string (5th audit — lexical 1.9.0 > 1.10.0)
grep -rn "sort.*version\b\|sorted(.*\.version" libs/

# Substring-vs-token schema/model name match (6th audit — "lvt" in name.lower()
# collapses a multi-vector ndarray to its first row on any future schema whose
# name contains the substring; pair with case-inconsistent siblings like
# `"_sv_" in name` vs `"lvt" in name.lower()`)
grep -rnP '"_sv_" in\b|"_mv_" in\b|"[a-z]{2,4}" in [^.]*\.lower\(\)' libs/

# Document v1 selection-expression interpolation — same injection hazard as
# YQL `contains "..."` but in the visit/selection path (6th audit)
grep -rnP '\.tenant_id=="\{[^}]*\}"|\.user_id=="\{[^}]*\}"|=="\{self\._' libs/

# Naive datetime explicitly passed to a query window kwarg (6th audit found
# 14+ sites regressed; the bare `datetime.now()` regex above is noisy, this
# narrower variant scopes to the call site)
grep -rnP "start_time\s*=\s*datetime\.now\(\)|end_time\s*=\s*datetime\.now\(\)" libs/

# Fire-and-forget asyncio.create_task — task reference dropped, CPython may
# GC the coroutine before it runs (6th audit, agent_dispatcher + telemetry)
grep -rnP "^\s*asyncio\.create_task\(" libs/

# Streamlit naive date/time input combined into a datetime without tz attach
# (6th audit — st.date_input/st.time_input return naive)
grep -rn "st\.date_input\|st\.time_input\|datetime\.combine(" libs/dashboard/

# float() coercion of an LM/predictor output field without try/except
# (6th audit — `"high"` / `"85%"` returned by real LMs crashes the route)
grep -rn "float(result\.\|float(prediction\.\|float(out\." libs/agents/

# isinstance(_, (int, float)) used to gate a timestamp without magnitude check
# (6th audit — seconds-vs-milliseconds confusion lands a document at 1970)
grep -rn "isinstance(.*, (int, float))" libs/vespa/ libs/core/

# Standalone-agent FastAPI app routes — `__main__`-launchable but bypassed by
# the unified runtime dispatcher; HIGH risk of untested-surface (6th audit
# Class B: search_agent / summarizer_agent / detailed_report_agent)
grep -rnP "^@app\.(post|get|put|delete)" libs/agents/

# Streamlit render-tab entry points (one per tabs/*.py) — confirm at least one
# streamlit-testing-library test invokes each (6th audit found 12 untested)
grep -rnP "^def render_.*_tab" libs/dashboard/

# Local _escape helpers that handle only `"` and miss `\` — use yql_quote
# from libs/vespa/cogniverse_vespa/_yql.py which handles both (6th audit)
grep -rnP 'def _escape\(.*\).*->\s*str|\.replace\(.*\\"' libs/ --include="*.py"

# YQL/selection value interpolated RAW where a sibling escapes the same field
# — convention-divergence injection (7th audit: graph_manager interpolated
# self._tenant_id raw into both `contains "{...}"` and the Document-v1
# `==="{...}"` selection while provenance_store/config_store wrapped the same
# value in yql_quote). The plain `contains "{` regex buries the unescaped
# offender among escaped siblings; the negative-lookahead isolates the raw ones.
grep -rnP 'contains "\{(?!.*(_escape|yql_quote))[^}]*\}"|=="\{(?!.*(_escape|yql_quote))[^}]*\}"' libs/ --include="*.py"
```

This is the only detection method that scales by **adding a regex** rather than **running another audit**.

### Class D — Edge-input / dimension / format fuzzing

Detection: for every function that takes a tensor / array / string / timestamp / dict, vary the input across:

- Shape: `(N, dim)` vs `(1, dim)` vs `(dim,)` (5th audit — single-row to `_mv_` schema).
- Timezone: naive `datetime` vs UTC vs non-UTC (4th audit — IST host).
- String contents: empty, embedded `"`, embedded `\`, embedded null byte.
- Dict shape: missing key, extra key, key with `None`, key with wrong type.
- Schema/document type: name with both `_sv_` and `_mv_` substrings, name with neither.

For each variation, run the function and verify it produces the SHAPE that downstream code expects. Don't grep — execute.

```python
# template for dimension fuzzing
for shape in [(N, dim), (1, dim), (dim,)]:
    arr = np.zeros(shape, dtype=np.float32)
    out = fn(arr, ...)
    assert <strong shape assertion based on shape>
```

### Class E — Silent context / config drop

Detection: for every `self._<config>` attribute set in `__init__` (or by a mixin like `DynamicDSPyMixin`), grep the rest of the class for **read** sites. An attribute that's set but never read is a silent fallback.

```bash
# find candidate attributes
grep -rn "self\._dspy_lm\|self\._llm_config\|self\._tenant_lm\|self\._artifact_tenant_id" libs/

# for each, count read vs write sites:
for attr in _dspy_lm _llm_config _tenant_lm _artifact_tenant_id; do
    echo "=== $attr ==="
    grep -rn "self\.$attr\b" libs/ | awk -F: '{print $1}' | sort -u
done
# attributes with one file (the class that sets them) and no reads elsewhere = candidates
```

The 5th audit's `text_analysis.analyze_text` finding (LM built per-tenant but never used) is Class E. Mock provider + auto-attribute access (5th's finetuning find) is the *test-side* of the same class.

---

## Single-pass protocol

```
PHASE 0 — INVENTORY (do once at the start, don't bury inside phases)
  • enumerate entry points (routes, CLIs, tabs, A2A endpoints, __main__)
  • enumerate every test file and what entry point it tests
  • enumerate every system boundary the codebase talks to
    (Phoenix, Vespa, MinIO, LM endpoint, Mem0, Redis, Argo, Streamlit)
  • record the inventory — every later phase consults it

PHASE 1 — CLASS A SWEEP (test suspects)
  • for each system boundary identified in Phase 0:
      list every test that mocks that boundary
      for each, check whether the asserted shape matches the real
      boundary's contract (read the real impl)
  • flag every test where the assertion encodes the code's payload
    rather than the boundary's contract

PHASE 2 — CLASS B SWEEP (untested surface)
  • for each entry point from Phase 0:
      grep tests/ for any test that reaches it
      if zero tests reach it, flag the surface as Class B
  • for entry points that ARE tested, check whether the test
    reaches the LIVE call (vs aborting at validation/header check)

PHASE 3 — CLASS C SWEEP (pattern hunt list)
  • run every regex in the hunt list (see Class C above; extend each cycle)
  • for each hit, manually verify (Class C is the noisiest — many false positives)

PHASE 4 — CLASS D SWEEP (edge-input fuzz)
  • for every function on the boundary list that takes a tensor /
    array / string / timestamp / dict, run with edge inputs
  • execute, don't grep — Class D is invisible to static analysis

PHASE 5 — CLASS E SWEEP (silent context drop)
  • for each candidate attribute (self._dspy_lm, self._llm_config,
    self._<provider>, self._<tenant>_*), count read vs write sites
  • flag attrs set in __init__ but never read in the class body

PHASE 6 — EXECUTE THE HAPPY PATH (4th-audit method, kept)
  • for each entry point from Phase 0, run with realistic inputs
    against the REAL boundary (not a mock)
  • the failures here are the 4th-audit class

PHASE 7 — REVIEW GATE (mandatory, applied to every finding)
  • re-verify each finding by reading the live caller and
    proving the bug fires on a real run
  • agents emit false positives ("renamed not deleted", "subpackage-
    hasattr false-negatives") — every finding must pass introspection
    against the live code before it earns a fix

PHASE 8 — FIX with the strict-commit protocol
  • see .claude/rules/strict-commit.md
  • every fix ships with a real-boundary regression test that would
    have caught the bug; never weaken or mock-away the assertion
```

Phases 1–6 are **independent** — run them in parallel via the `codebase-integrity-auditor` agent's fan-out partitions, one agent per partition × phase. Don't sequence them.

---

## Detection methodology rules

1. **Trust nothing in green.** A passing test is evidence the test executed, not that the code is correct. Class A exists because we kept trusting green tests.

2. **The boundary is the contract.** When auditing a code path that talks to a system boundary, the contract is what the boundary accepts, not what the code currently sends. If the test mocks the boundary, the test cannot prove the contract holds.

3. **Static methods see static problems.** grep + imports + AST find naming inconsistencies. They cannot find Class D (edge-input format), Class E (silent drop), or runtime-context bugs. Execute, don't grep, for those classes.

4. **Pattern hunts scale by regex, not by audit.** Class C extends by adding one regex per new footgun, not by running another audit cycle. The hunt list is part of this file — extend it.

5. **Surface inventory is not optional.** Phase 0 is where you find Class B. Skip Phase 0 → no Class B finding for the rest of the cycle, because Classes A/C/D/E all walk code that tests exist for.

6. **Review gate every finding.** Agents emit false positives. Renamed classes look deleted; submodule imports look missing; "the method is unreferenced via grep" is not proof the method is dead. The hard rule from the project memory applies — see `[[feedback-never-delete-on-grep-alone]]`.

---

## Findings format

Every audit finding goes into a table with these columns:

| Severity | Class | Where (file:line) | What's broken on the happy path | Reproduction | Confidence |
|---|---|---|---|---|---|

- **Severity:** CRIT / HIGH / MED / LOW — based on user-visible impact, not on how clever the finding is.
- **Class:** A / B / C / D / E — so the next audit knows which detection method earned each finding.
- **Where:** absolute file path + line; not "somewhere in the dispatcher".
- **What's broken on the happy path:** the failure mode, not the implementation detail. "404 on every cancel" not "missing `task_id` field".
- **Reproduction:** the exact command or code that demonstrates the failure. If you can't reproduce, the finding is speculation — lower confidence, mark it.
- **Confidence:** HIGH / MEDIUM / LOW — be honest. "MEDIUM — needs live Vespa to confirm 400 vs 200" is more useful than false HIGH.

---

## Done criteria

An audit is done when:

1. **All seven phases ran.** Skipping a phase = blind spot for that class.
2. **Every finding has a row in the findings table.** No verbal findings.
3. **Every CRIT / HIGH finding has a fix or a deferred-TODO with a written plan.** Pipeline-cache (5th audit) is the model: write `docs/development/<name>.md` with the approved plan + inline pointer-TODOs at the affected sites.
4. **Every fix shipped with a real-boundary regression test** (see strict-commit.md).
5. **The hunt list (Class C) is extended** with any new pattern this audit surfaced. The audit's deliverable includes additions to this file.

If any of (1)–(5) is missing, the next audit will need to catch what this one didn't — exactly the failure mode this protocol exists to break.

---

## What an audit must NOT do

- Delete production code without explicit user approval, even when (1) the audit flags it, (2) a live replacement is named, (3) tests pass without it. The user has enforced this three times. See `[[feedback-never-delete-on-grep-alone]]`.
- Dismiss failures as "pre-existing", "LLM-dependent", "transient", "infrastructure". See `CLAUDE.md` → Testing Best Practices.
- Stop at "the audit agent found N issues." The review gate runs against every finding. Agent false positives are routine.
- Defer a finding to "the next audit." That is the exact failure mode this protocol exists to break.
