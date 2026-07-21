# SINGLE-PASS CODEBASE AUDIT PROTOCOL

One audit, all detection methods in parallel. **Reject the assumption that each audit only catches the class of bug its method was designed to find.**

Every detection method trusts some artifact — imports, green tests, call graphs, execution against test doubles. Each of those artifacts can lie: code that imports cleanly can run the wrong logic; a green test can pass by mocking the broken thing away or by asserting the broken shape; execution against a stand-in of the wrong concrete type proves nothing about production. The protocol below enumerates those trust assumptions up front and attacks each with its own method, in one pass, instead of discovering them one audit at a time.

---

## The bug classes and their detection methods

Every finding belongs to one of these classes (A–G). Hunt every class in every audit. No class is optional.

### Class A — Tests that lock in the broken contract

Detection: walk every unit test that uses `MagicMock`, `AsyncMock`, `monkeypatch`, or `patch.object` on a SYSTEM BOUNDARY (HTTP route, telemetry provider, registry, LM, storage, message queue). For each, ask:

1. Does the test assert the *payload the code builds* (suspect — encodes whatever the code does, not what the boundary accepts)?
2. Does the test patch a class method on the SUT to force a passing path that wouldn't pass against the real implementation?
3. Does the test docstring or comment hedge ("flagged separately for review", "for now use mock data")?

If any answer is yes, replace that test with a real-boundary round-trip via `httpx.ASGITransport`, a real Vespa Docker, a real Phoenix Docker, or a real subprocess for the suspect contract.

Concrete sweep:

```bash
# tests that mock a routed payload and never hit the real route
grep -rn "AsyncMock\|MagicMock" tests/ | grep -iE "tenant_id|payload|route|process" | head -50

# tests that monkeypatch a registry to fake authorisation
grep -rn "monkeypatch.setattr.*build_default_registry\|patch.*KnowledgeRegistry" tests/

# tests asserting on a mock-shaped fallback (a prior bug encoded in the asserts)
grep -rn "test_returns_mock\|assert.*mock_spans\|mock_provider" tests/

# test mocks a pyvespa data-plane op to RETURN a non-2xx status_code WITHOUT
# raising — real pyvespa raise_for_status raises on every 4xx/5xx except 404,
# so the mock exercises a branch via a shape the boundary can't produce while
# the real raise->except path stays untested. Docstrings claiming the client
# "does not raise" are the tell.
grep -rnP 'status_code\s*=\s*(4|5)\d\d' tests/ --include="*.py" | grep -viE '404|raise|side_effect'
grep -rniP 'pyvespa[^\n]*(does not|doesn.t|not) raise' tests/ libs/ --include="*.py"
```

### Class A extension — real-boundary test using the wrong concrete TYPE

The "no mocks at the boundary" rule is necessary but NOT sufficient. A test can use a REAL library object + a REAL recording stub and still prove nothing, because it constructed a stand-in of a **different concrete type** than production builds. Two wrapper types of the same library can expose the same attribute through different access paths; a gate like `hasattr(obj, "attr")` then takes opposite branches for the test's type and production's type, and the test stays green while production silently no-ops.

Detection: for any test that constructs a stand-in of a type the production code also constructs (a DSPy module, a pydantic model, a client wrapper), assert the fixture's concrete type == the production type, OR drive the REAL production object. A DSPy overlay/optimization test MUST use the same predictor wrapper the served agent uses — better, instantiate the real served module (see `tests/runtime/integration/test_artefact_overlay_consumed_by_agent.py` `TestServedAgentModulesAreOverlayReachable`, which iterates the real serve targets). A real object of a convenient type is still a stand-in.

### Class B — Untested surface

Detection: enumerate every entry point — REST routes, CLI commands, A2A endpoints, Streamlit tabs, Argo workflow steps, `__main__` blocks — and confirm at least one test exercises each. Untested surface is invisible to every other detection method, because every other method walks code that tests exist for.

```bash
# enumerate routes
grep -rn "@router.post\|@router.get\|@app.post\|@app.get" libs/ | awk -F: '{print $1":"$3}' | sort -u

# enumerate CLI commands
grep -rn "@click.command\|argparse.ArgumentParser\|sys.argv" scripts/ libs/

# enumerate dashboard tabs
grep -rn "def _render.*_tab\|st\.tabs\|register_tab" libs/dashboard/

# standalone FastAPI apps inside agent packages — __main__-launchable but
# bypassed by the unified runtime dispatcher; high risk of untested surface
grep -rnP "^@app\.(post|get|put|delete)" libs/agents/

# Streamlit render-tab entry points — confirm at least one streamlit-testing
# test invokes each
grep -rnP "^def render_.*_tab" libs/dashboard/

# for each, grep tests/ for a test that hits it
```

For entry points that ARE tested, check whether the test reaches the LIVE call (vs aborting at validation/header check):

```bash
# route whose ONLY in-process test posts json={} and asserts just the 422/404
# guard — the handler body never executes in an ordinary pytest run and a
# regression ships green. For each route such files touch, confirm a NON-e2e
# test asserts a 2xx success body; cluster-gated e2e coverage does not count.
grep -rln 'json={}' tests/*/integration/ tests/*/unit/ 2>/dev/null | xargs grep -l 'status_code == 422\|status_code == 404' 2>/dev/null

# pytest DESELECTION gated on an infra-readiness probe — invisible to the
# "0 skipped" gate, unlike a skip; the surface silently loses its only test.
grep -rnP 'pytest_deselected|skip_substrings\.append|items\.remove' tests/ --include="*.py" -B3 \
  | grep -iE '_ready\(\)|_available\(\)|environ\.get|reachable'

# CI marker gates: for each workflow selection of the form
# `pytest <dir> -m "<expr>"`, list files in <dir> lacking the gating marker —
# those never run in that job. Each must be covered by another explicit
# selection or by per-change affected-test discovery; a file in neither rots
# red without anyone executing it.
grep -rhoP 'pytest \S*tests/\S+( -m "[^"]+")?' .github/workflows/*.yml | sort -u
# then per marker-gated selection: grep -rL "<marker>" <dir>/*.py
```

### Class C — Cross-file syntactic pattern hunts

Detection: maintain a list of footgun patterns and grep for each across the whole repo. Per-symbol introspection sees one function at a time; pattern hunts sweep the repo. Each entry below is a GENERIC class detector — it must be able to fire on future code, not only on an already-fixed instance. When a new footgun class surfaces, add its generic detector here (propose the addition in conversation first).

**Escaping / interpolation**

```bash
# Raw value interpolation into YQL — any f-string value inside contains "..."
grep -rn 'contains "\{[^}]*\}"' libs/

# Raw interpolation into a Document v1 selection expression — same injection
# hazard as YQL but in the visit/delete-selection path
grep -rnP '\.tenant_id=="\{[^}]*\}"|\.user_id=="\{[^}]*\}"|=="\{self\._' libs/

# Interpolation that is RAW where a sibling escapes the same field —
# convention-divergence injection; the negative lookahead isolates the raw
# offenders that a plain regex buries among escaped siblings
grep -rnP 'contains "\{(?!.*(_escape|yql_quote))[^}]*\}"|=="\{(?!.*(_escape|yql_quote))[^}]*\}"' libs/ --include="*.py"

# Local escape helpers (any name: _escape, _esc, _quote, lambdas) — each must
# escape BOTH backslash and the quote char it wraps with; the canonical helper
# is yql_quote in libs/vespa/cogniverse_vespa/_yql.py
grep -rnP 'def _escape\(.*\).*->\s*str|def _esc\b|_esc\s*=\s*lambda|def _quote\b|\.replace\(.*\\"' libs/ --include="*.py"
```

**Datetime / timezone**

```bash
# Naive datetime.now() anywhere near a query window or persisted timestamp
grep -rn "datetime\.now()" libs/ | grep -v "timezone\.utc\|datetime\.now(tz="
grep -rnP "start_time\s*=\s*datetime\.now\(\)|end_time\s*=\s*datetime\.now\(\)" libs/

# datetime.utcnow() — always NAIVE; LOOKS tz-correct but isn't. Compared
# against a fromisoformat()-parsed aware value it raises TypeError; a
# surrounding bare except then fail-closes. Use datetime.now(timezone.utc).
grep -rnP "datetime\.utcnow\(\)" libs/ --include="*.py"

# Streamlit date/time inputs are naive; combining them without attaching a tz
# produces naive query windows
grep -rn "st\.date_input\|st\.time_input\|datetime\.combine(" libs/dashboard/
```

**Type / shape coercion at boundaries**

```bash
# float() coercion of an LM/predictor output field without try/except — real
# LMs return "high" / "85%" and the route crashes
grep -rn "float(result\.\|float(prediction\.\|float(out\." libs/agents/

# isinstance((int,float)) gating a timestamp without a magnitude check —
# seconds-vs-milliseconds confusion lands documents at 1970
grep -rn "isinstance(.*, (int, float))" libs/vespa/ libs/core/

# Epoch passed through as bare int() with no seconds-vs-ms magnitude guard
# within ~3 lines (*1000, //1000, >1e11 check)
grep -rnP 'return int\((v|ts|value|timestamp|raw|created_at)\)\s*$' libs/ --include="*.py"

# numpy scalar reaching a scalar serializer un-coerced — np.float64 IS a float
# subclass (repr leaks into the query), np.int64 is NOT an int subclass (falls
# to the string branch). Confirm .item() coercion or np.integer/np.floating
# gating BEFORE the isinstance dispatch.
grep -rnP 'isinstance\([^,]+, \(int, float\)\)' libs/vespa/ libs/agents/ --include="*.py"

# (1,dim)->dict tensor conversion guarded only by ndim==2 without a
# shape[0]==1 flatten where a sibling branch flattens — a single-vector
# encoder emitting (1,dim) on a dense schema input becomes a nested dict
grep -rnP '\{str\(\w+\): \w+\.tolist\(\)' libs/ --include="*.py"

# Model/schema classification by bare substring instead of a delimiter-bracketed
# token — "xy" in name.lower() misfires on any future name embedding the
# letters; match the token form (e.g. "_xy_") instead
grep -rnP '"[a-z]{2,6}" in [^.]*\.lower\(\)|"_sv_" in\b|"_mv_" in\b' libs/ --include="*.py"

# Method typed `-> bool` returning a delegatee whose own signature returns
# Dict/other — the type lie ships green because no test calls it
grep -rnP 'def \w+\([^)]*\)\s*->\s*bool:\s*$' libs/ --include="*.py" -A3 | grep -B2 'return self\.\w+\.\w+\('

# dataclasses.asdict on a non-dataclass
grep -rn "dataclasses.asdict\|asdict(" libs/ | grep -v "@dataclass"

# Sort by version as plain string (lexical "1.9.0" > "1.10.0")
grep -rn "sort.*version\b\|sorted(.*\.version" libs/
```

**Backend response handling**

```bash
# Pyvespa response consumed without a status_code check within ~5 lines
grep -rn "vespa_client\..*_data\|self.app\..*_data" libs/ | head

# Query response consumed via .hits / root.children after ONLY a status guard,
# with NO root.errors / coverage.degraded check — soft timeouts arrive as
# HTTP 200 + root.errors + partial children, so degraded queries return
# partial/empty results recorded as SUCCESS. The shared helper is
# vespa_search_children(); direct consumers must raise on root.errors first.
grep -rnP 'response\.hits|\.get\("root", \{\}\)\.get\("children"|\.get\("root", \{\}\)' libs/ --include="*.py"

# A raise-on-degraded helper consumed INSIDE a broad except that returns [] —
# the except re-swallows what the helper raises. Each enclosing except must
# re-raise, never flatten.
grep -rnP '\.post\([^)]*/search/|vespa_search_children' libs/agents/ --include="*.py"

# Phoenix get_spans_dataframe without an explicit timeout kwarg — the client
# METHOD defaults to a short timeout that overrides client-level settings;
# loaded-project scans blow through it and read as "no spans"
grep -rnP 'get_spans_dataframe\(' libs/ tests/ --include="*.py" | grep -v "timeout="

# Whole-project span pull filtered client-side where the API supports
# filters={"name": ...} server-side — each scan re-downloads the full window
grep -rnP 'df\["name"\] ==|spans_df\["name"\] ==' libs/ --include="*.py"
```

**Fault-contract flattening**

```bash
# Backend/telemetry read flattened to []/None/{} the caller reads as a VALID
# no-data state — an outage becomes "no checkpoints"/"no configs"/"first run"
# and every guard keyed off the state silently opens. Reads raise; only a
# genuine not-found may map to None/[].
grep -rnP -A2 'except Exception' libs/ --include="*.py" | grep -B2 -E 'return (\[\]|\{\}|None|pd\.DataFrame\(\))'

# MIXED read-failure contracts within one file — a hardened get_X that raises
# next to a list_X/get_X_history that still flattens; the divergence is the tell
grep -rlP 'except Exception' libs/ --include="*.py" | while read f; do \
  grep -qP 'raise so callers|read FAILED|raise rather than mask' "$f" \
  && grep -qP 'return \[\]$|return None$' "$f" && echo "MIXED raise/flatten: $f"; done

# Asymmetric read-vs-write fault contract in one module — a _load_/_read_
# helper that `except -> return {}/[]/None` paired with a _save_/_write_
# sibling that propagates; the read should propagate too
grep -rlP 'def _(load|read)_\w+' libs/ --include="*.py" | while read f; do \
  grep -qP 'def _(save|write)_\w+' "$f" && grep -qP 'except Exception' "$f" \
  && grep -qP 'return \{\}$|return \[\]$|return None$' "$f" && echo "asym read/write fault: $f"; done

# Manager-layer re-flatten one layer ABOVE a hardened primitive — an except
# returning []/None/success around a backend call whose primitive RAISES on
# outage re-introduces the masked failure the primitive was hardened against
grep -rnPA3 'except Exception' libs/agents/ --include="*.py" \
  | grep -B3 -E 'return \[\]|return None|return \{\}' | grep -B1 'self\._backend\.'

# Config-resolving helper invoked with config_manager=None inside a broad
# except that degrades to a default/base-model — the feature is silently dead
# on arrival with only a warning log; a sibling that works via another seam is
# the tell
grep -rnP 'get_config\([^)]*config_manager=None' libs/ --include="*.py"
```

**Concurrency, caches, locks**

```bash
# Fire-and-forget asyncio.create_task — task reference dropped; CPython may
# GC the coroutine before it runs. Keep a strong reference (task set +
# done-callback discard).
grep -rnP "^\s*asyncio\.create_task\(" libs/

# Non-atomic get-then-set on a process cache/registry — cold-start builds must
# use set_if_absent + close-loser or get_or_set; the displaced-close causes
# use-after-close, not just a leak (canonical fix: backend_registry.py)
grep -rnPA6 'cls\._instances\.get\(|self\._instances\.get\(' libs/ --include="*.py" | grep -PB6 '_instances\.set\('

# Unlocked lazy init of a session-owning attr — `if self._x is None: self._x
# = build()` racing two first-touches leaks or double-builds
grep -rnPA3 'if self\._\w*(app|client|session) is None' libs/ --include="*.py" | grep -P '= make_|\.syncio\(|Client\('

# Async cache COLD-BUILD with no in-flight guard — a miss that awaits an
# expensive build between the check and the write, with no per-key
# Future/Lock, runs N full builds for N concurrent first-touches
grep -rnPA8 'get_or_set\(|\.get\(\w+\)\s*$' libs/ --include="*.py" | grep -PB8 'await self\._build_\w+\(' | grep -v 'inflight\|_build_inflight\|pending'

# First-wins process-global singleton handed to consumers with DIFFERENT
# configuration — the seed captures the first consumer's binding and every
# later consumer inherits it; confirm the seed's binding is compared against
# the requester's config before reuse
grep -rnP 'if cls\._\w+ is None.*and|cls\._shared\w+ = ' libs/ --include="*.py" | grep -i "shared\|singleton\|global"

# Module-level memo with NO eviction and/or returning the SHARED mutable value
# — keys accumulate forever; caller mutation poisons the cache. Require
# bounded/replace-on-write semantics + defensive copy on return.
grep -rnP '^_[A-Z_]+(CACHE|MEMO)\b\s*[:=]' libs/ --include="*.py"

# Single-instance TTL/request cache invalidated only by same-instance writes —
# out-of-band writers converge only when the TTL lapses; flag any consumer
# assuming immediate cross-instance visibility and any UNBOUNDED staleness
grep -rnP '_request_cache|_invalidate_request_cache|_REQUEST_CACHE_TTL' libs/ --include="*.py"

# Shared-state snapshot consumed under a serializing lock but COLLECTED before
# acquiring it — the race loser writes a stale complete-state, dropping
# whatever the winner added; no single-run test catches it. Confirm the
# snapshot is (re)collected INSIDE the lock body.
grep -rnPB10 'with .*[Ll]ock' libs/ --include="*.py" | grep -P '=\s*list\(|=\s*dict\(|\.copy\(\)|=\s*\[.+for\s'

# Sync blocking call NOT offloaded, adjacent to a to_thread in the same async
# def — the neighbor got offloaded, the hot call didn't; trace each bare call
# to the loop it blocks
grep -rnPB6 'await asyncio\.to_thread' libs/ --include="*.py" | grep -PE '(mgr|manager|wm|self\._\w+)\.\w+\([^)]*\)\s*$' | grep -v to_thread

# Async route/handler calling a blocking deploy/convergence primitive (with
# internal sleeps) directly on the loop while a sibling call site offloads the
# same call via to_thread
grep -rnPB10 'schema_registry\.deploy_schema\(|reload_config\(\)' libs/runtime/ --include="*.py" | grep -P 'async def|add_signal_handler|to_thread'

# DSPy module called without a per-tenant LM context — confirm a surrounding
# `with dspy.context(lm=...)`
grep -rn "self\._dspy_module(\|self\._module(\|module(text=" libs/ | head
```

**Retry, timeout, readiness**

```bash
# Retry count silently multiplying the timeout budget — worst-case hang is
# timeout x attempts, compounding across multi-feed paths; confirm no
# non-idempotent op is retried blindly
grep -rnPA3 'num_retries\w*\s*=' libs/ --include="*.py" | grep -P 'timeout'

# Readiness/convergence probe that times out and "proceeds anyway" — deploy/
# startup reports success for a resource that never became ready
grep -rniP 'proceeding anyway|continuing anyway|proceed despite' libs/ --include="*.py"

# LM-availability skip gate probing ONLY one server flavor — when the
# configured endpoint is the other flavor the gate FALSELY SKIPS the suite
# (an infra skip = a bug). The canonical gate probes both /api/tags and
# /v1/models after stripping a trailing /v1 (see tests/fixtures/llm.py).
grep -rnP '/api/tags(?!.*v1/models)' tests/ --include="*.py" | grep -iE 'available|skip|reachable'
```

**Multi-step writes, durable state**

```bash
# A durable GUARD MARKER — an idempotency / inflight / dedup / lock / claimed
# key — persisted BEFORE the side-effecting op it guards, with no
# clear-on-failure: a failure of the guarded op orphans the marker and masks a
# never-completed operation for the marker's TTL. Find every guard-marker
# WRITE; for each, the following submit/xadd/publish/post/enqueue must be
# compensated (try -> clear marker on failure -> re-raise).
grep -rnP '\b\w*(mark|set|persist|record|claim|reserve)\w*(inflight|pending|idempoten|dedup|lock|claimed|processing|reserved)\w*\(' libs/ --include="*.py"

# Multi-step write with no compensation — >=2 sequential backend writes in one
# activate/switch/promote method; a step-2 failure orphans step-1. Fixes are
# compensation (restore prior state before re-raising) or idempotent-retry.
grep -rnPA12 'def set_active|def .*activate|def .*switch' libs/ --include="*.py" | grep -cP '_update_\w+_fields\(|feed_data_point\('

# Multi-step RESTORE/copy helper called OUTSIDE the compensation try that
# wraps the state save — a failure mid-restore leaves half-advanced state
# uncompensated; the call site must sit INSIDE the compensation scope
grep -rnPB2 '_restore_\w+\(|_copy_\w+_into_active' libs/ --include="*.py" | grep -P 'await|='

# Sidecar/metadata unlink or utime BEFORE os.replace — a failed replace
# destroys the old entry's liveness metadata; require tmp-write + replace
# ordering that preserves the prior entry until the new one is durable
grep -rnPB4 'os\.replace\(' libs/ --include="*.py" | grep -P '\.unlink\(|os\.utime\('

# State/expiry encoded in a file's mtime via os.utime — a reader in the
# write->utime window sees mtime=write-time and expires a FRESH entry; any
# mtime-resetting op (cp, rsync, restore) silently expires everything.
# Require tmp-write + utime(tmp) + os.replace, or a sidecar/manifest.
grep -rnP 'os\.utime\(' libs/ --include="*.py"

# Atomic-write cleanup blanket-skipping .tmp with no age gate — orphan tmp
# files grow disk unboundedly across hard-kills
grep -rnPA2 '\.suffix (in \([^)]*\.tmp|== "\.tmp")' libs/ --include="*.py" | grep -i continue | grep -v st_mtime

# Work-queue consumer with no reclaim path for another consumer's abandoned
# work — applies to any exclusive-delivery mechanism (consumer groups,
# visibility-timeout queues, DB row claims): claiming new work without ever
# reclaiming orphaned in-flight work strands it forever when a consumer dies
# uncleanly. Any reclaim path must also bound redelivery attempts, or a
# consumer-killing poison item loops.
grep -rln "xreadgroup" libs/ --include="*.py" | while read f; do \
  grep -qrP "xautoclaim|xclaim" "$(dirname "$f")" || echo "no reclaim path for: $f"; done
```

**Dead knobs, dropped params**

```bash
# Config field advertising a capability the module never implements — ttl/
# expiry, compression, timeout, stats toggles, ceilings; for each hit, demand
# a read site that changes behavior
grep -rnE '(ttl|expire|_timeout_s|compress|enable_stats|serialization_format|_ceiling)\w*\s*[:=]' libs/ --include="*.py" | head -40

# Boolean capability toggle with zero behavior-changing readers
grep -rnP '\b(enable|disable)_\w+\s*:\s*bool\s*=' libs/ --include="*.py"

# Deps/config attribute assigned in __init__ but never functionally read —
# for each self.<knob> = deps.<knob>, demand a read site that changes behavior
grep -rnP 'self\.(max_\w+|.*_enabled|inference_service) = ' libs/agents/ libs/runtime/ --include="*.py"

# A function accepting a filter param (*_contains, name_*, query) never
# referenced in its body — callers believe they filtered; every row comes back
grep -rnP 'def \w+\([^)]*_contains[^)]*\)' libs/ --include="*.py"

# Validator whose try body is a comment + unconditional `return True`
grep -rnP 'def (validate|verify|check)_\w+' libs/ --include="*.py" -A6 | grep -B4 'return True' | grep -B2 '# .*would\|# TODO\|pass$'

# Profile-name allowlist gating a query-construction feature — any profile
# named outside the list silently loses the feature; derive structurally from
# the profile's own definition instead
grep -rnP 'profile_name in \[|profile_name in \(' libs/ --include="*.py"

# Startup registration HARDCODING a dict that duplicates a configs/ section
# inside a broad-except swallow — the copy drifts silently; read the loaded
# config and pin the identity with a test
grep -rnPA6 'add_backend_profile\(|register_\w+_profile\(' libs/runtime/ --include="*.py" | grep -P '=\s*\{'
```

**Tenant identity**

```bash
# Tenant-scoped reader that doesn't canonicalize — writers use the canonical
# org:tenant form; a reader deriving a project/schema/provider from a raw id
# queries an EMPTY namespace on real traffic. Confirm canonical_tenant_id at
# the reader's constructor entry.
grep -rnP 'get_project_name\(|f"cogniverse-\{[^}]*tenant' libs/ --include="*.py"

# Dashboard/CLI entrypoint feeding a raw tenant id into a per-tenant store —
# confirm canonical_tenant_id() before the value reaches get_provider /
# get_project_name / *Storage(tenant_id=...)
grep -rnP 'session_state\[.?current_tenant|st\.session_state\.get\("current_tenant"' libs/ --include="*.py"

# k8s label value built from a tenant id without sanitizing — canonical ids
# contain ':', which label values reject; confirm sanitization before
# metadata.labels
grep -rnP '"tenant[^"]*":\s*[a-z_]*tenant' libs/ --include="*.py" | grep -viE 'replace|sanitize|params|parameters'
```

**Confinement and construction**

```bash
# Raw Vespa document/v1 URL construction OUTSIDE libs/vespa — bypasses the
# backend abstraction's session reuse, error contract, and namespace handling.
# Enforced by tests/backends/unit/test_docv1_confined_to_vespa.py.
grep -rn 'document/v1/' libs/ --include="*.py" | grep -vE '^libs/vespa/|main\.py:'

# object.__new__(ProductionClass) outside tests — a production path building a
# partially-initialized instance that skips __init__ invariants
grep -rnP 'object\.__new__\(|__class__\.__new__\(' libs/ --include="*.py"
```

**Test-infra hygiene**

```bash
# Test-spawned docker containers without the owner-pid label — unlabeled
# spawns escape reap_dead_owner_containers; a SIGKILLed pytest session orphans
# heavy containers in host RAM. Filter on the label LITERAL the tests use
# (cogniverse-test-owner-pid), not a constant name; pre-filter to real
# `docker run` spawns (bare `"run"` matches uv/kubectl argv).
grep -rnP '"docker",\s*$' tests/ --include="*.py" -A4 | grep -B1 '"run"' | grep -v 'cogniverse-test-owner-pid'

# Unlabeled spawn where a SIBLING spawn in the SAME file IS labeled —
# convention divergence the per-file regex misses
grep -rlP '"docker",\s*$|docker run' tests/ --include="*.py" | while read f; do \
  grep -qP 'cogniverse-test-owner-pid' "$f" && \
  grep -cP '"run"|docker run' "$f" | grep -qv '^1$' && echo "MIXED labeled/unlabeled: $f"; done

# Test module importing a session fixture directly — pytest treats a
# module-level import as a SECOND FixtureDef with its own session cache,
# silently booting duplicate infra mid-sweep. Fixture re-exports belong in
# conftest.py ONLY.
grep -rn "from tests.conftest import" tests/ --include="*.py" | grep -v "conftest.py:"
```

This is the only detection method that scales by **adding a regex** rather than **running another audit**.

### Class D — Edge-input / dimension / format fuzzing

Detection: for every function that takes a tensor / array / string / timestamp / dict, vary the input across:

- Shape: `(N, dim)` vs `(1, dim)` vs `(dim,)`.
- Timezone: naive `datetime` vs UTC vs non-UTC host.
- String contents: empty, embedded `"`, embedded `\`, embedded null byte.
- Dict shape: missing key, extra key, key with `None`, key with wrong type.
- Schema/document type: name containing multiple classification tokens, name containing none.

For each variation, run the function and verify it produces the SHAPE that downstream code expects. Don't grep — execute.

```python
# template for dimension fuzzing
for shape in [(N, dim), (1, dim), (dim,)]:
    arr = np.zeros(shape, dtype=np.float32)
    out = fn(arr, ...)
    assert <strong shape assertion based on shape>
```

### Class E — Silent context / config drop

Detection: for every `self._<config>` attribute set in `__init__` (or by a mixin), grep the rest of the class for **read** sites. An attribute that's set but never read is a silent fallback — the caller believes the knob is wired.

```bash
# find candidate attributes, then count read vs write sites per attribute:
for attr in _dspy_lm _llm_config _tenant_lm _artifact_tenant_id; do
    echo "=== $attr ==="
    grep -rn "self\.$attr\b" libs/ | awk -F: '{print $1}' | sort -u
done
# attributes with one file (the class that sets them) and no reads elsewhere = candidates
```

The same class covers per-tenant resources built but never used (an LM constructed per-tenant while the module calls the default), and its test-side twin: a mock provider whose auto-attributes make every access succeed, hiding the drop.

### Class F — Concurrency, thread-safety, resource lifecycle, multi-process consistency

The other five classes audit code in isolation; Class F attacks what happens when it runs *together*. **Execute or full-read call chains — do NOT grep-heuristic** (a diff-grep for un-offloaded calls misses the case where the blocking call is in a *sync manager method* and the un-offloaded caller is an *adjacent* async route). Detect:

1. **Shared-session thread-safety** — for any long-lived client shared across threads, verify the wrapper holds no per-request mutable state, then hammer N threads × M ops through ONE instance against a real backend, asserting zero exceptions + zero cross-talk.
2. **Non-atomic get-then-set on a process cache/registry** — every cold-start build must use `set_if_absent`+close-loser or `get_or_set`, never `get()`+`set()`; the displaced-close causes a *use-after-close* on the loser's session, not just a leak.
3. **Sync blocking I/O on an async loop** — for every `async def`, confirm each pyvespa/`requests`/`time.sleep`/blocking-read is inside `await asyncio.to_thread(...)`; a `to_thread` call *adjacent* to a bare sync call in the same function is the tell. Trace to the loop: an API loop stalls all requests; a worker loop stalls shutdown and signal handling.
4. **Unlocked lazy init of a session-owning attr** — `if self._x is None: self._x = build()` racing two first-touches leaks the loser; a single-call "primer" is a tell the init is unguarded.
5. **Atomic-write orphans** — tmp→`os.replace` needs an age-gated `.tmp` reaper; a blanket suffix-skip in cleanup grows disk unboundedly across hard-kills.
6. **Multi-replica staleness** — flag only *unbounded* in-process staleness on correctness-bearing values (a TTL-bounded scoped cache is fine).

Prove every finding by executing the interleaving (barrier + counter), never by reasoning. A passing single-thread test proves nothing about concurrent use.

### Class G — Fault injection / failure-path execution

Every other class (including F's hammers) executes against HEALTHY boundaries. Class G runs the same paths with the boundary DOWN, HUNG, or FAILING MID-OPERATION and asserts the failure CONTRACT: raise-with-context (good) vs silent success / empty / None (bug) vs torn partial state (worst). Hardening the base layer is not enough — the SAME flatten re-appears one layer UP when a manager's `except Exception: return []` re-swallows a primitive that now raises; and torn state in multi-step writes is invisible to any single-boundary test.

Detection (EXECUTE — do not grep-conclude):

1. **Reads under outage** — point each store/manager at a DEAD PORT and a `docker pause`d container. Every read must raise (or return an explicit error object), never `[]`/`None`/`{}`/zero-counts indistinguishable from no-data.
2. **Multi-step writes** — enumerate every get-then-set / deactivate-then-activate / feed-then-backref / node-then-edge sequence; monkeypatch step N to raise and inspect persisted state for a torn result. Fixes are compensation or idempotent-retry safety.
3. **HTTP degradation** — stub server returning 503/429/507/connection-reset/slow-trickle; TIME the retry budget: per-attempt timeout × retry count is the real hang bound, and no non-idempotent op may be retried blindly.
4. **Disk failure** — monkeypatch `os.replace`/`os.utime`/write → ENOSPC/EACCES mid-`set()`: the OLD value must survive a failed overwrite, no partial entry readable, no orphan tmp beyond the reaper.
5. **LM/telemetry down** — error surfaced vs silently swallowed into a base-model/no-op fallback; distinguish build-time laziness from call-time failure.
6. **Partial batches** — doc k of N schema-invalid: the failure report names exactly the failed ids AND the good docs persist.

The greppable precursors live in the Class C hunt list (fault-contract flattening, multi-step writes); Class G executes the hits.

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
  • run every regex in the hunt list (see Class C above)
  • for each hit, manually verify (Class C is the noisiest — many false positives)

PHASE 4 — CLASS D SWEEP (edge-input fuzz)
  • for every function on the boundary list that takes a tensor /
    array / string / timestamp / dict, run with edge inputs
  • execute, don't grep — Class D is invisible to static analysis

PHASE 5 — CLASS E SWEEP (silent context drop)
  • for each candidate attribute, count read vs write sites
  • flag attrs set in __init__ but never read in the class body

PHASE 6 — EXECUTE THE HAPPY PATH
  • for each entry point from Phase 0, run with realistic inputs
    against the REAL boundary (not a mock)

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

1. **Trust nothing in green.** A passing test is evidence the test executed, not that the code is correct. Class A exists because green tests kept being trusted.

2. **The boundary is the contract.** When auditing a code path that talks to a system boundary, the contract is what the boundary accepts, not what the code currently sends. If the test mocks the boundary, the test cannot prove the contract holds.

3. **Static methods see static problems.** grep + imports + AST find naming inconsistencies. They cannot find Class D (edge-input format), Class E (silent drop), or runtime-context bugs. Execute, don't grep, for those classes.

4. **Pattern hunts scale by regex, not by audit.** Class C extends by adding one GENERIC regex per new footgun class — a detector that can fire on future code, never a memorial of the fixed instance. Propose additions in conversation before writing them here.

5. **Surface inventory is not optional.** Phase 0 is where Class B is found. Skip Phase 0 → no Class B finding for the rest of the cycle, because Classes A/C/D/E all walk code that tests exist for.

6. **Review gate every finding.** Agents emit false positives. Renamed classes look deleted; submodule imports look missing; "the method is unreferenced via grep" is not proof the method is dead. The hard rule from the project memory applies — see `[[feedback-never-delete-on-grep-alone]]`.

7. **Affected-test discovery covers observables, not just symbols.** When a fix changes a contract, grep tests/ for every assertion on the contract's OBSERVABLES (counters, lengths, response fields) — not only direct callers — and run every hit, e2e included, regardless of CI markers. Report the exact scope that ran; never summarize a targeted selection as "integration green".

---

## Findings format

Every audit finding goes into a table with these columns:

| Severity | Class | Where (file:line) | What's broken on the happy path | Reproduction | Confidence |
|---|---|---|---|---|---|

- **Severity:** CRIT / HIGH / MED / LOW — based on user-visible impact, not on how clever the finding is.
- **Class:** A / B / C / D / E / F / G — so the next audit knows which detection method earned each finding.
- **Where:** absolute file path + line; not "somewhere in the dispatcher".
- **What's broken on the happy path:** the failure mode, not the implementation detail. "404 on every cancel" not "missing `task_id` field".
- **Reproduction:** the exact command or code that demonstrates the failure. If you can't reproduce, the finding is speculation — lower confidence, mark it.
- **Confidence:** HIGH / MEDIUM / LOW — be honest. "MEDIUM — needs live Vespa to confirm 400 vs 200" is more useful than false HIGH.

---

## Done criteria

An audit is done when:

1. **All seven phases ran.** Skipping a phase = blind spot for that class.
2. **Every finding has a row in the findings table.** No verbal findings.
3. **Every CRIT / HIGH finding has a fix or a deferred-TODO with a written plan** (plan doc in `docs/plan/` + inline pointer-TODOs at the affected sites, per user approval).
4. **Every fix shipped with a real-boundary regression test** (see strict-commit.md).
5. **Any new footgun CLASS surfaced this cycle has a generic detector added** to the Class C hunt list — generic, able to fire on future code, never a memorial of the fixed instance.

If any of (1)–(5) is missing, the next audit will need to catch what this one didn't — exactly the failure mode this protocol exists to break.

---

## What an audit must NOT do

- Delete production code without explicit user approval, even when (1) the audit flags it, (2) a live replacement is named, (3) tests pass without it. See `[[feedback-never-delete-on-grep-alone]]`.
- Dismiss failures as "pre-existing", "LLM-dependent", "transient", "infrastructure". See `CLAUDE.md` → Testing Best Practices.
- Stop at "the audit agent found N issues." The review gate runs against every finding. Agent false positives are routine.
- Defer a finding to "the next audit." That is the exact failure mode this protocol exists to break.
