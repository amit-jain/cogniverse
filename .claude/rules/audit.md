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

**This protocol enumerates them up front.** The detection methods below (classes A–G) are orthogonal; run all of them in one audit cycle.

---

## The bug classes and their detection methods

Every finding so far falls into one of these classes (A–G). Hunt every class in every audit. No class is optional.

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

# Model-name single-vector/global classification by BARE `"lvt" in name.lower()`
# substring instead of the token-bracketed `_lvt_` (7th audit: encoders.py
# VideoPrismQueryEncoder.is_global — fixed via _videoprism_is_global helper;
# SIBLING sites still bare-substring: model_loaders.py:788 and
# videoprism_text_encoder.py:304 [the latter is a model-name remap elif chain,
# not is_global]). All real lvt model names carry `_lvt_`, so today it only
# misfires on a hypothetical name embedding the 3 letters — robustness, not a
# firing bug.
grep -rnP '"lvt" in [^.]*\.lower\(\)|"global" in [^.]*\.lower\(\)' libs/ --include="*.py"

# LM-availability skip gate that probes ONLY Ollama's /api/tags (7th audit:
# tests/runtime/integration/conftest.py + test_detailed_report_real.py +
# test_query_enhancement_real.py + tests/e2e/conftest.py). When the configured
# llm_config.primary.api_base ends in /v1 (vLLM/OAI-compat), /api/tags 404s and
# the gate FALSELY SKIPS the whole suite even though the LM is up — an infra
# skip = a bug. The canonical fix probes BOTH /api/tags and /v1/models after
# stripping a trailing /v1 (see tests/agents/integration/conftest.py:is_llm_available
# and tests/fixtures/llm.py). The negative-lookahead finds single-probe gates.
grep -rnP '/api/tags(?!.*v1/models)' tests/ --include="*.py" | grep -iE 'available|skip|reachable'

# FastAPI Query params sent as multipart FORM fields by test clients —
# FastAPI silently applies the defaults (8th: wait/wait_timeout/force posted
# in data= to /ingestion/upload; the test believed wait=true but the route
# returned immediately and force was never honored, so idempotency dedupe
# reused stale ingests). For each hit, check the route's declaration.
grep -rnP '"(wait|wait_timeout|force|limit|timeout_s)"\s*:' tests/ --include="*.py" -A0 | xargs -I{} true # then verify each data={...} block against the route's Form/Query split

# Optional enrichment step inside a larger pipeline called UNGUARDED — one
# unreachable sidecar throws away all prior computation (8th: face pipeline
# inside KG extraction; ConnectError destroyed entities+claims already
# extracted). Look for conditional `if <feature>_url:` calls without try.
grep -rnP 'if \w+_(url|endpoint):' libs/ --include="*.py" -A3 | grep -B2 -A2 "_run_\|_pipeline(" | grep -v "try:"

# Epoch value passed through as int() with NO seconds-vs-milliseconds magnitude
# guard nearby — the write side hardened ms-vs-s (_validate_ms/_validate_s) but
# read/filter sites reuse the same isinstance((int,float)) shape with a bare
# int(v) (search_agent _epoch_ms; memory _compute_age_seconds before the fix).
# For each hit, confirm a *1000 / //1000 / >1e11 magnitude check within ~3 lines.
grep -rnP 'return int\((v|ts|value|timestamp|raw|created_at)\)\s*$' libs/ --include="*.py"

# datetime.utcnow() (8th audit, messaging/auth.py) — always NAIVE, distinct
# from datetime.now() because it LOOKS tz-correct but isn't. A naive utcnow()
# compared against a fromisoformat()-parsed value raises TypeError the moment a
# tz-aware value is stored; a surrounding bare except then fail-closes (a valid
# token is rejected). Replace with datetime.now(timezone.utc) on both sides.
grep -rnP "datetime\.utcnow\(\)" libs/ --include="*.py"

# (1,dim)->dict tensor conversion guarded only by ndim==2 without a shape[0]==1
# flatten, where a sibling input branch flattens (8th audit: search_backend
# qt/qtb vs the q branch). A single-vector encoder emitting (1,dim) on a dense
# _sv_ schema input becomes a nested {"0": [...]} dict -> Vespa 400. For each
# hit, check whether the bound schema input is dense (needs flat) or
# multi-vector (needs the dict).
grep -rnP '\{str\(\w+\): \w+\.tolist\(\)' libs/ --include="*.py"

# Phoenix get_spans_dataframe without an explicit timeout kwarg (9th audit) —
# the client METHOD defaults to 5s and overrides any client-level timeout;
# loaded-project scans blow through it and the failure often reads as
# "no spans". Every call site must pass timeout=.
grep -rnP 'get_spans_dataframe\(' libs/ tests/ --include="*.py" | grep -v "timeout="

# Backend-read failure flattened to an empty result the caller treats as a
# VALID "no data" state (9th audit: checkpoint_storage list/latest, approval
# get_pending_batches, annotation storages — a Phoenix outage read as "no
# checkpoints" and silently restarted workflows). For each hit whose try
# wraps a provider/telemetry/backend query: log {e!r} and raise instead.
grep -rnP -A2 'except Exception' libs/ --include="*.py" | grep -B2 -E 'return (\[\]|\{\}|None|pd\.DataFrame\(\))' | grep -B2 -iE 'spans|telemetry|provider|phoenix'

# Whole-project span pull filtered client-side by name where get_spans
# supports filters={"name": ...} server-side (9th audit: ~14 storage +
# dashboard sites; each scan re-downloads the full project window).
grep -rnP 'df\["name"\] ==|spans_df\["name"\] ==' libs/ --include="*.py"
# then check the paired get_spans call for a filters= kwarg

# Test-spawned docker containers without the owner-pid label (9th audit) —
# unlabeled spawns escape reap_dead_owner_containers, so a SIGKILLed pytest
# session orphans model-weight/JVM containers in host RAM indefinitely
# (this starved a 126 GB host into a freeze once). NOTE (16th audit): the
# label literal the tests actually use is `cogniverse-test-owner-pid` — an
# earlier form of this hunt filtered on the constant name OWNER_LABEL, which
# appears nowhere in the spawn sites, so every correctly-labeled spawn read
# as unlabeled (6/6 false positives). Also pre-filter to real `docker run`
# spawns: bare `"run",` matches ["uv","run",...] and kubectl argv too.
grep -rnP '"docker",\s*$' tests/ --include="*.py" -A4 | grep -B1 '"run"' \
  | grep -v -e 'OWNER_LABEL' -e 'cogniverse-test-owner-pid'
# then confirm each docker-run block includes --label cogniverse-test-owner-pid={pid}

# Deps/config attribute assigned in __init__ but never functionally read
# (9th audit: summarizer/report max_*_length + thinking_enabled +
# technical_analysis_enabled; strategies dropping inference_service via
# **kwargs; dynamic_dspy_mixin rebuilding endpoints without seed/retries/
# timeout/headers). Not greppable in one pass — for each self.<knob> = deps.<knob>
# in an agent __init__, demand a read site that changes behavior.
grep -rnP 'self\.(max_\w+|.*_enabled|inference_service) = ' libs/agents/ libs/runtime/ --include="*.py"

# numpy scalar reaching a YQL/filter scalar serializer without .item() coercion
# (11th audit: _yql_scalar/_build_filter_conditions — np.float64 IS a float
# subclass so repr() emits `score = np.float64(0.5)`; np.int64 is NOT an int
# subclass so it falls to the string branch -> `count contains "5"`; both are
# Vespa 400). For each hit, confirm the value is coerced via .item() or gated by
# np.integer/np.floating/np.bool_ BEFORE the isinstance(int,float) dispatch.
grep -rnP 'isinstance\([^,]+, \(int, float\)\)' libs/vespa/ libs/agents/ --include="*.py"
# then check whether a numpy scalar can reach it un-coerced

# Unlabeled test docker spawn where a SIBLING spawn in the SAME file IS labeled
# (11th audit: face-embed + Phoenix in tests/conftest.py escaped reap while the
# Vespa spawn carried the owner-pid label — convention divergence the isolated
# per-file regex misses).
grep -rlP '"docker",\s*$|docker run' tests/ --include="*.py" | while read f; do \
  grep -qP 'OWNER_LABEL|cogniverse-test-owner-pid' "$f" && \
  grep -cP '"run"|docker run' "$f" | grep -qv '^1$' && echo "MIXED labeled/unlabeled: $f"; done

# Telemetry/backend read `except Exception: ...; return []/None` where a SIBLING
# in the same package was hardened to `raise` (11th: analytics.py get_spans
# returned [] while checkpoint_storage was made to raise; config_store/
# adapter_store flatten Vespa outage to None/[] indistinguishable from no-data).
grep -rlP 'except Exception' libs/ --include="*.py" | while read f; do \
  grep -qP 'get_spans_dataframe|\.spans\.|config_store|adapter' "$f" && \
  grep -qP 'return \[\]|return None' "$f" && echo "$f"; done

# pytest DESELECTION gated on an infra-readiness probe — invisible to the
# "0 skipped" gate, unlike a skip (11th: the only in-cluster sandbox code-exec
# e2e was deselected whenever _openshell_sandbox_ready() was false).
grep -rnP 'pytest_deselected|skip_substrings\.append|items\.remove' tests/ --include="*.py" -B3 \
  | grep -iE '_ready\(\)|_available\(\)|environ\.get|reachable'

# Config field advertising a capability the module never implements — ttl/
# expiry, compression, timeout, stats-toggle, serialization_format (11th:
# media.cache.ttl_days, media.backends.http.timeout_s, cache.enable_compression,
# chunk_processor.cache_chunks, quality_monitor NDCG/error/p95 ceilings all
# reach a config attr with zero behavior-changing reader). For each hit, grep
# the owning module for a read that changes behavior.
grep -rnE '(ttl|expire|_timeout_s|compress|enable_stats|serialization_format|_ceiling)\w*\s*[:=]' libs/ --include="*.py" | head -40

# FastAPI include_router mounting a mutating/admin/tenant router with NO
# dependencies=[Depends(auth)] (11th: the runtime mounts every router — incl.
# anonymous DELETE /admin/tenants and memory-wipe — behind only CORSMiddleware;
# no auth dependency exists in the codebase at all).
grep -rnP 'include_router\(' libs/ --include="*.py" | grep -viE 'dependencies\s*=' \
  | grep -iE 'admin|tenant|delete|ingest|graph|events'
# then confirm NO app-level auth middleware/dependency guards the app

# Store class with MIXED read-failure contracts — singular get_X hardened to
# raise on backend failure while list_X/get_X_history in the SAME file still
# flatten to []/None (12th audit: config_store get_config_history/list_configs/
# list_all_configs + adapter_store.list_adapters vs their hardened get_
# siblings — a Vespa outage reads as "no schemas registered" to schema_registry
# and silently reverts finetuned LoRA adapters to the base model; also
# backend.query_metadata_documents, whose []-on-400 contract a unit test
# locks in while test_store_read_outage_raises pins the opposite for get_).
grep -rlP 'except Exception' libs/ --include="*.py" | while read f; do \
  grep -qP 'read FAILED|raise so callers|raise rather than mask' "$f" \
  && grep -qP 'return \[\]$|return None$' "$f" && echo "MIXED raise/flatten: $f"; done

# Escape helper the `def _escape` hunt misses — short names and lambdas
# (12th audit: telemetry provider's nested `_esc` was found only by manual
# read; verify each escapes BOTH backslash and the quote char it wraps with).
grep -rnP 'def _esc\b|_esc\s*=\s*lambda|def _quote\b' libs/ --include="*.py"

# Boolean capability toggle with zero behavior-changing readers — the ttl/
# compress/timeout knob regex above misses bool toggles (12th audit:
# CacheConfig.enable_stats defined, stats always collected regardless).
grep -rnP '\b(enable|disable)_\w+\s*:\s*bool\s*=' libs/ --include="*.py"
# then for each, grep the owning package for a read that changes behavior

# Readiness/convergence probe that times out and "proceeds anyway" instead of
# raising — deploy/startup reports success for a resource that never became
# ready (12th audit: _wait_for_schema_convergence warns "proceeding anyway"
# and deploy_schemas returns True for a schema Vespa never activated;
# reproduced under sweep load where the 6 lifecycle tests fail together and
# all pass in isolation. Sibling: runtime lifespan "proceeding anyway").
grep -rniP 'proceeding anyway|continuing anyway|proceed despite' libs/ --include="*.py"

# Vespa query response consumed via .hits with no root.errors / coverage
# check — soft timeouts arrive as HTTP 200 + root.errors + partial/empty
# children, so degraded queries return [] AND get recorded as SUCCESS
# (12th audit: search_backend._process_results checks only hasattr(response,
# "hits") then record_search(True); the convergence probe in backend.py DOES
# check root.errors — the search path diverged from its own sibling).
grep -rnP 'response\.hits|\.get\("root", \{\}\)' libs/ --include="*.py"
# then verify an errors/coverage.degraded check inside the consuming function

# Test module importing a session fixture directly (12th audit remediation) —
# pytest treats a module-level import as a SECOND FixtureDef with its own
# session cache, silently booting a duplicate infra container mid-sweep;
# combined with a first-wins process singleton (_shared_schema_registry) this
# cross-wired schema deploys across two Vespa containers and produced 6
# chronic lifecycle failures. Fixture re-exports belong in conftest.py ONLY.
grep -rn "from tests.conftest import" tests/ --include="*.py" | grep -v "conftest.py:"

# First-wins process-global singleton handed to consumers with DIFFERENT
# configuration — the seed captures the first consumer's binding (endpoint,
# store, tenant) and every later consumer inherits it regardless of its own
# config (12th audit: BackendRegistry._shared_schema_registry; fixed via
# endpoint-scoped reuse). For each hit, check the seed's binding is compared
# against the requesting consumer's config before reuse.
grep -rnP 'if cls\._\w+ is None.*and|cls\._shared\w+ = ' libs/ --include="*.py" | grep -i "shared\|singleton\|global"

# Config-resolving helper invoked WITHOUT a config_manager, where the helper's
# get_config/ConfigUtils call RAISES on config_manager=None inside a broad
# except that degrades to nullcontext/base-model/defaults — the feature is
# silently DEAD ON ARRIVAL with only a warning log (13th audit:
# profile_selection_agent → adapter_lm_context → get_config(None) ValueError
# → base model; masked by tests that patched get_config). The sibling that
# works via a different seam is the tell.
grep -rnP 'get_config\([^)]*config_manager=None|adapter_lm_context\([^,)]+,\s*"[^"]+"\s*\)' libs/ --include="*.py"
# then check whether a wrapping try/except swallows the ValueError into a fallback

# State/expiry encoded in a file's mtime via os.utime and read back via
# stat().st_mtime. Two hazards: (a) a reader/cleanup in the write→utime window
# sees mtime=write-time and deletes a FRESH entry as expired — the writer's
# utime then hits a missing file (13th audit: structured_filesystem set(),
# 800/800 false-miss repro incl. never-expire entries); (b) any mtime-resetting
# op (cp without -p, rsync without -t/-a, restore) silently expires the whole
# cache. Require tmp-write + utime(tmp) + os.replace, or a sidecar/manifest.
grep -rnP 'os\.utime\(' libs/ --include="*.py"

# Hot-path per-request config read that BYPASSES the manager's scoped TTL
# cache — a get_*_config that calls store.get_config directly while siblings
# route through _cached_config_value → an uncached synchronous Vespa query on
# the event loop per dispatch (13th audit: get_agent_config via
# _agent_behavior_kwargs).
grep -rnP 'def get_\w+_config\(' libs/foundation/cogniverse_foundation/config/manager.py
# then for each, confirm the body routes through _cached_config_value

# Module-level memo keyed by a content signature (path,mtime)/(dir,mtime-sig)
# with NO eviction — every source edit adds a key, old keys never drop; and/or
# the memo returns the SHARED mutable value (caller mutation poisons the cache)
# (13th audit: _ALL_STRATEGIES_CACHE). Prefer clear()-then-set single-entry
# replace (as _JSON_CONFIG_CACHE does) + defensive copy on return.
grep -rnP '^_[A-Z_]+(CACHE|MEMO)\b\s*[:=]' libs/ --include="*.py"
# then check the write path for eviction and the return path for copying

# Method typed `-> bool` that returns a delegatee whose own signature returns
# Dict/other — the type lie ships green because no test calls it (13th audit:
# VespaBackend.health_check -> bool returning VespaSearchBackend.health_check's
# Dict).
grep -rnP 'def \w+\([^)]*\)\s*->\s*bool:\s*$' libs/ --include="*.py" -A3 | grep -B2 'return self\.\w+\.\w+\('

# Validator whose try body is a comment + unconditional `return True` — a
# validation method that never validates (13th audit: backend.validate_schema
# "# This would query Vespa" → return True; its except branch is dead).
grep -rnP 'def (validate|verify|check)_\w+' libs/ --include="*.py" -A6 | grep -B4 'return True' | grep -B2 '# .*would\|# TODO\|pass$'

# Raw Vespa document/v1 URL construction OUTSIDE libs/vespa — a caller
# hand-building /document/v1/<ns>/<schema>/docid/<id> HTTP bypasses the backend
# abstraction's session reuse, error contract, and namespace handling (14th
# audit migration: wiki_manager, graph_manager, ingestion router — all moved
# onto VespaBackend.put/get/update/delete_document_fields). The trailing slash
# distinguishes construction from prose mentions ("document/v1 URL"). Enforced
# by tests/backends/unit/test_docv1_confined_to_vespa.py.
grep -rn 'document/v1/' libs/ --include="*.py" | grep -vE '^libs/vespa/|main\.py:'

# Agent raw-requests Vespa /search/ consumer that reads root.children after
# ONLY a status_code!=200 guard, with NO root.errors/coverage.degraded check
# (14th audit: document_agent/image_search_agent/audio_analysis_agent all read
# data.get("root",{}).get("children",[]) directly — a Vespa soft-timeout is
# HTTP 200 + root.errors code 12 + partial children, so degraded results ship
# as complete; diverged from search_backend._process_results which raises on
# root.errors. Fixed via the shared vespa_search_children() helper).
grep -rnP 'data\.get\("root", \{\}\)\.get\("children"' libs/agents/ --include="*.py"
# then confirm the enclosing method raises on root.errors before iterating

# object.__new__(ProductionClass) OUTSIDE tests — a production path building a
# partially-initialized instance that skips __init__ invariants (14th audit
# hunt; none found in prod, but the pattern recurs in fixtures and can leak in).
grep -rnP 'object\.__new__\(|__class__\.__new__\(' libs/ --include="*.py"

# A raise-on-degraded search helper consumed INSIDE a broad except that
# returns [] — the except re-swallows the VespaSearchDegraded the helper
# raises, defeating soft-timeout surfacing (15th: document_agent + audio agent
# re-swallowed while image re-raised; the graph manager never adopted the
# helper at all). For each async search/find method consuming the helper,
# confirm the enclosing except RE-RAISES, never returns [].
grep -rnP '\.post\([^)]*/search/|vespa_search_children' libs/agents/ --include="*.py"
# then read each enclosing method: `except Exception` must `raise`

# Test mocks a pyvespa data-plane/query op to RETURN a non-2xx status_code
# WITHOUT raising — real pyvespa raise_for_status raises on every 4xx/5xx
# except 404, so the mock exercises a belt-and-braces branch via a shape the
# boundary can't produce while the real raise->except path stays untested
# (15th: delete 500-mock + query 400-mock, both with FALSE "pyvespa does not
# raise" docstrings the SUT's own comments contradicted).
grep -rnP 'status_code\s*=\s*(4|5)\d\d' tests/ --include="*.py" | grep -viE '404|raise|side_effect'
grep -rniP 'pyvespa[^\n]*(does not|doesn.t|not) raise' tests/ libs/ --include="*.py"

# Runtime startup add_*/register_* that HARDCODES a dict duplicating a
# configs/config.json section inside a broad-except swallow — the copy drifts
# silently and nothing pins the identity (15th: main.py wiki_semantic dup;
# fixed by reading the loaded config + a chart-vs-configs identity test).
grep -rnPA6 'add_backend_profile\(|register_\w+_profile\(' libs/runtime/ --include="*.py" | grep -P '=\s*\{'

# Async route/handler calling a blocking deploy/convergence primitive (with
# internal sleeps) DIRECTLY on the loop while a SIBLING call site offloads the
# same call via to_thread (15th: admin.py deploy_schema x2 vs tenant_manager's
# offload; the sigusr1 handler ran a Vespa round-trip inline against its own
# "must return immediately" comment).
grep -rnPB10 'schema_registry\.deploy_schema\(|reload_config\(\)' libs/runtime/ --include="*.py" | grep -P 'async def|add_signal_handler|to_thread'

# The offloaded-the-neighbor-not-the-hot-loop tell: an async def containing
# BOTH an await asyncio.to_thread AND a bare sync extractor/LM/HTTP call in a
# loop (15th: ingestion offloaded the 30s upsert while per-segment GLiNER
# [240s timeouts] + the face pipeline ran inline, deferring SIGTERM to a
# SIGKILL). Confirm each such call is itself inside to_thread.
grep -rnP '\.(extract_from_text|predict_entities)\(|_run_\w+_pipeline\(' libs/runtime/ --include="*.py"

# A function that ACCEPTS a *_contains/name_*/query filter param and never
# references it in its body — callers believe they filtered; every row comes
# back (15th: graph _visit(name_contains=query) returned ALL tenant nodes on
# the encoder-down fallback).
grep -rnP 'def \w+\([^)]*_contains[^)]*\)' libs/ --include="*.py"
# then confirm the param appears in the function body

# Profile-name allowlist gating a query-construction feature — any profile
# named outside the list silently loses the feature (15th: use_nearestneighbor
# allowlist killed semantic ranking for wiki/memory/audio profiles; replaced
# by structural derivation from the rank profile's first-phase expression).
grep -rnP 'profile_name in \[|profile_name in \(' libs/ --include="*.py"

# Tenant→Phoenix-project reader that doesn't canonicalize (16th audit:
# annotation_agent, approval_storage, orchestration_annotation_storage — the
# runtime WRITES under canonical "org:tenant" projects; a reader deriving the
# project/provider from a raw id queries an EMPTY project on real traffic.
# get_project_name is a pure template and canonicalizes nothing; the fix is
# canonical_tenant_id at the reader's constructor entry, as
# routing/annotation_storage.py does).
grep -rnP 'get_project_name\(|f"cogniverse-\{[^}]*tenant' libs/ --include="*.py"
# then confirm canonical_tenant_id was applied to that tenant_id in the same
# scope (constructor arg or immediate assignment)

# Dashboard/CLI entrypoint feeding a raw tenant id into a per-tenant store
# or provider (16th audit: libs/dashboard had ZERO canonical_tenant_id usage
# and passed st.session_state["current_tenant"] straight into the reader
# classes above — self-defending constructors close it, but new stores must
# canonicalize too).
grep -rnP 'session_state\[.?current_tenant|st\.session_state\.get\("current_tenant"' libs/ --include="*.py"
# then confirm canonical_tenant_id() is applied before the value reaches
# get_provider / get_project_name / *Storage(tenant_id=...)

# k8s label value built from a tenant id without sanitizing (16th audit:
# canonical ids contain ':', which k8s label values reject — the Argo submit
# path sanitizes via replace(":", "_") while the tenant router uses a fuller
# _sanitize_label_value; a NEW manifest-construction site that forgets either
# form fails the whole submit).
grep -rnP '"tenant[^"]*":\s*[a-z_]*tenant' libs/ --include="*.py" | grep -viE 'replace|sanitize|params|parameters'
# then confirm the value is sanitized before landing in metadata.labels

# Single-instance TTL/request cache invalidated only by same-instance writes
# (16th audit: the dispatcher's per-tenant ArtifactManager amortizes reads via
# a 5s request cache; promote_to_canary invalidates the SAME instance only, so
# an out-of-band promoter — admin API, CLI pod — converges only when the TTL
# lapses. Bounded staleness is the accepted contract; flag any NEW consumer
# that assumes immediate cross-instance visibility, and any cache whose
# staleness is UNBOUNDED).
grep -rnP '_request_cache|_invalidate_request_cache|_REQUEST_CACHE_TTL' libs/ --include="*.py"
# then check every writer that must be visible to cached readers either goes
# through the same instance or tolerates the TTL
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

### Class F — Concurrency, thread-safety, resource lifecycle, multi-process consistency

The other five classes audit code in isolation; Class F attacks what happens when it runs *together*. **Execute or full-read call chains — do NOT grep-heuristic** (a grep for "added backend calls without to_thread" misses the case where the blocking call is in a *sync manager method* and the un-offloaded caller is an *unchanged/adjacent* async route — the 14th audit's deep-read caught 3 such sites a diff-grep declared clean). Detect:

1. **Shared-session thread-safety** — for any long-lived client shared across threads, verify the wrapper holds no per-request mutable state, then hammer N threads × M ops through ONE instance against a real backend, asserting zero exceptions + zero cross-talk (14th: PersistentVespaOps over httpr.Client — safe; 2400 ops, 0 exceptions).
2. **Non-atomic get-then-set on a process cache/registry** — every cold-start build must use `set_if_absent`+close-loser or `get_or_set`, never `get()`+`set()`; the displaced-close causes a *use-after-close* on the loser's session, not just a leak (14th: EntryPointRegistry.get() — 24 threads → 23 use-after-close; canonical fix backend_registry.py `set_if_absent`).
3. **Sync blocking I/O on an async loop** — for every `async def`, confirm each pyvespa/`requests`/`time.sleep`/blocking-read is inside `await asyncio.to_thread(...)`; a `to_thread` call *adjacent* to a bare sync call in the same function is the tell (14th: wiki routes, ingestion.py graph upsert, lifecycle_scheduler pin lookup each offloaded the neighbor but not the call). Trace to the loop: main API loop stalls all requests; a sequential worker loop stalls shutdown/claim and `_time.sleep` retries freeze it for minutes.
4. **Unlocked lazy init of a session-owning attr** — `if self._x is None: self._x = build()` racing two first-touches leaks the loser; a single-call "primer" is a tell the init is unguarded (14th: `_metadata_vespa_app`, `get_config_manager_singleton`).
5. **Atomic-write orphans** — tmp→`os.replace` needs an age-gated `.tmp` reaper; a blanket suffix-skip in cleanup grows disk unboundedly across hard-kills (14th: structured_filesystem cleanup).
6. **Multi-replica staleness** — flag only *unbounded* in-process staleness on correctness-bearing values (a TTL-bounded scoped cache is fine).

```bash
# sync blocking call NOT offloaded, sitting next to a to_thread in the same async def
grep -rnPB6 'await asyncio\.to_thread' libs/ --include="*.py" | grep -PE '(mgr|manager|wm|self\._\w+)\.\w+\([^)]*\)\s*$' | grep -v to_thread
# non-atomic registry cold-start build (get then set instead of set_if_absent)
grep -rnPA6 'cls\._instances\.get\(|self\._instances\.get\(' libs/ --include="*.py" | grep -PB6 '_instances\.set\('
# unlocked lazy session init
grep -rnPA3 'if self\._\w*(app|client|session) is None' libs/ --include="*.py" | grep -P '= make_|\.syncio\(|Client\('
# atomic-write cleanup blanket-skipping .tmp with no age gate
grep -rnPA2 '\.suffix (in \([^)]*\.tmp|== "\.tmp")' libs/ --include="*.py" | grep -i continue | grep -v st_mtime
```

Class F is invisible to every isolation-based class: a passing single-thread test proves nothing about concurrent use. Run the hammer; read the whole async call chain.

### Class G — Fault injection / failure-path execution

Every other class (including F's hammers) executes against HEALTHY boundaries. Class G runs the same paths with the boundary DOWN, HUNG, or FAILING MID-OPERATION and asserts the failure CONTRACT: raise-with-context (good) vs silent success / empty / None (bug) vs torn partial state (worst). The 12th cycle hardened base-store reads to raise; Class G catches the SAME bug re-introduced one layer UP (a manager's `except Exception: return []` re-flattens a primitive that now raises — 15th: wiki_manager + graph_manager did exactly this over the hardened doc primitives) and TORN state in multi-step writes no single-boundary test can see (15th: adapter set_active stranded a tenant with ZERO active adapters → silent base-model reversion).

Detection (EXECUTE — do not grep-conclude):

1. **Reads under outage** — point each store/manager at a DEAD PORT and a `docker pause`d container. Every read must raise (or return an explicit error object), never `[]`/`None`/`{}`/zero-counts indistinguishable from no-data ("empty graph" / "healthy wiki" illusions).
2. **Multi-step writes** — enumerate every get-then-set / deactivate-then-activate / feed-then-backref / node-then-edge sequence; monkeypatch step N to raise and inspect persisted state for a torn result. Fixes are compensation (restore prior state before re-raising) or idempotent-retry safety.
3. **HTTP degradation** — stub server returning 503/429/507/connection-reset/SLOW-trickle; TIME the retry budget: per-attempt timeout × retry count is the real hang bound (15th: "15s fail-fast" was 45s hung — 3 attempts), and confirm no non-idempotent op is retried blindly.
4. **Disk failure** — monkeypatch `os.replace`/`os.utime`/write → ENOSPC/EACCES mid-`set()`: the OLD value must survive a failed overwrite (15th: sidecar unlinked BEFORE replace destroyed legacy entries), no partial entry readable, no orphan tmp beyond the reaper.
5. **LM/telemetry down** — error surfaced vs silently swallowed into a base-model/no-op fallback; distinguish build-time laziness from call-time failure.
6. **Partial batches** — doc k of N schema-invalid: failure report names exactly the failed ids AND the good docs persist.

Greppable precursors (Class C sweeps them; G then executes the hits):

```bash
# Manager-layer re-flatten: except returning []/None/success around a backend
# call whose primitive RAISES on outage (15th: wiki_manager save/search/lint,
# graph_manager reads — router replied 200 "saved" with nothing persisted).
grep -rnPA3 'except Exception' libs/agents/ --include="*.py" \
  | grep -B3 -E 'return \[\]|return None|return \{\}|logger\.exception' \
  | grep -B1 'self\._backend\.'

# Multi-step write with no compensation: >=2 sequential backend writes in one
# method — step-2 failure orphans step-1 (15th: adapter set_active).
grep -rnPA12 'def set_active|def .*activate|def .*switch' libs/ --include="*.py" \
  | grep -cP '_update_\w+_fields\(|feed_data_point\('   # >=2 in one method = suspect

# Sidecar/metadata unlink or utime BEFORE os.replace — a failed replace
# destroys the old entry's liveness metadata (15th: structured_filesystem).
grep -rnPB4 'os\.replace\(' libs/ --include="*.py" | grep -P '\.unlink\(|os\.utime\('

# Retry count silently multiplying the timeout budget — worst-case hang is
# timeout x attempts, not timeout (15th: 15s x 3 = 45s per op, compounding
# across multi-feed paths like save_session).
grep -rnPA3 'num_retries\w*\s*=' libs/ --include="*.py" | grep -P 'timeout'
```

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
