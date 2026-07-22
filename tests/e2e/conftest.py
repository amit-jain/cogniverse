"""Shared fixtures and helpers for E2E tests.

Provides availability checks, skip markers, and Streamlit interaction helpers
for both API (httpx) and dashboard (Playwright) E2E tests.

Test artifact paths (real data used for ingestion tests):
- Video: data/testset/evaluation/sample_videos/v_-nl4G-00PtA.mp4
- Image: data/testset/evaluation/processed/keyframes/big_buck_bunny_clip/frame_0000.jpg
- Audio: extracted via ffmpeg from sample video (real speech/sound)
- PDF: Video-ChatGPT paper from arxiv (related to the test dataset)
- Document: data/testset/dataset_summary.md (real markdown about the evaluation set)
"""

import json
import os
import re
import subprocess
import tempfile
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

# Deployment-lifecycle tests bring up their own port-forward-based cluster
# and are exercised via a dedicated ``pytest tests/e2e/deployment/`` run —
# never as part of the main suite, which boots its own NodePort stack via
# ``e2e_stack`` (two cluster creates in one session would double a
# 20-minute boot and the subsuite's own stack-conflict guard would fire).
collect_ignore_glob: list[str] = ["deployment/*"]


def _openshell_sandbox_ready() -> bool:
    """True iff the openshell-mtls Secret is present in-cluster.

    Runtime needs it to dial the host gateway; without it the coding
    agent's sandbox dispatch raises a deterministic RuntimeError that
    can't be resolved from the test harness.
    """
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "secret",
                "openshell-mtls",
                "-n",
                "cogniverse",
                "--ignore-not-found",
                "-o",
                "name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def pytest_collection_modifyitems(config, items):
    """Run async tests before playwright, and deselect tests whose external
    deps aren't available.

    pytest-playwright leaves a registered asyncio event loop after teardown,
    which trips later pytest-asyncio tests with "cannot be called from a
    running event loop". Ordering pure-async first sidesteps the collision.

    Telegram tests need a bot token + chat ID, and the coding-sandbox test
    needs the openshell-mtls Secret in-cluster. Deselect (not skip) when
    those aren't set up.
    """
    skip_substrings: list[str] = []
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get(
        "TELEGRAM_TEST_CHAT_ID"
    ):
        skip_substrings.append("TestTelegramRealFlow")

    if not _openshell_sandbox_ready():
        skip_substrings.append("test_coding_agent_full_execution_with_sandbox")

    # Teacher-model optimization e2e requires scaling up cogniverse-vllm-llm-teacher
    # which is a 1-2 hour run end-to-end. Off by default; opt in via
    # RUN_TEACHER_OPTIMIZATION_E2E=1 (or invoke pytest with
    # ``-m requires_teacher_model`` to bypass this deselection).
    teacher_optim_explicit = os.environ.get(
        "RUN_TEACHER_OPTIMIZATION_E2E"
    ) == "1" or "requires_teacher_model" in (config.option.markexpr or "")
    if not teacher_optim_explicit:
        skip_substrings.append("test_router_optimization_e2e")

    # Non-router optimizer persistence tests are slow (each invokes
    # optimization_cli end-to-end against the cluster) and need real
    # telemetry traces to produce meaningful output. Off by default;
    # opt in via RUN_OPTIMIZER_PERSISTENCE_E2E=1 or
    # ``pytest -m requires_optimizer_data``.
    optimizer_data_explicit = os.environ.get(
        "RUN_OPTIMIZER_PERSISTENCE_E2E"
    ) == "1" or "requires_optimizer_data" in (config.option.markexpr or "")
    if not optimizer_data_explicit:
        skip_substrings.append("test_optimizer_persistence_e2e")

    if skip_substrings:
        keep = []
        deselected = []
        for item in items:
            if any(s in item.nodeid for s in skip_substrings):
                deselected.append(item)
            else:
                keep.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = keep

    def _priority(item):
        path = str(item.fspath)
        if "test_dashboard_e2e" in path:  # playwright lives here
            return 2
        # Pure async tests that would trip over the playwright loop:
        if any(
            mark in path
            for mark in (
                "test_messaging_gateway_e2e",
                "test_tenant_extensibility_e2e",
                "test_wiki_e2e",
            )
        ):
            return 0
        return 1

    items.sort(key=_priority)


# k3d NodePort URLs — defined in charts/cogniverse/values.yaml
RUNTIME = "http://localhost:33000"  # runtime.service.nodePort
DASHBOARD = "http://localhost:33501"  # dashboard.service.nodePort
PHOENIX_URL = "http://localhost:33006"  # phoenix.service.nodePort
TENANT_ID = "flywheel_org:production"

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
E2E_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "cogniverse_e2e_artifacts"


def runtime_available() -> bool:
    # /health/live is cheap; /health does backend + registry lookups and
    # can block under LLM load, producing false-negative skips.
    try:
        r = httpx.get(f"{RUNTIME}/health/live", timeout=30.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
        return False


def dashboard_available() -> bool:
    try:
        r = httpx.get(DASHBOARD, timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
        return False


def _ensure_stack_running() -> bool:
    """Verify the stack is running. Does NOT redeploy — a transient probe
    blip used to trigger a mid-suite helm upgrade that cascaded every
    downstream test into a pod-restart failure.

    Retries each probe a few times because k3d-serverlb intermittently
    drops the first connection with ``RemoteProtocolError`` even when
    the pod is healthy (same reason ``_runtime_already_up_for_collect``
    already retries 3× at collect time). Without the retry here, one
    such blip skipped the entire session.
    """
    import time as _t

    for attempt in range(5):
        if runtime_available() and dashboard_available():
            return True
        if attempt < 4:
            _t.sleep(3.0)
    return False


def _skip_if_no_runtime():
    """Skip marker that checks runtime availability at test time, not import time."""
    if not runtime_available():
        pytest.skip(
            "Runtime not available at localhost:33000 — run 'cogniverse up' first"
        )


def _skip_if_no_dashboard():
    """Skip marker that checks dashboard availability at test time, not import time."""
    if not dashboard_available():
        pytest.skip(
            "Dashboard not available at localhost:33501 — run 'cogniverse up' first"
        )


# Keep the old names for backward compat but make them no-ops
# The actual check happens in e2e_stack fixture (autouse, session-scoped)
skip_if_no_runtime = pytest.mark.e2e
skip_if_no_dashboard = pytest.mark.e2e


_MINTED_TENANTS_THIS_TEST: list[str] = []


def unique_id(prefix: str = "e2e") -> str:
    """Mint a per-test tenant id and register it for end-of-test cleanup.

    The session-end ``_cleanup_test_tenants`` sweep can't keep up with
    the per-test churn — Vespa accumulates 200+ tenant schemas mid-run
    and new deploys time out. Recording every mint here lets the
    autouse ``_drain_test_tenants_after_each_test`` fixture DELETE
    each tenant as soon as the test finishes, keeping the cluster
    schema count flat through the whole sweep.
    """
    tid = f"{prefix}_{uuid.uuid4().hex[:8]}"
    if any(tid.startswith(p) for p in _TEST_TENANT_PREFIXES):
        _MINTED_TENANTS_THIS_TEST.append(tid)
    return tid


# Vespa config-server URL. The e2e suite ASSUMES a k3d cluster with the
# config-server NodePort-mapped at localhost:33071 (see
# charts/cogniverse/values.k3s.yaml). Override via VESPA_CONFIG_URL
# only if running against a non-k3d topology.
_VESPA_SCHEMAS_LIST_URL = os.environ.get(
    "VESPA_CONFIG_URL",
    "http://localhost:33071",
).rstrip("/") + (
    "/application/v2/tenant/default/application/default/"
    "environment/prod/region/default/instance/default/content/schemas/"
)


def _vespa_config_server_reachable() -> bool:
    """One-shot probe of the Vespa config-server. Cached after first hit."""
    try:
        resp = httpx.get(_VESPA_SCHEMAS_LIST_URL, timeout=5.0)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _vespa_deployed_schema_names() -> set[str]:
    """Read the live deployed-schemas list straight from Vespa's config-server.

    Returns the set of base names (without the .sd suffix). Empty set
    on probe failure so callers treat the lookup as "don't know" and
    fall through.
    """
    try:
        resp = httpx.get(_VESPA_SCHEMAS_LIST_URL, timeout=10.0)
        resp.raise_for_status()
        entries = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return set()
    names: set[str] = set()
    for entry in entries:
        tail = entry.rsplit("/", 1)[-1]
        if tail.endswith(".sd"):
            names.add(tail[: -len(".sd")])
    return names


def _tenant_schema_names_in_vespa(tenant_id: str, deployed: set[str]) -> set[str]:
    """Subset of ``deployed`` whose name carries the tenant's suffix.

    Vespa-side tenant schemas are named ``<base>_<tenant_with_:_to_>``
    (e.g. ``agent_memories_kagent_kg_abc_t1``). We don't know the base
    set up front, so just match by suffix.
    """
    suffix = "_" + tenant_id.replace(":", "_")
    return {name for name in deployed if name.endswith(suffix)}


@pytest.fixture(autouse=True)
def _reset_event_loop_state_before_each_test():
    """Clear leaked thread-attached event loops before every test.

    Some upstream code paths in cogniverse + dspy + dspy/lite-llm call
    ``asyncio.set_event_loop(asyncio.new_event_loop())`` for a quick
    ``run_until_complete`` and never undo it. ``set_event_loop`` writes
    the loop into the thread-local ``_event_loop_policy.current_loop``
    slot, so after the call every future ``asyncio.get_event_loop()``
    on the same thread returns that leaked loop. When pytest-asyncio
    later tries to set up an async test, its ``Runner.run`` checks
    ``events._get_running_loop()``: if the leaked loop happens to still
    be "running" (e.g. partially closed via the leaker's cleanup path
    but with a coroutine still scheduled) it raises
    ``RuntimeError: Runner.run() cannot be called from a running event
    loop`` and the test fails before its body ever runs.

    Reset the thread's loop slot to ``None`` (and any leaked policy
    state) at the start of every test, so pytest-asyncio always sees a
    clean thread when it constructs its per-test runner.
    """
    import asyncio

    # Drop any leaked thread-current loop. Wrapping in try/except
    # because asyncio's API for "give me the leaked loop without
    # creating one" differs across 3.10/3.11/3.12.
    try:
        leaked = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        leaked = None
    if leaked is not None and not leaked.is_closed():
        try:
            leaked.close()
        except Exception:  # noqa: BLE001 — defensive
            pass
    try:
        asyncio.set_event_loop(None)
    except RuntimeError:
        pass
    yield


@pytest.fixture(autouse=True)
def _drain_test_tenants_after_each_test():
    """Delete every test tenant minted via ``unique_id`` after each test
    AND wait for Vespa to actually drop the tenant's schemas.

    Cleanup contract: every schema MUST be created via the
    SchemaRegistry deploy path AND removed via the tenant-delete path.
    A timed-out HTTP DELETE that left the runtime mid-redeploy
    silently produced the registry-vs-Vespa drift the deploy guard
    keeps tripping over. Replace the blind 30 s timeout with: send
    the DELETE (60 s for the runtime to ACK), then poll Vespa's
    schemas list every 2 s until none of the tenant's schemas remain.
    Hard cap at 10 minutes per tenant so a hung Vespa can't wedge the
    suite indefinitely.
    """
    _MINTED_TENANTS_THIS_TEST.clear()
    yield
    minted = list(_MINTED_TENANTS_THIS_TEST)
    _MINTED_TENANTS_THIS_TEST.clear()
    if not minted:
        return
    # Vespa config-server polling is part of the cleanup contract — the
    # only safe completion signal that the runtime DELETE actually
    # removed the schemas. Outside k3d (or a topology that exposes the
    # config-server at $VESPA_CONFIG_URL) we can't poll, so fail loudly
    # rather than silently leak schemas across the suite.
    if not _vespa_config_server_reachable():
        raise RuntimeError(
            f"_drain_test_tenants_after_each_test cannot reach Vespa "
            f"config-server at {_VESPA_SCHEMAS_LIST_URL!r}. The e2e suite "
            f"is k3d-only — start it with `cogniverse up`, or set "
            f"VESPA_CONFIG_URL to the config-server base URL of your "
            f"deployed cluster."
        )
    # Tests that mint via unique_id("<base>") may construct derived
    # tenants like f"{base}:t1". Cover the common shapes so we delete
    # the actual tenant the test wrote under. ``:_org_trunk`` is the
    # federation promotion target (org_trunk_tenant_id maps "<org>:x" to
    # "<org>:_org_trunk"): the promote route creates it as a side effect,
    # the test never mints it, so without reaping it here every
    # promotion test leaks one org-trunk schema set forever.
    targets: set[str] = set()
    for tid in minted:
        targets.add(tid)
        for suf in (":t1", ":t2", ":t3", ":production", ":org_admin", ":_org_trunk"):
            targets.add(tid + suf)
    for full_tid in targets:
        # Skip tenants that aren't actually in Vespa — most derived
        # suffixes (`:t2`, `:t3`, etc.) won't apply to a given test, so
        # the DELETE would 404 and we'd waste a 60 s timeout + poll
        # window per non-existent tenant.
        deployed = _vespa_deployed_schema_names()
        if not _tenant_schema_names_in_vespa(full_tid, deployed):
            continue
        try:
            with httpx.Client(timeout=60.0) as client:
                client.delete(f"{RUNTIME}/admin/tenants/{full_tid}")
        except (httpx.HTTPError, OSError):
            # Server may have started the redeploy anyway. The poll
            # below is the actual completion signal.
            pass
        # Poll Vespa until the tenant's schemas are gone from the
        # deployed app package. 10 min cap, 2 s interval.
        deadline = _time.monotonic() + 600.0
        last_remaining: set[str] = set()
        while _time.monotonic() < deadline:
            deployed = _vespa_deployed_schema_names()
            remaining = _tenant_schema_names_in_vespa(full_tid, deployed)
            if not remaining:
                break
            last_remaining = remaining
            _time.sleep(2.0)
        else:
            print(
                f"_drain_test_tenants_after_each_test: gave up waiting on "
                f"{full_tid!r} — Vespa still shows {sorted(last_remaining)} "
                f"after 600 s"
            )


def register_tenant_and_wait(
    tenant_id: str,
    *,
    created_by: str = "e2e",
    timeout_s: float = 600.0,
) -> None:
    """POST /admin/tenants and poll until the tenant is fully visible.

    Mirrors the deletion-side contract in
    ``_drain_test_tenants_after_each_test``: send the create, then poll
    Vespa's config-server schemas list every 2 s until the tenant's
    per-tenant schemas appear (read-after-write consistent with
    prepareandactivate), AND poll ``GET /admin/tenants/{tid}`` until the
    tenant_metadata search-side row is queryable. Hard cap at 10 minutes
    so a hung Vespa can't wedge the suite.

    Why: the bare 60 s tenant_metadata poll in the older test helpers
    was overrun by the cluster-wide schema-count growth (per-tenant
    deploy is O(N) in deployed schemas). The schemas-list poll uses the
    same definitive Vespa signal the cleanup contract already relies on,
    just inverted (presence instead of absence).
    """
    if not _vespa_config_server_reachable():
        raise RuntimeError(
            f"register_tenant_and_wait cannot reach Vespa config-server "
            f"at {_VESPA_SCHEMAS_LIST_URL!r}. The e2e suite is k3d-only — "
            f"start it with `cogniverse up`, or set VESPA_CONFIG_URL to "
            f"the config-server base URL of your deployed cluster."
        )

    # Send the create — give the runtime up to 5 min to ACK; the actual
    # readiness signal is the poll below, not the response code.
    with httpx.Client(timeout=300.0) as client:
        try:
            resp = client.post(
                f"{RUNTIME}/admin/tenants",
                json={"tenant_id": tenant_id, "created_by": created_by},
            )
        except (httpx.HTTPError, OSError) as exc:
            # Server may have started the deploy anyway; the poll below
            # is the actual completion signal. Don't fail here.
            print(
                f"register_tenant_and_wait: POST raised {exc!r} — "
                f"continuing to schemas poll"
            )
            resp = None
        if resp is not None and resp.status_code not in (200, 201, 409):
            raise RuntimeError(
                f"register_tenant_and_wait: POST /admin/tenants for "
                f"{tenant_id!r} returned {resp.status_code} {resp.text}"
            )

    deadline = _time.monotonic() + timeout_s
    saw_schema = False
    saw_metadata = False
    while _time.monotonic() < deadline:
        if not saw_schema:
            deployed = _vespa_deployed_schema_names()
            if _tenant_schema_names_in_vespa(tenant_id, deployed):
                saw_schema = True
        if not saw_metadata:
            try:
                with httpx.Client(timeout=10.0) as client:
                    r = client.get(f"{RUNTIME}/admin/tenants/{tenant_id}")
                    if r.status_code == 200:
                        saw_metadata = True
            except (httpx.HTTPError, OSError):
                pass
        if saw_schema and saw_metadata:
            return
        _time.sleep(2.0)
    raise RuntimeError(
        f"register_tenant_and_wait: tenant {tenant_id!r} not ready after "
        f"{timeout_s:.0f} s — saw_schema={saw_schema} "
        f"saw_metadata={saw_metadata}"
    )


def run_async(coro):
    """Run a coroutine to completion in a fresh OS thread.

    pytest.ini sets ``asyncio_mode = auto`` so pytest-asyncio enters an
    event loop on the calling thread for every test. A sync test body
    that calls ``asyncio.get_event_loop().run_until_complete(coro)`` or
    ``asyncio.new_event_loop().run_until_complete(coro)`` raises
    ``RuntimeError: This event loop is already running`` because asyncio
    refuses ``run_until_complete`` while the thread is inside another.
    Worse, the leaked-loop state cascades into subsequent
    ``@pytest.mark.asyncio`` tests which then fail with
    ``Runner.run() cannot be called from a running event loop``.

    Running the coroutine in a separate OS thread isolates it from
    pytest-asyncio's loop — ``asyncio.run`` in the worker creates a
    fresh loop, runs the coroutine, closes the loop, returns the result
    (or re-raises the exception) on the calling thread.
    """
    import asyncio
    import threading

    box: dict = {}

    def _runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — propagate verbatim
            box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture(scope="session")
def phoenix_client_session():
    """Single PhoenixClient reused across the session.

    Every span-polling test was rebuilding `PhoenixClient(base_url=...)` per
    invocation (e.g. test_a2a_gateway_e2e.py:1041, test_batch_optimization_e2e.py).
    Sharing one client over the e2e session avoids the per-test connection
    setup and keeps every assertion path against Phoenix using the same wire.
    """
    from phoenix.client import Client as PhoenixClient

    return PhoenixClient(base_url=PHOENIX_URL)


def wait_for_span(
    phoenix_client,
    project: str,
    name_substr: str,
    attribute_predicate=None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 2.0,
):
    """Poll a Phoenix project until a matching span lands or the deadline expires.

    Mirrors the polling logic at test_a2a_gateway_e2e.py:1045-1062 so every
    Phase that asserts on spans goes through the same helper. Returns the
    first matching pandas Series row (the span's record) or None on timeout.

    `name_substr` is a case-insensitive substring match on `span.name`.
    `attribute_predicate`, if given, is `(attrs_dict) -> bool` evaluated on
    each candidate span's attributes column. The first span that satisfies
    BOTH name match AND predicate is returned.
    """
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            spans_df = phoenix_client.spans.get_spans_dataframe(
                project_identifier=project,
                limit=200,
            )
            if spans_df is not None and not spans_df.empty:
                matches = spans_df[
                    spans_df["name"].str.contains(name_substr, case=False, na=False)
                ]
                if not matches.empty:
                    if attribute_predicate is None:
                        return matches.iloc[0]
                    for _, row in matches.iterrows():
                        attrs = row.get("attributes")
                        if attrs is None:
                            continue
                        # Phoenix returns attributes as a dict-like; coerce.
                        attrs_dict = (
                            dict(attrs) if not isinstance(attrs, dict) else attrs
                        )
                        if attribute_predicate(attrs_dict):
                            return row
        except Exception:
            # Phoenix can transiently 5xx during heavy ingest; keep polling
            # until the deadline. Failures past the deadline surface to caller.
            pass
        _time.sleep(poll_interval_s)
    return None


def restart_runtime(timeout_s: int = 60) -> bool:
    """Restart the runtime pod via kubectl.

    When the runtime is deployed on k3d, use kubectl to restart the pod
    instead of killing local processes. K8s will create a new pod automatically.
    Returns True if the new runtime is healthy.
    """
    try:
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "restart",
                "deployment/cogniverse-runtime",
                "-n",
                "cogniverse",
            ],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Wait for the new pod to become ready. /health/live is cheap and
    # doesn't queue behind uvicorn workers under LLM load.
    for _ in range(timeout_s):
        _time.sleep(1)
        try:
            r = httpx.get(f"{RUNTIME}/health/live", timeout=5.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            pass
    return False


def _bootstrap_tenant_and_schemas() -> None:
    """Create the E2E tenant and deploy schemas if not already done.

    Called once per session. Idempotent — 409 (already exists) is fine.
    """
    # Read profile definitions from config.json
    config_path = DATA_ROOT.parent / "configs" / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    active_profile = config.get(
        "active_video_profile", "video_colpali_smol500_mv_frame"
    )
    all_profiles = config.get("backend", {}).get("profiles", {})
    profile_def = all_profiles.get(active_profile, {})

    # Step 1: Create tenant (409 = already exists)
    try:
        resp = httpx.post(
            f"{RUNTIME}/admin/tenants",
            json={"tenant_id": TENANT_ID, "created_by": "e2e-test"},
            timeout=30,
        )
        if resp.status_code not in (200, 201, 409):
            print(f"Tenant creation returned {resp.status_code}: {resp.text[:200]}")
    except (httpx.HTTPError, OSError) as exc:
        print(f"Tenant creation failed: {exc}")

    # Delete-then-create so config.json edits take effect: POST rejects
    # re-creation and PUT can't change embedding_model. delete_schema=false
    # avoids redeploying the Vespa schema on every session.
    if profile_def:
        try:
            httpx.delete(
                f"{RUNTIME}/admin/profiles/{active_profile}",
                params={"tenant_id": TENANT_ID, "delete_schema": "false"},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            print(f"Profile pre-delete failed (non-fatal): {exc}")

        try:
            payload = {
                "profile_name": active_profile,
                "tenant_id": TENANT_ID,
                "type": profile_def.get("type", "video"),
                "description": profile_def.get("description", ""),
                "schema_name": profile_def.get("schema_name", active_profile),
                "embedding_model": profile_def.get("embedding_model", ""),
                "pipeline_config": profile_def.get("pipeline_config", {}),
                "strategies": profile_def.get("strategies", {}),
                "embedding_type": profile_def.get("embedding_type", "multi_vector"),
                "schema_config": profile_def.get("schema_config", {}),
                "model_specific": profile_def.get("model_specific"),
                "deploy_schema": True,
            }
            resp = httpx.post(
                f"{RUNTIME}/admin/profiles",
                json=payload,
                timeout=60,
            )
            if resp.status_code not in (200, 201, 409):
                print(
                    f"Profile registration returned {resp.status_code}: {resp.text[:200]}"
                )
        except (httpx.HTTPError, OSError) as exc:
            print(f"Profile registration failed: {exc}")


LLM_URL = "http://localhost:33434"
LLM_MODEL = "qwen3:4b"


def _ensure_llm_model() -> None:
    """Pull the LLM model if not already available."""
    try:
        resp = httpx.get(f"{LLM_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if any(LLM_MODEL in m.get("name", "") for m in models):
                return  # Model already loaded
        # Pull the model
        print(f"Pulling LLM model {LLM_MODEL}...")
        httpx.post(
            f"{LLM_URL}/api/pull",
            json={"name": LLM_MODEL},
            timeout=600,
        )
        print(f"Model {LLM_MODEL} pulled")
    except (httpx.HTTPError, OSError) as exc:
        print(f"LLM model pull failed (non-fatal): {exc}")


def _ingest_sample_video() -> None:
    """Ingest a sample video so search tests have data to query.

    Uses the runtime's POST /ingestion/upload endpoint with the sample
    ActivityNet video. Idempotent — re-ingesting the same video is harmless.
    """
    video_path = (
        DATA_ROOT / "testset" / "evaluation" / "sample_videos" / "v_-nl4G-00PtA.mp4"
    )
    if not video_path.exists():
        print(f"Sample video not found at {video_path}, skipping ingestion")
        return

    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    active_profile = config.get(
        "active_video_profile", "video_colpali_smol500_mv_frame"
    )

    # Skip re-ingest when search already finds this video for the tenant —
    # ColPali serialises through one pod queue, and a redundant ingest
    # blocks every other inference request behind it.
    try:
        probe = httpx.post(
            f"{RUNTIME}/search/",
            json={
                "query": video_path.stem,
                "profile": active_profile,
                "top_k": 1,
                "tenant_id": TENANT_ID,
            },
            timeout=20,
        )
        if probe.status_code == 200 and probe.json().get("results_count", 0) > 0:
            print(
                f"Sample video already in Vespa for tenant {TENANT_ID}; skipping ingest"
            )
            return
    except (httpx.HTTPError, OSError):
        pass

    try:
        with open(video_path, "rb") as f:
            resp = httpx.post(
                f"{RUNTIME}/ingestion/upload",
                files={"file": (video_path.name, f, "video/mp4")},
                data={
                    "profile": active_profile,
                    "backend": "vespa",
                    "tenant_id": TENANT_ID,
                },
                timeout=1800,
            )
        if resp.status_code == 200:
            data = resp.json()
            fed = data.get("documents_fed", 0)
            chunks = data.get("chunks_created", 0)
            print(f"Sample video ingested: {chunks} chunks, {fed} documents fed")
            if fed == 0 and chunks > 0:
                print(
                    "WARNING: chunks created but 0 documents fed to Vespa — "
                    "search tests will have no data"
                )
        else:
            print(f"Video ingestion returned {resp.status_code}: {resp.text[:200]}")
    except (httpx.HTTPError, OSError) as exc:
        print(f"Video ingestion failed (non-fatal): {exc}")


E2E_CLUSTER_NAME = "cogniverse-e2e"
DEV_CLUSTER_NAME = "cogniverse"

# Host-side loadbalancer ports of the e2e cluster. The right-hand side is
# the chart's canonical NodePort (unchanged); the host side is offset into
# 33xxx so the e2e stack never collides with a dev cluster's 28xxx/8080
# mappings or the 29xxx test-sidecar range. Every localhost URL in this
# suite uses the 33xxx side.
E2E_HOST_PORTS = {
    33080: 8080,  # vespa http
    33071: 19071,  # vespa config
    33000: 28000,  # runtime
    33501: 28501,  # dashboard
    33006: 26006,  # phoenix ui
    33317: 4317,  # otel grpc
    33434: 11434,  # llm (ollama-compat)
    33746: 2746,  # argo server
    33901: 29001,  # inference sidecars
    33902: 29002,
    33904: 29004,
    33905: 29005,
    33906: 29006,
    33910: 29010,
    33911: 29011,
}


def _port_bound(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _stop_dev_cluster_and_free_ports() -> None:
    """Stop (never delete) a running dev ``cogniverse`` k3d cluster.

    The host cannot fit two clusters' pods in RAM, and the dev cluster's
    loadbalancer holds the canonical NodePorts the e2e stack maps. Data
    survives the stop — bring it back with ``k3d cluster start cogniverse``
    (or ``cogniverse up``) after the e2e run.

    The e2e cluster's own host ports (33xxx) never overlap the dev
    cluster's, but a stray process could hold one — verify they are all
    free before the create, which would otherwise fail the whole session.
    """
    result = subprocess.run(
        ["k3d", "cluster", "list", DEV_CLUSTER_NAME],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0 and DEV_CLUSTER_NAME in result.stdout:
        print(
            f"Stopping dev cluster {DEV_CLUSTER_NAME!r} — the e2e stack needs "
            "the host's RAM and NodePorts. Restart it afterwards with "
            f"'k3d cluster start {DEV_CLUSTER_NAME}'."
        )
        subprocess.run(
            ["k3d", "cluster", "stop", DEV_CLUSTER_NAME],
            capture_output=True,
            text=True,
            timeout=180,
        )

    deadline = _time.time() + 90
    while _time.time() < deadline:
        busy = [p for p in E2E_HOST_PORTS if _port_bound(p)]
        if not busy:
            return
        _time.sleep(2)
    pytest.fail(
        f"e2e host ports {busy} are bound by another process — free them and re-run."
    )


_E2E_FINGERPRINT_CM = "e2e-build-fingerprint"


def _e2e_deploy_fingerprint() -> str:
    """Content hash of everything baked into the e2e images / affecting the
    deploy: committed HEAD + the diff of ``libs``/``configs``/``charts`` +
    any untracked files under them. ``tests/`` is deliberately excluded —
    pytest reads test files fresh each run, so iterating on test assertions
    must NOT force a cluster rebuild; a change to deployed code must."""
    import hashlib

    repo_root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout

    paths = ["libs", "configs", "charts", "pyproject.toml"]
    material = "\n".join(
        [
            _git("rev-parse", "HEAD").strip(),
            _git("diff", "HEAD", "--", *paths),
            _git("status", "--porcelain", "--untracked-files=all", "--", *paths),
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _kubectl_e2e(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--context", f"k3d-{E2E_CLUSTER_NAME}", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _read_e2e_fingerprint() -> str:
    got = _kubectl_e2e(
        "-n",
        "cogniverse",
        "get",
        "configmap",
        _E2E_FINGERPRINT_CM,
        "-o",
        "jsonpath={.data.fingerprint}",
    )
    return got.stdout.strip() if got.returncode == 0 else ""


def _stamp_e2e_fingerprint(fingerprint: str) -> None:
    rendered = _kubectl_e2e(
        "-n",
        "cogniverse",
        "create",
        "configmap",
        _E2E_FINGERPRINT_CM,
        f"--from-literal=fingerprint={fingerprint}",
        "--dry-run=client",
        "-o",
        "yaml",
    )
    subprocess.run(
        ["kubectl", "--context", f"k3d-{E2E_CLUSTER_NAME}", "apply", "-f", "-"],
        input=rendered.stdout,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _e2e_cluster_reusable(fingerprint: str) -> bool:
    """A warm ``cogniverse-e2e`` cluster can be reused iff it is running,
    reachable, its runtime answers, and it was built from the SAME deployed
    content (fingerprint). Any of those failing → boot fresh."""
    listed = subprocess.run(
        ["k3d", "cluster", "list", E2E_CLUSTER_NAME],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if listed.returncode != 0 or E2E_CLUSTER_NAME not in listed.stdout:
        return False
    if (
        _kubectl_e2e("get", "ns", "cogniverse", "-o", "name", timeout=15).returncode
        != 0
    ):
        return False
    if not runtime_available():
        return False
    return _read_e2e_fingerprint() == fingerprint


@pytest.fixture(scope="session", autouse=True)
def e2e_stack():
    """Provide a healthy, bootstrapped e2e stack — reusing a warm cluster
    when the deployed content is unchanged, booting fresh otherwise.

    The cluster is a dedicated k3d deployment whose loadbalancer maps the
    offset 33xxx HOST ports onto the chart's canonical NodePorts (see
    ``E2E_HOST_PORTS``), Helm-installed with devMode OFF so the pods run the
    code baked into images built from the working tree — never a bind-mounted
    tree with a stale interpreter. Host storage is NOT shared, so a fresh
    boot starts on clean data and cannot touch a dev cluster's state.

    Lifecycle:
      * REUSE — if a running ``cogniverse-e2e`` whose stamped deploy
        fingerprint matches the current ``libs``/``configs``/``charts`` is
        already up, reuse it (~seconds). Editing only ``tests/`` keeps the
        fingerprint, so assertion iteration is fast.
      * FRESH — otherwise (no cluster / unhealthy / deployed code changed /
        ``E2E_FRESH`` set) stop any dev cluster (RAM + ports), rebuild
        images, create the cluster, deploy, wait, and stamp the fingerprint.
        The fresh path is what proves the first-install contract.
      * TEARDOWN — the cluster is LEFT WARM for the next run; only
        ``E2E_FRESH=1`` deletes it. Reset manually with
        ``k3d cluster delete cogniverse-e2e``.

    After the stack is healthy (reused or fresh) the E2E tenant + schemas +
    sample data are (idempotently) bootstrapped and CronWorkflows suspended
    for the session.
    """
    from tests.e2e.deployment.conftest import (
        create_test_cluster,
        delete_test_cluster,
        deploy_stack,
    )

    fingerprint = _e2e_deploy_fingerprint()
    force_fresh = os.environ.get("E2E_FRESH", "").lower() in ("1", "true", "yes")

    if not force_fresh and _e2e_cluster_reusable(fingerprint):
        print(
            f"Reusing warm e2e cluster {E2E_CLUSTER_NAME} "
            f"(deploy fingerprint {fingerprint} unchanged)"
        )
    else:
        # A crashed previous session leaves its disposable cluster — and its
        # 33xxx port bindings — behind, and a stale-fingerprint cluster must
        # be replaced. Reap it before the port check, or the leftover reads
        # as "another process holds the ports".
        delete_test_cluster(E2E_CLUSTER_NAME)
        _stop_dev_cluster_and_free_ports()

        create_test_cluster(
            E2E_CLUSTER_NAME,
            ports=[f"{host}:{node}" for host, node in E2E_HOST_PORTS.items()],
            share_host_storage=False,
        )
        # Test-cluster-only helm overrides (never touch the shipped chart):
        #  - teacher LM off: only the opt-in teacher-optimization e2e uses it,
        #    and it can't coexist with colpali + student during GPU weight
        #    load; the dev cluster keeps it scaled to zero for the same reason.
        #  - vLLM liveness grace widened: on a COLD cluster the GPU engines
        #    load weights from disk for ~12 min (vs instant off the warm dev
        #    cache), then profile — overrunning the shipped 22-min liveness
        #    budget, so the kubelet kills them mid-init and they never
        #    converge. ~50 min (initialDelay 1200s + 60×30s) covers a cold
        #    load under memory contention. Only LIVENESS matters — readiness
        #    never kills, it just gates the Available condition the wait below
        #    keys off, so it is left shipped-default.
        gpu_vllm = ("vllm_colpali", "vllm_asr", "vllm_llm_student")
        extra_set = {"inference.vllm_llm_teacher.enabled": "false"}
        for svc in gpu_vllm:
            extra_set[f"inference.{svc}.livenessProbe.initialDelaySeconds"] = "1200"
            extra_set[f"inference.{svc}.livenessProbe.failureThreshold"] = "60"
        deploy_stack(E2E_CLUSTER_NAME, "cogniverse", extra_set=extra_set)
        if not _ensure_stack_running():
            pytest.fail("e2e stack did not become healthy after deploy")

        # GPU inference deployments (colpali embed, ASR, student) load
        # weights from a cold disk for ~12 min each, then profile — the
        # ingest path depends on them and deploy_stack's own pod wait is
        # best-effort at 300s. Without this, the first upload runs while the
        # embed service refuses connections, every segment's embedding
        # fails, and the job (correctly) terminates failed instead of
        # exercising the pipeline. Budget 40 min to cover the cold-load +
        # profile chain under GPU-memory contention.
        wait = subprocess.run(
            [
                "kubectl",
                "--context",
                f"k3d-{E2E_CLUSTER_NAME}",
                "wait",
                "--for=condition=available",
                "deployment",
                "--all",
                "-n",
                "cogniverse",
                "--timeout=2400s",
            ],
            capture_output=True,
            text=True,
            timeout=2460,
        )
        if wait.returncode != 0:
            pytest.fail(
                "e2e stack deployments not all available within 40m: "
                f"{(wait.stdout or '')[-600:]} {(wait.stderr or '')[-300:]}"
            )
        # Stamp the deployed content so the next session can reuse this
        # cluster iff the deployed code is still identical.
        _stamp_e2e_fingerprint(fingerprint)

    try:
        cron_restore = _suspend_cronworkflows_for_session()
        _bootstrap_tenant_and_schemas()
        _ingest_sample_video()
        _ensure_llm_model()
        _ensure_sandbox_gateway()
        try:
            yield
        finally:
            _restore_cronworkflows(cron_restore)
    finally:
        # Leave the cluster warm for reuse; only a forced-fresh run tears
        # it down (so CI / first-install verification stays hermetic).
        if force_fresh:
            delete_test_cluster(E2E_CLUSTER_NAME)


# Prefixes used by per-test tenants. Anything else (bootstrap, system,
# real customer tenants) MUST NOT match.
_TEST_TENANT_PREFIXES = (
    "graph_e2e_",
    "iso_",
    "mix_",
    "rev_",
    "sch_",
    "load_",
    "del_",
    "conc_",
    "both_",
    "apiorg_",
    "apinorm_",
    "search_e2e_",
    "ingest_e2e_",
    # Knowledge-system e2e prefixes (added with the Section A/B/C/D coverage).
    # Each phase claims one prefix; tests mint via unique_id("<prefix>") so the
    # session-end sweep at _cleanup_test_tenants reaps them automatically.
    "know_",  # KnowledgeRegistry / lifecycle / pinning
    "prov_",  # Provenance round-trip
    "confl_",  # Contradiction detection
    "trust_",  # Trust ranking
    "fed_",  # Federation + cross-tenant
    "rlm_",  # RLM telemetry / A-B / deep-synthesis
    "opt_",  # Optimizer canary / variants / rollback
    "sbx_",  # Sandbox policy + health probe
    "kagent_",  # Nine knowledge agents
    "cron_e2e_org_",  # CronWorkflow execution e2e (org+tenant pair, both sides matched)
    # Smoke-test / CLI bootstrap prefixes observed in operator runs of
    # ``cogniverse up`` / smoke commands. These create orgs with epoch
    # suffixes (e.g. ``smk_1778946797``) that previously survived every
    # e2e teardown and accumulated to 320+ rows.
    "boot_",
    "canonsmoke_",
    "canontest_",
    "smk_",
    "smk2_",
)


def _cleanup_test_tenants() -> None:
    """Delete every test-prefixed tenant AND parent org so the next run starts clean.

    Tests mint per-test tenants and orgs and don't tear them down;
    without this they accumulate. Symptoms observed:
      * 321 orgs after a few days of runs — slows ``list_organizations``
        and turns the daily-cleanup CronWorkflow into a 10-min crawl
        because it instantiates one ``Mem0MemoryManager`` per tenant.
      * Vespa orphan rollback trips on stale schemas left behind.

    Only entities matching ``_TEST_TENANT_PREFIXES`` are touched —
    real customer orgs / tenants must never be eligible.

    Waits for the runtime to be ready before sweeping. Tests that
    trigger a runtime rollout (e.g. the daily-gateway cron e2e) leave
    the runtime mid-restart at teardown time; without this wait the
    sweep would flood the log with ``Server disconnected without
    sending a response`` for every test tenant.
    """
    import time as _t

    deadline = _t.monotonic() + 180.0
    while _t.monotonic() < deadline and not runtime_available():
        _t.sleep(3.0)
    # 1. Tenant sweep — query Vespa for every schema_registry row and
    # delete via runtime so the registry tombstone + Vespa schema both
    # land atomically.
    vespa_url = os.environ.get("VESPA_URL", "http://localhost:33080")
    yql = (
        "select tenant_id from config_metadata "
        'where scope contains "schema" '
        'and service contains "schema_registry"'
    )
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{vespa_url}/search/", params={"yql": yql, "hits": 400})
            if resp.status_code != 200:
                return
            hits = resp.json().get("root", {}).get("children", []) or []
    except (httpx.HTTPError, OSError):
        return

    tenants_seen: set[str] = set()
    for hit in hits:
        tid = (hit.get("fields") or {}).get("tenant_id", "")
        if tid and any(tid.startswith(p) for p in _TEST_TENANT_PREFIXES):
            tenants_seen.add(tid)

    # Delete serially (one worker) under a total budget. Each delete undeploys
    # a Vespa schema, which redeploys the WHOLE application package; concurrent
    # undeploys race on that shared package rebuild — the loser sees a stale
    # survivor set and Vespa rejects it ("services.xml does not exist" /
    # "schema-removal ... loss of all data"). Serial avoids the race; the
    # budget keeps a large backlog from hanging setup/teardown, and what isn't
    # reaped this run is picked up by the next sweep.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _delete_one(tid: str) -> None:
        try:
            with httpx.Client(timeout=20.0) as client:
                client.delete(f"{RUNTIME}/admin/tenants/{tid}")
        except (httpx.HTTPError, OSError) as exc:
            print(f"Cleanup failed for tenant {tid}: {exc}")

    sweep_deadline = _t.monotonic() + 180.0
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        futures = [pool.submit(_delete_one, tid) for tid in sorted(tenants_seen)]
        for _fut in as_completed(futures):
            if _t.monotonic() > sweep_deadline:
                print("Tenant cleanup budget exhausted; remainder left for next run")
                break
    finally:
        # Don't block on the queued/in-flight deletes — cancel the rest so a
        # large backlog can't hang session setup/teardown past the budget.
        pool.shutdown(wait=False, cancel_futures=True)

    # 2. Org sweep — DELETE /admin/organizations/{org_id}. Tenants
    # have been removed above so org delete is unblocked. Skip orgs
    # whose id doesn't match a test prefix so flywheel_org / customer
    # orgs survive.
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{RUNTIME}/admin/organizations")
            if r.status_code != 200:
                return
            orgs = (r.json() or {}).get("organizations") or []
    except (httpx.HTTPError, OSError) as exc:
        print(f"Cleanup failed listing organizations: {exc}")
        return

    org_ids = sorted(
        o["org_id"]
        for o in orgs
        if o.get("org_id")
        and any(o["org_id"].startswith(p) for p in _TEST_TENANT_PREFIXES)
    )
    for org_id in org_ids:
        try:
            with httpx.Client(timeout=60.0) as client:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
        except (httpx.HTTPError, OSError) as exc:
            print(f"Cleanup failed for org {org_id}: {exc}")


def _reconcile_vespa_orphans() -> None:
    """Drop tenants whose schemas survive in Vespa with no registry record.

    Test-only — production runtimes must not auto-drop orphans because
    they may represent half-completed deploys of real customer data.

    Calls ``/admin/reconcile-orphans?dry_run=false`` so all orphan
    tenants land in a single redeploy. Iterating per-tenant DELETE
    fails in the multi-orphan case: each individual delete refuses on
    the others' presence, so atomic bulk-drop is required.
    """
    try:
        with httpx.Client(timeout=300.0) as client:
            dry = client.post(
                f"{RUNTIME}/admin/reconcile-orphans", params={"dry_run": "true"}
            )
    except (httpx.HTTPError, OSError) as exc:
        print(f"Session pre-flight: reconcile dry-run failed: {exc}")
        return

    if dry.status_code == 404:
        # Older runtime image without the endpoint — fall through silently
        # so a partially-deployed cluster doesn't block the rest of the
        # session-start fixture.
        print(
            "Session pre-flight: /admin/reconcile-orphans not available on "
            "this runtime; skipping orphan reconciliation."
        )
        return
    if dry.status_code != 200:
        print(
            f"Session pre-flight: reconcile dry-run returned "
            f"{dry.status_code}: {dry.text[:200]}"
        )
        return

    diff = dry.json()
    orphans = diff.get("orphan_schemas") or []
    if not orphans:
        return
    orphan_tenants = diff.get("orphan_tenants") or []
    unrecovered = diff.get("unrecovered_schemas") or []
    print(
        f"Session pre-flight: dropping {len(orphans)} Vespa orphan schema(s) "
        f"across {len(orphan_tenants)} tenant(s) in one atomic redeploy."
    )
    if unrecovered:
        print(
            f"  {len(unrecovered)} schema(s) with unknown base prefixes will "
            f"NOT be dropped: {unrecovered}"
        )

    try:
        with httpx.Client(timeout=300.0) as client:
            confirm = client.post(
                f"{RUNTIME}/admin/reconcile-orphans", params={"dry_run": "false"}
            )
    except (httpx.HTTPError, OSError) as exc:
        print(f"Session pre-flight: reconcile confirm failed: {exc}")
        return

    if confirm.status_code != 200:
        print(
            f"Session pre-flight: reconcile confirm returned "
            f"{confirm.status_code}: {confirm.text[:200]}"
        )


_CRON_NAMESPACE = "cogniverse"


def _suspend_cronworkflows_for_session() -> list[str]:
    """Suspend every Argo CronWorkflow in the cogniverse namespace.

    Returns the list of CronWorkflow names that were toggled from
    ``spec.suspend != true`` to ``true`` so the matching
    ``_restore_cronworkflows`` call only re-enables what this fixture
    actually changed. Workflows that were already suspended (by the
    user or a previous session) stay suspended on teardown.

    No-op when:
      * ``kubectl`` is not on PATH (CI runs without a real cluster)
      * the Argo CRD is not installed (``no resources found``)
    """
    import json as _json
    import shutil
    import subprocess

    if shutil.which("kubectl") is None:
        return []

    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "cronworkflows",
                "-n",
                _CRON_NAMESPACE,
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Session pre-flight: kubectl get cronworkflows failed: {exc}")
        return []

    if result.returncode != 0:
        if "could not find the requested resource" in (result.stderr or ""):
            return []
        if "the server doesn't have a resource type" in (result.stderr or ""):
            return []
        print(
            "Session pre-flight: kubectl get cronworkflows returned "
            f"rc={result.returncode}: {result.stderr.strip()[:200]}"
        )
        return []

    try:
        payload = _json.loads(result.stdout or "{}")
    except _json.JSONDecodeError as exc:
        print(f"Session pre-flight: cronworkflows JSON parse failed: {exc}")
        return []

    toggled: list[str] = []
    for item in payload.get("items") or []:
        name = (item.get("metadata") or {}).get("name")
        if not name:
            continue
        suspended = bool((item.get("spec") or {}).get("suspend"))
        if suspended:
            continue
        patch = '{"spec":{"suspend":true}}'
        patch_result = subprocess.run(
            [
                "kubectl",
                "patch",
                "cronworkflow",
                name,
                "-n",
                _CRON_NAMESPACE,
                "--type",
                "merge",
                "-p",
                patch,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if patch_result.returncode != 0:
            print(
                f"Session pre-flight: failed to suspend cronworkflow {name}: "
                f"{patch_result.stderr.strip()[:200]}"
            )
            continue
        toggled.append(name)

    if toggled:
        print(
            f"Session pre-flight: suspended {len(toggled)} cronworkflow(s) "
            f"for the duration of the e2e session: {sorted(toggled)}"
        )
    return toggled


def _restore_cronworkflows(names: list[str]) -> None:
    """Re-enable CronWorkflows previously suspended by the fixture."""
    import shutil
    import subprocess

    if not names or shutil.which("kubectl") is None:
        return

    patch = '{"spec":{"suspend":false}}'
    for name in names:
        result = subprocess.run(
            [
                "kubectl",
                "patch",
                "cronworkflow",
                name,
                "-n",
                _CRON_NAMESPACE,
                "--type",
                "merge",
                "-p",
                patch,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"Session teardown: failed to restore cronworkflow {name}: "
                f"{result.stderr.strip()[:200]}"
            )


def _ensure_sandbox_gateway() -> None:
    """Start the OpenShell sandbox gateway and sync mTLS certs into k3d."""
    try:
        from cogniverse_cli.sandbox import ensure_sandbox_ready
    except ImportError:
        print(
            "OpenShell sandbox setup unavailable (cogniverse_cli.sandbox not importable)"
        )
        return
    try:
        ensure_sandbox_ready()
    except Exception as exc:  # pragma: no cover — infra failure, not fatal
        print(f"OpenShell gateway bootstrap raised (non-fatal): {exc}")


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1920, "height": 1080}}


def wait_for_streamlit(page, timeout: int = 30_000):
    """Wait for Streamlit app to fully render."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)
    page.wait_for_load_state("networkidle")


def _strip_emoji(text: str) -> str:
    """Strip leading emoji + whitespace from tab text for clean comparison."""
    import re

    return re.sub(
        r"^[\U0001f300-\U0001faff\u2600-\u27bf\ufe0f\u200d]+\s*", "", text
    ).strip()


def _click_tab_by_label(page, label: str, retries: int = 6, settle_ms: int = 3_000):
    """Click a Streamlit tab by matching its visible text (ignoring emojis).

    Prefers exact matches over substring matches to avoid ambiguity
    (e.g., "Synthetic Data" should match the sub-tab, not
    "Synthetic Data & Optimization").

    Prefers visible tabs over hidden ones to handle duplicate sub-tab
    labels across different parent tabs.
    """
    for attempt in range(retries):
        tabs = page.locator('button[role="tab"]')
        count = tabs.count()

        # Collect tab info once per attempt
        tab_info = []
        for i in range(count):
            tab = tabs.nth(i)
            raw = tab.text_content() or ""
            clean = _strip_emoji(raw).lower()
            visible = tab.is_visible()
            tab_info.append((tab, raw, clean, visible))

        target = label.lower()

        # Pass 1: exact match on visible tabs
        for tab, raw, clean, visible in tab_info:
            if clean == target and visible:
                tab.scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                tab.click()
                page.wait_for_timeout(1_000)
                # Verify tab is now selected (aria-selected="true")
                if tab.get_attribute("aria-selected") != "true":
                    # Click didn't register — try JS dispatch
                    tab.dispatch_event("click")
                    page.wait_for_timeout(1_000)
                page.wait_for_timeout(settle_ms)
                page.wait_for_load_state("networkidle")
                return

        # Pass 2: substring match on visible tabs
        for tab, raw, clean, visible in tab_info:
            if target in clean and visible:
                tab.scroll_into_view_if_needed()
                tab.click()
                page.wait_for_timeout(settle_ms)
                page.wait_for_load_state("networkidle")
                return

        # Pass 3: exact match on hidden tabs (force-click)
        for tab, raw, clean, visible in tab_info:
            if clean == target:
                tab.click(force=True)
                page.wait_for_timeout(settle_ms)
                page.wait_for_load_state("networkidle")
                return

        # Pass 4: substring match on hidden tabs (force-click)
        for tab, raw, clean, visible in tab_info:
            if target in clean:
                tab.click(force=True)
                page.wait_for_timeout(settle_ms)
                page.wait_for_load_state("networkidle")
                return

        if attempt < retries - 1:
            page.wait_for_timeout(3_000)
    tab_texts = [tabs.nth(i).text_content() or "" for i in range(tabs.count())]
    raise ValueError(
        f"Tab '{label}' not found after {retries} attempts. Available tabs: {tab_texts}"
    )


def click_top_tab(page, label: str):
    """Click a top-level Streamlit tab."""
    start = _time.monotonic()
    _click_tab_by_label(page, label)
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op("click_top_tab", label, elapsed_ms=elapsed)


def click_sub_tab(page, label: str):
    """Click a sub-level Streamlit tab.

    Uses a longer settle time than top-level tabs because sub-tabs
    often trigger heavy Streamlit reruns (API calls, data loading).
    """
    start = _time.monotonic()
    _click_tab_by_label(page, label, settle_ms=4_000)
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op("click_sub_tab", label, elapsed_ms=elapsed)


def fill_input(locator, value: str):
    """Fill a Streamlit input, handling both visible and hidden elements.

    Uses keyboard approach (click + type) for visible elements to ensure
    Streamlit picks up the value. Falls back to JS for hidden elements.
    """
    start = _time.monotonic()
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )
        # Streamlit text_input requires Enter to commit the value
        locator.press("Enter")
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op("fill_input", "text_input", value, elapsed)


def fill_textarea(locator, value: str):
    """Fill a Streamlit textarea, handling both visible and hidden elements.

    Uses keyboard approach for visible elements. Streamlit textareas
    commit their value on Ctrl+Enter (Enter just adds a newline).
    Falls back to JS for hidden elements.
    """
    start = _time.monotonic()
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Control+Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op("fill_textarea", "textarea", value, elapsed)


def click_button(page, text: str):
    """Click a Streamlit button by text, excluding tab buttons.

    Uses JS click to bypass visibility checks. Excludes buttons with
    role="tab" to avoid accidentally clicking tabs instead of form buttons.
    """
    start = _time.monotonic()
    btn = page.locator(f'button:not([role="tab"]):has-text("{text}")')
    if btn.count() > 0:
        btn.first.evaluate("el => el.click()")
        page.wait_for_timeout(2_000)
        page.wait_for_load_state("networkidle")
        elapsed = (_time.monotonic() - start) * 1000
        if _report_collector:
            _report_collector.record_browser_op(
                "click_button", text, elapsed_ms=elapsed
            )
        return True
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op(
            "click_button (not found)", text, elapsed_ms=elapsed
        )
    return False


def expand_sidebar(page):
    """Expand the sidebar if it's collapsed (common in headless mode)."""
    # Streamlit collapses sidebar in narrow viewports / headless
    collapse_btn = page.locator(
        '[data-testid="stSidebarCollapsedControl"], '
        'button[aria-label="Open sidebar"], '
        '[data-testid="collapsedControl"]'
    )
    if collapse_btn.count() > 0 and collapse_btn.first.is_visible():
        collapse_btn.first.click()
        page.wait_for_timeout(1_000)


def set_tenant(page, tenant_id: str, retries: int = 3):
    """Set the active tenant in the sidebar with retry.

    Targets the 'Active Tenant' input specifically (not 'Tenant ID').
    Retries if the value doesn't stick (Streamlit session state timing).
    """
    start = _time.monotonic()
    expand_sidebar(page)

    sidebar = page.locator('[data-testid="stSidebar"]')
    tenant_input = sidebar.locator('input[aria-label="Active Tenant"]')

    for attempt in range(retries):
        tenant_input.click(click_count=3, force=True)
        page.keyboard.press("Delete")
        tenant_input.type(tenant_id, delay=30)
        tenant_input.press("Enter")
        page.wait_for_timeout(4_000)
        page.wait_for_load_state("networkidle")

        # Verify tenant was committed to Streamlit session state
        # by checking for the confirmation alert
        tenant_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Current tenant")'
        )
        if tenant_alert.count() > 0:
            elapsed = (_time.monotonic() - start) * 1000
            if _report_collector:
                _report_collector.record_browser_op(
                    "set_tenant", "sidebar", tenant_id, elapsed
                )
            return
    raise RuntimeError(
        f"set_tenant failed: tenant '{tenant_id}' was not committed to "
        f"Streamlit session state after {retries} attempts. "
        "Expected 'Current tenant' confirmation alert to appear."
    )


# Video-ChatGPT paper (arxiv) — directly related to the test dataset
ARXIV_PDF_URL = "https://arxiv.org/pdf/2306.05424"


def _download_if_missing(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a file if it doesn't already exist (cached across test runs)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


@pytest.fixture(scope="session")
def real_pdf_path():
    """Download Video-ChatGPT arxiv paper (related to test videos)."""
    dest = E2E_ARTIFACT_DIR / "video_chatgpt_2306.05424.pdf"
    try:
        return _download_if_missing(ARXIV_PDF_URL, dest)
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Cannot download test PDF: {exc}")


@pytest.fixture(scope="session")
def real_image_path():
    """Real 1280x720 Big Buck Bunny keyframe — cached, or extracted from
    big_buck_bunny_clip.mp4 via PyAV on fresh checkouts.
    """
    cached = (
        DATA_ROOT
        / "testset"
        / "evaluation"
        / "processed"
        / "keyframes"
        / "big_buck_bunny_clip"
        / "frame_0000.jpg"
    )
    if cached.exists():
        return cached

    source_video = (
        DATA_ROOT
        / "testset"
        / "evaluation"
        / "sample_videos"
        / "big_buck_bunny_clip.mp4"
    )
    if not source_video.exists():
        pytest.skip(f"Cannot generate keyframe — source video missing: {source_video}")

    dest = E2E_ARTIFACT_DIR / "big_buck_bunny_frame_0000.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import av

        with av.open(str(source_video)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frame.to_image().save(str(dest), format="JPEG", quality=92)
                break
    except Exception as exc:
        pytest.skip(f"Cannot extract first frame via PyAV: {exc}")
    if not dest.exists() or dest.stat().st_size == 0:
        pytest.skip(f"Frame extraction produced no output at {dest}")
    return dest


@pytest.fixture(scope="session")
def real_video_path():
    """Real 874KB ActivityNet sample video."""
    path = DATA_ROOT / "testset" / "evaluation" / "sample_videos" / "v_-nl4G-00PtA.mp4"
    if not path.exists():
        pytest.skip(f"Sample video not found: {path}")
    return path


@pytest.fixture(scope="session")
def real_document_path():
    """Real dataset summary markdown from the repo."""
    path = DATA_ROOT / "testset" / "dataset_summary.md"
    if not path.exists():
        pytest.skip(f"Document not found: {path}")
    return path


@pytest.fixture(scope="session")
def extracted_audio_path(real_video_path):
    """Extract 10s mono 16kHz PCM WAV from sample video via PyAV+wave.

    Uses PyAV instead of subprocess ffmpeg so the fixture works on
    machines without a system ffmpeg binary.
    """
    dest = E2E_ARTIFACT_DIR / "extracted_video_audio.wav"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import wave

        import av
        import numpy as np

        with av.open(str(real_video_path)) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                pytest.skip(
                    f"Sample video {real_video_path} has no audio stream — "
                    f"audio fixture cannot generate test wav"
                )
            stream = audio_streams[0]
            target_rate = 16000
            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
            chunks: list[np.ndarray] = []
            collected = 0
            need = target_rate * 10  # 10s of mono samples
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().reshape(-1)
                    chunks.append(arr)
                    collected += arr.size
                    if collected >= need:
                        break
                if collected >= need:
                    break

        if not chunks:
            pytest.skip(f"PyAV decoded no audio frames from {real_video_path}")

        samples = np.concatenate(chunks)[:need].astype(np.int16, copy=False)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(target_rate)
            wav.writeframes(samples.tobytes())
    except Exception as exc:
        pytest.skip(f"Cannot extract audio via PyAV: {exc}")
    return dest


E2E_REPORT_DIR = Path("/tmp")
E2E_REPORT_JSON = E2E_REPORT_DIR / "e2e_report.json"
E2E_REPORT_MD = E2E_REPORT_DIR / "e2e_report.md"


class E2EReportCollector:
    """Collects HTTP operations and test outcomes for E2E reporting.

    Automatically captures every httpx call to the runtime (localhost:33000)
    by monkeypatching httpx.Client.send. Groups operations by test name
    and writes JSON + markdown reports at session end.
    """

    def __init__(self):
        self.operations: list[dict] = []
        self.test_results: dict[str, dict] = {}
        self._current_test: str | None = None
        self._original_send = None
        self._session_start = datetime.now(timezone.utc)

    def start_test(self, nodeid: str):
        self._current_test = nodeid

    def end_test(self, nodeid: str, outcome: str, duration: float):
        self.test_results[nodeid] = {
            "outcome": outcome,
            "duration_s": round(duration, 3),
        }
        self._current_test = None

    def record_browser_op(
        self, action: str, target: str, value: str = "", elapsed_ms: float = 0
    ):
        """Record a Playwright browser interaction (tab click, input fill, button click)."""
        self.operations.append(
            {
                "test": self._current_test or "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "BROWSER",
                "url": action,
                "status_code": 200,
                "elapsed_ms": round(elapsed_ms, 1),
                "request": {"target": target, "value": value}
                if value
                else {"target": target},
                "response": {"status_code": 200, "status": "ok"},
            }
        )

    def record(
        self, request: httpx.Request, response: httpx.Response, elapsed_ms: float
    ):
        url = str(request.url)
        # Only capture calls to the runtime, not external downloads
        if "localhost:33000" not in url and "127.0.0.1:8000" not in url:
            return

        # Parse request body — guard against streaming requests that
        # haven't been read yet (multipart file uploads use streaming)
        req_body = self._parse_request_body(request)
        # Parse response body
        resp_body = self._safe_json(response)

        self.operations.append(
            {
                "test": self._current_test or "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "url": self._short_url(url),
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 1),
                "request": self._extract_request_fields(req_body, url),
                "response": self._extract_response_fields(
                    resp_body, url, response.status_code
                ),
            }
        )

    def install_hook(self):
        """Monkeypatch httpx.Client.send to record all HTTP calls."""
        collector = self
        original = httpx.Client.send
        self._original_send = original

        def recording_send(client_self, request, **kwargs):
            start = _time.monotonic()
            response = original(client_self, request, **kwargs)
            elapsed = (_time.monotonic() - start) * 1000
            collector.record(request, response, elapsed)
            return response

        httpx.Client.send = recording_send

    def uninstall_hook(self):
        if self._original_send is not None:
            httpx.Client.send = self._original_send

    def write_reports(self):
        report = self._build_report()
        # JSON report
        E2E_REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
        # Markdown report
        E2E_REPORT_MD.write_text(self._render_markdown(report))

    @staticmethod
    def _short_url(url: str) -> str:
        """Strip base URL, keep path + query."""
        return re.sub(r"https?://[^/]+", "", url)

    @staticmethod
    def _parse_request_body(request: httpx.Request) -> dict | None:
        try:
            content = request.content
        except httpx.RequestNotRead:
            # Streaming request (multipart file uploads) — content not yet buffered.
            # Extract what we can from headers.
            ct = request.headers.get("content-type", "")
            if "multipart" in ct:
                return {"_multipart": True}
            return None
        if not content:
            return None
        try:
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Multipart form data — extract field names from raw bytes
            ct = request.headers.get("content-type", "")
            if "multipart" in ct:
                fields = {"_multipart": True}
                # Extract form field values from multipart body
                for match in re.finditer(rb'name="(\w+)"\r\n\r\n([^\r]+)', content):
                    key = match.group(1).decode()
                    val = match.group(2).decode(errors="replace")
                    if len(val) < 200:
                        fields[key] = val
                # Extract filename from file field
                fn_match = re.search(rb'filename="([^"]+)"', content)
                if fn_match:
                    fields["filename"] = fn_match.group(1).decode()
                return fields
            return None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict | None:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, httpx.ResponseNotRead):
            return None

    @staticmethod
    def _extract_request_fields(body: dict | None, url: str) -> dict:
        """Extract semantically meaningful request fields based on endpoint."""
        if body is None:
            return {}
        if body.get("_multipart"):
            return {"type": "file_upload"}

        fields = {}
        # Agent process requests
        for key in (
            "query",
            "agent_name",
            "top_k",
            "profile",
            "strategy",
            "tenant_id",
            "video_dir",
            "max_videos",
            "batch_size",
            "org_id",
            "org_name",
            "tenant_name",
            "profile_name",
            "schema_name",
            "base_schemas",
        ):
            if key in body:
                fields[key] = body[key]
        # Nested context
        ctx = body.get("context", {})
        if isinstance(ctx, dict) and "tenant_id" in ctx:
            fields["tenant_id"] = ctx["tenant_id"]
        # A2A params
        params = body.get("params", {})
        if isinstance(params, dict):
            msg = params.get("message", {})
            if isinstance(msg, dict):
                parts = msg.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    fields["query"] = parts[0].get("text", "")
        # Form data fields
        if "data" in body:
            fields.update(body["data"])
        return fields

    @staticmethod
    def _extract_response_fields(body: dict | None, url: str, status_code: int) -> dict:
        """Extract semantically meaningful response fields based on endpoint."""
        if body is None:
            return {"status_code": status_code}

        # Some endpoints return a list instead of a dict (e.g., /events/queues)
        if isinstance(body, list):
            return {"status_code": status_code, "items_count": len(body)}

        fields = {"status_code": status_code}

        # Common fields across many endpoints
        for key in (
            "status",
            "agent",
            "recommended_agent",
            "confidence",
            "reasoning",
            "enhanced_query",
            "entity_count",
            "has_entities",
            "dominant_types",
            "results_count",
            "query",
            "profile",
            "strategy",
            "session_id",
            "job_id",
            "videos_processed",
            "videos_total",
            "filename",
            "video_id",
            "chunks_created",
            "documents_fed",
            "processing_time",
            "total_agents",
            "count",
            "org_id",
            "tenant_full_id",
            "tenants_deleted",
            "service",
            "protocolVersion",
        ):
            if key in body:
                fields[key] = body[key]

        # Nested structures — summarize counts rather than full data
        if "entities" in body and isinstance(body["entities"], list):
            fields["entities_count"] = len(body["entities"])
            if body["entities"]:
                fields["entity_types"] = list(
                    {
                        e.get("type", "unknown")
                        for e in body["entities"]
                        if isinstance(e, dict)
                    }
                )
        if "results" in body and isinstance(body["results"], list):
            fields["results_returned"] = len(body["results"])
        if "strategies" in body and isinstance(body["strategies"], list):
            fields["strategies_count"] = len(body["strategies"])
        if "profiles" in body and isinstance(body["profiles"], list):
            fields["profiles_count"] = len(body["profiles"])
        if "agents" in body and isinstance(body["agents"], (list, dict)):
            agents = body["agents"]
            fields["agents_count"] = (
                len(agents) if isinstance(agents, list) else len(agents)
            )
        if "backends" in body and isinstance(body["backends"], dict):
            fields["backends_count"] = len(body["backends"])
        if "organizations" in body and isinstance(body["organizations"], list):
            fields["organizations_count"] = len(body["organizations"])
        if "tenants" in body and isinstance(body["tenants"], list):
            fields["tenants_count"] = len(body["tenants"])
        if "relationships" in body and isinstance(body["relationships"], list):
            fields["relationships_count"] = len(body["relationships"])
        if "query_variants" in body and isinstance(body["query_variants"], list):
            fields["query_variants_count"] = len(body["query_variants"])
        if "generators" in body:
            fields["generators"] = body["generators"]
        if "optimizers" in body:
            fields["optimizers_count"] = (
                len(body["optimizers"])
                if isinstance(body["optimizers"], (list, dict))
                else 0
            )
        if "skills" in body and isinstance(body["skills"], list):
            fields["skills_count"] = len(body["skills"])
        # A2A result
        result = body.get("result")
        if isinstance(result, dict):
            fields["task_id"] = result.get("id")
            fields["context_id"] = result.get("contextId")
            status = result.get("status", {})
            if isinstance(status, dict):
                fields["task_state"] = status.get("state")

        # Error detail
        if "detail" in body:
            fields["error_detail"] = str(body["detail"])[:200]

        return fields

    def _build_report(self) -> dict:
        """Build the full report structure."""
        session_end = datetime.now(timezone.utc)
        elapsed = (session_end - self._session_start).total_seconds()

        # Group operations by test
        ops_by_test: dict[str, list[dict]] = {}
        for op in self.operations:
            test = op["test"]
            ops_by_test.setdefault(test, []).append(op)

        # Group tests by class
        tests_by_class: dict[str, list[str]] = {}
        for nodeid in {
            **self.test_results,
            **{op["test"]: None for op in self.operations},
        }:
            if nodeid == "unknown":
                continue
            parts = nodeid.split("::")
            cls = parts[1] if len(parts) >= 2 else "module"
            tests_by_class.setdefault(cls, []).append(nodeid)

        # Summary counts
        outcomes = [r["outcome"] for r in self.test_results.values()]
        passed = outcomes.count("passed")
        failed = outcomes.count("failed")
        skipped = outcomes.count("skipped")

        return {
            "session": {
                "start": self._session_start.isoformat(),
                "end": session_end.isoformat(),
                "duration_s": round(elapsed, 1),
                "runtime_url": RUNTIME,
            },
            "summary": {
                "total_tests": len(self.test_results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total_http_operations": len(self.operations),
                "total_http_time_ms": round(
                    sum(op["elapsed_ms"] for op in self.operations), 1
                ),
            },
            "tests_by_class": {
                cls: [
                    {
                        "nodeid": nid,
                        **self.test_results.get(
                            nid, {"outcome": "unknown", "duration_s": 0}
                        ),
                        "operations": ops_by_test.get(nid, []),
                    }
                    for nid in sorted(set(tests))
                ]
                for cls, tests in sorted(tests_by_class.items())
            },
        }

    def _render_markdown(self, report: dict) -> str:
        """Render the report as markdown with summary + per-test details."""
        lines = []
        s = report["summary"]
        sess = report["session"]

        lines.append("# E2E Test Report")
        lines.append("")
        lines.append(f"**Date**: {sess['start'][:19]}Z")
        lines.append(f"**Duration**: {sess['duration_s']}s")
        lines.append(f"**Runtime**: {sess['runtime_url']}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Tests | {s['total_tests']} |")
        lines.append(f"| Passed | {s['passed']} |")
        lines.append(f"| Failed | {s['failed']} |")
        lines.append(f"| Skipped | {s['skipped']} |")
        lines.append(f"| HTTP Operations | {s['total_http_operations']} |")
        lines.append(f"| Total HTTP Time | {s['total_http_time_ms']:.0f}ms |")
        lines.append("")

        # Per-class sections
        for cls, tests in report["tests_by_class"].items():
            lines.append(f"## {cls}")
            lines.append("")

            for test in tests:
                outcome = test["outcome"]
                icon = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(
                    outcome, "?"
                )
                method = (
                    test["nodeid"].split("::")[-1]
                    if "::" in test["nodeid"]
                    else test["nodeid"]
                )
                lines.append(f"### [{icon}] {method} ({test['duration_s']}s)")
                lines.append("")

                ops = test.get("operations", [])
                if not ops:
                    lines.append("_No HTTP operations recorded._")
                    lines.append("")
                    continue

                # Operations table
                lines.append("| Method | Endpoint | Status | Time | Key Results |")
                lines.append("|--------|----------|--------|------|-------------|")

                for op in ops:
                    resp = op["response"]
                    req = op["request"]
                    if op["method"] == "BROWSER":
                        target = req.get("target", "")
                        value = req.get("value", "")
                        detail = f"{target}"
                        if value:
                            detail += (
                                f'="{value[:30]}{"..." if len(value) > 30 else ""}"'
                            )
                        lines.append(
                            f"| UI | `{op['url']}` | - "
                            f"| {op['elapsed_ms']:.0f}ms | {detail} |"
                        )
                    else:
                        key_results = self._format_key_results(req, resp, op["url"])
                        lines.append(
                            f"| {op['method']} | `{op['url'][:60]}` | {op['status_code']} "
                            f"| {op['elapsed_ms']:.0f}ms | {key_results} |"
                        )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_key_results(req: dict, resp: dict, url: str) -> str:
        """Format the most important results for a single operation."""
        parts = []

        # Agent operations
        if resp.get("recommended_agent"):
            parts.append(f"agent={resp['recommended_agent']}")
        if resp.get("confidence") is not None:
            parts.append(f"conf={resp['confidence']:.2f}")
        if resp.get("entities_count"):
            parts.append(f"entities={resp['entities_count']}")
        if resp.get("enhanced_query"):
            eq = resp["enhanced_query"]
            parts.append(f'enhanced="{eq[:40]}{"..." if len(eq) > 40 else ""}"')

        # Search operations
        if resp.get("results_count") is not None:
            parts.append(f"results={resp['results_count']}")
        if resp.get("strategies_count"):
            parts.append(f"strategies={resp['strategies_count']}")
        if resp.get("profiles_count"):
            parts.append(f"profiles={resp['profiles_count']}")

        # Ingestion operations
        if resp.get("chunks_created") is not None:
            parts.append(f"chunks={resp['chunks_created']}")
        if resp.get("documents_fed") is not None and resp["documents_fed"] > 0:
            parts.append(f"docs_fed={resp['documents_fed']}")
        if resp.get("processing_time") is not None:
            parts.append(f"proc={resp['processing_time']:.1f}s")
        if resp.get("filename"):
            parts.append(f"file={resp['filename']}")

        # Job tracking
        if resp.get("job_id"):
            parts.append(f"job={resp['job_id'][:8]}")

        # Tenant/org
        if resp.get("org_id") and "organizations" not in url:
            parts.append(f"org={resp['org_id']}")
        if resp.get("tenant_full_id"):
            parts.append(f"tenant={resp['tenant_full_id']}")
        if resp.get("organizations_count") is not None:
            parts.append(f"orgs={resp['organizations_count']}")
        if resp.get("tenants_count") is not None:
            parts.append(f"tenants={resp['tenants_count']}")

        # Health/registry
        if resp.get("service"):
            parts.append(f"svc={resp['service']}")
        if resp.get("total_agents") is not None:
            parts.append(f"agents={resp['total_agents']}")
        if resp.get("agents_count") is not None and resp.get("total_agents") is None:
            parts.append(f"agents={resp['agents_count']}")
        if resp.get("backends_count") is not None:
            parts.append(f"backends={resp['backends_count']}")

        # A2A
        if resp.get("task_state"):
            parts.append(f"state={resp['task_state']}")
        if resp.get("skills_count"):
            parts.append(f"skills={resp['skills_count']}")

        # Synthetic
        if resp.get("optimizers_count"):
            parts.append(f"optimizers={resp['optimizers_count']}")

        # Errors
        if resp.get("error_detail"):
            parts.append(f'err="{resp["error_detail"][:50]}"')

        # Fallback: status field
        if not parts and resp.get("status"):
            parts.append(f"status={resp['status']}")

        return (
            ", ".join(parts[:5]) if parts else f"status={resp.get('status_code', '?')}"
        )


# Singleton collector — created once per session
_report_collector: E2EReportCollector | None = None


def _ensure_playwright_browsers() -> None:
    """Install Chromium for Playwright tests on first use.

    pytest-playwright declares the Python package dep but the Chromium
    binary is a separate download (~150MB) that lives in a user cache
    dir. Doing it here means ``uv sync --dev`` + ``pytest`` is all a
    fresh dev machine needs — no out-of-band ``playwright install``
    step to remember. Idempotent: if the binary is already on disk
    the launch succeeds immediately.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return  # pytest-playwright not installed; nothing to do
    try:
        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        return
    except Exception:
        pass
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            capture_output=True,
            timeout=600,
            check=True,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        # Let the individual dashboard tests surface the error; don't
        # abort the whole suite just because one optional install failed.
        pass


def pytest_configure(config):
    """Install the HTTP recording hook and Playwright browsers at session start."""
    global _report_collector
    _report_collector = E2EReportCollector()
    _report_collector.install_hook()
    _ensure_playwright_browsers()


def pytest_unconfigure(config):
    """Write reports and uninstall hook at session end."""
    global _report_collector
    if _report_collector is not None:
        _report_collector.uninstall_hook()
        if _report_collector.operations or _report_collector.test_results:
            _report_collector.write_reports()
            print(f"\n{'=' * 60}")
            print(f"E2E REPORT: {E2E_REPORT_JSON}")
            print(f"E2E REPORT: {E2E_REPORT_MD}")
            s = _report_collector._build_report()["summary"]
            print(
                f"  {s['total_tests']} tests | "
                f"{s['passed']} passed | {s['failed']} failed | "
                f"{s['skipped']} skipped | "
                f"{s['total_http_operations']} HTTP ops | "
                f"{s['total_http_time_ms']:.0f}ms total"
            )
            print(f"{'=' * 60}")
        _report_collector = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test outcomes for the report."""
    outcome = yield
    rep = outcome.get_result()

    if _report_collector is None:
        return

    if rep.when == "setup":
        _report_collector.start_test(item.nodeid)
        if rep.skipped:
            _report_collector.end_test(item.nodeid, "skipped", 0.0)
    elif rep.when == "call":
        _report_collector.end_test(
            item.nodeid,
            rep.outcome,
            rep.duration,
        )
