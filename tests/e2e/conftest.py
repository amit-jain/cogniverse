"""Shared fixtures and helpers for E2E tests.

Provides stack checks, artifact generation, and Streamlit interaction helpers
for both API (httpx) and dashboard (Playwright) E2E tests.

Test artifact paths (real data used for ingestion tests):
- Video: tests/system/resources/videos/v_-D1gdv_gQyw.mp4
- Ingested corpus: tests/system/resources/videos/v_-6dz6tBH77I.mp4 (sha-pinned
  session fixture, plus its first frame ingested as image content)
- Image: first frame extracted from that tracked video
- Audio: ten seconds extracted from that tracked video's real audio stream
- PDF: repository evaluation-dataset content written as a deterministic PDF
- Document: data/testset/dataset_summary.md (real markdown about the evaluation set)
"""

import dataclasses
import functools
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import time as _time
import uuid
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Iterator

import httpx
import pytest
from cogniverse_cli.argo import (
    ARGO_NAMESPACE,
    ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
)
from cogniverse_cli.images import DEPLOY_INPUT_PATHS
from cogniverse_cli.secrets import read_secret

from cogniverse_agents.gateway_agent import SIMPLE_ROUTE_MAP, GatewayAgent
from cogniverse_core.memory.provenance import DerivationKind
from cogniverse_foundation.config.bootstrap import INFERENCE_API_KEY_ENV
from cogniverse_foundation.config.inference_auth import (
    endpoint_root,
    is_modal_inference_url,
)
from tests.e2e import backend_env, cron_guard
from tests.e2e.tab_selection import tab_candidates_in_scope

# Deployment-lifecycle tests bring up their own port-forward-based cluster
# and are exercised via a dedicated ``pytest tests/e2e/deployment/`` run —
# never as part of the main suite, which boots its own NodePort stack via
# ``e2e_stack`` (two cluster creates in one session would double a
# 20-minute boot and the subsuite's own stack-conflict guard would fire).
collect_ignore_glob: list[str] = ["deployment/*"]

# tests/conftest.py::backend_config_env defaults BACKEND_PORT to a dead sentinel
# so a unit test resolving config without a real store fails loudly. E2E runs
# against a live cluster, so publish its Vespa endpoint before that session
# fixture reads the default; otherwise every in-process config read here fails
# with ConnectionRefused on the sentinel port.
backend_env.export_backend_env()


def _modal_inference_deselections(config, items):
    explicit = os.environ.get("RUN_MODAL_INFERENCE_E2E") == "1" or (
        "requires_modal_inference" in (config.option.markexpr or "")
    )
    if explicit:
        return []
    return [
        item
        for item in items
        if any(item.iter_markers(name="requires_modal_inference"))
    ]


def _telegram_real_flow_deselections(items):
    # read_secret is the shared lookup: env var, then ./.env, then ~/.env, each
    # of which may be a directory of per-key <VAR>.env files. Reading os.environ
    # here instead would ignore a secret provisioned the documented way.
    token = read_secret("TELEGRAM_BOT_TOKEN")
    chat_id = read_secret("TELEGRAM_TEST_CHAT_ID")
    if token and chat_id:
        return [], None
    deselected = [
        item for item in items if any(item.iter_markers(name="requires_telegram_bot"))
    ]
    if not deselected:
        return [], None
    missing = []
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not chat_id:
        missing.append("TELEGRAM_TEST_CHAT_ID")
    return deselected, "missing " + " and ".join(missing)


def argo_workflow_controller_probe_command(
    namespace: str = ARGO_NAMESPACE,
) -> list[str]:
    return [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "-n",
        namespace,
        "get",
        "pods",
        "-l",
        ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR,
        "--field-selector=status.phase=Running",
        "-o",
        "name",
    ]


def argo_workflow_controller_probe_failure_message(
    *,
    command: list[str],
    namespace: str = ARGO_NAMESPACE,
) -> str:
    return (
        "Argo workflow controller unavailable after E2E stack setup; "
        f"command={' '.join(command)!r}; context={KUBECTL_CONTEXT!r}; "
        f"namespace={namespace!r}; "
        f"selector={ARGO_WORKFLOW_CONTROLLER_LABEL_SELECTOR!r}"
    )


def _require_modal_inference_endpoints(items, endpoints) -> None:
    for item in items:
        for marker in item.iter_markers(name="requires_modal_inference"):
            if len(marker.args) != 1 or not isinstance(marker.args[0], str):
                raise pytest.UsageError(
                    "requires_modal_inference must name exactly one inference service"
                )
            service = marker.args[0]
            endpoint = endpoints.get(service)
            provider = endpoint.provider if endpoint is not None else None
            if provider != "modal":
                pytest.fail(
                    f"{item.nodeid} requires Modal provider for {service!r}; "
                    f"resolved {provider!r}",
                    pytrace=False,
                )


def expected_gateway_routing(query: str, gw: dict) -> tuple[str, str]:
    """Return the expected gateway complexity and route for a response."""
    confidence = gw["confidence"]
    threshold = gw["fast_path_confidence_threshold"]
    if confidence < threshold:
        complexity = "complex"
    elif gw["modality"] == "both":
        complexity = "complex"
    elif gw["generation_type"] == "detailed_report":
        complexity = "complex"
    else:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        if query_words & GatewayAgent._COMPLEXITY_KEYWORDS:
            complexity = "complex"
        elif any(marker in query_lower for marker in GatewayAgent._MULTI_STEP_MARKERS):
            complexity = "complex"
        elif query_lower.count(",") >= 3 or query_lower.count(" and ") >= 2:
            complexity = "complex"
        else:
            complexity = "simple"

    if complexity == "complex":
        routed_to = "orchestrator_agent"
    else:
        route_key = (gw["modality"], gw["generation_type"])
        routed_to = SIMPLE_ROUTE_MAP.get(route_key, "orchestrator_agent")
    return complexity, routed_to


def assert_orchestrated(data: dict, query: str, gw: dict) -> None:
    """The complex-route payload the dispatcher builds for an orchestrated query."""
    assert data["status"] == "success", data
    assert data["agent"] == "orchestrator_agent", data
    assert data["message"] == f"Orchestrated '{query[:50]}' via A2A pipeline", data
    assert data["gateway_context"] == {
        "modality": gw["modality"],
        "generation_type": gw["generation_type"],
        "confidence": gw["confidence"],
    }, data
    orchestration = data["orchestration_result"]
    assert orchestration["query"] == query, orchestration
    assert set(orchestration) == {
        "query",
        "workflow_id",
        "plan_steps",
        "parallel_groups",
        "plan_reasoning",
        "agent_results",
        "final_output",
        "execution_summary",
        "metadata",
    }, sorted(orchestration)


# Video queries GLiNER tags video_content at >= 0.44 across the calibrator's
# whole 0.15-0.5 threshold range and nothing else, so on any tenant whose
# fast-path threshold is at or below that score each routes simple to
# search_agent with the GLiNER score as its confidence.
GATEWAY_VIDEO_QUERIES = (
    "search for animal videos",
    "search for video content about AI",
    "find videos about machine learning",
)


def expected_gateway_calibration(decisions: list[tuple[str, float]]) -> dict:
    """The gateway-thresholds calibrator's result for error-free decisions.

    ``decisions`` are ``(complexity, confidence)`` pairs, one per gateway span
    the tenant produced. Mirrors optimization_cli._compute_gateway_thresholds
    for spans that carried no ERROR status: the fast-path threshold drops
    to 0.35 only when the complex error rate is below 0.05 AND the mean
    confidence exceeds 0.8, otherwise it stays at the 0.4 default; the GLiNER
    threshold is max(0.15, min(p25 * 0.8, 0.5)).
    """
    import statistics

    confidences = [confidence for _, confidence in decisions]
    ordered = sorted(confidences)
    n = len(ordered)
    pos = 0.25 * (n - 1)  # pandas' default linear quantile
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    p25 = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    mean = statistics.fmean(confidences)
    simple_count = sum(1 for complexity, _ in decisions if complexity == "simple")
    complex_count = sum(1 for complexity, _ in decisions if complexity == "complex")
    fast_path = 0.35 if mean > 0.8 else 0.4
    return {
        "fast_path_confidence_threshold": fast_path,
        "gliner_threshold": round(max(0.15, min(p25 * 0.8, 0.5)), 3),
        "analysis": {
            "total_spans": n,
            "simple_count": simple_count,
            "complex_count": complex_count,
            "simple_error_rate": 0.0,
            "complex_error_rate": 0.0,
            "mean_confidence": round(mean, 4),
            "p25_confidence": round(p25, 4),
        },
    }


def pytest_collection_modifyitems(config, items):
    """Run async tests before playwright and select paid model tests explicitly.

    pytest-playwright leaves a registered asyncio event loop after teardown,
    which trips later pytest-asyncio tests with "cannot be called from a
    running event loop". Ordering pure-async first sidesteps the collision.

    """
    skip_substrings: list[str] = []

    # Teacher-model optimization e2e requires scaling up cogniverse-vllm-llm-teacher
    # which is a 1-2 hour run end-to-end. Off by default; opt in via
    # RUN_TEACHER_OPTIMIZATION_E2E=1 (or invoke pytest with
    # ``-m requires_teacher_model`` to bypass this deselection).
    teacher_optim_explicit = os.environ.get(
        "RUN_TEACHER_OPTIMIZATION_E2E"
    ) == "1" or "requires_teacher_model" in (config.option.markexpr or "")
    if not teacher_optim_explicit:
        skip_substrings.append("test_router_optimization_e2e")

    modal_deselections = _modal_inference_deselections(config, items)
    telegram_deselections, telegram_reason = _telegram_real_flow_deselections(items)
    if skip_substrings or modal_deselections:
        keep = []
        deselected = []
        for item in items:
            if item in modal_deselections or any(
                substring in item.nodeid for substring in skip_substrings
            ):
                deselected.append(item)
            else:
                keep.append(item)
        if deselected:
            # Deselection is invisible beyond the summary count — name what
            # was dropped and why so a green run can't silently omit these.
            reporter = config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(
                    "e2e conftest deselected (explicit requirement unavailable): "
                    + ", ".join(sorted({item.nodeid for item in deselected})),
                    yellow=True,
                )
            config.hook.pytest_deselected(items=deselected)
            items[:] = keep

    if telegram_deselections:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"e2e conftest deselected ({telegram_reason}): "
                + ", ".join(sorted({item.nodeid for item in telegram_deselections})),
                yellow=True,
            )
        config.hook.pytest_deselected(items=telegram_deselections)
        items[:] = [item for item in items if item not in telegram_deselections]

    def _priority(item):
        if _requests_browser(item):
            return 2
        path = str(item.fspath)
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
GLINER_URL = "http://localhost:33907"  # gliner NodePort 29007 via E2E_HOST_PORTS
TENANT_ID = "flywheel_org:production"
IN_POD_TELEMETRY_PRELUDE = (
    "from cogniverse_runtime.entrypoint_env import resolve_library_env_defaults; "
    "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
    "get_telemetry_manager(otlp_endpoint=resolve_library_env_defaults()['telemetry_otlp_endpoint']); "
)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
SAMPLE_VIDEO_PATH = (
    Path(__file__).parent.parent
    / "system"
    / "resources"
    / "videos"
    / "v_-6dz6tBH77I.mp4"
)
SAMPLE_VIDEO_CONTENT_ID = (
    "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"
)
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
    such blip made the session health gate fail even though the stack was ready.
    """
    import time as _t

    for attempt in range(5):
        if runtime_available() and dashboard_available():
            return True
        if attempt < 4:
            _t.sleep(3.0)
    return False


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


def _clear_thread_event_loop() -> None:
    """Detach any loop registered as current on this thread.

    The code that created a loop owns closing it. This reset only prevents a
    stale policy registration from becoming the implicit loop of a later test.
    """
    import asyncio

    asyncio.set_event_loop(None)


def _clear_stale_running_loop():
    """Detach a *running-loop* thread-local that no task is executing on.

    ``set_event_loop(None)`` clears the policy slot read by
    ``get_event_loop()``. ``Runner.run()`` reads a different thread-local
    via ``events._get_running_loop()``, so a loop left in that slot makes
    every later pytest-asyncio test raise ``RuntimeError: Runner.run()
    cannot be called from a running event loop`` before its body runs.

    Two things leave that slot set with no task executing: a runner that
    never unwound (``run_forever``'s ``finally`` resets ``_thread_id`` and
    this slot together, so a leak keeps both), and Playwright's sync API,
    whose ``_sync`` deliberately re-sets the slot to its own loop after
    every call. Both are detached here; the caller decides whether to
    re-attach for a test that needs the sync-API sentinel. A loop with a
    task executing on it right now is genuinely running and left alone.
    Returns the detached loop, or ``None``.
    """
    import asyncio
    import warnings

    leaked = asyncio.events._get_running_loop()
    if leaked is None:
        return None
    if not leaked.is_closed() and asyncio.current_task(loop=leaked) is not None:
        return None
    warnings.warn(
        f"Detached a leaked running event loop {leaked!r} left by "
        f"{_LAST_FINISHED_TEST or 'session setup'}; that test must unwind "
        "the loop it started",
        RuntimeWarning,
        stacklevel=2,
    )
    asyncio.events._set_running_loop(None)
    return leaked


_LAST_FINISHED_TEST: str | None = None
_PARKED_RUNNING_LOOP = None
_BROWSER_FIXTURES = frozenset({"browser", "context", "page", "new_context"})


def _requests_browser(request) -> bool:
    return bool(_BROWSER_FIXTURES.intersection(getattr(request, "fixturenames", ())))


@pytest.fixture(autouse=True)
def _reset_event_loop_state_before_each_test(request):
    """Clear thread-attached event-loop state before every test.

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

    global _LAST_FINISHED_TEST, _PARKED_RUNNING_LOOP
    _clear_thread_event_loop()
    if _requests_browser(request):
        # Playwright's sync API needs its own loop back in the running-loop
        # slot; a pytest-asyncio test in between parked it here.
        parked = _PARKED_RUNNING_LOOP
        if (
            asyncio.events._get_running_loop() is None
            and parked is not None
            and not parked.is_closed()
        ):
            asyncio.events._set_running_loop(parked)
    else:
        # Detach a leaked loop from this non-browser test, but keep the
        # cached Playwright loop intact for later browser teardown.
        _clear_stale_running_loop()
    yield
    _LAST_FINISHED_TEST = request.node.nodeid
    if _requests_browser(request):
        # Playwright's sync API leaves its own loop in the running-loop slot
        # after every call. Cache that loop, and only that loop: it is the one
        # ``browser.close()`` needs at session teardown. Caching a loop leaked
        # by a non-browser test instead is what left teardown holding a closed
        # loop and raising ``no running event loop``.
        current = asyncio.events._get_running_loop()
        if current is not None and not current.is_closed():
            _PARKED_RUNNING_LOOP = current
    # Session-scoped teardown (Playwright's ``browser.close()``) runs after
    # the last test; give it back the cached Playwright loop.
    parked = _PARKED_RUNNING_LOOP
    if (
        asyncio.events._get_running_loop() is None
        and parked is not None
        and not parked.is_closed()
    ):
        asyncio.events._set_running_loop(parked)


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

    # Send the create; the runtime rolls back on failure, so a transient 502
    # can be retried safely here without leaving a torn tenant behind. The
    # readiness signal is still the poll below, not the response code.
    deadline = _time.monotonic() + timeout_s
    last_failure = ""
    with httpx.Client(timeout=300.0) as client:
        while True:
            try:
                resp = client.post(
                    f"{RUNTIME}/admin/tenants",
                    json={"tenant_id": tenant_id, "created_by": created_by},
                )
            except (httpx.HTTPError, OSError) as exc:
                last_failure = f"raised {exc!r}"
                if _time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"register_tenant_and_wait: POST /admin/tenants for "
                        f"{tenant_id!r} {last_failure}"
                    ) from exc
                print(
                    f"register_tenant_and_wait: POST raised {exc!r} — "
                    f"retrying tenant creation"
                )
                _time.sleep(2.0)
                continue

            if resp.status_code in (200, 201, 409):
                break

            last_failure = f"returned {resp.status_code} {resp.text}"
            if resp.status_code in (502, 503, 504) and _time.monotonic() < deadline:
                print(
                    f"register_tenant_and_wait: POST /admin/tenants for "
                    f"{tenant_id!r} {last_failure} — retrying"
                )
                _time.sleep(2.0)
                continue
            raise RuntimeError(
                f"register_tenant_and_wait: POST /admin/tenants for "
                f"{tenant_id!r} {last_failure}"
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


RUN_ASYNC_TIMEOUT_S = 600.0


def run_async(coro, timeout_s: float | None = None):
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

    The join is bounded by ``timeout_s`` (default ``RUN_ASYNC_TIMEOUT_S``):
    a coroutine that outlives it raises ``TimeoutError`` naming the
    coroutine, and the daemon worker is abandoned — the pytest main
    thread must never block forever in ``Thread.join``.
    """
    import asyncio

    if timeout_s is None:
        timeout_s = RUN_ASYNC_TIMEOUT_S
    box: dict = {}

    def _runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — propagate verbatim
            box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        name = getattr(coro, "__qualname__", repr(coro))
        raise TimeoutError(
            f"run_async: coroutine {name!r} did not complete within "
            f"{timeout_s:g}s; abandoning its daemon worker thread"
        )
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
    span_name: str,
    start_time: datetime,
    attribute_predicate=None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 2.0,
):
    """Poll Phoenix for an exact span name within a bounded time window.

    Returns the first matching pandas Series row or None when the exact span
    never appears before the deadline. If Phoenix read attempts fail during
    polling, raises with the last read error so outage and not-found stay
    distinguishable.

    `attribute_predicate`, if given, is `(attrs_dict) -> bool` evaluated on
    each matching span's attributes column. The first row that satisfies the
    predicate is returned.
    """
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    from phoenix.client.types.spans import SpanQuery

    def _esc(value: object) -> str:
        # Backslash first, then quote — quoting first would let a trailing
        # backslash re-escape the closing quote.
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    predicate = f"name == '{_esc(span_name)}'"
    query = SpanQuery().where(predicate)
    deadline = _time.monotonic() + timeout_s
    query_timeout_s = max(1, ceil(timeout_s))
    last_error = None

    while _time.monotonic() < deadline:
        try:
            spans_df = phoenix_client.spans.get_spans_dataframe(
                project_identifier=project,
                query=query,
                start_time=start_time,
                timeout=query_timeout_s,
            )
            last_error = None
            if spans_df is not None and not spans_df.empty:
                if attribute_predicate is None:
                    return spans_df.iloc[0]
                for _, row in spans_df.iterrows():
                    attrs = row.get("attributes")
                    if attrs is None:
                        continue
                    # Phoenix returns attributes as a dict-like; coerce.
                    attrs_dict = dict(attrs) if not isinstance(attrs, dict) else attrs
                    if attribute_predicate(attrs_dict):
                        return row
        except Exception as exc:
            # Phoenix can transiently 5xx during heavy ingest; keep polling
            # until the deadline. If every read fails, raise with context.
            last_error = exc
        _time.sleep(poll_interval_s)

    if last_error is not None:
        raise RuntimeError(
            "wait_for_span read failed while polling Phoenix: "
            f"project={project!r} span_name={span_name!r} "
            f"start_time={start_time.isoformat()} timeout_s={timeout_s:g} "
            f"poll_interval_s={poll_interval_s:g} last_error={last_error!r}"
        ) from last_error
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
                "--context",
                KUBECTL_CONTEXT,
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


def _configured_profile_name(
    config: dict,
    *,
    profile_type: str | None = None,
    schema_name: str | None = None,
) -> str:
    profiles = config.get("backend", {}).get("profiles", {})
    for profile_name, profile_config in profiles.items():
        if not isinstance(profile_config, dict):
            continue
        if profile_type is not None and profile_config.get("type") != profile_type:
            continue
        if schema_name is not None and profile_config.get("schema_name") != schema_name:
            continue
        return profile_name
    criteria = []
    if profile_type is not None:
        criteria.append(f"type={profile_type!r}")
    if schema_name is not None:
        criteria.append(f"schema_name={schema_name!r}")
    raise AssertionError(f"configured profile not found ({', '.join(criteria)})")


def _active_video_profile_name(config: dict) -> str:
    active_profile = config.get("active_video_profile")
    if isinstance(active_profile, dict):
        name = active_profile.get("name")
        if isinstance(name, str) and name.strip():
            return name
    if isinstance(active_profile, str) and active_profile.strip():
        return active_profile
    return _configured_profile_name(config, profile_type="video")


def _configured_image_profile_name(config: dict) -> str:
    return _configured_profile_name(config, profile_type="image")


def _configured_audio_profile_name(config: dict) -> str:
    return _configured_profile_name(config, profile_type="audio")


def _configured_document_profile_name(
    config: dict, *, schema_name: str = "document_text"
) -> str:
    return _configured_profile_name(
        config, profile_type="document", schema_name=schema_name
    )


def _synthetic_fixture_profiles(config: dict) -> list[str]:
    profiles = [
        _active_video_profile_name(config),
        _configured_image_profile_name(config),
    ]
    configured_profiles = config.get("backend", {}).get("profiles", {})
    missing = [name for name in profiles if name not in configured_profiles]
    if missing:
        raise AssertionError(
            f"Synthetic E2E fixture profiles are not configured: {missing}"
        )
    modalities = [configured_profiles[name].get("type") for name in profiles]
    if modalities != ["video", "image"]:
        raise AssertionError(
            f"Synthetic E2E fixtures require video and image profiles, got {modalities}"
        )
    return profiles


def _profile_selection_video_profiles(config: dict) -> list[str]:
    profiles = [
        _active_video_profile_name(config),
        _configured_profile_name(
            config,
            profile_type="video",
            schema_name="video_colqwen_omni_mv_chunk_30s",
        ),
    ]
    configured_profiles = config.get("backend", {}).get("profiles", {})
    missing = [name for name in profiles if name not in configured_profiles]
    if missing:
        raise AssertionError(
            f"Profile-selection video profiles are not configured: {missing}"
        )
    modalities = [configured_profiles[name].get("type") for name in profiles]
    if modalities != ["video", "video"]:
        raise AssertionError(
            f"Profile-selection fixtures require two video profiles, got {modalities}"
        )
    return profiles


def _expected_chunk_count(
    duration_s: float, chunk_duration: float, chunk_overlap: float
) -> int:
    """Mirror ChunkProcessor.extract_chunks: one chunk per loop iteration from 0
    while start < duration, stepping by chunk_duration - chunk_overlap."""
    if duration_s <= 0:
        raise AssertionError(f"video duration must be positive, got {duration_s!r}")
    step = chunk_duration - chunk_overlap
    if step <= 0:
        raise AssertionError(
            f"chunk_duration {chunk_duration!r} must exceed chunk_overlap {chunk_overlap!r}"
        )
    count = 0
    start = 0.0
    while start < duration_s:
        count += 1
        start += step
    return count


def _video_duration_seconds(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        raise AssertionError(
            f"Could not determine duration for tracked video {path!r}: "
            f"fps={video_fps!r}, frames={total_frames!r}"
        )
    return total_frames / video_fps


def _expected_sample_documents_fed(path: Path, profile: str, media_type: str) -> int:
    if not media_type.startswith("video/"):
        return 1

    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    profile_def = config.get("backend", {}).get("profiles", {}).get(profile, {})
    segmentation = profile_def.get("strategies", {}).get("segmentation", {})
    if segmentation.get("class") == "ChunkSegmentationStrategy":
        # Multi-vector chunk profiles feed one document per chunk
        # (strategy.py: num_patches > 1 -> multi_doc).
        params = segmentation.get("params", {})
        return _expected_chunk_count(
            _video_duration_seconds(path),
            float(params.get("chunk_duration", 30.0)),
            float(params.get("chunk_overlap", 0.0)),
        )
    pipeline_config = profile_def.get("pipeline_config", {}) if profile_def else {}
    target_fps = pipeline_config.get("keyframe_fps")
    if not isinstance(target_fps, (int, float)) or target_fps <= 0:
        target_fps = (
            profile_def.get("strategies", {})
            .get("segmentation", {})
            .get("params", {})
            .get("fps", 0.5)
        )
    if not isinstance(target_fps, (int, float)) or target_fps <= 0:
        raise AssertionError(
            f"Could not determine keyframe fps for profile {profile!r}: {profile_def}"
        )

    import cv2

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        raise AssertionError(
            f"Could not determine frame count for tracked video {path!r}: "
            f"fps={video_fps!r}, frames={total_frames!r}"
        )
    frame_interval = int(video_fps / target_fps) if video_fps > target_fps else 1
    return sum(
        1 for frame_idx in range(total_frames) if frame_idx % frame_interval == 0
    )


def _bootstrap_tenant_and_schemas() -> None:
    """Create the E2E tenant and deploy schemas if not already done.

    Called once per session. Idempotent — 409 (already exists) is fine.
    """
    # Read profile definitions from config.json
    config_path = DATA_ROOT.parent / "configs" / "config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    all_profiles = config.get("backend", {}).get("profiles", {})
    # Register the profile families the seeded fixtures actually use; the
    # tenant's profile-selection and admin listings must reflect live config.
    profile_names = tuple(
        dict.fromkeys(
            (
                *_synthetic_fixture_profiles(config),
                *_profile_selection_video_profiles(config),
                _configured_document_profile_name(config),
                _configured_document_profile_name(
                    config, schema_name="document_visual"
                ),
                _configured_audio_profile_name(config),
            )
        )
    )

    # Step 1: Create tenant (409 = already exists)
    try:
        resp = httpx.post(
            f"{RUNTIME}/admin/tenants",
            json={"tenant_id": TENANT_ID, "created_by": "e2e-test"},
            timeout=30,
        )
        if resp.status_code not in (200, 201, 409):
            # Every later step assumes this tenant exists, so a swallowed
            # failure here resurfaces as an unrelated-looking 404 from the
            # first request that uses it (a Vespa write block reads as
            # "tenant not registered"). Fail on the real cause instead.
            pytest.fail(
                f"e2e tenant {TENANT_ID!r} could not be created: "
                f"HTTP {resp.status_code}: {resp.text[:400]}"
            )
    except (httpx.HTTPError, OSError) as exc:
        pytest.fail(
            f"e2e tenant {TENANT_ID!r} creation failed: {type(exc).__name__}: {exc}"
        )

    # Delete-then-create so config.json edits take effect: POST rejects
    # re-creation and PUT can't change embedding_model. delete_schema=false
    # avoids redeploying the Vespa schema on every session.
    for profile_name in profile_names:
        profile_def = all_profiles.get(profile_name, {})
        if not profile_def:
            continue

        try:
            httpx.delete(
                f"{RUNTIME}/admin/profiles/{profile_name}",
                params={"tenant_id": TENANT_ID, "delete_schema": "false"},
                timeout=30,
            )
        except (httpx.HTTPError, OSError) as exc:
            print(f"Profile pre-delete failed (non-fatal): {exc}")

        try:
            payload = {
                "profile_name": profile_name,
                "tenant_id": TENANT_ID,
                "type": profile_def.get("type", "video"),
                "description": profile_def.get("description", ""),
                "schema_name": profile_def.get("schema_name", profile_name),
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


def _content_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


EVALUATION_QUERY_ASSET = (
    DATA_ROOT / "testset" / "evaluation" / "sample_videos_retrieval_queries.json"
)


@functools.lru_cache(maxsize=1)
def _evaluation_query_rows() -> tuple[dict[str, object], ...]:
    rows = json.loads(EVALUATION_QUERY_ASSET.read_text())
    if not isinstance(rows, list):
        raise AssertionError(f"{EVALUATION_QUERY_ASSET} did not load a JSON list")
    return tuple(row for row in rows if isinstance(row, dict))


def profile_selection_corpus_videos() -> tuple[Path, ...]:
    """Every video the profile-selection truth asset references, sorted by id.

    Ids normalize through the same production helper the label rule uses, so
    the corpus and the labels cannot disagree about what counts as a video id.
    """
    from cogniverse_runtime.optimization_cli import _profile_selection_expected_videos

    sample_videos_dir = _EVALUATION_CORPUS_DIR / "evaluation" / "sample_videos"
    expected_ids = sorted(
        {
            video_id
            for row in _evaluation_query_rows()
            for video_id in _profile_selection_expected_videos(row)
        }
    )
    if not expected_ids:
        pytest.fail(
            f"Profile-selection truth asset {EVALUATION_QUERY_ASSET} yielded no "
            "expected videos"
        )

    missing_ids: list[str] = []
    duplicate_ids: list[str] = []
    corpus_paths: list[Path] = []
    for video_id in expected_ids:
        matches = sorted(
            path for path in sample_videos_dir.glob(f"{video_id}.*") if path.is_file()
        )
        if len(matches) == 1:
            corpus_paths.append(matches[0])
        elif not matches:
            missing_ids.append(video_id)
        else:
            duplicate_ids.append(video_id)
    if missing_ids or duplicate_ids:
        details = []
        if missing_ids:
            details.append(f"missing ids: {missing_ids!r}")
        if duplicate_ids:
            details.append(f"duplicate ids: {duplicate_ids!r}")
        pytest.fail(
            f"Profile-selection sample video corpus mismatch in {sample_videos_dir}: "
            + "; ".join(details)
        )
    return tuple(corpus_paths)


_SAMPLE_VIDEO_MEDIA_TYPES = {".mp4": "video/mp4", ".mkv": "video/x-matroska"}


def _sample_video_media_type(path: Path) -> str:
    """Upload MIME for a sampled video, by suffix (no system mime database)."""
    try:
        return _SAMPLE_VIDEO_MEDIA_TYPES[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported sample video suffix {path.suffix!r} for {path.name!r}"
        ) from None


def _sample_frame_path() -> Path:
    """Materialize the first real video frame as the image-modality fixture."""
    destination = E2E_ARTIFACT_DIR / f"{SAMPLE_VIDEO_CONTENT_ID}_frame_0000.jpg"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        import av

        with av.open(str(SAMPLE_VIDEO_PATH)) as container:
            stream = container.streams.video[0]
            frame = next(container.decode(stream))
            frame.to_image().save(destination, format="JPEG", quality=95)
    except (ImportError, IndexError, OSError, StopIteration) as exc:
        pytest.fail(f"Could not extract tracked sample video frame: {exc}")
    if not destination.exists() or destination.stat().st_size == 0:
        pytest.fail(f"Tracked sample video frame was not created: {destination}")
    return destination


def _source_url_matches(
    source_url: object,
    *,
    content_id: str,
    tenant_id: str,
    suffix: str,
) -> bool:
    if not isinstance(source_url, str) or not source_url.startswith("s3://"):
        return False
    return source_url.rsplit("/", 2)[-2:] == [tenant_id, f"{content_id}{suffix}"]


def _sample_source_location_matches(
    metadata: dict,
    *,
    content_id: str,
    tenant_id: str,
    suffix: str,
    family: str,
) -> bool:
    """Prove the hit carries the tenant-scoped location of the exact fixture.

    Media fixtures record an S3 ``source_url``; text fixtures record the
    tenant-partitioned ``document_path`` of the cached original.
    """
    if family in ("image", "video", "audio"):
        return _source_url_matches(
            metadata.get("source_url"),
            content_id=content_id,
            tenant_id=tenant_id,
            suffix=suffix,
        )
    document_path = metadata.get("document_path")
    if not isinstance(document_path, str):
        return False
    tenant_segments = f"/{tenant_id.replace(':', '/')}/"
    return tenant_segments in document_path and document_path.endswith(
        f"/{content_id}{suffix}"
    )


def _matching_sample_results(
    search_body: dict,
    *,
    content_id: str,
    tenant_id: str,
    profile: str,
    suffix: str,
    media_type: str,
) -> list[dict]:
    """Return only hits that prove the exact tenant-scoped fixture identity."""
    assert search_body.get("query") == content_id, search_body
    assert search_body.get("profile") == profile, search_body
    assert search_body.get("strategy") == "default", search_body
    results = search_body.get("results")
    assert isinstance(results, list), search_body
    assert search_body.get("results_count") == len(results), search_body

    family = media_type.split("/", 1)[0]
    if family == "image":
        identity_field, expected_document_prefix = "image_id", f"{content_id}_seg_"
    elif family == "video":
        identity_field, expected_document_prefix = "video_id", f"{content_id}_seg_"
    elif family == "text":
        identity_field, expected_document_prefix = "document_id", f"{content_id}_"
    elif family == "audio":
        identity_field, expected_document_prefix = "audio_id", f"{content_id}_"
    else:
        raise AssertionError(f"unsupported sample media_type {media_type!r}")

    matches = []
    for result in results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if (
            result.get("source_id") == content_id
            and metadata.get(identity_field) == content_id
            and isinstance(result.get("document_id"), str)
            and result["document_id"].startswith(expected_document_prefix)
            and _sample_source_location_matches(
                metadata,
                content_id=content_id,
                tenant_id=tenant_id,
                suffix=suffix,
                family=family,
            )
        ):
            matches.append(result)
    return matches


def _validate_sample_ingestion_result(
    result: dict,
    *,
    content_id: str,
    tenant_id: str,
    suffix: str,
    expected_documents_fed: int,
) -> int:
    """Validate the terminal worker result and return its persisted count."""
    assert result.get("video_id") == content_id, result
    assert _source_url_matches(
        result.get("source_url"),
        content_id=content_id,
        tenant_id=tenant_id,
        suffix=suffix,
    ), result
    chunks = result.get("chunks")
    documents_fed = result.get("documents_fed")
    assert type(chunks) is int and chunks == expected_documents_fed, result
    assert type(documents_fed) is int and documents_fed == expected_documents_fed, (
        result
    )
    assert chunks == documents_fed, result
    return documents_fed


def _search_sample_content(
    *,
    content_id: str,
    tenant_id: str = TENANT_ID,
    profile: str,
    suffix: str,
    media_type: str,
) -> tuple[list[dict] | None, str | None]:
    """Return (matches, error). A search-API failure is reported as an
    error string, never flattened into 'no matches'.

    Asks for segment granularity explicitly: the caller counts persisted
    DOCUMENTS against documents_fed, and video profiles default to source
    granularity, which collapses every frame of a video into one result.
    """
    try:
        response = httpx.post(
            f"{RUNTIME}/search/",
            json={
                "query": content_id,
                "profile": profile,
                "strategy": "default",
                "top_k": 1000,
                "tenant_id": tenant_id,
                "result_granularity": "segment",
            },
            timeout=60,
        )
    except (httpx.HTTPError, OSError) as exc:
        return None, f"search request failed: {exc!r}"
    if response.status_code != 200:
        return None, (f"search returned {response.status_code}: {response.text[:500]}")
    matches = _matching_sample_results(
        response.json(),
        content_id=content_id,
        tenant_id=tenant_id,
        profile=profile,
        suffix=suffix,
        media_type=media_type,
    )
    return matches, None


def _ensure_sample_content_ingested(
    path: Path,
    *,
    profile: str,
    media_type: str,
    tenant_id: str = TENANT_ID,
) -> str:
    if not path.exists():
        pytest.fail(f"Tracked sample content not found: {path}")

    content_id = _content_sha256(path)
    existing_matches, _ = _search_sample_content(
        content_id=content_id,
        tenant_id=tenant_id,
        profile=profile,
        suffix=path.suffix,
        media_type=media_type,
    )
    if existing_matches:
        print(
            f"Exact {content_id} fixture already has {len(existing_matches)} "
            f"documents in {profile} for {tenant_id}; skipping ingest"
        )
        return content_id

    try:
        with path.open("rb") as source:
            response = httpx.post(
                f"{RUNTIME}/ingestion/upload",
                files={"file": (path.name, source, media_type)},
                data={
                    "profile": profile,
                    "backend": "vespa",
                    "tenant_id": tenant_id,
                },
                # The search above proved these documents are absent, so this
                # call needs real work done. A done marker outliving the
                # documents it describes (Vespa redeployed, Redis kept) would
                # otherwise satisfy the submit without re-ingesting anything.
                params={"force": "true"},
                timeout=60,
            )
    except (httpx.HTTPError, OSError) as exc:
        pytest.fail(f"Sample content upload failed for {path}: {exc}")
    if response.status_code != 200:
        pytest.fail(
            f"Sample content upload returned {response.status_code} for "
            f"{path}: {response.text[:500]}"
        )

    submission = response.json()
    assert submission.get("filename") == path.name, submission
    assert submission.get("existing") is False, submission
    assert _source_url_matches(
        submission.get("source_url"),
        content_id=content_id,
        tenant_id=tenant_id,
        suffix=path.suffix,
    ), submission
    ingest_id = submission.get("ingest_id")
    if not isinstance(ingest_id, str) or not re.fullmatch(
        r"ingest_[0-9a-f]{32}", ingest_id
    ):
        pytest.fail(f"Sample content upload returned invalid ingest_id: {submission}")

    duration_s = (
        _video_duration_seconds(path) if media_type.startswith("video/") else 0.0
    )
    latest: dict = {}
    documents_fed = 0
    expected_documents_fed = _expected_sample_documents_fed(path, profile, media_type)
    deadline_s = max(
        300.0,
        duration_s * (120.0 / 5.0),
        expected_documents_fed * 120.0,
    )
    deadline = _time.monotonic() + deadline_s
    while _time.monotonic() < deadline:
        try:
            status_response = httpx.get(
                f"{RUNTIME}/ingestion/{ingest_id}/status", timeout=10
            )
        except httpx.HTTPError as exc:
            pytest.fail(f"Ingestion status request failed for {ingest_id}: {exc}")
        if status_response.status_code != 200:
            pytest.fail(
                f"Ingestion status returned {status_response.status_code} for "
                f"{ingest_id}: {status_response.text[:500]}"
            )
        snapshot = status_response.json()
        if snapshot.get("ingest_id") != ingest_id:
            pytest.fail(
                f"Ingestion status identity mismatch for {ingest_id}: {snapshot}"
            )
        latest = snapshot.get("latest", {})
        if snapshot.get("state") == "complete":
            documents_fed = _validate_sample_ingestion_result(
                latest.get("result", {}),
                content_id=content_id,
                tenant_id=tenant_id,
                suffix=path.suffix,
                expected_documents_fed=expected_documents_fed,
            )
            break
        if snapshot.get("state") == "failed":
            pytest.fail(f"Sample content ingestion failed: {latest}")
        _time.sleep(5)
    else:
        pytest.fail(
            f"Sample content ingestion {ingest_id} did not complete within "
            f"{deadline_s:.0f}s: {latest}"
        )

    search_deadline = _time.monotonic() + max(120.0, duration_s * 5.0)
    matches: list[dict] = []
    search_error: str | None = None
    while _time.monotonic() < search_deadline:
        found, search_error = _search_sample_content(
            content_id=content_id,
            tenant_id=tenant_id,
            profile=profile,
            suffix=path.suffix,
            media_type=media_type,
        )
        matches = found or []
        if len(matches) == documents_fed:
            print(
                f"Sample {content_id} persisted {documents_fed} exact documents "
                f"in {profile} for {tenant_id}"
            )
            return content_id
        _time.sleep(2)

    failure = (
        f"Sample {content_id} reported {documents_fed} documents_fed but search "
        f"found {len(matches)} exact persisted documents in {profile} for {tenant_id}"
    )
    if search_error:
        failure += f"; last search error: {search_error}"
    pytest.fail(failure)


def _ingest_sample_video() -> None:
    """Ensure the tracked sample videos are persisted for E2E tests."""
    actual_content_id = _content_sha256(SAMPLE_VIDEO_PATH)
    if actual_content_id != SAMPLE_VIDEO_CONTENT_ID:
        pytest.fail(
            f"Tracked sample video content changed: expected "
            f"{SAMPLE_VIDEO_CONTENT_ID}, got {actual_content_id}"
        )

    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    candidate_paths = (
        SAMPLE_VIDEO_PATH,
        _TRACKED_E2E_VIDEO,
        *profile_selection_corpus_videos(),
    )
    ingest_paths: list[Path] = []
    seen_content_ids: set[str] = set()
    for path in candidate_paths:
        content_id = _content_sha256(path)
        if content_id in seen_content_ids:
            continue
        seen_content_ids.add(content_id)
        ingest_paths.append(path)

    for profile in _profile_selection_video_profiles(config):
        for path in ingest_paths:
            persisted_content_id = _ensure_sample_content_ingested(
                path,
                profile=profile,
                media_type=_sample_video_media_type(path),
            )
            assert persisted_content_id == _content_sha256(path)


def _sample_audio_path() -> Path:
    """Ten seconds of the tracked video's real audio stream, extracted once."""
    return _extract_audio_fixture(
        _TRACKED_E2E_VIDEO, E2E_ARTIFACT_DIR / "tracked_video_audio.wav"
    )


def sample_audio_content_id() -> str:
    """Document id the seeded corpus assigns the tracked audio clip."""
    return _content_sha256(_sample_audio_path())


def _ingest_sample_audio() -> str:
    """Ensure the tracked video's audio is persisted as audio content."""
    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return _ensure_sample_content_ingested(
        _sample_audio_path(),
        profile=_configured_audio_profile_name(config),
        media_type="audio/wav",
    )


SAMPLE_DOCUMENT_TITLES = ("v_-nl4G-00PtA.txt", "v_0BtHd6dvm78.txt")
_CAPTION_CORPUS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "testset"
    / "Test_Human_Annotated_Captions"
)
_EVALUATION_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "testset"
_EVALUATION_TEXT_CORPUS_DIR = _EVALUATION_CORPUS_DIR / "evaluation" / "processed"


def _evaluation_text_corpus_paths() -> tuple[Path, ...]:
    # sample_videos_retrieval_queries.json is deliberately NOT ingested. It is the
    # ground truth this tenant is evaluated against -- profile labels derive from its
    # expected_videos and the quality monitor uses it as its golden set. Ingesting it
    # puts a document holding every evaluation query verbatim into the corpus being
    # searched, so it matches any of those queries by construction and outranks the
    # content that should answer them.
    return (
        _EVALUATION_CORPUS_DIR / "dataset_summary.md",
        *_sorted_evaluation_corpus_paths("descriptions"),
        *_sorted_evaluation_corpus_paths("transcripts"),
    )


def _sorted_evaluation_corpus_paths(subdir: str) -> tuple[Path, ...]:
    return tuple(sorted((_EVALUATION_TEXT_CORPUS_DIR / subdir).glob("*.json")))


def _ingest_sample_documents(tenant_id: str = TENANT_ID) -> dict[str, str]:
    """Ensure the two human-annotated captions that describe washing dishes
    are persisted as document content; returns ``{title: content_id}``."""
    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        title: _ensure_sample_content_ingested(
            _CAPTION_CORPUS_DIR / title,
            profile=_configured_document_profile_name(config),
            media_type="text/plain",
            tenant_id=tenant_id,
        )
        for title in SAMPLE_DOCUMENT_TITLES
    }


def _ingest_evaluation_text_corpus() -> dict[str, str]:
    """Ensure the evaluation text corpus is persisted as document content.

    The corpus stores prose inside JSON structures; document profiles accept
    prose formats only, so JSON members are materialized to text first.
    """
    from tests.e2e.corpus_text import materialize_corpus_text

    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    profile = _configured_document_profile_name(config)
    dest_dir = Path(tempfile.gettempdir()) / "cogniverse-e2e-corpus-text"

    ingested: dict[str, str] = {}
    for path in _evaluation_text_corpus_paths():
        key = path.relative_to(DATA_ROOT).as_posix()
        if path.suffix.lower() == ".json":
            source = materialize_corpus_text(path, key, dest_dir)
            if source is None:
                continue
        else:
            source = path
        ingested[key] = _ensure_sample_content_ingested(
            source,
            profile=profile,
            media_type="text/plain",
        )
    return ingested


def _ingest_sample_frame() -> str:
    """Ensure a real frame from the tracked video is persisted as image content."""
    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return _ensure_sample_content_ingested(
        _sample_frame_path(),
        profile=_configured_image_profile_name(config),
        media_type="image/jpeg",
    )


E2E_CLUSTER_NAME = "cogniverse-e2e"
KUBECTL_CONTEXT = f"k3d-{E2E_CLUSTER_NAME}"
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
    33881: 28081,  # semantic-router envoy
    33901: 29001,  # inference sidecars
    33902: 29002,
    33903: 29003,
    33904: 29004,
    33905: 29005,
    33906: 29006,
    33907: 29007,
    33908: 29008,
    33909: 29009,
    33910: 29010,
    33911: 29011,
    33912: 29012,  # video_embed (X-CLIP)
}

_E2E_TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"
_E2E_ASR_MODELS = {
    "cpu": "openai/whisper-tiny",
    "cuda": "openai/whisper-large-v3-turbo",
    "rocm": "openai/whisper-large-v3-turbo",
}
SEMANTIC_ROUTER_ENVOY = "http://localhost:33881"


def _e2e_docker_network_gateway_ip() -> str:
    network_name = f"k3d-{E2E_CLUSTER_NAME}"
    command = [
        "docker",
        "network",
        "inspect",
        network_name,
        "-f",
        "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "docker network gateway inspection failed: "
            f"{shlex.join(command)}\nstderr: {(result.stderr or '').strip()}"
        )
    gateway_ip = result.stdout.strip()
    if not gateway_ip:
        raise RuntimeError(
            "docker network gateway inspection returned an empty IP: "
            f"{shlex.join(command)}"
        )
    return gateway_ip


# The teacher needs 20Gi of the node's 123.5Gi. Video embedding and
# transcription cannot make room for it -- the session fixture ingests the
# corpus, and the shipped profiles bind embedding to vllm_colpali and
# transcription to vllm_asr. Only the code retriever is unused here, so the
# rest of the room comes from right-sizing requests to measured usage.
_E2E_DISABLED_INFERENCE_SERVICES = frozenset({"code_colbert_pylate"})


def _e2e_deployment_overrides() -> dict[str, str]:
    from cogniverse_cli.sandbox import active_gateway_metadata, pod_gateway_endpoint

    overrides = {
        **{
            f"inference.{service}.enabled": "false"
            for service in sorted(_E2E_DISABLED_INFERENCE_SERVICES)
        },
        "runtime.sandbox.enabled": "true",
        "runtime.sandbox.inCluster.enabled": "false",
        "runtime.sandbox.gatewayEndpoint": pod_gateway_endpoint(
            active_gateway_metadata()
        ),
        "runtime.sandbox.hostGatewayIP": _e2e_docker_network_gateway_ip(),
    }
    for service in (
        "vllm_colpali",
        "vllm_asr",
        "vllm_llm_student",
        "vllm_llm_teacher",
    ):
        overrides[f"inference.{service}.livenessProbe.initialDelaySeconds"] = "1200"
        overrides[f"inference.{service}.livenessProbe.failureThreshold"] = "60"
    return overrides


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


_E2E_DEPLOY_STATE_CM = "e2e-deploy-state"
_E2E_DEPLOY_DIFF_PATHS = DEPLOY_INPUT_PATHS


def _e2e_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_e2e_command(repo_root: Path, *args: str) -> list[str]:
    return ["git", "-C", str(repo_root), *args]


def _git_e2e(
    repo_root: Path, *args: str, timeout: int = 30
) -> subprocess.CompletedProcess:
    return subprocess.run(
        _git_e2e_command(repo_root, *args),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require_git_success(
    result: subprocess.CompletedProcess, command: list[str]
) -> None:
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"git command failed with exit {result.returncode}: "
        f"{shlex.join(command)}\nstderr: {(result.stderr or '').strip()}"
    )


def _current_e2e_deploy_sha(repo_root: Path | None = None) -> str:
    repo_root = repo_root or _e2e_repo_root()
    result = _git_e2e(repo_root, "rev-parse", "HEAD")
    _require_git_success(result, _git_e2e_command(repo_root, "rev-parse", "HEAD"))
    return result.stdout.strip()


def _normalize_e2e_deployment_identity(
    identity: dict[str, object],
) -> dict[str, object]:
    normalized = dict(identity)
    for key in ("values_files", "image_tags"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = tuple(value)
    return normalized


def _effective_e2e_deployment_identity(repo_root: Path) -> dict:
    from cogniverse_cli.images import dev_deployment_identity

    from tests.e2e.deployment.conftest import deployment_helm_inputs

    inputs = deployment_helm_inputs(
        repo_root,
        extra_set=_e2e_deployment_overrides(),
    )
    return dev_deployment_identity(
        repo_root,
        torch_backend=inputs["backend"],
        values_files=inputs["helm_values"],
        set_overrides=inputs["helm_set_overrides"],
        versions=inputs["image_versions"],
    )


def _current_e2e_deploy_state(repo_root: Path | None = None) -> dict:
    repo_root = repo_root or _e2e_repo_root()
    return _effective_e2e_deployment_identity(repo_root)


def _require_clean_e2e_worktree(repo_root: Path | None = None) -> None:
    repo_root = repo_root or _e2e_repo_root()
    result = _git_e2e(repo_root, "status", "--porcelain")
    _require_git_success(result, _git_e2e_command(repo_root, "status", "--porcelain"))
    if result.stdout.strip():
        raise RuntimeError(
            "refusing to deploy from a dirty git tree; commit first "
            "(a WIP commit is fine, amend it later), then rerun"
        )


def _read_e2e_deploy_state() -> dict | None:
    got = _kubectl_e2e(
        "-n",
        "cogniverse",
        "get",
        "configmap",
        _E2E_DEPLOY_STATE_CM,
        "-o",
        "jsonpath={.data.stamp}",
    )
    if got.returncode != 0:
        return None
    raw = got.stdout.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return (
        _normalize_e2e_deployment_identity(payload)
        if isinstance(payload, dict)
        else None
    )


def _stamp_e2e_deploy_state(deploy_state: dict) -> None:
    render_args = (
        "-n",
        "cogniverse",
        "create",
        "configmap",
        _E2E_DEPLOY_STATE_CM,
        f"--from-literal=stamp={json.dumps(deploy_state, sort_keys=True, separators=(',', ':'))}",
        "--dry-run=client",
        "-o",
        "yaml",
    )
    rendered = _kubectl_e2e(*render_args)
    _require_kubectl_success(rendered, _kubectl_e2e_command(*render_args))

    apply_command = _kubectl_e2e_command("apply", "-f", "-")
    applied = subprocess.run(
        apply_command,
        input=rendered.stdout,
        capture_output=True,
        text=True,
        timeout=30,
    )
    _require_kubectl_success(applied, apply_command)


def _e2e_action_for_cluster_state(cluster_state: str) -> str:
    """Map a cluster state to the action the session fixture takes.

    ``stale`` deploys rather than failing: deploy_stack rebuilds only the
    changed images and helm upgrades an existing release, so the repair keeps
    every seeded corpus. Deleting the cluster costs the recreate plus all of
    that data, so it is never the automatic remedy.
    """
    if cluster_state == "reusable":
        return "reuse"
    if cluster_state in ("absent", "stale"):
        return "deploy"
    return "fail"


def _e2e_deploy_reuse_state(
    repo_root: Path,
    deployed_state: dict | None,
    *,
    current_identity: dict | None = None,
) -> tuple[str, str]:
    if not isinstance(deployed_state, dict):
        return "stale", "deploy stamp is missing or malformed"
    if set(deployed_state) != {
        "backend",
        "values_files",
        "set_overrides",
        "image_tags",
        "chart_digest",
    }:
        return "stale", "deploy stamp is missing or malformed"
    if current_identity is None:
        current_identity = _effective_e2e_deployment_identity(repo_root)
    if _normalize_e2e_deployment_identity(
        deployed_state
    ) == _normalize_e2e_deployment_identity(current_identity):
        return "reusable", ""
    return "stale", "deployment identity changed"


def _kubectl_e2e_command(*args: str) -> list[str]:
    return ["kubectl", "--context", KUBECTL_CONTEXT, *args]


def _kubectl_e2e(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _kubectl_e2e_command(*args),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require_kubectl_success(
    result: subprocess.CompletedProcess, command: list[str]
) -> None:
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"kubectl command failed with exit {result.returncode}: "
        f"{shlex.join(command)}\nstderr: {(result.stderr or '').strip()}"
    )


def _e2e_required_model_probes(backend: str) -> list[tuple[str, str]]:
    """Model endpoints this cluster must serve, derived from the enabled set.

    A service switched off by the deployment overrides has no pod to probe, so
    gating on it would fail readiness for a model nothing deployed.
    """
    probes: list[tuple[str, str]] = []
    if (
        backend in {"cuda", "rocm"}
        and "vllm_colpali" not in _E2E_DISABLED_INFERENCE_SERVICES
    ):
        probes.append(("http://127.0.0.1:33901", _E2E_TOMORO_MODEL))
    if "vllm_asr" not in _E2E_DISABLED_INFERENCE_SERVICES:
        probes.append(
            (
                "http://127.0.0.1:33905",
                _E2E_ASR_MODELS.get(backend, "openai/whisper-large-v3-turbo"),
            )
        )
    return probes


# Sidecars the session fixture's own bootstrap exercises: video embedding
# (colpali), transcription (asr), document and text embeddings
# (colbert_pylate, denseon), audio embedding (clap_embed) and graph extraction
# (gliner). Only the first two speak OpenAI's /v1/models, so the rest were
# ungated -- on 2026-09-01 a run proceeded with gliner still starting and the
# seven selected tests ERRORed reporting "ingestion did not complete", which
# names the symptom rather than the missing model. Every sidecar serves
# /health, so all six are gateable.
_E2E_GATED_INFERENCE_SERVICES = (
    "vllm_colpali",
    "vllm_asr",
    "colbert_pylate",
    "denseon",
    "clap_embed",
    "gliner",
)

# The e2e loadbalancer maps NodePort 290NN to host port 339NN (E2E_HOST_PORTS).
_E2E_NODEPORT_TO_HOST = 33900 - 29000


def _chart_inference_node_ports() -> dict[str, int]:
    """service -> nodePort, read from the shipped chart rather than restated."""

    import re as _re

    values = Path(__file__).resolve().parents[2] / "charts/cogniverse/values.yaml"
    ports: dict[str, int] = {}
    current: str | None = None
    for line in values.read_text().splitlines():
        header = _re.match(r"^  ([a-z_]+):\s*$", line)
        if header:
            current = header.group(1)
        port = _re.search(r"nodePort:\s*(\d+)", line)
        if port and current:
            ports[current] = int(port.group(1))
    return ports


def e2e_required_health_probes(backend: str) -> list[tuple[str, str]]:
    """(service, base_url) for every enabled sidecar this cluster must serve.

    A service switched off by the deployment overrides has no pod to probe, so
    gating on it would fail readiness for something nothing deployed.
    """

    node_ports = _chart_inference_node_ports()
    probes: list[tuple[str, str]] = []
    for service in _E2E_GATED_INFERENCE_SERVICES:
        if service in _E2E_DISABLED_INFERENCE_SERVICES:
            continue
        node_port = node_ports.get(service)
        if node_port is None:
            raise RuntimeError(
                f"{service} is gated for e2e readiness but the shipped chart "
                "declares no nodePort for it, so the gate cannot reach it. "
                "Either the service was renamed or its Service lost nodePort."
            )
        probes.append(
            (service, f"http://localhost:{node_port + _E2E_NODEPORT_TO_HOST}")
        )
    return probes


def _required_e2e_models_ready(backend: str | None = None) -> tuple[bool, str]:
    from cogniverse_cli.images import detect_torch_backend

    from tests.utils.vllm_sidecar import serves_exact_model

    resolved_backend = backend or detect_torch_backend()
    for url, model in _e2e_required_model_probes(resolved_backend):
        if not serves_exact_model(url, model, timeout=5.0):
            return False, f"{model} is not served exactly at {url}/v1/models"
    # The remaining sidecars do not speak /v1/models, but the session fixture's
    # own bootstrap calls them: leaving them ungated turns "a model never came
    # up" into an ingestion timeout reported against every selected test.
    for service, base_url in e2e_required_health_probes(resolved_backend):
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"{service} is not reachable at {base_url}/health: {exc}"
        if response.status_code != 200:
            return (
                False,
                f"{service} is not healthy at {base_url}/health: "
                f"HTTP {response.status_code}",
            )
    return True, ""


def _required_e2e_semantic_router_ready() -> tuple[bool, str]:
    # GET /v1/models, not a chat completion: the completion path requires the
    # caller identity headers the runtime injects (x-authz-user-id and friends),
    # so a bare probe returns 403 and never reaches the router. /v1/models needs
    # no headers and still separates the states we care about -- it answers 500
    # while the routing classifier model is missing and 200 once it is loaded.
    url = f"{SEMANTIC_ROUTER_ENVOY}/v1/models"
    try:
        response = httpx.get(url, timeout=10.0)
    except (httpx.HTTPError, OSError) as exc:
        return (
            False,
            f"semantic-router envoy readiness failed at {url}; error={exc}",
        )
    if response.status_code != 200:
        return (
            False,
            "semantic-router envoy readiness failed at "
            f"{url}; status={response.status_code}; body={response.text[:200]!r}",
        )
    return True, ""


def _e2e_cluster_state() -> tuple[str, str]:
    """Inspect the shared cluster without changing its lifecycle."""
    from cogniverse_cli.cluster import list_cluster_states

    try:
        cluster = next(
            (
                state
                for state in list_cluster_states()
                if state["name"] == E2E_CLUSTER_NAME
            ),
            None,
        )
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        return "unhealthy", f"cluster inventory inspection failed: {exc}"
    if cluster is None:
        return "absent", ""
    servers_running = cluster["servers_running"]
    servers_count = cluster["servers_count"]
    if servers_running == 0:
        return "stopped", ""
    if servers_running != servers_count:
        return (
            "unhealthy",
            f"{servers_running} of {servers_count} server nodes are running",
        )
    if (
        _kubectl_e2e("get", "ns", "cogniverse", "-o", "name", timeout=15).returncode
        != 0
    ):
        return "unhealthy", "the cogniverse namespace is unreachable"
    deployed_state = _read_e2e_deploy_state()
    state, detail = _e2e_deploy_reuse_state(_e2e_repo_root(), deployed_state)
    if state == "stale":
        return state, detail
    if not runtime_available():
        return "unhealthy", f"runtime readiness failed at {RUNTIME}"
    models_ready, model_detail = _required_e2e_models_ready()
    if not models_ready:
        return "unhealthy", model_detail
    semantic_router_ready, semantic_router_detail = (
        _required_e2e_semantic_router_ready()
    )
    if not semantic_router_ready:
        return "unhealthy", semantic_router_detail
    return "reusable", ""


def _wait_for_e2e_reuse_convergence(
    *,
    timeout_s: float = 2400,
    poll_interval_s: float = 5,
) -> tuple[str, str]:
    deadline = _time.monotonic() + timeout_s
    state = "unhealthy"
    detail = "cluster did not report state"
    while True:
        state, detail = _e2e_cluster_state()
        if state in {"reusable", "stale"}:
            return state, detail
        if _time.monotonic() >= deadline:
            return (
                "unhealthy",
                f"cluster did not converge within {timeout_s:g}s; last state was "
                f"{state}: {detail or '<no detail>'}",
            )
        _time.sleep(poll_interval_s)


def _reconcile_orphan_schemas() -> None:
    """Drop schemas left in Vespa by runs that never reached their teardown.

    Every tenant fixture deletes what it created, but a killed run never gets
    to its finally, and one leftover schema is enough to refuse every later
    deploy -- including this session's own bootstrap. A per-tenant delete
    cannot clear it, because that leaves the other orphans in its survivor set
    and is refused for the same reason; the reconciler drops them together, so
    its survivor set is reconstructable.

    Safe because it only drops schemas with no registry record, and it runs
    before this session creates anything. It must stay at the head of the
    pre-flight: a schema registered by a concurrent session would look like an
    orphan while its registry write is still in flight, which is why two
    suites may not share this cluster.
    """
    with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
        preview = client.post("/admin/reconcile-orphans", params={"dry_run": True})
        if preview.status_code != 200:
            pytest.fail(
                "Session pre-flight: could not list orphan schemas "
                f"(HTTP {preview.status_code}): {preview.text[:300]}"
            )
        orphans = sorted(preview.json().get("orphan_schemas") or [])
        if not orphans:
            return

        print(
            f"Session pre-flight: dropping {len(orphans)} orphan schema(s) left "
            f"by earlier runs: {orphans}"
        )
        dropped = client.post("/admin/reconcile-orphans", params={"dry_run": False})
        if dropped.status_code != 200:
            pytest.fail(
                "Session pre-flight: failed to drop orphan schemas "
                f"(HTTP {dropped.status_code}): {dropped.text[:500]}"
            )

        # Read back: the reconciler reports what it dropped, but the state that
        # matters is whether any orphan survives to refuse the bootstrap.
        after = client.post("/admin/reconcile-orphans", params={"dry_run": True})
        if after.status_code != 200:
            pytest.fail(
                "Session pre-flight: could not re-list orphan schemas after "
                f"dropping them (HTTP {after.status_code}): {after.text[:300]}"
            )
        remaining = sorted(after.json().get("orphan_schemas") or [])
        if remaining:
            pytest.fail(
                "Session pre-flight: orphan schemas survived reconciliation and "
                f"will refuse every deploy this session makes: {remaining}"
            )


@pytest.fixture(scope="session", autouse=True)
def e2e_stack(request, resolved_inference_endpoints):
    """Provide a healthy, bootstrapped e2e stack without replacing shared state.

    The cluster is a dedicated k3d deployment whose loadbalancer maps the
    offset 33xxx HOST ports onto the chart's canonical NodePorts (see
    ``E2E_HOST_PORTS``), Helm-installed with devMode OFF so the pods run the
    code baked into images built from the working tree — never a bind-mounted
    tree with a stale interpreter. Host storage is NOT shared, so a fresh
    boot starts on clean data and cannot touch a dev cluster's state.

    Lifecycle:
      * REUSE — if a running ``cogniverse-e2e`` whose stamped deploy
        identity matches the current repo state, reuse it (~seconds).
        Editing only ``tests/`` keeps the deploy identity unchanged, so
        assertion iteration is fast.
      * CREATE — only when no ``cogniverse-e2e`` cluster exists, stop any dev
        cluster (RAM + ports), build, deploy, wait, and stamp the deploy state.
      * START — resume a stopped shared cluster through the supported project
        lifecycle, then inspect it again before reuse.
      * REJECT — a stale or unhealthy existing shared cluster is never
        replaced. Repair it or explicitly delete it before rerunning.
        ``E2E_FRESH`` likewise requires the shared cluster to be absent.
      * TEARDOWN — the cluster is left warm unless this session created it for
        an ``E2E_FRESH`` run. A session never deletes a cluster it did not own.

    Tests marked ``requires_modal_inference(service)`` are checked against the
    resolved endpoints before any cluster work, so a locally-provisioned
    service fails the run instead of quietly standing in for Modal.

    After the stack is healthy (reused or fresh) the E2E tenant + schemas +
    sample data are (idempotently) bootstrapped and CronWorkflows suspended
    for the session.
    """
    _require_modal_inference_endpoints(
        request.session.items,
        resolved_inference_endpoints,
    )

    from tests.e2e import run_lock

    if run_lock.acquire(run_lock.default_lock_path()):
        request.addfinalizer(lambda: run_lock.release(run_lock.default_lock_path()))
    run_lock.ensure_e2e_gpu_residency()

    from cogniverse_cli.cluster import start_cluster

    from tests.e2e.deployment.conftest import (
        create_test_cluster,
        delete_test_cluster,
        deploy_stack,
    )

    repo_root = _e2e_repo_root()
    deploy_sha = _current_e2e_deploy_sha(repo_root)
    force_fresh = os.environ.get("E2E_FRESH", "").lower() in ("1", "true", "yes")
    _ensure_host_sandbox_gateway()
    cluster_state, state_detail = _e2e_cluster_state()
    created_this_session = False
    reset_command = f"k3d cluster delete {E2E_CLUSTER_NAME}"

    if cluster_state != "absent" and force_fresh:
        pytest.fail(
            f"E2E_FRESH cannot replace existing shared e2e cluster "
            f"{E2E_CLUSTER_NAME!r}. Delete it explicitly with `{reset_command}`, "
            "then rerun."
        )
    if cluster_state == "stopped":
        _stop_dev_cluster_and_free_ports()
        start_cluster(E2E_CLUSTER_NAME)
        cluster_state, state_detail = _wait_for_e2e_reuse_convergence()
    if cluster_state == "unhealthy":
        pytest.fail(
            f"Existing shared e2e cluster {E2E_CLUSTER_NAME!r} is unhealthy "
            f"({state_detail}). Repair it, or delete it explicitly with "
            f"`{reset_command}`, then rerun."
        )
    if cluster_state == "stale":
        print(
            f"Repairing stale e2e cluster {E2E_CLUSTER_NAME} in place "
            f"({state_detail}): rebuilding changed images, upgrading the "
            "release, restamping. Seeded data is preserved."
        )

    if _e2e_action_for_cluster_state(cluster_state) == "reuse":
        print(
            f"Reusing warm e2e cluster {E2E_CLUSTER_NAME} (deploy identity unchanged)"
        )
        _sync_sandbox_into_cluster(KUBECTL_CONTEXT, roll_runtime=True)
    else:
        _require_clean_e2e_worktree(repo_root)
        if cluster_state == "absent":
            _stop_dev_cluster_and_free_ports()
            create_test_cluster(
                E2E_CLUSTER_NAME,
                ports=[f"{host}:{node}" for host, node in E2E_HOST_PORTS.items()],
                share_host_storage=False,
            )
            created_this_session = True
        deploy_identity = _effective_e2e_deployment_identity(repo_root)
        sandbox_overrides = _e2e_deployment_overrides()
        # Test-cluster-only helm overrides (never touch the shipped chart):
        #  - vLLM liveness grace widened: on a COLD cluster the GPU engines
        #    load weights from disk for ~12 min (vs instant off the warm dev
        #    cache), then profile — overrunning the shipped 22-min liveness
        #    budget, so the kubelet kills them mid-init and they never
        #    converge. ~50 min (initialDelay 1200s + 60×30s) covers a cold
        #    load under memory contention. Only LIVENESS matters — readiness
        #    never kills, it just gates the Available condition the wait below
        #    keys off, so it is left shipped-default.
        #  - host-mode sandbox wiring: the gateway's mTLS secret and metadata
        #    configmaps must exist before Helm installs the runtime so its
        #    subPath mounts resolve at pod start; hostAliases maps
        #    host.docker.internal to the k3d network gateway.
        _sync_sandbox_into_cluster(KUBECTL_CONTEXT, roll_runtime=False)
        deploy_stack(
            E2E_CLUSTER_NAME,
            "cogniverse",
            extra_set=sandbox_overrides,
        )
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
                KUBECTL_CONTEXT,
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
        models_ready, model_detail = _required_e2e_models_ready()
        if not models_ready:
            pytest.fail(
                "e2e stack required model identity did not converge after deploy: "
                f"{model_detail}"
            )
        finished_sha = _current_e2e_deploy_sha(repo_root)
        if finished_sha != deploy_sha:
            pytest.fail(
                "working-tree deployment inputs changed while the e2e stack was "
                f"being built: started with {deploy_sha!r}, finished with "
                f"{finished_sha!r}; rerun against a stable tree"
            )
        # Stamp the deployed identity.
        _stamp_e2e_deploy_state(deploy_identity)

    cron_restore: list[str] = []
    try:
        _cron_session_token()
        stale_restore = _restore_stale_cronworkflows()
        if _cron_result_failures(stale_restore):
            pytest.fail(
                "Session pre-flight: failed to restore stale cronworkflow(s): "
                f"{', '.join(_cron_result_failures(stale_restore))}"
            )
        suspend_result = _suspend_cronworkflows_for_session()
        cron_restore = _cron_result_names(suspend_result)
        if _cron_result_failures(suspend_result):
            pytest.fail(
                "Session pre-flight: failed to suspend cronworkflow(s): "
                f"{', '.join(_cron_result_failures(suspend_result))}"
            )
        _reconcile_orphan_schemas()
        _bootstrap_tenant_and_schemas()
        _ingest_sample_video()
        _ingest_sample_frame()
        _ingest_sample_audio()
        _ingest_sample_documents()
        _ingest_evaluation_text_corpus()
        try:
            yield
        finally:
            restore_result = _restore_cronworkflows(cron_restore)
            if _cron_result_failures(restore_result):
                pytest.fail(
                    "Session teardown: failed to restore cronworkflow(s): "
                    f"{', '.join(_cron_result_failures(restore_result))}"
                )
    finally:
        # Only delete a disposable cluster created and owned by this session.
        if force_fresh and created_this_session:
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


def _sweep_tenant_deletes(
    tenants: set[str],
    *,
    budget_s: float = 180.0,
    delete_one=None,
) -> None:
    """Serially delete test tenants via the runtime under a hard budget.

    Serial (one worker) because each delete undeploys a Vespa schema,
    which redeploys the WHOLE application package; concurrent undeploys
    race on that shared rebuild — the loser sees a stale survivor set and
    Vespa rejects it. The budget bounds both the completed-work loop and
    the wait itself, so a hung delete cannot block session teardown; the
    sweep reports what it left behind, and the next run picks it up.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if delete_one is None:

        def delete_one(tid: str) -> None:
            try:
                with httpx.Client(timeout=20.0) as client:
                    client.delete(f"{RUNTIME}/admin/tenants/{tid}")
            except (httpx.HTTPError, OSError) as exc:
                print(f"Cleanup failed for tenant {tid}: {exc}")

    sweep_deadline = _time.monotonic() + budget_s
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        futures = [pool.submit(delete_one, tid) for tid in sorted(tenants)]
        try:
            for _fut in as_completed(futures, timeout=budget_s):
                if _time.monotonic() > sweep_deadline:
                    print(
                        "Tenant cleanup budget exhausted; remainder left for next run"
                    )
                    break
        except TimeoutError:
            pending = sum(1 for f in futures if not f.done())
            print(
                f"Tenant cleanup budget exhausted after {budget_s:g}s with "
                f"{pending} deletes still pending; remainder left for next run"
            )
    finally:
        # Don't block on the queued/in-flight deletes — cancel the rest so a
        # large backlog can't hang session setup/teardown past the budget.
        pool.shutdown(wait=False, cancel_futures=True)


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
    vespa_url = backend_env.vespa_url()
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

    _sweep_tenant_deletes(tenants_seen)

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


_CRON_SESSION_TOKEN: str | None = None


def _cron_session_token() -> str:
    global _CRON_SESSION_TOKEN
    if _CRON_SESSION_TOKEN is None:
        _CRON_SESSION_TOKEN = cron_guard.new_session_token()
    return _CRON_SESSION_TOKEN


def _cron_result_names(result) -> list[str]:  # noqa: ANN001
    if result is None:
        return []
    names = getattr(result, "restore_names", None)
    if names is None:
        names = getattr(result, "restored_names", None)
    if names is None:
        names = result
    return list(names)


def _cron_result_failures(result) -> list[str]:  # noqa: ANN001
    if result is None:
        return []
    failures = getattr(result, "failures", ())
    return list(failures)


def _restore_stale_cronworkflows():
    if _CRON_SESSION_TOKEN is None:
        return cron_guard.CronRestoreResult(())
    return cron_guard.restore_stale_cronworkflows(_CRON_SESSION_TOKEN)


def _suspend_cronworkflows_for_session() -> cron_guard.CronSuspendResult:
    return cron_guard.suspend_cronworkflows_for_session(_cron_session_token())


def _restore_cronworkflows(names: list[str]) -> cron_guard.CronRestoreResult:
    return cron_guard.restore_cronworkflows(names)


def _ensure_host_sandbox_gateway() -> None:
    """Start (or reuse) the host OpenShell gateway.

    Runs before the deploy identity is computed: the pod-facing gateway
    endpoint in the identity comes from the active gateway's own metadata.
    """
    from cogniverse_cli.sandbox import ensure_host_gateway

    try:
        ready = ensure_host_gateway()
    except Exception as exc:
        pytest.fail(f"OpenShell host gateway bootstrap raised: {exc!r}", pytrace=False)
    if not ready:
        pytest.fail(
            "OpenShell host gateway bootstrap returned false; expected=True",
            pytrace=False,
        )


def _openshell_mtls_fingerprint(kube_context: str) -> str:
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            "cogniverse",
            "get",
            "secret",
            "openshell-mtls",
            "-o",
            "jsonpath={.data.tls\\.crt}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _sync_sandbox_into_cluster(kube_context: str, *, roll_runtime: bool) -> None:
    """Sync the active gateway's mTLS certs + metadata into the e2e cluster.

    The runtime mounts them with subPath, which never refreshes in place, so
    on a reused deployment a changed secret rolls the runtime deployment.
    """
    from cogniverse_cli.sandbox import sync_gateway_certs_to_cluster

    before = _openshell_mtls_fingerprint(kube_context)
    try:
        synced = sync_gateway_certs_to_cluster(kube_context=kube_context)
    except Exception as exc:
        pytest.fail(
            f"OpenShell cert sync raised; kube_context={kube_context!r}; error={exc!r}",
            pytrace=False,
        )
    if not synced:
        pytest.fail(
            f"OpenShell cert sync returned false; kube_context={kube_context!r}; "
            "expected=True",
            pytrace=False,
        )
    if not roll_runtime:
        return
    after = _openshell_mtls_fingerprint(kube_context)
    if after == before:
        return
    for args in (
        ["rollout", "restart", "deployment/cogniverse-runtime"],
        ["rollout", "status", "deployment/cogniverse-runtime", "--timeout=900s"],
    ):
        result = subprocess.run(
            ["kubectl", "--context", kube_context, "-n", "cogniverse", *args],
            capture_output=True,
            text=True,
            timeout=960,
        )
        if result.returncode != 0:
            pytest.fail(
                "runtime rollout after OpenShell cert change failed: "
                f"kubectl {' '.join(args)}\nstdout={result.stdout!r}\n"
                f"stderr={result.stderr!r}",
                pytrace=False,
            )


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1920, "height": 1080}}


def wait_for_script_idle(page, timeout_ms: int = 480_000, settle_ms: int = 1_500):
    """Block until no Streamlit script run is in flight.

    Streamlit streams a run's output over the websocket, so
    ``wait_for_load_state("networkidle")`` reports idle while the run is still
    executing and the later tabs have not been written yet. The status widget
    is present for exactly the duration of a run, and absent both before and
    after one, so completion is a continuous absence of ``settle_ms`` rather
    than a bare ``state="detached"`` wait.
    """
    deadline = _time.monotonic() + timeout_ms / 1000
    idle_since = None
    while _time.monotonic() < deadline:
        if page.locator('[data-testid="stStatusWidget"]').count():
            idle_since = None
        elif idle_since is None:
            idle_since = _time.monotonic()
        elif _time.monotonic() - idle_since >= settle_ms / 1000:
            return
        page.wait_for_timeout(250)
    raise TimeoutError(
        f"Streamlit was still running a script after {timeout_ms / 1000:g}s. "
        "The run never completed, so any widget assertion here would be "
        "reading a partially rendered page."
    )


def wait_for_streamlit(page, timeout: int = 30_000):
    """Wait for the Streamlit app to finish rendering."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)
    wait_for_script_idle(page)


def _strip_emoji(text: str) -> str:
    """Strip leading emoji + whitespace from tab text for clean comparison."""
    import re

    return re.sub(
        r"^[\U0001f300-\U0001faff\u2600-\u27bf\ufe0f\u200d]+\s*", "", text
    ).strip()


def _activate_tab(page, tab, settle_ms: int) -> bool:
    """Click a tab and confirm it became the selected one.

    A click that lands while Streamlit is mid-rerun is swallowed. Returning
    anyway leaves the previous tab open, and every later locator then reads a
    panel the test never asked for, which surfaces as a missing widget rather
    than as a missed click.

    Visibility is read here rather than taken from the caller's snapshot.
    Trying one candidate re-renders the strip, so a tab recorded as visible
    when the candidates were collected can be hidden by the time it is
    reached; an unforced click then raises instead of yielding the next
    candidate.
    """
    from playwright.sync_api import Error as PlaywrightError

    try:
        visible = tab.is_visible()
        if visible:
            tab.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
        tab.click(force=not visible)
        page.wait_for_timeout(1_000)

        if tab.get_attribute("aria-selected") != "true":
            tab.dispatch_event("click")
            page.wait_for_timeout(1_000)
            if tab.get_attribute("aria-selected") != "true":
                return False
    except PlaywrightError:
        return False

    page.wait_for_timeout(settle_ms)
    # Selecting a tab reruns the script; wait for that run rather than for
    # networkidle, which the websocket satisfies while the panel is still
    # being written.
    wait_for_script_idle(page)
    return True


def _click_tab_by_label(
    page, label: str, scope: str, retries: int = 6, settle_ms: int = 3_000
):
    """Click a Streamlit tab by matching its visible text (ignoring emojis).

    ``scope`` is ``"top"`` or ``"sub"`` and is what disambiguates a label that
    names two tabs: the dashboard nests a "Synthetic Data" sub-tab inside a
    "Synthetic Data & Optimization" parent, so a page-wide search finds an
    exact match on the sub-tab and only a substring match on the parent. A
    caller asking for the parent is then sent into a panel that is not open,
    and every click is swallowed.

    Within a scope, exact matches come before substring ones and visible tabs
    before hidden ones.
    """
    # The tab strip renders only after the tenant gate resolves, which calls
    # the runtime. While the cluster is warming that call is slow, so the strip
    # can be empty for far longer than the retry loop below allows. Wait for it
    # to exist before searching it; a genuine absence still falls through to the
    # loop and is reported by the empty-strip branch at the end.
    #
    # Wait for the tab the caller actually asked for, in the containment the
    # scope selects. Waiting on any tab is satisfied immediately by the top
    # strip; waiting on any nested tab is satisfied by whichever panel's sub
    # strip is already open, which is generally not the one being opened here.
    # Either way the loop then searches a strip the target is absent from and
    # spends every attempt on an empty candidate list. `has-text` matches a
    # case-insensitive substring, the same way the scoping below does.
    escaped = label.replace('"', '\\"')
    strip_selector = (
        f'[role="tabpanel"] button[role="tab"]:has-text("{escaped}")'
        if scope == "sub"
        else f'button[role="tab"]:has-text("{escaped}")'
    )
    # Let the in-flight run finish before reading the strip. Otherwise the
    # search runs against however much of the script has been written so far,
    # and a tab that has merely not been rendered yet is indistinguishable
    # from one that is absent.
    wait_for_script_idle(page)
    try:
        page.locator(strip_selector).first.wait_for(state="attached", timeout=60_000)
    except Exception:
        pass

    for attempt in range(retries):
        tabs = page.locator('button[role="tab"]')

        # Collect label / visibility / containment for every tab in one round
        # trip. ``closest('[role="tabpanel"]')`` is the containment fact: a tab
        # rendered inside another tab's panel is a sub-tab. Read from ARIA
        # rather than a Streamlit-version-specific attribute.
        raw_info = page.eval_on_selector_all(
            'button[role="tab"]',
            """els => els.map(el => ({
                text: el.textContent || "",
                visible: !!(el.offsetParent || el.getClientRects().length),
                nested: !!el.closest('[role="tabpanel"]'),
            }))""",
        )
        scoped = [
            (entry["text"], entry["visible"], entry["nested"]) for entry in raw_info
        ]

        for idx in tab_candidates_in_scope(scoped, label, scope):
            if _activate_tab(page, tabs.nth(idx), settle_ms):
                return

        if attempt < retries - 1:
            page.wait_for_timeout(3_000)
    tab_texts = [tabs.nth(i).text_content() or "" for i in range(tabs.count())]
    if not tab_texts:
        raise ValueError(
            f"No tabs rendered at all, so tab '{label}' could not be found. "
            "The dashboard renders its tab strip only once the tenant gate "
            "is satisfied, which calls the runtime; an empty strip means that "
            "gate never completed, not that the tab is missing."
        )
    in_scope = [text for text, _visible, nested in scoped if nested is (scope == "sub")]
    raise ValueError(
        f"Tab '{label}' was never activated in the '{scope}' strip after "
        f"{retries} attempts. A tab that is present but never reports "
        "aria-selected has had every click swallowed by an in-flight rerun; a "
        "tab absent from this strip is in the other one. "
        f"Tabs in the '{scope}' strip: {in_scope}. All tabs: {tab_texts}"
    )


def _wait_for_visible_panel(page, timeout: int = 60_000):
    """Wait for the clicked tab's panel to become visible.

    Streamlit renders a tab body lazily after the click. The dashboard holds 49
    panels and the one under test can be well down the DOM, so under load the
    panel can lag the click by longer than a caller's own timeout. Settle here,
    once, rather than in every caller.
    """
    try:
        page.locator('[role="tabpanel"]:visible').first.wait_for(
            state="visible", timeout=timeout
        )
    except Exception:
        # A tab that renders no panel is the caller's assertion to make.
        pass


def click_top_tab(page, label: str):
    """Click a top-level Streamlit tab."""
    start = _time.monotonic()
    _click_tab_by_label(page, label, scope="top")
    _wait_for_visible_panel(page)
    elapsed = (_time.monotonic() - start) * 1000
    if _report_collector:
        _report_collector.record_browser_op("click_top_tab", label, elapsed_ms=elapsed)


def click_sub_tab(page, label: str):
    """Click a sub-level Streamlit tab.

    Uses a longer settle time than top-level tabs because sub-tabs
    often trigger heavy Streamlit reruns (API calls, data loading).
    """
    start = _time.monotonic()
    _click_tab_by_label(page, label, scope="sub", settle_ms=4_000)
    _wait_for_visible_panel(page)
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


def active_tab_panel(page, timeout: int = 60_000):
    """The visible top-level tab panel.

    Streamlit renders every tab body into the DOM, so a page-wide locator
    also matches widgets belonging to tabs that are not open: at the time of
    writing the Configuration tab alone contributes 14 selectboxes and 16 text
    inputs. Scope widget assertions to this panel so a test can only observe
    the tab it opened. Nested panels are children of their parent, so the
    first visible one in DOM order is the top-level panel.
    """
    panel = page.locator('[role="tabpanel"]:visible').first
    try:
        panel.wait_for(state="visible", timeout=timeout)
    except Exception as exc:
        # No panel at all is almost never a tab problem: the dashboard
        # st.stop()s before rendering any tab when the tenant gate is not
        # satisfied, so report what the page actually shows rather than a bare
        # selector timeout.
        body = ""
        try:
            body = (page.inner_text("body") or "").strip()
        except Exception:
            pass
        raise AssertionError(
            "No tab panel became visible. The dashboard renders its tabs only "
            "after the tenant gate passes, so an empty page means that gate "
            f"stopped the script rather than that a tab is missing. Page text: "
            f"{body[:600]!r}"
        ) from exc
    # The element turns visible before Streamlit streams its children in, so a
    # query issued right after networkidle can read an empty panel. Settle on
    # the panel carrying content rather than merely existing.
    deadline = _time.monotonic() + timeout / 1000.0
    while _time.monotonic() < deadline:
        if (panel.inner_text() or "").strip():
            break
        page.wait_for_timeout(250)
    return panel


def active_sub_tab_panel(page, timeout: int = 60_000):
    """The visible sub-tab panel inside the open top-level tab.

    ``active_tab_panel`` narrows to the open top-level tab, which is not
    enough for an assertion about a sub-tab: Streamlit renders every sub-tab
    body into that panel too, and a CSS locator matches a subtree whether or
    not it is displayed. Counting metrics in the Optimization tab's Overview
    sub-tab found seven -- its own four plus three belonging to sub-tabs the
    test never opened.

    Narrowing by CONTAINMENT rather than by ``:visible``. The two are not
    interchangeable: Streamlit collapses expanders by default, so their
    contents are in the open panel and undisplayed, and a visibility filter
    drops the very widgets a test means to count.
    """
    panel = (
        active_tab_panel(page, timeout=timeout)
        .locator('[role="tabpanel"]:visible')
        .first
    )
    panel.wait_for(state="visible", timeout=timeout)
    deadline = _time.monotonic() + timeout / 1000.0
    while _time.monotonic() < deadline:
        if (panel.inner_text() or "").strip():
            break
        page.wait_for_timeout(250)
    return panel


def panel_widget(page, testid: str, label: str, timeout: int = 20_000):
    """Locator for the Streamlit widget of ``testid`` labelled ``label``,
    scoped to the visible tab panel.

    A bare ``[data-testid="stSelectbox"]`` matches every selectbox the app
    renders, so asserting it is non-empty proves nothing about the tab under
    test. Naming the widget makes the assertion able to fail.
    """
    located = (
        active_tab_panel(page, timeout=timeout)
        .locator(f'[data-testid="{testid}"]')
        .filter(has_text=label)
    )
    # Wait for the widget itself: a panel can carry content while this
    # particular widget is still streaming. A genuine absence is the caller's
    # assertion to report, so a timeout here is not an error.
    try:
        located.first.wait_for(state="attached", timeout=timeout)
    except Exception:
        pass
    return located


def click_button(page, text: str):
    """Click a Streamlit button by text, excluding tab buttons.

    Uses JS click to bypass visibility checks. Excludes buttons with
    role="tab" to avoid accidentally clicking tabs instead of form buttons.

    Prefers a VISIBLE match. Streamlit renders every tab body, not just the
    selected one, and ``has-text`` is a case-insensitive substring, so a
    label routinely matches buttons in other panels: "Load" matches six
    buttons, of which the first is a hidden "Upload". Combined with the JS
    click, ``.first`` silently actuated the wrong widget and the calling
    test saw nothing happen.
    """
    start = _time.monotonic()
    btn = page.locator(f'button:not([role="tab"]):has-text("{text}")')
    visible = page.locator(f'button:not([role="tab"]):has-text("{text}"):visible')
    target = visible.first if visible.count() > 0 else btn.first
    if btn.count() > 0:
        target.evaluate("el => el.click()")
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
        # Setting the tenant reruns the whole script, which is the expensive
        # one: every tab body re-executes. Wait for that run to finish.
        wait_for_script_idle(page)

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


_TRACKED_E2E_VIDEO = (
    DATA_ROOT.parent / "tests" / "system" / "resources" / "videos" / "v_-D1gdv_gQyw.mp4"
)


def _atomic_artifact(dest: Path, writer) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
    try:
        writer(staged)
        if not staged.exists() or staged.stat().st_size == 0:
            raise RuntimeError(f"E2E artifact writer produced an empty file: {dest}")
        staged.replace(dest)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return dest


def _write_pdf_fixture(dest: Path, text: str) -> Path:
    lines = []
    for line in text.splitlines():
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines.append(f"({escaped}) Tj")
    content = "BT\n/F1 12 Tf\n72 720 Td\n14 TL\n" + "\nT*\n".join(lines) + "\nET\n"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, value in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n{value}\nendobj\n".encode("latin-1"))
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return _atomic_artifact(dest, lambda staged: staged.write_bytes(payload))


def _extract_image_fixture(source_video: Path, dest: Path) -> Path:
    if not source_video.exists():
        raise FileNotFoundError(f"E2E source video does not exist: {source_video}")

    def write_image(staged: Path) -> None:
        import av

        with av.open(str(source_video)) as container:
            for frame in container.decode(video=0):
                frame.to_image().save(staged, format="JPEG", quality=92)
                return
        raise RuntimeError(
            f"E2E source video contains no decodable frame: {source_video}"
        )

    return _atomic_artifact(dest, write_image)


def _extract_audio_fixture(
    source_video: Path, dest: Path, duration_seconds: int = 10
) -> Path:
    if not source_video.exists():
        raise FileNotFoundError(f"E2E source video does not exist: {source_video}")

    def write_audio(staged: Path) -> None:
        import wave

        import av
        import numpy as np

        target_rate = 16_000
        required_samples = target_rate * duration_seconds
        chunks: list[np.ndarray] = []
        collected = 0
        with av.open(str(source_video)) as container:
            audio_streams = [
                stream for stream in container.streams if stream.type == "audio"
            ]
            if len(audio_streams) != 1:
                raise RuntimeError(
                    f"E2E source video must contain exactly one audio stream: {source_video}"
                )
            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
            for frame in container.decode(audio_streams[0]):
                for resampled in resampler.resample(frame):
                    samples = resampled.to_ndarray().reshape(-1)
                    chunks.append(samples)
                    collected += samples.size
                    if collected >= required_samples:
                        break
                if collected >= required_samples:
                    break
        if collected < required_samples:
            raise RuntimeError(
                f"E2E source video yielded {collected} audio samples; "
                f"expected {required_samples}: {source_video}"
            )
        samples = np.concatenate(chunks)[:required_samples].astype(np.int16, copy=False)
        with wave.open(str(staged), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(target_rate)
            wav.writeframes(samples.tobytes())

    return _atomic_artifact(dest, write_audio)


@pytest.fixture(scope="session")
def real_document_path():
    path = DATA_ROOT / "testset" / "dataset_summary.md"
    if not path.exists():
        raise FileNotFoundError(f"E2E document does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def real_pdf_path(real_document_path):
    summary = real_document_path.read_text(encoding="utf-8")
    required_lines = (
        "Evaluation Dataset",
        "Video-ChatGPT Benchmark",
        "Provides 500 test videos from ActivityNet-200.",
    )
    for line in required_lines[:2]:
        if line not in summary:
            raise RuntimeError(f"Repository dataset summary is missing {line!r}")
    return _write_pdf_fixture(
        E2E_ARTIFACT_DIR / "evaluation_dataset.pdf",
        "\n".join(required_lines),
    )


@pytest.fixture(scope="session")
def real_video_path():
    if not _TRACKED_E2E_VIDEO.exists():
        raise FileNotFoundError(
            f"E2E source video does not exist: {_TRACKED_E2E_VIDEO}"
        )
    return _TRACKED_E2E_VIDEO


@pytest.fixture(scope="session")
def real_image_path(real_video_path):
    return _extract_image_fixture(
        real_video_path,
        E2E_ARTIFACT_DIR / "tracked_video_frame.jpg",
    )


@pytest.fixture(scope="session")
def extracted_audio_path(real_video_path):
    return _extract_audio_fixture(
        real_video_path,
        E2E_ARTIFACT_DIR / "tracked_video_audio.wav",
    )


E2E_REPORT_DIR = Path("/tmp")
E2E_REPORT_JSON = E2E_REPORT_DIR / "e2e_report.json"
E2E_REPORT_MD = E2E_REPORT_DIR / "e2e_report.md"


class E2EReportCollector:
    """Collects HTTP operations and test outcomes for E2E reporting.

    Automatically captures every httpx call to the runtime (localhost:33000)
    by monkeypatching httpx.Client.send. Groups operations by test name
    and writes JSON + markdown reports at session end.

    The in-memory operation log is hard-capped at ``MAX_OPERATIONS``;
    overflow is counted and surfaced in the report, never silently
    discarded.
    """

    # One record deep-sizes at ~2.7 KiB (polling loops record every poll,
    # so an hours-long sweep reaches 10^5-10^6 ops unbounded); 50k ops
    # holds the resident log near 135 MiB.
    MAX_OPERATIONS = 50_000

    def __init__(self):
        self.operations: list[dict] = []
        self.operations_dropped = 0
        self.test_results: dict[str, dict] = {}
        self._current_test: str | None = None
        self._original_send = None
        self._session_start = datetime.now(timezone.utc)
        self._ops_lock = threading.Lock()

    def _append_op(self, op: dict) -> None:
        with self._ops_lock:
            if len(self.operations) >= self.MAX_OPERATIONS:
                self.operations_dropped += 1
                return
            self.operations.append(op)

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
        self._append_op(
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

        self._append_op(
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
    def _extract_request_fields(body: dict | list | None, url: str) -> dict:
        """Extract semantically meaningful request fields based on endpoint."""
        if body is None:
            return {}
        # Some endpoints take a JSON array rather than an object (e.g. the
        # profile-selection ground-truth upload). This wraps every request the
        # suite makes, so an unhandled shape fails the fixture and errors the
        # whole module rather than one test.
        if isinstance(body, list):
            return {"type": "json_array", "items_count": len(body)}
        if not isinstance(body, dict):
            # A JSON body is any JSON type, not just an object.
            return {"type": "json_scalar", "value_type": type(body).__name__}
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
    def _extract_response_fields(
        body: dict | list | None, url: str, status_code: int
    ) -> dict:
        """Extract semantically meaningful response fields based on endpoint."""
        if body is None:
            return {"status_code": status_code}

        # Some endpoints return a list instead of a dict (e.g., /events/queues)
        if isinstance(body, list):
            return {"status_code": status_code, "items_count": len(body)}
        if not isinstance(body, dict):
            # A JSON body is any JSON type, not just an object.
            return {
                "status_code": status_code,
                "type": "json_scalar",
                "value_type": type(body).__name__,
            }

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
                "operations_dropped": self.operations_dropped,
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
        if s["operations_dropped"]:
            lines.append(
                f"**OPERATION LOG TRUNCATED**: {s['operations_dropped']} "
                f"operations dropped after cap of {self.MAX_OPERATIONS}; "
                "per-test operation tables are incomplete."
            )
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


def expected_initial_trust(kind: str, derivation_kind: DerivationKind) -> float:
    """The trust score a fresh ``kind`` write under ``derivation_kind`` persists.

    Written out from the shipped schema defaults and derivation weights rather
    than imported from the product, so an e2e expectation cannot drift along
    with the table it is meant to pin. Unknown keys raise.
    """
    default_trust = {
        "conversation_turn": 0.4,
        "learned_strategy": 0.6,
        "tenant_instruction": 0.95,
        "external_doc": 0.7,
        "entity_fact": 0.5,
        "kg_node": 0.6,
        "kg_edge": 0.6,
        "session_scratch": 0.3,
    }[kind]
    weight = {
        DerivationKind.DIRECT_INGEST: 1.20,
        DerivationKind.USER_ASSERT: 1.10,
        DerivationKind.EXTRACTION: 1.00,
        DerivationKind.SUMMARIZATION: 0.90,
        DerivationKind.SYNTHESIS: 0.85,
        DerivationKind.AGENT_INFERENCE: 0.70,
    }[derivation_kind]
    return min(1.0, max(0.0, default_trust * weight))


def pytest_configure(config):
    """Install the HTTP recording hook and Playwright browsers at session start."""
    global _report_collector
    config.addinivalue_line(
        "markers",
        "requires_modal_inference(service): require a live Modal provider",
    )
    config.addinivalue_line(
        "markers",
        "requires_telegram_bot: require a configured Telegram bot token and chat id",
    )
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


@dataclasses.dataclass(frozen=True)
class StudentLLM:
    api_base: str
    api_key: str
    model: str


def _runtime_deployment_env() -> dict[str, str]:
    command = _kubectl_e2e_command(
        "get",
        "deploy",
        "cogniverse-runtime",
        "-n",
        "cogniverse",
        "-o",
        'jsonpath={range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\\n"}{end}',
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    _require_kubectl_success(result, command)
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            env[name] = value
    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_models(api_base: str, headers: dict[str, str], deadline_s: float) -> None:
    deadline = _time.monotonic() + deadline_s
    last = ""
    while _time.monotonic() < deadline:
        try:
            r = httpx.get(f"{api_base}/models", headers=headers, timeout=10)
            if r.status_code == 200 and r.json().get("data"):
                return
            last = f"{r.status_code} {r.text[:120]}"
        except httpx.HTTPError as exc:
            last = repr(exc)
        _time.sleep(2)
    pytest.fail(
        f"student LLM at {api_base} listed no models within {deadline_s:g}s "
        f"(last: {last})"
    )


@pytest.fixture(scope="module")
def student_llm() -> Iterator[StudentLLM]:
    """The student LLM the deployed runtime uses, reachable from the host.

    LLM_ENDPOINT / LLM_MODEL come from the runtime Deployment. A Modal
    endpoint is used directly with its bearer and the runtime's boot
    deadline for a scale-from-zero start; an in-cluster vLLM service is
    reached through a kubectl port-forward that lives for the module.
    """
    env = _runtime_deployment_env()
    api_base, model = env["LLM_ENDPOINT"], env["LLM_MODEL"]
    root = endpoint_root(api_base)
    if is_modal_inference_url(root):
        api_key = read_secret(INFERENCE_API_KEY_ENV)
        if not api_key:
            pytest.fail(
                f"student LLM {api_base} is a Modal endpoint; "
                f"{INFERENCE_API_KEY_ENV} must be set in the environment or ./.env"
            )
        _wait_for_models(
            api_base,
            {"Authorization": f"Bearer {api_key}"},
            float(env["INFERENCE_HEALTH_BOOT_DEADLINE_SECONDS"]),
        )
        yield StudentLLM(api_base=api_base, api_key=api_key, model=model)
        return
    port = _free_port()
    command = _kubectl_e2e_command(
        "port-forward",
        "-n",
        "cogniverse",
        "svc/cogniverse-vllm-llm-student",
        f"{port}:8000",
    )
    proc = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    local = f"http://127.0.0.1:{port}/v1"
    try:
        _wait_for_models(local, {}, 30.0)
        yield StudentLLM(api_base=local, api_key="not-required", model=model)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def assert_telegram_chunks(chunks, response):
    """The formatted chunks must carry the agent's own content.

    ``len(chunks) >= 1`` cannot fail: format_agent_response returns at least
    one element on every branch, including its warming, unavailable, error and
    "No results found." branches, so it passed just as happily on a degraded
    reply as on a real answer.
    """
    from cogniverse_messaging.telegram_handler import MAX_MESSAGE_LENGTH

    assert len(chunks) == 1, (
        f"a top_k<=3 answer formats to a single chunk; got {len(chunks)} of "
        f"lengths {[len(chunk) for chunk in chunks]}"
    )
    assert all(0 < len(chunk) <= MAX_MESSAGE_LENGTH for chunk in chunks), chunks

    joined = "\n\n".join(chunks)
    assert response["message"] in joined, (
        f"the reply must carry the agent's message {response['message']!r}; "
        f"got {joined!r}"
    )
    for result in (response.get("results") or [])[:5]:
        title = (
            result.get("video_title")
            or result.get("title")
            or result.get("source_id", "Unknown")
        )
        assert title in joined, (
            f"the reply must list the returned result {title!r}; got {joined!r}"
        )
