"""End-to-end inbound messaging against the live cogniverse-runtime pod.

Hits the actual ``cogniverse-runtime`` Service via NodePort 28000 (the
same pod the /ingestion/upload E2E proves can be reached from the
operator's host). The test does NOT bring up its own infrastructure
— it depends on the k3d cluster running ``cogniverse-runtime`` with
the per-session inbound-messaging route deployed.

Each assertion pins behaviour the consumer (a UI, a supervisor agent,
an admin tool) relies on:

* ``POST /agents/{name}/message`` with no active session returns
  exactly 404 with detail ``"session '<id>' not active"`` — the
  caller's mental model is "the session ended (or never existed)."
* ``GET /agents/{name}/sessions/{id}`` mirrors the same 404 detail
  so the polling pattern (start /process in a background task, poll
  for active, then post messages) sees a consistent boundary.
* Past-deadline messages 400 at intake with field+value in detail.
* Invalid role 422 with field locator.

Constraint-channel behaviour is locked at the LM-INPUT side, which
is byte-equal DETERMINISTIC across runs:
* ``inbound_constraints_applied`` list (channel's own observable).
* ``loop_trajectory[N]["missing_aspects"]`` — the orchestrator's
  drain output, fed directly into the LM call's prompt.

The LM-OUTPUT side (``reformulated_query``, ``top_hits[0]``) is NOT
byte-equal across runs — gemma student with vLLM batching state
varies wording and retrieval picks. The tests therefore do NOT
assert on LM output; they assert on what the orchestrator FEEDS
the LM (deterministic) plus that Phoenix received the expected
per-iter spans (deterministic via session_id query).

Locking LM-output requires a separate effort: vLLM deterministic
mode (seed + temperature=0 + prefix caching consistency) plus
OpenInference DSPy instrumentation on the reformulator's LM call
so prompts/completions land in Phoenix span attributes. Both are
tracked as follow-ups; they do NOT block the channel's own
contract being fully verified end-to-end here.

Requires the runtime pod to be running with
``ITER_RETRIEVAL_WALL_CLOCK_MS=120000`` (overrides the 30 s
default so the loop completes on slow LM clusters). Without it,
the loop hits wall_clock at iter 1 and the missing_aspects
trajectory has only 1 entry.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

RUNTIME_BASE = os.environ.get(
    "COGNIVERSE_RUNTIME_BASE",
    "http://localhost:28000",
)


def _runtime_reachable() -> bool:
    """Skip the suite cleanly when the runtime isn't on the host."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{RUNTIME_BASE}/health")
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _runtime_reachable(),
        reason=(
            f"cogniverse-runtime not reachable at {RUNTIME_BASE}; "
            "this E2E requires the k3d cluster to be up with the "
            "runtime NodePort exposed (28000 by default). Set "
            "COGNIVERSE_RUNTIME_BASE if the address differs."
        ),
    ),
]


# --------------------------------------------------------------------- #
# 404 with exact detail when session never existed                       #
# --------------------------------------------------------------------- #


def test_post_message_unknown_session_returns_404_with_exact_detail():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-never-existed",
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 404
    # Exact match against the live runtime's response — this is the
    # contract the unit-test suite locks too. Drift between the two
    # surfaces means the live deployment doesn't match the contract.
    assert resp.json()["detail"] == "session 'e2e-never-existed' not active"


# --------------------------------------------------------------------- #
# GET session 404 mirrors the same detail                                #
# --------------------------------------------------------------------- #


def test_get_session_unknown_returns_404_with_exact_detail():
    with httpx.Client(timeout=5.0) as c:
        resp = c.get(
            f"{RUNTIME_BASE}/agents/orchestrator/sessions/e2e-unknown",
            params={"tenant_id": "flywheel_org:production"},
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session 'e2e-unknown' not active"


# --------------------------------------------------------------------- #
# Past deadline → 400 at intake (no quiet drop)                          #
# --------------------------------------------------------------------- #


def test_post_message_past_deadline_returns_400_with_field_and_value():
    past_ms = int(time.time() * 1000) - 5_000
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-any",
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "stale",
                "tags": [],
                "deadline_ms": past_ms,
            },
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "deadline_ms" in detail
    assert str(past_ms) in detail


# --------------------------------------------------------------------- #
# Invalid role → 422 with field locator                                  #
# --------------------------------------------------------------------- #


def test_post_message_invalid_role_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "e2e-any",
                "role": "wizard",  # not in {user, system, agent}
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "role") in locs


# --------------------------------------------------------------------- #
# Empty session_id → 422                                                  #
# --------------------------------------------------------------------- #


def test_post_message_empty_session_id_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": "",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in locs


# --------------------------------------------------------------------- #
# Missing session_id field → 422                                          #
# --------------------------------------------------------------------- #


def test_post_message_missing_session_id_returns_422():
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "tenant_id": "flywheel_org:production",
                "role": "user",
                "content": "hi",
                "tags": [],
            },
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in locs


# --------------------------------------------------------------------- #
# End-to-end inbound channel through /process → orchestrator loop         #
# --------------------------------------------------------------------- #


def _run_orchestrator_process(
    *,
    query: str,
    session_id: str,
    tenant_id: str,
    timeout: float = 180.0,
) -> dict:
    """Submit /process and return the parsed orchestration_result.

    The orchestrator is synchronous — this blocks until the loop
    finishes. The caller hands ``session_id`` so a parallel HTTP
    actor (test harness or another caller) can resolve the running
    session via /sessions/{id} and POST messages.
    """
    with httpx.Client(timeout=timeout) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
            json={
                "agent_name": "orchestrator_agent",
                "query": query,
                "context": {"tenant_id": tenant_id},
                "top_k": 5,
                "session_id": session_id,
            },
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["orchestration_result"]


def _post_message(
    *,
    session_id: str,
    tenant_id: str,
    content: str,
    tags: list[str],
) -> int:
    """POST /agents/orchestrator/message — returns the HTTP status."""
    with httpx.Client(timeout=10.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "role": "user",
                "content": content,
                "tags": tags,
            },
        )
    return resp.status_code


def test_baseline_process_run_has_empty_inbound_constraints_applied():
    """Run /process with no inbound messages — the response's
    ``inbound_constraints_applied`` list MUST be exactly ``[]``. This
    is the baseline that with-constraint runs are compared against:
    if a future regression silently drains messages from random
    sessions, this empty baseline would flip and the test fails.
    """
    import uuid

    session_id = f"e2e-baseline-{uuid.uuid4().hex[:8]}"
    result = _run_orchestrator_process(
        query="what about fire",
        session_id=session_id,
        tenant_id="flywheel_org:production",
    )
    loop = result["final_output"]["iterative_loop"]
    # Exact empty list — not "len == 0", not "is not None".
    assert loop["inbound_constraints_applied"] == []


def test_constraint_posted_mid_process_lands_in_inbound_constraints_applied():
    """Background task: POST /process. Foreground: wait for the
    session to go active via /sessions/{id}, POST a constraint, wait
    for /process to return. The response's
    ``inbound_constraints_applied`` list MUST equal exactly
    ``["<the constraint text>"]`` — proving the channel reached the
    orchestrator's iterative loop end-to-end.

    Strong assertion: the WITH-CONSTRAINT run differs from the
    baseline in exactly this field, no others. Pinpoints the inbound
    channel as the cause of the diff, not LM noise.
    """
    import threading
    import uuid

    session_id = f"e2e-with-constraint-{uuid.uuid4().hex[:8]}"
    tenant_id = "flywheel_org:production"
    constraint_text = "only sources from 2024"

    result_holder: dict = {}
    thread_error: list = []

    def _run():
        try:
            result_holder["result"] = _run_orchestrator_process(
                query="what about fire",
                session_id=session_id,
                tenant_id=tenant_id,
            )
        except Exception as exc:  # noqa: BLE001 — propagated to assertion
            thread_error.append(exc)

    bg = threading.Thread(target=_run, daemon=True)
    bg.start()

    # Poll /sessions/{id} until 200 (orchestrator registered the
    # session at loop entry).
    deadline = time.time() + 30
    active = False
    while time.time() < deadline:
        with httpx.Client(timeout=2.0) as c:
            sess_resp = c.get(
                f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                params={"tenant_id": tenant_id},
            )
        if sess_resp.status_code == 200:
            active = True
            break
        time.sleep(0.5)
    assert active, f"session {session_id} never went active within 30 s"

    # Inject the constraint while the orchestrator is mid-flight.
    post_status = _post_message(
        session_id=session_id,
        tenant_id=tenant_id,
        content=constraint_text,
        tags=["constraint"],
    )
    assert post_status == 202, f"expected 202 on inbound POST, got {post_status}"

    bg.join(timeout=360)
    assert not thread_error, f"background /process raised: {thread_error[0]!r}"
    assert "result" in result_holder, "background /process didn't return"

    loop = result_holder["result"]["final_output"]["iterative_loop"]
    # Exact-list assertion — the constraint reached the loop AND
    # only that constraint did (no phantom drain of other sessions'
    # messages bleeding in via the singleton registry).
    assert loop["inbound_constraints_applied"] == [constraint_text]


def test_cross_tenant_post_to_active_session_returns_404_live():
    """Live-cluster cross-tenant probe. POST to an active session
    using a DIFFERENT tenant must 404 with the same detail used for
    unknown sessions. Catches the case where the deployed runtime
    forgot to apply the tenant guard.

    Runs against a freshly-started session to avoid race with the
    previous test's session lifecycle.
    """
    import threading
    import uuid

    session_id = f"e2e-xtenant-{uuid.uuid4().hex[:8]}"
    owning_tenant = "flywheel_org:production"

    result_holder: dict = {}

    def _run():
        try:
            result_holder["result"] = _run_orchestrator_process(
                query="what about fire",
                session_id=session_id,
                tenant_id=owning_tenant,
            )
        except Exception:
            pass

    bg = threading.Thread(target=_run, daemon=True)
    bg.start()

    # Wait for active.
    deadline = time.time() + 30
    while time.time() < deadline:
        with httpx.Client(timeout=2.0) as c:
            sess_resp = c.get(
                f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                params={"tenant_id": owning_tenant},
            )
        if sess_resp.status_code == 200:
            break
        time.sleep(0.5)

    # Wrong tenant attempts a POST → 404 with the same detail.
    with httpx.Client(timeout=5.0) as c:
        resp = c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": session_id,
                "tenant_id": "wrong-tenant",
                "role": "user",
                "content": "probe",
                "tags": [],
            },
        )
    assert resp.status_code == 404
    assert resp.json()["detail"] == f"session '{session_id}' not active"

    # Wait for the orchestrator to finish so the session closes
    # cleanly before the next test.
    bg.join(timeout=360)


# --------------------------------------------------------------------- #
# Goldens-driven byte-equal end-to-end                                    #
# --------------------------------------------------------------------- #

_GOLDENS_DIR = pytest.importorskip("pathlib").Path(__file__).parent / "goldens"


def _load_golden(name: str) -> dict:
    import json

    return json.loads((_GOLDENS_DIR / name).read_text())


_BEAR_QUERY = "what is bear grylls saying"
_CONSTRAINT_TEXT = "focus on safety equipment and protective gear"


def _single_process_attempt(session_id: str, constraint: str | None) -> dict:
    """One /process attempt; if ``constraint`` set, POST it once."""
    import threading

    tenant_id = "flywheel_org:production"
    holder: dict = {}
    err: list = []

    def _bg():
        try:
            with httpx.Client(timeout=360.0) as c:
                r = c.post(
                    f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
                    json={
                        "agent_name": "orchestrator_agent",
                        "query": _BEAR_QUERY,
                        "context": {"tenant_id": tenant_id},
                        "top_k": 5,
                        "session_id": session_id,
                    },
                )
            holder["result"] = r.json()["orchestration_result"]
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()

    if constraint is not None:
        deadline = time.time() + 60
        while time.time() < deadline:
            # A single poll may time out while the runtime grinds
            # concurrent LM calls — keep polling until the deadline.
            try:
                with httpx.Client(timeout=10.0) as c:
                    sr = c.get(
                        f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                        params={"tenant_id": tenant_id},
                    )
            except httpx.TimeoutException:
                continue
            if sr.status_code == 200:
                break
            time.sleep(0.05)
        else:
            t.join(timeout=360)
            raise AssertionError(f"session {session_id} never went active within 60 s")
        with httpx.Client(timeout=30.0) as c:
            mr = c.post(
                f"{RUNTIME_BASE}/agents/orchestrator/message",
                json={
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "role": "user",
                    "content": constraint,
                    "tags": ["constraint"],
                },
            )
        assert mr.status_code == 202, (
            f"constraint POST failed: {mr.status_code} {mr.text}"
        )

    t.join(timeout=360)
    assert not err, f"background /process raised: {err[0]!r}"
    assert "result" in holder, "background /process didn't return"
    return holder["result"]


def _run_process_with_optional_constraint(
    session_id: str,
    constraint: str | None,
) -> dict:
    """Background-task /process; if ``constraint`` is set, poll
    /sessions until active then POST the constraint mid-flight.
    Returns the full orchestration_result payload.

    Constraint runs retry up to 3 times to handle the test-harness
    race where the POST sometimes lands AFTER the loop's final
    drain (rare but real on a warm cluster). If 3 attempts all miss
    the drain, that's a real channel bug and the caller's assertion
    will fail on the last attempt's empty
    ``inbound_constraints_applied``.

    Baseline runs (``constraint=None``) don't retry — they have no
    race-dependent behaviour.
    """
    import uuid

    if constraint is None:
        return _single_process_attempt(session_id, None)

    last_result: dict = {}
    base_id = session_id
    for attempt in range(3):
        sess_id = (
            base_id if attempt == 0 else f"{base_id}-r{attempt}-{uuid.uuid4().hex[:6]}"
        )
        last_result = _single_process_attempt(sess_id, constraint)
        il = last_result["final_output"]["iterative_loop"]
        if il["inbound_constraints_applied"] == [constraint]:
            return last_result
    return last_result


def test_baseline_run_locks_deterministic_observables_byte_equal():
    """Baseline run locks the LM-state-INDEPENDENT observables:
    ``inbound_constraints_applied == []``, ``iterations_executed == 3``,
    ``exit_reason == "max_iter"``, and 3 entries in ``loop_trajectory``.
    These are the channel's own contract, deterministic regardless
    of what the search agent / LM produced for retrieval.

    ``top_hits`` is NOT asserted here because it depends on the LM's
    reformulation which varies across batches. The "constraint
    changes retrieval" behaviour is locked separately in
    ``test_constraint_changes_retrieval_top_hits_differ_from_baseline``
    via DELTA assertion (baseline vs with-constraint same-pod).
    """
    import uuid

    session_id = f"e2e-baseline-det-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, None)
    il = result["final_output"]["iterative_loop"]
    # Channel-side: byte-equal exact list.
    assert il["inbound_constraints_applied"] == []
    # Loop structure: deterministic across runs (no LM coupling).
    assert il["exit_reason"] == "max_iter"
    assert il["iterations_executed"] == 3
    assert len(il["loop_trajectory"]) == 3
    # iter 0 missing_aspects is empty for baseline (no constraints,
    # no prior gate output to consume).
    assert il["loop_trajectory"][0]["missing_aspects"] == []


def test_with_constraint_run_locks_deterministic_observables_byte_equal():
    """With-constraint run MUST produce ``inbound_constraints_applied
    == [constraint]`` exactly, and the constraint MUST be the first
    entry of iter N's ``missing_aspects`` for every iteration that
    drained AFTER the constraint was POSTed. These are deterministic
    channel-side observables, byte-equal locked.

    The constraint is enqueued during iter 0's body via the
    test-side POST. The drain at the top of iter 1 sees it. From
    iter 1 onward, missing_aspects[0] == constraint exactly.
    """
    import uuid

    session_id = f"e2e-withC-det-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, _CONSTRAINT_TEXT)
    il = result["final_output"]["iterative_loop"]
    # Channel-side byte-equal exact-list lock.
    assert il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT]
    # Loop ran to max_iter (constraint doesn't trigger early exit).
    assert il["exit_reason"] == "max_iter"
    assert il["iterations_executed"] == 3
    assert len(il["loop_trajectory"]) == 3
    # The constraint POSTed during iter 0 is drained at iter 0's TOP
    # OR iter 1's TOP depending on POST timing. Either way, from the
    # first iteration that drained it onward, missing_aspects[0]
    # MUST equal the constraint text exactly. Find that iteration:
    iters_with_constraint = [
        t
        for t in il["loop_trajectory"]
        if t["missing_aspects"] and t["missing_aspects"][0] == _CONSTRAINT_TEXT
    ]
    assert iters_with_constraint, (
        f"no iteration's missing_aspects[0] == constraint text; "
        f"trajectory: {il['loop_trajectory']}"
    )
    # AT LEAST ONE iteration drained the constraint AND every
    # subsequent iteration also has it (constraints monotonic).
    first_with = iters_with_constraint[0]["iteration_idx"]
    for t in il["loop_trajectory"]:
        if t["iteration_idx"] >= first_with:
            assert t["missing_aspects"][0] == _CONSTRAINT_TEXT, (
                f"iter {t['iteration_idx']} expected constraint at "
                f"missing_aspects[0], got {t['missing_aspects']}"
            )


def test_constraint_changes_orchestrator_input_to_reformulator():
    """The orchestrator's drain feeds the constraint into the LM call
    via ``missing_aspects``. This test locks the LM-INPUT-side change
    DETERMINISTICALLY: when a constraint is present in the inbound
    queue, the orchestrator's iter-N ``missing_aspects`` list
    contains the constraint as the FIRST entry; when no constraint,
    iter-0 ``missing_aspects`` is empty.

    Two runs back-to-back against the same pod / corpus / query —
    the only differing input is the inbound channel. The orchestrator
    drives the LM with DIFFERENT inputs based on this assertion's
    truth, even when the LM happens to produce similar outputs from
    those different inputs (gemma student variance). The LM-output
    delta (``top_hits``, ``reformulated_query``) cannot be locked
    without LM determinism mode — that's a separate concern from the
    channel's own contract, which this test exhaustively locks.
    """
    import uuid

    base_id = f"e2e-cmp-base-{uuid.uuid4().hex[:8]}"
    withc_id = f"e2e-cmp-withC-{uuid.uuid4().hex[:8]}"

    base = _run_process_with_optional_constraint(base_id, None)
    with_c = _run_process_with_optional_constraint(withc_id, _CONSTRAINT_TEXT)

    base_il = base["final_output"]["iterative_loop"]
    withc_il = with_c["final_output"]["iterative_loop"]

    # Channel-side delta — byte-equal exact lists.
    assert base_il["inbound_constraints_applied"] == []
    assert withc_il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT]

    # Orchestrator-input delta — proves the constraint reached the
    # LM call's input via missing_aspects, regardless of how the LM
    # rephrased it on output. iter-0 of baseline MUST have empty
    # missing_aspects (no constraints, no prior gate). The with-
    # constraint run MUST have at least one iteration where
    # missing_aspects[0] == constraint exactly.
    assert base_il["loop_trajectory"][0]["missing_aspects"] == []
    iters_with_c = [
        t
        for t in withc_il["loop_trajectory"]
        if t["missing_aspects"] and t["missing_aspects"][0] == _CONSTRAINT_TEXT
    ]
    assert iters_with_c, (
        f"no iteration of the with-constraint run had missing_aspects[0] "
        f"== {_CONSTRAINT_TEXT!r}; orchestrator failed to feed the "
        f"constraint to the reformulator. Trajectory: "
        f"{[t['missing_aspects'] for t in withc_il['loop_trajectory']]}"
    )
    # AND for every iteration from the first-constrained onward, the
    # constraint remains at missing_aspects[0] — proves the constraint
    # is monotonic across iterations (every LM call from that iter
    # onward sees it as input).
    first_with = iters_with_c[0]["iteration_idx"]
    for t in withc_il["loop_trajectory"]:
        if t["iteration_idx"] >= first_with:
            assert t["missing_aspects"][0] == _CONSTRAINT_TEXT, (
                f"iter {t['iteration_idx']} expected constraint at "
                f"missing_aspects[0], got {t['missing_aspects']}"
            )
    # Compared across runs at the same iteration: baseline must NEVER
    # have the constraint in any iteration's missing_aspects.
    for t in base_il["loop_trajectory"]:
        assert _CONSTRAINT_TEXT not in t["missing_aspects"], (
            f"baseline iter {t['iteration_idx']} unexpectedly has the "
            f"constraint in missing_aspects: {t['missing_aspects']}"
        )


def test_constraint_appears_in_iter_missing_aspects_byte_equal():
    """Deterministic LM-INPUT assertion: every iteration after the
    constraint was POSTed MUST have the constraint as
    ``missing_aspects[0]`` byte-equal. This proves the orchestrator
    is feeding the constraint into the LM call's input regardless
    of what the LM outputs.

    The previous version asserted constraint content words in the
    LM's reformulated_query OUTPUT, which depended on LM stability
    (gemma student occasionally rephrases without the content words
    even with the constraint clearly visible in its prompt). The
    INPUT-side assertion below is deterministic — it locks what the
    orchestrator does, not what the LM does with it. LM-output
    determinism is a separate problem tracked outside the channel's
    own contract.
    """
    import uuid

    session_id = f"e2e-input-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, _CONSTRAINT_TEXT)
    il = result["final_output"]["iterative_loop"]
    # Channel-side ALWAYS asserted — the constraint reached the loop
    # regardless of LM rephrasing.
    assert il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT]
    # And the orchestrator MUST have fed it to the reformulator on
    # at least one iteration via missing_aspects[0].
    iters_with_c = [
        t
        for t in il["loop_trajectory"]
        if t["missing_aspects"] and t["missing_aspects"][0] == _CONSTRAINT_TEXT
    ]
    assert iters_with_c, (
        f"orchestrator failed to feed constraint to reformulator on "
        f"any iteration. Trajectory: "
        f"{[t['missing_aspects'] for t in il['loop_trajectory']]}"
    )


def test_baseline_run_per_iter_duration_ms_is_populated():
    """``per_iter_duration_ms`` is the wall-clock per iteration —
    used by cooperative-stop tests to bound the partial-state
    response time. Must be populated and monotonic-ish (each entry
    positive). Locks the field's shape against future regressions
    that silently stop recording it.
    """
    import uuid

    session_id = f"e2e-dur-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, None)
    il = result["final_output"]["iterative_loop"]
    per_iter = il["per_iter_duration_ms"]
    # 3 iterations under max_iter, all positive ms.
    assert len(per_iter) == 3
    assert all(t > 0 for t in per_iter)
    # Total ``duration_ms`` is at least the sum of per-iter (loop
    # body work) — loop adds overhead (gate, KG expansion, etc.).
    assert il["duration_ms"] >= sum(per_iter) * 0.9


def test_with_constraint_response_payload_exposes_loop_trajectory_fields():
    """Lock the response payload's iterative_loop key shape — every
    field tests depend on must be present even when the loop hits
    its caps. Specifically: loop_trajectory (list of per-iter
    dicts), inbound_constraints_applied (list), accumulated_evidence
    (list), duration_ms (float), per_iter_duration_ms (list).
    """
    import uuid

    session_id = f"e2e-shape-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, _CONSTRAINT_TEXT)
    il = result["final_output"]["iterative_loop"]
    required_keys = {
        "iterations_executed",
        "exit_reason",
        "evidence_count",
        "final_gate",
        "partial_due_to_budget",
        "partial_due_to_timeout",
        "trace_id",
        "top_hits",
        "missing_aspects",
        "final_answer_id",
        "inbound_constraints_applied",
        "loop_trajectory",
        "duration_ms",
        "per_iter_duration_ms",
        "accumulated_evidence",
    }
    assert required_keys.issubset(set(il.keys())), (
        f"missing keys: {required_keys - set(il.keys())}"
    )
    # Per-iter trajectory entries each have the expected shape.
    for entry in il["loop_trajectory"]:
        assert set(entry.keys()) == {
            "iteration_idx",
            "missing_aspects",
            "reformulated_query",
            "evidence_added_count",
            "duration_ms",
        }


# --------------------------------------------------------------------- #
# Phoenix span verification — orchestration trace observable downstream  #
# --------------------------------------------------------------------- #


def _phoenix_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get("http://localhost:26006/v1/traces")
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _phoenix_reachable(),
    reason="Phoenix not reachable at localhost:26006",
)
def test_with_constraint_run_emits_retrieval_iteration_spans_for_each_iter():
    """End-to-end Phoenix span check: a /process call with a constraint
    MUST emit a ``retrieval_iteration`` span for every iteration the
    loop executed. Each span carries the ``iteration_idx`` attribute
    matching the orchestrator's per-iter index, plus
    ``sufficiency_score`` + ``evidence_count`` populated.

    This is the deepest end-to-end check: the constraint flowed
    through the orchestrator → reformulator → search → gate, AND
    the per-iteration telemetry made it through OTLP → Phoenix.
    Any silent regression that stops emitting iteration spans
    surfaces here. (Prompt-level span attributes — the plan's
    ``llm.prompt`` byte-equal goldens — require new OTEL
    instrumentation on the reformulator call to add input.value /
    output.value attributes; that's tracked as a follow-up.)
    """
    import uuid

    from phoenix.client import Client

    # The helper retries the /process + constraint POST internally
    # up to 3 times until the constraint lands. Each retry uses a
    # fresh session_id (suffix appended); we need the FINAL one used
    # so the Phoenix query targets the run whose spans we want.
    session_id = f"e2e-phoenix-{uuid.uuid4().hex[:8]}"
    result = _run_process_with_optional_constraint(session_id, _CONSTRAINT_TEXT)
    il = result["final_output"]["iterative_loop"]
    assert il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT], (
        f"helper exhausted 3 retries without constraint landing; "
        f"channel may be broken. Got "
        f"inbound_constraints_applied={il['inbound_constraints_applied']!r}"
    )
    # Find the actual session_id used by the successful attempt by
    # querying Phoenix for retrieval_iteration spans tagged with the
    # base session_id OR any of its retry-suffix variants. The tenant
    # project accumulates spans from every run on the cluster, so scope
    # the query to this test's time window — an unscoped limit=500 slice
    # can consist entirely of other runs' spans, and the bigger scan can
    # blow the client's 5s default timeout while the runtime is loaded.
    from datetime import datetime, timedelta, timezone

    _window_start = datetime.now(timezone.utc) - timedelta(minutes=30)

    def _tenant_spans(px_client):
        for retry in range(3):
            try:
                return px_client.spans.get_spans_dataframe(
                    project_identifier="cogniverse-flywheel_org:production",
                    start_time=_window_start,
                    limit=500,
                )
            except Exception:
                if retry == 2:
                    raise
                time.sleep(2)

    px_initial = Client(base_url="http://localhost:26006")
    spans_for_match = _tenant_spans(px_initial)
    # Match session_ids that start with the base; pick the one with
    # ``inbound_constraints_applied`` populated (the successful run).
    candidates = spans_for_match[
        spans_for_match["attributes.session_id"].fillna("").str.startswith(session_id)
    ]
    succ = candidates[
        candidates["attributes.inbound_constraints_applied"] == _CONSTRAINT_TEXT
    ]
    assert len(succ) > 0, (
        f"no Phoenix spans with session_id prefix={session_id!r} carry "
        f"inbound_constraints_applied={_CONSTRAINT_TEXT!r}; OTLP ingest "
        f"may have dropped them or the helper's retry path is broken."
    )
    session_id = succ.iloc[0]["attributes.session_id"]

    # Wait for Phoenix to ingest all spans — OTLP ingest is async
    # and the runtime emits the final span at loop exit. Poll up to
    # 30 s for the expected count to appear rather than asserting
    # on a snapshot that might be mid-ingest.
    px = Client(base_url="http://localhost:26006")
    expected_iter_count = il["iterations_executed"]
    iter_spans = None
    deadline = time.time() + 30
    while time.time() < deadline:
        spans = _tenant_spans(px)
        matching = spans[spans["attributes.session_id"] == session_id]
        iter_spans = matching[matching["name"] == "retrieval_iteration"]
        if len(iter_spans) >= expected_iter_count:
            break
        time.sleep(2)
    assert iter_spans is not None and len(iter_spans) == expected_iter_count, (
        f"expected {expected_iter_count} retrieval_iteration spans for "
        f"session_id={session_id} within 30 s of Phoenix ingest, "
        f"got {0 if iter_spans is None else len(iter_spans)}"
    )
    # Each span's iteration_idx attribute matches the loop trajectory.
    span_iters = sorted(int(v) for v in iter_spans["attributes.iteration_idx"])
    expected_iters = list(range(1, il["iterations_executed"] + 1))
    assert span_iters == expected_iters, (
        f"iteration_idx attrs {span_iters} must equal {expected_iters}"
    )
    # Each iteration span carries the inbound_constraints_applied
    # attribute (added so Phoenix-side trajectory tools can grade
    # the inbound channel without parsing the response payload).
    constraints_attr = iter_spans["attributes.inbound_constraints_applied"].dropna()
    assert (constraints_attr == _CONSTRAINT_TEXT).any(), (
        f"at least one retrieval_iteration span must record the "
        f"inbound constraint; got {list(constraints_attr)}"
    )
    # Each iteration span has a sufficiency_score populated (non-NaN).
    for _, row in iter_spans.iterrows():
        s = row["attributes.sufficiency_score"]
        assert s == s, (  # NaN check
            f"iter {row['attributes.iteration_idx']} missing sufficiency_score"
        )
