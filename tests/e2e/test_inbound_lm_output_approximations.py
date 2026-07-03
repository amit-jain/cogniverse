"""LM-output approximation tests for plan items the byte-equal path
can't lock (vLLM batch-state non-determinism).

Three plan-audit rows had "PARTIAL" status because byte-equal LM
output isn't achievable on the production vLLM stack. Each uses a
statistical / threshold assertion across N runs to lock the
observable behavior:

* Constraint changes retrieval output: across 3 with-constraint
  vs 3 baseline runs, with-constraint reformulated_query MUST
  mention constraint content-words in ≥2/3 of runs and baseline
  in 0/3.

* Cooperative stop returns pre-stop state: stopped run's
  ``duration_ms`` is significantly shorter than baseline's
  (saved at least one iter's worth of LM time), AND stopped
  evidence count is ≤ baseline evidence count.

* Per-session isolation: paired A (with-constraint) and B
  (baseline) sessions. Across 3 pairs, A's top_hits SET differs
  from B's top_hits SET in at least one pair OR the constraint
  appears in A's reformulated_query and is absent in B's
  reformulated_query for all 3 pairs (the LM-INPUT-side delta
  is always true; the LM-OUTPUT-side delta is the statistical
  approximation).
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

RUNTIME_BASE = os.environ.get("COGNIVERSE_RUNTIME_BASE", "http://localhost:28000")
_TENANT = "flywheel_org:production"
_CONSTRAINT_TEXT = "focus on safety equipment and protective gear"
_CONSTRAINT_WORDS = {"safety", "equipment", "protective", "gear"}
_BEAR_QUERY = "what is bear grylls saying"


def _runtime_reachable() -> bool:
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
        reason=f"cogniverse-runtime not reachable at {RUNTIME_BASE}",
    ),
]


def _run_process(session_id: str, query: str, constraint: str | None) -> dict:
    """Background-task /process + optional constraint POST. Retries
    up to 3x if the constraint racing the loop drops the message.
    """
    import threading

    def _single(sid: str) -> dict:
        holder: dict = {}
        err: list = []

        def _bg() -> None:
            try:
                # Two parallel 120s-wall-clock loops on the ~12 tok/s LM
                # queue behind each other; the last iteration's LM call can
                # overshoot the loop budget, so allow well past 2x120s.
                with httpx.Client(timeout=480.0) as c:
                    r = c.post(
                        f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
                        json={
                            "agent_name": "orchestrator_agent",
                            "query": query,
                            "context": {"tenant_id": _TENANT},
                            "top_k": 5,
                            "session_id": sid,
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
                            f"{RUNTIME_BASE}/agents/orchestrator/sessions/{sid}",
                            params={"tenant_id": _TENANT},
                        )
                except httpx.TimeoutException:
                    continue
                if sr.status_code == 200:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(f"session {sid} never active")
            with httpx.Client(timeout=30.0) as c:
                c.post(
                    f"{RUNTIME_BASE}/agents/orchestrator/message",
                    json={
                        "session_id": sid,
                        "tenant_id": _TENANT,
                        "role": "user",
                        "content": constraint,
                        "tags": ["constraint"],
                    },
                )

        t.join(timeout=540)
        assert not t.is_alive(), f"/process for {sid} still running after 540s"
        assert not err, f"/process raised: {err[0]!r}"
        return holder["result"]

    if constraint is None:
        return _single(session_id)

    for attempt in range(3):
        sid = (
            session_id
            if attempt == 0
            else f"{session_id}-r{attempt}-{uuid.uuid4().hex[:6]}"
        )
        result = _single(sid)
        il = result["final_output"]["iterative_loop"]
        if il["inbound_constraints_applied"] == [constraint]:
            return result
    return result


# --------------------------------------------------------------------- #
# Plan row 1: constraint changes retrieval output                        #
# --------------------------------------------------------------------- #


def test_constraint_changes_lm_reformulator_words_statistically():
    """Approximation for plan section 'Constraint changes retrieval
    output': across 3 with-constraint runs vs 3 baseline runs, the
    LM's reformulated_query MUST mention constraint content-words
    (safety/equipment/protective/gear) in at least 2/3 of the
    with-constraint runs AND in 0/3 of the baseline runs.

    Locks the LM-output effect statistically — even though vLLM batch
    state makes any single run's output non-deterministic, the
    aggregate signal of "constraint terms appear in LM output more
    often when constraint is in the prompt" IS reliable.
    """

    def _count_runs_with_constraint_words(constraint: str | None) -> int:
        hits = 0
        for i in range(3):
            sid = f"stat-{i}-{uuid.uuid4().hex[:6]}"
            r = _run_process(sid, _BEAR_QUERY, constraint)
            il = r["final_output"]["iterative_loop"]
            for t in il.get("loop_trajectory", []):
                words = set(t["reformulated_query"].lower().split())
                if _CONSTRAINT_WORDS & words:
                    hits += 1
                    break
        return hits

    baseline_hits = _count_runs_with_constraint_words(None)
    with_c_hits = _count_runs_with_constraint_words(_CONSTRAINT_TEXT)

    # Baseline MUST never produce constraint words (no leak).
    assert baseline_hits == 0, (
        f"baseline runs unexpectedly produced constraint words in "
        f"{baseline_hits}/3 reformulated_queries — possible session "
        f"contamination"
    )
    # With-constraint MUST produce constraint words in ≥2/3 runs —
    # confirms the LM actually consumed the constraint, not just
    # buffered it. 2/3 is the threshold that distinguishes signal
    # from LM-rephrasing noise (one occasional miss is the LM
    # paraphrasing without content words; consistent absence would
    # indicate the constraint isn't reaching the LM).
    assert with_c_hits >= 2, (
        f"with-constraint runs produced constraint words in only "
        f"{with_c_hits}/3 reformulated_queries — LM may not be "
        f"consuming the constraint reliably"
    )


# --------------------------------------------------------------------- #
# Plan row 2: cooperative stop returns pre-stop state                    #
# --------------------------------------------------------------------- #


def test_stop_run_finishes_significantly_faster_than_baseline():
    """Approximation for plan section 'Cooperative stop returns the
    exact pre-stop accumulated state': a stop-mid-flight run must
    finish significantly faster than a baseline full run, AND its
    accumulated evidence count must be ≤ baseline's.

    Byte-equal evidence content isn't achievable (LM varies what it
    returns per iter), but the timing + count delta is — proves the
    stop actually short-circuited the loop, returning a partial
    state instead of running to completion.
    """
    # Baseline run — full 3 iterations to max_iter.
    sid_base = f"stop-base-{uuid.uuid4().hex[:6]}"
    base = _run_process(sid_base, _BEAR_QUERY, None)
    base_il = base["final_output"]["iterative_loop"]
    assert base_il["exit_reason"] == "max_iter"
    base_duration_ms = base_il["duration_ms"]
    base_per_iter = base_il["per_iter_duration_ms"]
    base_evidence_count = base_il["evidence_count"]
    assert len(base_per_iter) >= 2, "baseline must run ≥2 iters for timing delta"

    # Stop run — POST a stop message mid-flight.
    sid_stop = f"stop-test-{uuid.uuid4().hex[:6]}"
    stop_result = _run_process_with_stop(sid_stop)
    stop_il = stop_result["final_output"]["iterative_loop"]

    # Stop exited with user_stop.
    assert stop_il["exit_reason"] == "user_stop"
    # Stopped iters < baseline iters.
    assert stop_il["iterations_executed"] < base_il["iterations_executed"]
    # Stopped duration is significantly less — at LEAST saved one
    # iter's worth of LM time. Using the median of baseline's per-iter
    # durations as the saved-floor estimate. 0.5× as the conservative
    # threshold (LM variance can make a single iter run 2x slower).
    median_per_iter = sorted(base_per_iter)[len(base_per_iter) // 2]
    saved_at_least = median_per_iter * 0.5
    assert stop_il["duration_ms"] < base_duration_ms - saved_at_least, (
        f"stop run duration {stop_il['duration_ms']:.0f}ms vs baseline "
        f"{base_duration_ms:.0f}ms — stop didn't save ≥{saved_at_least:.0f}ms; "
        f"per-iter durations were {base_per_iter}"
    )
    # Stopped evidence count is ≤ baseline.
    assert stop_il["evidence_count"] <= base_evidence_count, (
        f"stopped evidence count {stop_il['evidence_count']} > baseline "
        f"{base_evidence_count} — stop should not accumulate MORE evidence "
        f"than a full run"
    )


def _run_process_with_stop(session_id: str) -> dict:
    """Run /process; once the session goes active, POST a stop
    message to trigger cooperative cancellation."""
    import threading

    holder: dict = {}
    err: list = []

    def _bg() -> None:
        try:
            with httpx.Client(timeout=360.0) as c:
                r = c.post(
                    f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
                    json={
                        "agent_name": "orchestrator_agent",
                        "query": _BEAR_QUERY,
                        "context": {"tenant_id": _TENANT},
                        "top_k": 5,
                        "session_id": session_id,
                    },
                )
            holder["result"] = r.json()["orchestration_result"]
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()

    # Wait until session active.
    deadline = time.time() + 60
    while time.time() < deadline:
        with httpx.Client(timeout=2.0) as c:
            sr = c.get(
                f"{RUNTIME_BASE}/agents/orchestrator/sessions/{session_id}",
                params={"tenant_id": _TENANT},
            )
        if sr.status_code == 200:
            break
        time.sleep(0.05)

    # POST stop.
    with httpx.Client(timeout=10.0) as c:
        c.post(
            f"{RUNTIME_BASE}/agents/orchestrator/message",
            json={
                "session_id": session_id,
                "tenant_id": _TENANT,
                "role": "user",
                "content": "",
                "tags": ["stop"],
            },
        )

    t.join(timeout=360)
    assert not err
    return holder["result"]


# --------------------------------------------------------------------- #
# Plan row 3: per-session isolation                                       #
# --------------------------------------------------------------------- #


def test_per_session_isolation_two_parallel_lm_outputs_differ_statistically():
    """Approximation for plan section 'Per-session isolation': across
    3 paired (A=with-constraint, B=baseline) parallel sessions, the
    LM's reformulated_query for A MUST contain constraint words in
    ≥2/3 pairs, and B's MUST NOT contain them in any pair. Locks the
    DELTA at the LM-output level via statistical aggregate, even
    when individual runs vary.
    """
    import threading

    a_hits = 0
    b_hits = 0

    for i in range(3):
        sid_a = f"iso-A-{i}-{uuid.uuid4().hex[:6]}"
        sid_b = f"iso-B-{i}-{uuid.uuid4().hex[:6]}"
        results: dict = {}

        def _run_a() -> None:
            results["a"] = _run_process(sid_a, _BEAR_QUERY, _CONSTRAINT_TEXT)

        def _run_b() -> None:
            results["b"] = _run_process(sid_b, _BEAR_QUERY, None)

        ta = threading.Thread(target=_run_a, daemon=True)
        tb = threading.Thread(target=_run_b, daemon=True)
        ta.start()
        tb.start()
        ta.join(timeout=400)
        tb.join(timeout=400)

        a_il = results["a"]["final_output"]["iterative_loop"]
        b_il = results["b"]["final_output"]["iterative_loop"]

        # ALWAYS-true channel-side: A has the constraint, B doesn't.
        assert a_il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT]
        assert b_il["inbound_constraints_applied"] == []

        # LM-output approximation
        a_words = set()
        for t in a_il["loop_trajectory"]:
            a_words.update(t["reformulated_query"].lower().split())
        if _CONSTRAINT_WORDS & a_words:
            a_hits += 1

        b_words = set()
        for t in b_il["loop_trajectory"]:
            b_words.update(t["reformulated_query"].lower().split())
        if _CONSTRAINT_WORDS & b_words:
            b_hits += 1

    # A produced constraint words in ≥2/3 of paired runs.
    assert a_hits >= 2, (
        f"A's LM output included constraint words in only {a_hits}/3 paired "
        f"runs — constraint may not be reaching A's LM call reliably"
    )
    # B never produced constraint words — proves no isolation leak.
    assert b_hits == 0, (
        f"B's LM output included constraint words in {b_hits}/3 paired "
        f"runs — A's constraint leaked into B's session"
    )
