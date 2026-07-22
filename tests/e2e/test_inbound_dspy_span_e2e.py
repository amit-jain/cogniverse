"""Phoenix DSPy LM-span byte-equal assertions.

Closes the last LM-OUTPUT-side plan gap: when a constraint is
injected, it MUST appear in the LM's prompt (input.value) on the
iteration that drained the constraint. This is byte-equal
DETERMINISTIC because the orchestrator constructs the prompt
from ``missing_aspects`` which is byte-equal deterministic.

Requires the runtime pod to be running with:
* ``OPENINFERENCE_DSPY=1`` env var
* ``openinference-instrumentation-dspy`` installed (in the runtime image)
* DSPy re-instrumented with the cogniverse-dspy-instrumentation
  Phoenix tracer (done in main.py's lifespan)

The runtime emits DSPy LM spans to project ``cogniverse-dspy-
instrumentation`` with ``input.value`` (the full prompt) and
``output.value`` (the completion). Tests query the project, filter
to spans whose ``input.value`` contains the unique per-test query,
and assert the constraint text appears in the with-constraint run
AND is absent from the baseline run.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx
import pytest

RUNTIME_BASE = os.environ.get("COGNIVERSE_RUNTIME_BASE", "http://localhost:33000")
PHOENIX_BASE = os.environ.get("COGNIVERSE_PHOENIX_BASE", "http://localhost:33006")
_TENANT = "flywheel_org:production"
_CONSTRAINT_TEXT = "focus on safety equipment and protective gear"


def _runtime_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{RUNTIME_BASE}/health")
        return r.status_code == 200
    except Exception:
        return False


def _phoenix_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{PHOENIX_BASE}/v1/traces")
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (_runtime_reachable() and _phoenix_reachable()),
        reason=(
            f"requires cogniverse-runtime at {RUNTIME_BASE} AND Phoenix at "
            f"{PHOENIX_BASE}"
        ),
    ),
]


def _run_process(session_id: str, query: str, constraint: str | None) -> dict:
    """Run /process with optional constraint POST. Returns the
    orchestration_result. Retries up to 3x if the constraint fails
    to land (rare race when LM is hot)."""
    import threading

    def _bg(holder: dict, err: list) -> None:
        try:
            with httpx.Client(timeout=360.0) as c:
                r = c.post(
                    f"{RUNTIME_BASE}/agents/orchestrator_agent/process",
                    json={
                        "agent_name": "orchestrator_agent",
                        "query": query,
                        "context": {"tenant_id": _TENANT},
                        "top_k": 5,
                        "session_id": session_id,
                    },
                )
            holder["result"] = r.json()["orchestration_result"]
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    holder: dict = {}
    err: list = []
    t = threading.Thread(target=_bg, args=(holder, err), daemon=True)
    t.start()

    if constraint is not None:
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
        else:
            raise AssertionError(f"session {session_id} never active")
        with httpx.Client(timeout=10.0) as c:
            mr = c.post(
                f"{RUNTIME_BASE}/agents/orchestrator/message",
                json={
                    "session_id": session_id,
                    "tenant_id": _TENANT,
                    "role": "user",
                    "content": constraint,
                    "tags": ["constraint"],
                },
            )
        assert mr.status_code == 202, f"constraint POST failed: {mr.text}"

    t.join(timeout=360)
    assert not err, f"/process raised: {err[0]!r}"
    return holder["result"]


def _query_dspy_lm_spans_with_text(text: str, timeout_s: float = 30.0) -> list:
    """Query Phoenix DSPy LM spans whose input.value contains ``text``.

    Polls up to ``timeout_s`` for OTLP ingest. Returns a list of
    spans (each a dict with name + attributes).
    """
    from datetime import datetime, timedelta, timezone

    from phoenix.client import Client

    px = Client(base_url=PHOENIX_BASE)
    # The shared instrumentation project accumulates spans across every
    # run on the cluster; without a time window the unsorted `limit`
    # slice can consist entirely of historic spans and hide this run's.
    window_start = datetime.now(timezone.utc) - timedelta(minutes=30)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            spans = px.spans.get_spans_dataframe(
                project_identifier="cogniverse-dspy-instrumentation",
                start_time=window_start,
                limit=500,
                timeout=90,
            )
        except Exception:
            time.sleep(0.5)
            continue
        if len(spans) == 0:
            time.sleep(0.5)
            continue
        # LM.__call__ spans have the full prompt in input.value
        lm_spans = spans[spans["name"] == "LM.__call__"]
        matching = lm_spans[
            lm_spans["attributes.input.value"]
            .fillna("")
            .str.contains(text, na=False, regex=False)
        ]
        if len(matching) > 0:
            return [
                {
                    "input": row["attributes.input.value"],
                    "output": row["attributes.output.value"],
                    "name": row["name"],
                    "span_id": row["context.span_id"],
                }
                for _, row in matching.iterrows()
            ]
        time.sleep(0.5)
    return []


# --------------------------------------------------------------------- #
# Phase-1 plan-section 7: RLM injection measurably changes LM prompt    #
# --------------------------------------------------------------------- #


def test_with_constraint_run_appears_in_dspy_lm_span_input_byte_equal():
    """End-to-end LM-prompt-level proof: when a constraint is POSTed
    mid-flight, the orchestrator MUST feed it to a DSPy LM call whose
    ``input.value`` Phoenix span attribute contains the constraint
    text byte-equal.

    Uses a unique-per-run query string as the anchor so the test's
    spans are findable even when many concurrent test runs share the
    Phoenix project.
    """
    # Unique anchor — the query string uniquely identifies this test's
    # spans across the shared cogniverse-dspy-instrumentation project.
    unique_id = uuid.uuid4().hex[:12]
    query = f"unique-test-marker-{unique_id} what is bear grylls saying"
    session_id = f"e2e-dspy-{unique_id}"

    # With-constraint run. Retry up to 3x if constraint POST races.
    result = None
    for attempt in range(3):
        sid = (
            session_id
            if attempt == 0
            else f"{session_id}-r{attempt}-{uuid.uuid4().hex[:6]}"
        )
        result = _run_process(sid, query, _CONSTRAINT_TEXT)
        il = result["final_output"]["iterative_loop"]
        if il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT]:
            break
    il = result["final_output"]["iterative_loop"]
    assert il["inbound_constraints_applied"] == [_CONSTRAINT_TEXT], (
        f"constraint never landed in 3 retries; "
        f"got {il['inbound_constraints_applied']!r}"
    )

    # Query Phoenix DSPy spans that mention the unique query anchor.
    spans = _query_dspy_lm_spans_with_text(unique_id, timeout_s=30.0)
    assert spans, (
        f"no DSPy LM spans found containing unique anchor {unique_id!r}; "
        f"runtime may not have OPENINFERENCE_DSPY=1 enabled or DSPy "
        f"re-instrumentation against Phoenix tracer failed"
    )

    # At least one DSPy LM span's input.value MUST contain the
    # constraint text byte-equal. This is the strong "constraint
    # reached the LM's prompt" assertion — the LM-INPUT side is
    # deterministic regardless of LM-OUTPUT variability.
    spans_with_constraint = [s for s in spans if _CONSTRAINT_TEXT in str(s["input"])]
    assert spans_with_constraint, (
        f"no DSPy LM span input.value contained constraint text "
        f"{_CONSTRAINT_TEXT!r}; inputs were: "
        f"{[str(s['input'])[:200] for s in spans[:3]]}"
    )

    # Specifically, the JSON-encoded input MUST have the constraint
    # in a messages-list user content field (not as random substring
    # in some other LM call). Parse the JSON and confirm.
    found_in_message = False
    for s in spans_with_constraint:
        try:
            inp = json.loads(s["input"])
        except Exception:
            continue
        messages = inp.get("messages") or []
        for m in messages:
            content = m.get("content", "")
            if _CONSTRAINT_TEXT in content:
                found_in_message = True
                break
        if found_in_message:
            break
    assert found_in_message, (
        "constraint appeared in DSPy LM span text but not in a parsed "
        "messages[].content — may be in metadata instead of the actual "
        "LM prompt"
    )


def test_baseline_run_dspy_lm_spans_do_not_contain_constraint_text():
    """Baseline (no inbound constraint) MUST NOT have the constraint
    text in any DSPy LM span input.value. Proves the LM doesn't
    accidentally see the constraint from a previous test's
    orchestration leaking through.
    """
    unique_id = uuid.uuid4().hex[:12]
    query = f"unique-baseline-marker-{unique_id} what is bear grylls saying"
    session_id = f"e2e-dspy-baseline-{unique_id}"

    result = _run_process(session_id, query, None)
    il = result["final_output"]["iterative_loop"]
    assert il["inbound_constraints_applied"] == []

    spans = _query_dspy_lm_spans_with_text(unique_id, timeout_s=30.0)
    assert spans, f"no DSPy LM spans found for baseline anchor {unique_id!r}"

    # Baseline MUST NOT have the constraint anywhere.
    leaks = [s for s in spans if _CONSTRAINT_TEXT in str(s["input"])]
    assert not leaks, (
        f"baseline run's DSPy LM spans leaked the constraint text "
        f"{_CONSTRAINT_TEXT!r} — cross-session contamination?"
    )


def test_dspy_lm_spans_carry_output_value_byte_equal():
    """Sanity check: DSPy LM spans MUST have output.value populated
    (the LM's completion). Locks the OpenInference DSPy
    instrumentation surface — if a future version stops emitting
    output, our LM-output-level tests would silently lose teeth.
    """
    unique_id = uuid.uuid4().hex[:12]
    query = f"unique-output-check-{unique_id} what is bear grylls saying"
    session_id = f"e2e-dspy-out-{unique_id}"

    result = _run_process(session_id, query, None)
    _ = result

    spans = _query_dspy_lm_spans_with_text(unique_id, timeout_s=30.0)
    assert spans
    populated = [s for s in spans if s.get("output") and str(s["output"]).strip()]
    assert len(populated) == len(spans), (
        f"some DSPy LM spans had empty output.value: "
        f"{len(spans) - len(populated)}/{len(spans)} empty"
    )
