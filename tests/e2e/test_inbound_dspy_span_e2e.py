"""Phoenix DSPy LM-span byte-equal assertions.

Closes the last LM-OUTPUT-side plan gap: when a constraint is
injected, it MUST appear in the LM's prompt (input.value) on the
iteration that drained the constraint. This is byte-equal
DETERMINISTIC because the orchestrator constructs the prompt
from ``missing_aspects`` which is byte-equal deterministic.

Requires the runtime pod to be running with:
* ``OPENINFERENCE_DSPY=1`` env var
* ``openinference-instrumentation-dspy`` installed (in the runtime image)
* DSPy re-instrumented with the tenant-routing tracer provider
  Phoenix tracer (done in main.py's lifespan)

The runtime emits DSPy LM spans to the tenant's project ``cogniverse-
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
from dataclasses import dataclass

import dspy
import httpx
import pytest

RUNTIME_BASE = os.environ.get("COGNIVERSE_RUNTIME_BASE", "http://localhost:33000")
PHOENIX_BASE = os.environ.get("COGNIVERSE_PHOENIX_BASE", "http://localhost:33006")
_TENANT = "flywheel_org:production"
_CONSTRAINT_TEXT = "focus on safety equipment and protective gear"
# Keep the outer deadline above the 90s Phoenix fetch timeout so one slow
# attempt can still finish before we give up.
_PHOENIX_QUERY_TIMEOUT_S = 120.0
_PHOENIX_QUERY_SLEEP_S = 0.5


@dataclass
class _SpanLookupDiagnostics:
    phoenix_error: Exception | None = None
    chain_span_count: int = 0
    matching_chain_count: int = 0
    trace_ids: tuple[str, ...] = ()
    lm_child_count: int = 0
    matching_lm_count: int = 0
    stage: str = "chain_query_error"

    def _describe_error(self) -> str:
        if self.phoenix_error is None:
            return "unknown Phoenix error"
        return f"{type(self.phoenix_error).__name__}: {self.phoenix_error}"

    def render(self, text: str) -> str:
        if self.stage == "chain_query_error":
            return (
                f"Phoenix error while querying ChainOfThought.forward spans "
                f"for anchor {text!r}: {self._describe_error()}"
            )
        if self.stage == "chain_anchor_missing":
            message = (
                f"ChainOfThought.forward spans for anchor {text!r} had "
                f"chain_span_count={self.chain_span_count} and "
                f"matching_chain_count={self.matching_chain_count}"
            )
            if self.phoenix_error is not None:
                message += f"; last Phoenix error was {self._describe_error()}"
            return message
        if self.stage == "lm_query_error":
            message = (
                f"Phoenix error while querying LM.__call__ spans for trace_ids="
                f"{list(self.trace_ids)!r} and anchor {text!r}: "
                f"{self._describe_error()}"
            )
            if self.chain_span_count or self.matching_chain_count:
                message += (
                    f"; chain_span_count={self.chain_span_count}; "
                    f"matching_chain_count={self.matching_chain_count}"
                )
            return message
        if self.stage == "lm_anchor_missing":
            message = (
                f"LM.__call__ spans for trace_ids={list(self.trace_ids)!r} "
                f"and anchor {text!r} had lm_child_count={self.lm_child_count} "
                f"and matching_lm_count={self.matching_lm_count}"
            )
            if self.phoenix_error is not None:
                message += f"; last Phoenix error was {self._describe_error()}"
            return message
        return f"timed out searching Phoenix for {text!r}"


def _require_http_prerequisite(name: str, endpoint: str) -> None:
    try:
        with httpx.Client(timeout=2.0) as c:
            response = c.get(endpoint)
    except httpx.HTTPError as exc:
        pytest.fail(
            f"{name} prerequisite request failed after E2E stack setup; "
            f"method='GET'; url={endpoint!r}; timeout=2.0s; error={exc!r}",
            pytrace=False,
        )
    assert response.status_code == 200, (
        f"{name} prerequisite returned HTTP {response.status_code}; "
        f"method='GET'; url={endpoint!r}; body={response.text[:500]!r}"
    )


@pytest.fixture(scope="module", autouse=True)
def _require_runtime_and_phoenix() -> None:
    _require_http_prerequisite("cogniverse runtime", f"{RUNTIME_BASE}/health")
    _require_http_prerequisite("Phoenix", f"{PHOENIX_BASE}/v1/traces")


pytestmark = pytest.mark.e2e


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


def _query_dspy_lm_spans_with_text(
    text: str, timeout_s: float = _PHOENIX_QUERY_TIMEOUT_S
) -> list:
    """Query the reformulator's DSPy LM spans whose input.value contains ``text``.

    The reformulator is the ``ChainOfThought.forward`` family emitted by
    OpenInference DSPy instrumentation. We first locate that family, then
    keep only the LM child spans from the same trace so we do not match the
    sufficiency gate's unrelated LM traffic.
    """
    from datetime import datetime, timedelta, timezone

    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    px = Client(base_url=PHOENIX_BASE)
    chain_span_name = f"{dspy.ChainOfThought.__name__}.forward"
    lm_span_name = f"{dspy.LM.__name__}.__call__"
    query = SpanQuery().where(f"name == '{chain_span_name}'")
    # The tenant project accumulates spans across every run on the
    # cluster; keep the time window so the reformulator lookup stays on
    # this run's traffic.
    window_start = datetime.now(timezone.utc) - timedelta(minutes=30)
    deadline = time.time() + timeout_s
    diagnostics = _SpanLookupDiagnostics()
    while time.time() < deadline:
        try:
            chain_spans = px.spans.get_spans_dataframe(
                project_identifier=f"cogniverse-{_TENANT}",
                start_time=window_start,
                query=query,
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics.phoenix_error = exc
            diagnostics.stage = "chain_query_error"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue
        diagnostics.chain_span_count = len(chain_spans)
        if len(chain_spans) == 0:
            diagnostics.stage = "chain_anchor_missing"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue
        # ChainOfThought.forward spans carry the reformulator inputs.
        matching_chains = chain_spans[
            chain_spans["attributes.input.value"]
            .fillna("")
            .str.contains(text, na=False, regex=False)
        ]
        diagnostics.matching_chain_count = len(matching_chains)
        if len(matching_chains) == 0:
            diagnostics.stage = "chain_anchor_missing"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue

        trace_ids = tuple(sorted(set(matching_chains["context.trace_id"].dropna())))
        diagnostics.trace_ids = trace_ids
        if not trace_ids:
            diagnostics.stage = "chain_anchor_missing"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue

        try:
            lm_spans = px.spans.get_spans_dataframe(
                project_identifier=f"cogniverse-{_TENANT}",
                start_time=window_start,
                query=SpanQuery().where(f"name == '{lm_span_name}'"),
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics.phoenix_error = exc
            diagnostics.stage = "lm_query_error"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue
        diagnostics.lm_child_count = len(lm_spans)
        if len(lm_spans) == 0:
            diagnostics.stage = "lm_anchor_missing"
            time.sleep(_PHOENIX_QUERY_SLEEP_S)
            continue

        matching = lm_spans[
            lm_spans["context.trace_id"].isin(trace_ids)
            & lm_spans["attributes.input.value"]
            .fillna("")
            .str.contains(text, na=False, regex=False)
        ]
        diagnostics.matching_lm_count = len(matching)
        if len(matching) > 0:
            return [
                {
                    "input": row["attributes.input.value"],
                    "output": row["attributes.output.value"],
                    "name": row["name"],
                    "span_id": row["context.span_id"],
                    "trace_id": row["context.trace_id"],
                }
                for _, row in matching.iterrows()
            ]
        diagnostics.stage = "lm_anchor_missing"
        time.sleep(_PHOENIX_QUERY_SLEEP_S)
    raise AssertionError(diagnostics.render(text))


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
    tenant Phoenix project.
    """
    # Unique anchor — the query string uniquely identifies this test's
    # spans across the tenant project.
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

    # Query Phoenix for the reformulator trace that mentions the unique
    # query anchor, then keep the LM child spans from that trace.
    spans = _query_dspy_lm_spans_with_text(
        unique_id, timeout_s=_PHOENIX_QUERY_TIMEOUT_S
    )
    # The helper raises with a stage-specific message rather than returning an
    # empty list, so these pin what a SUCCESSFUL lookup guarantees: the exact
    # record shape, and that every returned span carried the anchor it filtered on.
    assert [sorted(s) for s in spans] == [
        ["input", "name", "output", "span_id", "trace_id"]
    ] * len(spans), spans
    assert [s for s in spans if unique_id not in str(s["input"])] == [], spans

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

    spans = _query_dspy_lm_spans_with_text(
        unique_id, timeout_s=_PHOENIX_QUERY_TIMEOUT_S
    )
    # The helper raises with a stage-specific message rather than returning an
    # empty list, so these pin what a SUCCESSFUL lookup guarantees: the exact
    # record shape, and that every returned span carried the anchor it filtered on.
    assert [sorted(s) for s in spans] == [
        ["input", "name", "output", "span_id", "trace_id"]
    ] * len(spans), spans
    assert [s for s in spans if unique_id not in str(s["input"])] == [], spans

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

    spans = _query_dspy_lm_spans_with_text(
        unique_id, timeout_s=_PHOENIX_QUERY_TIMEOUT_S
    )
    # The helper raises with a stage-specific message rather than returning an
    # empty list, so these pin what a SUCCESSFUL lookup guarantees: the exact
    # record shape, and that every returned span carried the anchor it filtered on.
    assert [sorted(s) for s in spans] == [
        ["input", "name", "output", "span_id", "trace_id"]
    ] * len(spans), spans
    assert [s for s in spans if unique_id not in str(s["input"])] == [], spans
    populated = [s for s in spans if s.get("output") and str(s["output"]).strip()]
    assert len(populated) == len(spans), (
        f"some DSPy LM spans had empty output.value: "
        f"{len(spans) - len(populated)}/{len(spans)} empty"
    )
