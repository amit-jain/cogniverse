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

The "constraint actually changes retrieval output" end-to-end
behaviour requires a real BRIGHT-shaped query against the live
gemma + Vespa stack — too slow + LM-flaky to lock byte-equal here.
That assertion is covered by the iterative-loop integration test
which exercises the SHIPPED loop body deterministically; this E2E
file scopes only the HTTP boundary contract against the running
pod.
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
    pytest.mark.integration,
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
        resp = c.get(f"{RUNTIME_BASE}/agents/orchestrator/sessions/e2e-unknown")
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
            json={"role": "user", "content": "hi", "tags": []},
        )
    assert resp.status_code == 422
    locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in locs
