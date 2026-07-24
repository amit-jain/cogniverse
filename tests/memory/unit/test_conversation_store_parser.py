"""ConversationStore.get_history parses hostile row shapes without crashing.

get_history reads a Mem0 partition whose rows are untrusted: Mem0 can return
metadata as a dict OR a JSON string, and a foreign/corrupt row can carry a
non-numeric ``seq`` or a non-dict metadata payload. A single such row must be
skipped, never crash the whole read — otherwise the agent dispatcher (which
treats a raised load as "no history") silently loses a context's entire
history to one bad row. This drives the real parser through a recording stub
manager; no Mem0 infrastructure is needed to exercise the parse path.
"""

from __future__ import annotations

import json

import pytest

from cogniverse_core.conversation import ConversationStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CTX = "c1"


class _RecordingManager:
    """Returns a fixed row list from get_all_memories, like Mem0 would."""

    def __init__(self, rows):
        self._rows = rows

    def get_all_memories(self, tenant_id, agent_name):
        return list(self._rows)


def _row(memory, metadata):
    return {"memory": memory, "metadata": metadata}


def _turn(role, context_id, seq):
    return {
        "type": "conversation",
        "context_id": context_id,
        "turn_role": role,
        "seq": seq,
    }


def test_wellformed_turns_round_trip_in_seq_order():
    rows = [
        _row("[ctx:c1] [assistant] hi there", _turn("assistant", CTX, 2.0)),
        _row("[ctx:c1] [user] hello", _turn("user", CTX, 1.0)),
        # a valid row whose metadata arrives as a JSON string (Mem0 does this)
        _row("[ctx:c1] [user] third", json.dumps(_turn("user", CTX, 3.0))),
    ]
    history = ConversationStore(_RecordingManager(rows), "acme:acme").get_history(CTX)
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "third"},
    ]


def test_malformed_rows_are_skipped_not_fatal():
    rows = [
        _row("[ctx:c1] [user] hello", _turn("user", CTX, 1.0)),
        # metadata is a non-empty list — must not raise on meta.get(...)
        _row("junk", [1, 2, 3]),
        # metadata is a bare int
        _row("junk", 5),
        # metadata is a non-JSON string
        _row("junk", "{not valid json"),
        # seq is a non-numeric string — must not crash the sort with a mixed key
        _row("[ctx:c1] [user] bad-seq", _turn("user", CTX, "not-a-number")),
        # seq is None
        _row("[ctx:c1] [user] none-seq", _turn("user", CTX, None)),
        # a foreign role
        _row("[ctx:c1] [system] sys", _turn("system", CTX, 4.0)),
        # a valid assistant turn after all the junk
        _row("[ctx:c1] [assistant] hi", _turn("assistant", CTX, 2.0)),
    ]
    history = ConversationStore(_RecordingManager(rows), "acme:acme").get_history(CTX)
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_other_context_rows_do_not_leak():
    rows = [
        _row("[ctx:c1] [user] mine", _turn("user", CTX, 1.0)),
        _row("[ctx:other] [user] theirs", _turn("user", "other", 2.0)),
    ]
    history = ConversationStore(_RecordingManager(rows), "acme:acme").get_history(CTX)
    assert history == [{"role": "user", "content": "mine"}]
