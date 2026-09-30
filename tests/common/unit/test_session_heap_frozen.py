"""The per-test ``gc.collect`` walks only what tests allocate after collection."""

from __future__ import annotations

import gc
import sys
import weakref


class _Node:
    def __init__(self) -> None:
        self.peer: _Node | None = None


def _tracked_ids() -> set[int]:
    return {id(obj) for obj in gc.get_objects()}


def test_objects_alive_after_collection_are_frozen():
    this_module = sys.modules[__name__]
    made_in_test = _Node()

    tracked = _tracked_ids()

    assert id(this_module) not in tracked
    assert id(made_in_test) in tracked


def test_a_cycle_made_by_a_test_is_still_reclaimed():
    first, second = _Node(), _Node()
    first.peer, second.peer = second, first
    ref = weakref.ref(first)
    del first, second

    gc.collect()

    assert ref() is None
