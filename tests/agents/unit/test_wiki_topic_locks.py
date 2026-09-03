"""Topic locks are striped: same key always serializes, memory stays bounded."""

import pytest

from cogniverse_agents.wiki.wiki_manager import (
    _TOPIC_LOCK_STRIPES,
    _topic_lock,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_same_key_always_returns_the_same_lock():
    first = _topic_lock("acme:prod", "entity_robot_arm")
    second = _topic_lock("acme:prod", "entity_robot_arm")
    assert first is second


def test_lock_population_is_bounded_regardless_of_key_count():
    """One threading.Lock per (tenant, doc_id) was retained for process
    lifetime; a tenant with many wiki topics grew the map without bound."""
    locks = {
        id(_topic_lock(f"tenant-{i}:tenant-{i}", f"doc-{j}"))
        for i in range(100)
        for j in range(100)
    }
    assert len(locks) <= _TOPIC_LOCK_STRIPES
    assert _TOPIC_LOCK_STRIPES == 256
