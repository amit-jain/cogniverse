"""Unit tests for SandboxSessionPool reuses sessions across calls."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.sandbox_pool import (
    SandboxPoolConfig,
    SandboxSessionPool,
)


class _FakeSession:
    def __init__(self, name: str):
        self.id = name
        self.sandbox = MagicMock()
        self.sandbox.name = name
        self.delete_count = 0

    def delete(self):
        self.delete_count += 1


class _CountingClient:
    """Records create_session + wait_ready calls so tests can assert reuse."""

    def __init__(self):
        self.create_calls = 0
        self.wait_calls = 0
        self._next = 0
        self.created: list[_FakeSession] = []

    def create_session(self) -> _FakeSession:
        self._next += 1
        s = _FakeSession(name=f"sandbox-{self._next}")
        self.created.append(s)
        self.create_calls += 1
        return s

    def wait_ready(self, name: str, timeout_seconds: int = 0):
        self.wait_calls += 1


class TestPoolConfig:
    def test_defaults(self):
        cfg = SandboxPoolConfig()
        assert cfg.enabled is True
        assert cfg.max_pool_size == 8
        assert cfg.max_idle_seconds == 60.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_ENABLED", "false")
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_SIZE", "3")
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POOL_IDLE_S", "5.5")
        cfg = SandboxPoolConfig.from_environment()
        assert cfg.enabled is False
        assert cfg.max_pool_size == 3
        assert cfg.max_idle_seconds == 5.5


class TestReusePerAgent:
    def test_second_call_for_same_agent_reuses_session(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))

        sessions_seen = []
        pool.with_session("search_agent", lambda s: sessions_seen.append(s))
        pool.with_session("search_agent", lambda s: sessions_seen.append(s))

        # Same physical session both times.
        assert sessions_seen[0] is sessions_seen[1]
        # Only one create / one wait_ready in total.
        assert client.create_calls == 1
        assert client.wait_calls == 1

    def test_different_agents_get_separate_sessions(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))

        seen = {}
        pool.with_session("search_agent", lambda s: seen.setdefault("a", s))
        pool.with_session("summarizer_agent", lambda s: seen.setdefault("b", s))

        assert seen["a"] is not seen["b"]
        assert client.create_calls == 2


class TestIdleEviction:
    def test_idle_eviction_destroys_old_sessions(self):
        client = _CountingClient()
        pool = SandboxSessionPool(
            client,
            config=SandboxPoolConfig(max_pool_size=4, max_idle_seconds=0.05),
        )

        seen = []
        pool.with_session("search_agent", lambda s: seen.append(s))
        first_session = seen[0]

        # Sleep past the idle threshold; entry must evict + destroy.
        time.sleep(0.1)
        evicted = pool.evict_idle()
        assert evicted == 1
        assert first_session.delete_count == 1

        # Next checkout creates a fresh session.
        pool.with_session("search_agent", lambda s: seen.append(s))
        assert client.create_calls == 2
        assert seen[1] is not first_session


class TestCapacityCap:
    def test_oldest_idle_evicted_when_pool_full(self):
        client = _CountingClient()
        pool = SandboxSessionPool(
            client,
            config=SandboxPoolConfig(max_pool_size=2, max_idle_seconds=60.0),
        )

        # Fill capacity 2.
        s1 = []
        pool.with_session("a", lambda s: s1.append(s))
        # Tiny pause so a's last_used_at < b's
        time.sleep(0.01)
        pool.with_session("b", lambda s: s1.append(s))

        # Adding a third agent must evict 'a' (oldest idle).
        pool.with_session("c", lambda s: s1.append(s))
        stats = pool.stats()
        assert stats["pool_size"] == 2
        assert "a" not in stats["agents"]
        assert "b" in stats["agents"]
        assert "c" in stats["agents"]


class TestErrorHandling:
    def test_callback_exception_drops_pooled_session(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig())

        first_seen: list = []

        def callback_raise(s):
            first_seen.append(s)
            raise RuntimeError("simulated callback failure")

        with pytest.raises(RuntimeError, match="simulated"):
            pool.with_session("search_agent", callback_raise)

        # The errored-out session is destroyed and removed from the pool.
        assert first_seen[0].delete_count == 1
        # Next checkout creates a fresh session.
        seen = []
        pool.with_session("search_agent", lambda s: seen.append(s))
        assert seen[0] is not first_seen[0]
        assert client.create_calls == 2


class TestDisabledPool:
    def test_disabled_pool_creates_and_destroys_per_call(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(enabled=False))

        seen = []
        pool.with_session("search_agent", lambda s: seen.append(s))
        pool.with_session("search_agent", lambda s: seen.append(s))

        # Per-call: two creates, two destroys, NO entries in the pool.
        assert client.create_calls == 2
        for s in seen:
            assert s.delete_count == 1
        assert pool.stats()["pool_size"] == 0


class TestCloseAll:
    def test_close_all_destroys_every_pooled_session(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))

        sessions = []
        pool.with_session("a", lambda s: sessions.append(s))
        pool.with_session("b", lambda s: sessions.append(s))

        pool.close_all()
        for s in sessions:
            assert s.delete_count == 1
        assert pool.stats()["pool_size"] == 0


class TestReturnValue:
    def test_callback_return_value_propagates(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig())
        out = pool.with_session("agent", lambda s: {"result": "ok", "name": s.id})
        assert out == {"result": "ok", "name": "sandbox-1"}


class TestConcurrentCheckoutRace:
    def test_overprovisioned_checkout_does_not_overwrite_pooled_slot(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=8))

        e1 = pool._checkout("agent")
        e2 = pool._checkout("agent")  # e1 still in_use → over-provisioned transient

        assert e1 is not e2
        assert e1.session is not e2.session
        # The pooled slot keeps the first entry; the transient never replaces it.
        assert pool._entries["agent"] is e1

        pool._release(e2)  # transient → its session is destroyed, not pooled
        assert e2.session.delete_count == 1
        assert pool._entries["agent"] is e1

        pool._release(e1)  # canonical → pooled and reusable
        assert e1.session.delete_count == 0
        assert e1.in_use is False

        e3 = pool._checkout("agent")
        assert e3 is e1
        assert client.create_calls == 2

    def test_concurrent_checkouts_destroy_the_loser_no_leak(self):
        import threading

        create_barrier = threading.Barrier(2)
        callback_barrier = threading.Barrier(2)

        class _BarrierClient(_CountingClient):
            def create_session(self):
                # Both threads pass the empty-slot check, then create at once.
                create_barrier.wait(timeout=10)
                return super().create_session()

        client = _BarrierClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=8))

        results: list[str] = []

        def callback(session):
            results.append(session.id)
            # Hold the session in_use until BOTH callbacks are running, so the
            # second checkout observes the first as in_use and takes the
            # over-provision path deterministically.
            callback_barrier.wait(timeout=10)
            return session.id

        def worker():
            pool.with_session("agent", callback)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(results) == 2
        assert client.create_calls == 2  # both raced past the empty-slot check
        assert pool.stats()["pool_size"] == 1
        assert pool.stats()["in_use"] == 0
        # The over-provisioned loser's session is destroyed, not orphaned.
        destroyed = sum(s.delete_count for s in client.created)
        assert destroyed == 1
