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

    def test_evict_idle_destroys_outside_the_lock(self):
        """session.delete() is an un-timed gateway RPC; evict_idle must run it
        OUTSIDE the pool lock (like close_all) so a hung gateway can't freeze
        every checkout/release behind it."""
        client = _CountingClient()
        pool = SandboxSessionPool(
            client,
            config=SandboxPoolConfig(max_pool_size=4, max_idle_seconds=0.05),
        )

        lock_free_during_destroy = []
        seen = []
        pool.with_session("search_agent", lambda s: seen.append(s))
        session = seen[0]

        def _delete_checking_lock():
            # If the pool lock is acquirable here, the destroy runs off-lock.
            acquired = pool._lock.acquire(blocking=False)
            lock_free_during_destroy.append(acquired)
            if acquired:
                pool._lock.release()
            session.delete_count += 1

        session.delete = _delete_checking_lock
        time.sleep(0.1)
        assert pool.evict_idle() == 1
        assert lock_free_during_destroy == [True]


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

    def test_capacity_eviction_destroys_outside_the_lock(self):
        """The at-capacity checkout eviction must destroy the victim OUTSIDE
        the pool lock, like evict_idle/close_all — session.delete() is an
        un-timed gateway RPC; holding the lock across it would block every
        concurrent checkout/release behind a hung gateway."""
        client = _CountingClient()
        pool = SandboxSessionPool(
            client,
            config=SandboxPoolConfig(max_pool_size=1, max_idle_seconds=60.0),
        )

        seen: list = []
        pool.with_session("a", lambda s: seen.append(s))
        victim = seen[0]

        lock_free_during_destroy = []

        def _delete_checking_lock():
            acquired = pool._lock.acquire(blocking=False)
            lock_free_during_destroy.append(acquired)
            if acquired:
                pool._lock.release()
            victim.delete_count += 1

        victim.delete = _delete_checking_lock

        # Checkout for a second agent at capacity 1 → evicts 'a'.
        pool.with_session("b", lambda s: seen.append(s))

        assert lock_free_during_destroy == [True]
        assert victim.delete_count == 1
        stats = pool.stats()
        assert stats["pool_size"] == 1
        assert "a" not in stats["agents"]
        assert "b" in stats["agents"]

    def test_hung_gateway_during_eviction_does_not_block_other_checkouts(self):
        """Executable interleaving: while the evicted victim's delete() hangs,
        a concurrent checkout for another agent must complete — the hang is
        confined to the evicting thread."""
        import threading

        client = _CountingClient()
        pool = SandboxSessionPool(
            client,
            config=SandboxPoolConfig(max_pool_size=1, max_idle_seconds=60.0),
        )

        seen: list = []
        pool.with_session("a", lambda s: seen.append(s))
        victim = seen[0]

        delete_entered = threading.Event()
        release_delete = threading.Event()

        def _hanging_delete():
            delete_entered.set()
            assert release_delete.wait(timeout=10), "test hung"
            victim.delete_count += 1

        victim.delete = _hanging_delete

        evictor = threading.Thread(
            target=pool.with_session, args=("b", lambda s: None)
        )
        evictor.start()
        assert delete_entered.wait(timeout=5), "eviction never reached delete"

        # The gateway is hung mid-delete; another agent's checkout must
        # still complete promptly.
        done = threading.Event()

        def _other_checkout():
            pool.with_session("c", lambda s: None)
            done.set()

        other = threading.Thread(target=_other_checkout)
        other.start()
        assert done.wait(timeout=5), (
            "checkout blocked behind a hung gateway delete — the destroy "
            "is running under the pool lock"
        )
        release_delete.set()
        evictor.join(timeout=10)
        other.join(timeout=10)
        assert victim.delete_count == 1


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

    def test_close_all_leaves_checked_out_session_alive_until_release(self):
        """close_all during an in-flight exec (cert-rotation reconnect calls
        it via _drop_stale_pool) must not delete the session out from under
        the exec — the session drains: it stays usable until the callback
        returns, then is destroyed on release instead of re-pooled."""
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        observed = {}

        def callback(session):
            pool.close_all()
            # Still alive and usable mid-exec.
            observed["deleted_during_exec"] = session.delete_count
            observed["session"] = session
            return "exec-result"

        assert pool.with_session("coding_agent", callback) == "exec-result"

        assert observed["deleted_during_exec"] == 0
        # Destroyed exactly once on release, never re-pooled.
        assert observed["session"].delete_count == 1
        assert pool.stats()["pool_size"] == 0

    def test_close_all_destroys_idle_now_and_defers_busy_to_release(self):
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        idle_seen = []
        pool.with_session("idle_agent", lambda s: idle_seen.append(s))
        busy_entry = pool._checkout("busy_agent")

        pool.close_all()

        assert idle_seen[0].delete_count == 1
        assert busy_entry.session.delete_count == 0

        pool._release(busy_entry)
        assert busy_entry.session.delete_count == 1
        assert pool.stats()["pool_size"] == 0

    def test_closed_pool_stays_draining_and_never_repools(self):
        """After close_all the pool object is being discarded (shutdown or
        reconnect swap) — a later checkout must not park a fresh session in
        it forever; the session serves its one call and is destroyed."""
        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        pool.close_all()

        seen = []
        pool.with_session("agent", lambda s: seen.append(s))

        assert client.create_calls == 1
        assert seen[0].delete_count == 1
        assert pool.stats()["pool_size"] == 0

    def test_manager_close_tears_down_pool_and_client(self):
        """SandboxManager.close() (wired into the runtime lifespan shutdown so a
        restart doesn't orphan gateway containers) must destroy pooled sessions
        and close the client."""
        from cogniverse_runtime.sandbox_manager import SandboxManager

        client = _CountingClient()
        pool = SandboxSessionPool(client, config=SandboxPoolConfig(max_pool_size=4))
        sessions = []
        pool.with_session("a", lambda s: sessions.append(s))

        closed = {"client": False}

        class _Client:
            def close(self):
                closed["client"] = True

        mgr = SandboxManager(policy="disabled")
        mgr._pool = pool
        mgr._client = _Client()
        mgr._available = True

        mgr.close()

        assert sessions[0].delete_count == 1
        assert closed["client"] is True
        assert mgr._pool is None
        assert mgr._client is None


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


class TestManagerPoolLifecycle:
    """SandboxManager's lazy pool init and reconnect must keep exactly one
    pool, bound to the CURRENT client — a pool built on a pre-reconnect
    client keeps failing auth after a cert rotation while health looks
    green, and racing cold inits orphan pools whose live gateway sessions
    shutdown never reaps."""

    def test_concurrent_cold_init_builds_exactly_one_pool(self, monkeypatch):
        import threading
        import time as _time

        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        built: list = []

        class _SlowPool:
            def __init__(self, client, config=None, gateway_breaker=None):
                _time.sleep(0.02)
                built.append(self)
                self.client = client

            def close_all(self):
                pass

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _SlowPool)
        mgr = SandboxManager(policy="disabled")
        mgr._client = object()
        mgr._available = True

        results: list = []

        def grab():
            results.append(mgr._get_or_create_pool())

        threads = [threading.Thread(target=grab) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1
        assert len(results) == 16
        assert all(r is results[0] for r in results)

    def test_reconnect_rebuilds_pool_on_the_new_client(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_manager as sm_mod
        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        new_clients: list = []

        class _FakeClient:
            def __init__(self, endpoint=None, tls=None):
                new_clients.append(self)

        class _FakeTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                pass

        monkeypatch.setitem(
            sys.modules,
            "openshell",
            SimpleNamespace(SandboxClient=_FakeClient, TlsConfig=_FakeTls),
        )
        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "gw.invalid:19999")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(sm_mod, "_probe_gateway_endpoint", lambda ep: None)

        class _Pool:
            def __init__(self, client, config=None, gateway_breaker=None):
                self.client = client
                self.closed = False

            def close_all(self):
                self.closed = True

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _Pool)

        mgr = SandboxManager(policy="optional")
        old_client = object()
        mgr._client = old_client
        mgr._available = True
        stale = mgr._get_or_create_pool()
        assert stale is not None
        assert stale.client is old_client

        assert mgr.reconnect() is True

        assert stale.closed
        assert mgr._pool is None

        rebuilt = mgr._get_or_create_pool()
        assert rebuilt is not stale
        assert rebuilt.client is mgr._client
        assert rebuilt.client is new_clients[-1]

    def test_close_waits_for_in_flight_pool_build_and_reaps_it(self, monkeypatch):
        """close() racing a cold pool build must block on the pool lock and
        then reap the freshly built pool — an unlocked close slips past the
        builder and leaves a live pool behind on a closed manager."""
        import threading
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_pool as sp_mod
        from cogniverse_runtime.sandbox_manager import SandboxManager

        build_entered = threading.Event()
        build_release = threading.Event()
        close_all_calls: list = []

        class _BlockingPool:
            def __init__(self, client, config=None, gateway_breaker=None):
                self.config = config
                build_entered.set()
                assert build_release.wait(5)

            def close_all(self):
                close_all_calls.append(self)

        monkeypatch.setattr(sp_mod, "SandboxSessionPool", _BlockingPool)
        mgr = SandboxManager(policy="disabled")
        mgr._client = SimpleNamespace(close=lambda: None)
        mgr._available = True

        builder = threading.Thread(target=mgr._get_or_create_pool)
        builder.start()
        try:
            assert build_entered.wait(5)

            closer = threading.Thread(target=mgr.close)
            closer.start()
            closer.join(0.3)
            assert closer.is_alive(), "close() must block until the pool build ends"
        finally:
            build_release.set()
        builder.join(5)
        closer.join(5)
        assert not builder.is_alive()
        assert not closer.is_alive()
        assert mgr._pool is None
        assert len(close_all_calls) == 1
        assert mgr._client is None


class TestManagerConnectSerialization:
    """Concurrent _connect calls (cert-rotation tick vs exec-error trigger)
    must serialize, and the client each reconnect displaces must be closed
    exactly once — otherwise the loser's grpc channel leaks."""

    @pytest.fixture
    def gateway_env(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime import sandbox_manager as sm_mod

        clients: list = []

        class _FakeClient:
            def __init__(self, endpoint=None, tls=None):
                self.close_calls = 0
                clients.append(self)

            def close(self):
                self.close_calls += 1

        class _FakeTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                pass

        monkeypatch.setitem(
            sys.modules,
            "openshell",
            SimpleNamespace(SandboxClient=_FakeClient, TlsConfig=_FakeTls),
        )
        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "gw.invalid:19999")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(sm_mod, "_probe_gateway_endpoint", lambda ep: None)
        return SimpleNamespace(clients=clients, client_cls=_FakeClient)

    def test_reconnect_closes_displaced_client_exactly_once(self, gateway_env):
        from cogniverse_runtime.sandbox_manager import SandboxManager

        mgr = SandboxManager(policy="disabled")
        mgr._connect()
        mgr._connect()

        clients = gateway_env.clients
        assert len(clients) == 2
        assert mgr._client is clients[1]
        assert clients[0].close_calls == 1
        assert clients[1].close_calls == 0
        assert mgr._available is True

    def test_concurrent_connects_serialize_and_close_the_loser(
        self, gateway_env, monkeypatch
    ):
        import threading

        from cogniverse_runtime.sandbox_manager import SandboxManager

        first_entered = threading.Event()
        second_entered = threading.Event()
        release = threading.Event()
        guard = threading.Lock()

        orig_init = gateway_env.client_cls.__init__

        def blocking_init(client_self, endpoint=None, tls=None):
            orig_init(client_self, endpoint=endpoint, tls=tls)
            with guard:
                first = len(gateway_env.clients) == 1
            if first:
                first_entered.set()
                assert release.wait(5)
            else:
                second_entered.set()

        monkeypatch.setattr(gateway_env.client_cls, "__init__", blocking_init)

        mgr = SandboxManager(policy="disabled")

        t_rotation = threading.Thread(target=mgr._connect)
        t_rotation.start()
        try:
            assert first_entered.wait(5)

            t_exec_error = threading.Thread(target=mgr._connect)
            t_exec_error.start()
            assert not second_entered.wait(0.3), (
                "second _connect must wait for the first, not dial concurrently"
            )
        finally:
            release.set()
        t_rotation.join(5)
        t_exec_error.join(5)
        assert not t_rotation.is_alive()
        assert not t_exec_error.is_alive()

        clients = gateway_env.clients
        assert len(clients) == 2
        assert mgr._client is clients[1]
        assert clients[0].close_calls == 1
        assert clients[1].close_calls == 0
        assert mgr._available is True


class TestResolveTlsConfig:
    """Client mTLS certs must be read from the same OPENSHELL_CONFIG_DIR tree
    the cert rotator watches — a hardcoded home path makes connect read certs
    rotation never refreshes."""

    def test_certs_resolved_from_openshell_config_dir(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime.sandbox_manager import SandboxManager

        class _RecordingTls:
            def __init__(self, ca_path=None, cert_path=None, key_path=None):
                self.ca_path = ca_path
                self.cert_path = cert_path
                self.key_path = key_path

        monkeypatch.setitem(
            sys.modules, "openshell", SimpleNamespace(TlsConfig=_RecordingTls)
        )
        mtls = tmp_path / "gateways" / "gw-a" / "mtls"
        mtls.mkdir(parents=True)
        for name in ("ca.crt", "tls.crt", "tls.key"):
            (mtls / name).write_text("pem")
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))

        mgr = SandboxManager(policy="disabled")
        tls = mgr._resolve_tls_config()

        assert isinstance(tls, _RecordingTls)
        assert tls.ca_path == mtls / "ca.crt"
        assert tls.cert_path == mtls / "tls.crt"
        assert tls.key_path == mtls / "tls.key"

    def test_incomplete_cert_set_returns_none(self, monkeypatch, tmp_path):
        import sys
        from types import SimpleNamespace

        from cogniverse_runtime.sandbox_manager import SandboxManager

        monkeypatch.setitem(sys.modules, "openshell", SimpleNamespace(TlsConfig=object))
        mtls = tmp_path / "gateways" / "gw-a" / "mtls"
        mtls.mkdir(parents=True)
        (mtls / "ca.crt").write_text("pem")  # tls.crt / tls.key missing
        monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(tmp_path))

        mgr = SandboxManager(policy="disabled")
        assert mgr._resolve_tls_config() is None
